#!/usr/bin/env python3
"""Localize Target-posterior drift across four verification execution paths.

This is a numerical correctness diagnostic, not a throughput benchmark.  For
each frozen teacher-forced prefix and proposal tree it compares:

1. one-token causal AR,
2. an unpacked single-request tree forward,
3. a packed single-request tree forward, and
4. a packed multi-request tree forward.

The same probes can be repeated with the default SDPA dispatcher and the math
SDPA backend.  The latter retains BF16 model inputs/weights while using the
more stable math attention path (including FP32 intermediates for BF16).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_ours_greedy_same_prefix_h20 import clone_cache, same_bytes
from benchmark_continuous_tree_block_e2e import prepared_prompts
from gbv_experiments.config import Variant, load_config
from gbv_experiments.continuous_tree_block_decode import (
    ContinuousDecodeRequest,
    _pack_cache_sequence,
    _packed_block_inputs,
    _prefix_tree,
    _propose_many,
)
from gbv_experiments.engine import load_models
from gbv_experiments.terminal_formal import allocation_gate, model_gate
from gbv_experiments.tree_block_skip import TreeBlock


def distribution_gap(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Return stable logit and posterior discrepancies for one vocabulary row."""
    left = reference.detach().float().cpu()
    right = candidate.detach().float().cpu()
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Expected equal one-dimensional vocabulary logits")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise FloatingPointError("Non-finite logits in posterior audit")
    delta = (left - right).abs()
    p = torch.softmax(left.double(), dim=-1)
    q = torch.softmax(right.double(), dim=-1)
    probability_delta = (p - q).abs()
    return {
        "max_logit_error": float(delta.max()),
        "mean_logit_error": float(delta.mean()),
        "rms_logit_error": float(delta.square().mean().sqrt()),
        "total_variation": float(probability_delta.sum() / 2),
        "max_probability_error": float(probability_delta.max()),
        "mean_probability_error": float(probability_delta.mean()),
        "top1_agreement": int(left.argmax()) == int(right.argmax()),
        "reference_top1": int(left.argmax()),
        "candidate_top1": int(right.argmax()),
    }


def logit_summary(logits: torch.Tensor) -> dict:
    values, indices = logits.detach().float().cpu().topk(5)
    return {
        "top1": int(indices[0]),
        "top5_tokens": [int(value) for value in indices],
        "top5_logits": [float(value) for value in values],
        "top1_margin": float(values[0] - values[1]),
    }


@contextmanager
def attention_backend(name: str):
    if name == "default":
        with nullcontext():
            yield
        return
    if name != "math":
        raise ValueError(f"Unknown attention backend: {name}")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        yield


def parse_positive_csv(value: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    minimum = 0 if allow_zero else 1
    if not result or any(item < minimum for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integers")
    return result


def frozen_prefix(engine, prompt: torch.Tensor, extra_tokens: int) -> tuple[torch.Tensor, list[int]]:
    """Create a deterministic teacher-forced prefix and its next root token."""
    cache = engine.cache_factory()
    output = engine.target_forward(prompt, cache, hidden=False, last_only=True)
    logits = output.logits[0, -1]
    generated = []
    for index in range(extra_tokens + 1):
        token = int(logits.argmax())
        generated.append(token)
        if index == extra_tokens:
            break
        ids = torch.tensor([[token]], dtype=torch.long, device=engine.device)
        positions = torch.tensor([[cache.get_seq_length()]], device=engine.device)
        output = engine.target_forward(
            ids, cache, hidden=False, positions=positions, last_only=True,
        )
        logits = output.logits[0, -1]
    if extra_tokens:
        additions = torch.tensor(
            [generated[:-1]], dtype=torch.long, device=engine.device,
        )
        prefix = torch.cat((prompt, additions), dim=1)
    else:
        prefix = prompt
    return prefix, generated


def proposal_tree(engine, prompt, prefix, generated, maximum_rows):
    cache = engine.cache_factory()
    initial = engine.target_forward(prefix, cache, hidden=True, last_only=True)
    state = ContinuousDecodeRequest(0, prompt, 17)
    state.target_cache = cache
    state.full_features = engine.features(initial.hidden_states)
    state.generated = list(generated)
    variant = Variant(
        name="posterior_path_probe", method="ddtree", length=15, paths=1,
        temperature=0.0, draft_temperature=1.0,
        probability_dtype="float32", tree_budget=maximum_rows - 1,
    )
    _propose_many(
        engine, [state], variant, proposal_probability_dtype="float32",
    )
    if state.tree is None or len(state.tree.parents) < maximum_rows:
        raise RuntimeError("Draft did not expose the requested tree rows")
    return state.tree


def local_tree(tree, rows: int):
    if rows < 1 or rows > len(tree.parents):
        raise ValueError("Requested tree row count is unavailable")
    if rows == 1:
        from gbv_experiments.tree import Tree
        return Tree(tokens=[], parents=[-1], depths=[0], path_nodes=[])
    return _prefix_tree(tree, rows - 1)


def as_block(tree) -> TreeBlock:
    rows = len(tree.parents)
    return TreeBlock(
        root=0,
        nodes=tuple(range(rows)),
        parents=tuple(tree.parents),
        tokens=tuple(tree.tokens),
        depths=tuple(tree.depths),
    )


def base_cache(engine, prefix):
    cache = engine.cache_factory()
    engine.target_hidden_forward(prefix, cache)
    return cache


def project_target_logits(engine, hidden: torch.Tensor) -> torch.Tensor:
    head = engine.target.get_output_embeddings()
    return head(hidden.to(dtype=head.weight.dtype))


def detach_target_output_head(engine, dtype: str) -> None:
    """Raise projection precision without converting tied input embeddings."""
    original = engine.target.get_output_embeddings()
    detached = copy.deepcopy(original).to(dtype=getattr(torch, dtype))
    engine.target.set_output_embeddings(detached)
    if engine.target.get_input_embeddings().weight.dtype != next(
        parameter.dtype for parameter in engine.target.model.parameters()
        if parameter.is_floating_point()
    ):
        raise RuntimeError("Detached output head changed input-embedding precision")


def causal_ar_logits(engine, cache, prefix_length: int, root: int):
    ids = torch.tensor([[root]], dtype=torch.long, device=engine.device)
    positions = torch.tensor([[prefix_length]], device=engine.device)
    output = engine.target_hidden_forward(
        ids, clone_cache(engine, cache), positions=positions,
    )
    return project_target_logits(engine, output.last_hidden_state)[0, -1].float()


def unpacked_tree_logits(engine, cache, prefix_length: int, root: int, tree, *, all_nodes=False):
    ids = torch.tensor(
        [[root] + list(tree.tokens)], dtype=torch.long, device=engine.device,
    )
    positions = (
        torch.tensor([tree.depths], dtype=torch.long, device=engine.device)
        + prefix_length
    )
    mask = tree.mask(
        prefix_length, next(engine.target.parameters()).dtype, engine.device,
    )
    output = engine.target_hidden_forward(
        ids, clone_cache(engine, cache), positions=positions, mask=mask,
    )
    logits = project_target_logits(engine, output.last_hidden_state)[0]
    return logits.float() if all_nodes else logits[0].float()


def ancestor_path(tree, node: int) -> list[int]:
    if not 0 <= node < len(tree.parents) or tree.parents[0] != -1:
        raise ValueError("Invalid tree node or root")
    path = [node]
    while node:
        parent = tree.parents[node]
        if not 0 <= parent < node:
            raise ValueError("Tree parent must precede child")
        node = parent
        path.append(node)
    path.reverse()
    return path


def check_packed_structure(items, packed, lengths, offsets, total, ids, positions, mask, queries):
    expected = torch.zeros_like(mask[0, 0], dtype=torch.bool)
    for item, length, offset, query in zip(items, lengths, offsets, queries):
        tree = item["tree"]
        if length != item["prefix_length"]:
            raise RuntimeError("Packed prefix length mismatch")
        if ids[0, query:query + len(tree.parents)].tolist() != [item["root"]] + list(tree.tokens):
            raise RuntimeError("Packed token mismatch")
        for node in range(len(tree.parents)):
            path = ancestor_path(tree, node)
            if tree.depths[node] != len(path) - 1 or int(positions[0, query + node]) != length + len(path) - 1:
                raise RuntimeError("Packed position ID mismatch")
            expected[query + node, offset:offset + length] = True
            expected[query + node, [total + query + parent for parent in path]] = True
        for original, combined in zip(item["cache"].layers, packed.layers):
            if not same_bytes(original.keys, combined.keys[..., offset:offset + length, :]) or not same_bytes(original.values, combined.values[..., offset:offset + length, :]):
                raise RuntimeError("Packed KV contents mismatch")
    if not torch.equal(torch.isfinite(mask[0, 0]), expected):
        raise RuntimeError("Packed ancestor attention support mismatch")
    if not bool((mask[0, 0][expected] == 0).all()) or not bool(torch.isneginf(mask[0, 0][~expected]).all()):
        raise RuntimeError("Packed attention mask values mismatch")


def causal_tree_logits(engine, cache, prefix_length, root, tree):
    rows = []
    for node in range(len(tree.parents)):
        path = ancestor_path(tree, node)
        tokens = [root] + [tree.tokens[parent - 1] for parent in path[1:]]
        replay = clone_cache(engine, cache)
        for depth, token in enumerate(tokens):
            output = engine.target_hidden_forward(
                torch.tensor([[token]], device=engine.device), replay,
                positions=torch.tensor([[prefix_length + depth]], device=engine.device),
            )
        rows.append(project_target_logits(engine, output.last_hidden_state)[0, -1].float())
    return torch.stack(rows)


def packed_tree_logits(engine, items, *, all_nodes=False, check_structure=False):
    states, blocks, caches = [], [], []
    for index, item in enumerate(items):
        cache = clone_cache(engine, item["cache"])
        state = ContinuousDecodeRequest(index, item["prompt"], 17 + index)
        state.target_cache = cache
        state.round_prefix_len = item["prefix_length"]
        state.generated = [item["root"]]
        state.tree = item["tree"]
        states.append(state)
        blocks.append(as_block(item["tree"]))
        caches.append(cache)
    packed, lengths, offsets, total = _pack_cache_sequence(
        caches, engine.cache_factory, engine.device,
    )
    ids, positions, mask, queries = _packed_block_inputs(
        states, blocks, lengths, offsets, total,
        next(engine.target.parameters()).dtype, engine.device,
    )
    if check_structure:
        check_packed_structure(items, packed, lengths, offsets, total, ids, positions, mask, queries)
    output = engine.target_hidden_forward(
        ids, packed, positions=positions, mask=mask,
    )
    logits = project_target_logits(engine, output.last_hidden_state)[0]
    roots = [logits[query].float() for query in queries]
    for slot, item in enumerate(items):
        expected = lengths[slot] + 1
        actual = int(torch.isfinite(mask[0, 0, queries[slot]]).sum())
        if actual != expected:
            raise RuntimeError("Packed root attention support is not request-local")
    if all_nodes:
        return [logits[query:query + len(item["tree"].parents)].float()
                for query, item in zip(queries, items)]
    return roots


def aggregate(records: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for record in records:
        for comparison, values in record["comparisons"].items():
            buckets[(record["attention_backend"], record["rows"], comparison)].append(values)
    result = []
    for (backend, rows, comparison), values in sorted(buckets.items()):
        result.append({
            "attention_backend": backend,
            "rows": rows,
            "comparison": comparison,
            "probes": len(values),
            "top1_agreement_fraction": sum(v["top1_agreement"] for v in values) / len(values),
            "maximum_logit_error": max(v["max_logit_error"] for v in values),
            "mean_of_mean_logit_error": sum(v["mean_logit_error"] for v in values) / len(values),
            "maximum_total_variation": max(v["total_variation"] for v in values),
            "mean_total_variation": sum(v["total_variation"] for v in values) / len(values),
            "maximum_probability_error": max(v["max_probability_error"] for v in values),
        })
    return result


def timing_summary(milliseconds: list[float]) -> dict:
    if not milliseconds or any(value <= 0 for value in milliseconds):
        raise ValueError("Timing samples must be positive")
    ordered = sorted(milliseconds)
    return {
        "samples": len(ordered),
        "median_ms": statistics.median(ordered),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "mean_ms": statistics.mean(ordered),
    }


def benchmark_packed_wave(engine, probes, rows: int, repeats: int) -> dict:
    """Time one production-shaped packed Target wave with stable source caches."""
    if repeats < 1 or len(probes) < 2:
        raise ValueError("Packed-wave timing needs repeats and multiple probes")
    items = []
    for probe in probes:
        items.append(probe | {
            "cache": base_cache(engine, probe["prefix"]),
            "tree": local_tree(probe["tree"], rows),
        })
    for _ in range(3):
        packed_tree_logits(engine, items)
    torch.cuda.synchronize(engine.device)
    samples = []
    torch.cuda.reset_peak_memory_stats(engine.device)
    for _ in range(repeats):
        torch.cuda.synchronize(engine.device)
        started = time.perf_counter()
        packed_tree_logits(engine, items)
        torch.cuda.synchronize(engine.device)
        samples.append(1000 * (time.perf_counter() - started))
    result = timing_summary(samples)
    result.update({
        "scope": "synchronized packed wave including cache clone/pack, inputs, Target, and LM head",
        "slots": len(items),
        "rows_per_slot": rows,
        "total_target_rows": len(items) * rows,
        "target_dtype": str(next(engine.target.parameters()).dtype),
        "peak_allocated_bytes_during_timing": torch.cuda.max_memory_allocated(engine.device),
        "raw_ms": samples,
    })
    return result


def parameter_dtypes(model) -> list[str]:
    return sorted({
        str(parameter.dtype)
        for parameter in model.parameters()
        if parameter.is_floating_point()
    })


def diagnostic_model_gate(engine, dtype: str) -> dict:
    """Keep the formal BF16 gate; explicitly gate FP32 diagnostics separately."""
    if dtype == "bfloat16":
        model_gate(engine)
        return {"kind": "registered_formal_model_gate", "passed": True}
    expected = torch.float32
    observed = {
        "target": parameter_dtypes(engine.target),
        "draft": parameter_dtypes(engine.draft),
    }
    if observed != {"target": [str(expected)], "draft": [str(expected)]}:
        raise RuntimeError(f"FP32 diagnostic model dtype mismatch: {observed}")
    return {
        "kind": "diagnostic_fp32_dtype_gate",
        "passed": True,
        "observed_parameter_dtypes": observed,
        "formal_benchmark_eligible": False,
    }


def target_verification_precision_gate(engine, dtype: str) -> dict:
    """Certify the post-proposal Target precision used by this diagnostic."""
    expected_target = str(getattr(torch, dtype))
    observed = {
        "target": parameter_dtypes(engine.target),
        "draft": parameter_dtypes(engine.draft),
    }
    if observed["target"] != [expected_target]:
        raise RuntimeError(
            f"Target verification dtype mismatch: expected {expected_target}, "
            f"observed {observed['target']}"
        )
    return {
        "kind": "diagnostic_target_verification_precision_gate",
        "passed": True,
        "observed_parameter_dtypes": observed,
        "draft_was_used_before_target_precision_conversion": True,
        "formal_benchmark_eligible": dtype == "bfloat16",
    }


def target_head_precision_gate(engine, body_dtype: str, head_dtype: str) -> dict:
    """Verify an isolated higher-precision LM head without hidden body changes."""
    head = engine.target.get_output_embeddings()
    head_ids = {id(parameter) for parameter in head.parameters()}
    observed_head = sorted({
        str(parameter.dtype) for parameter in head.parameters()
        if parameter.is_floating_point()
    })
    observed_body = sorted({
        str(parameter.dtype) for parameter in engine.target.parameters()
        if parameter.is_floating_point() and id(parameter) not in head_ids
    })
    expected = {
        "body": [str(getattr(torch, body_dtype))],
        "head": [str(getattr(torch, head_dtype))],
    }
    observed = {"body": observed_body, "head": observed_head}
    if observed != expected:
        raise RuntimeError(
            f"Target head diagnostic dtype mismatch: expected {expected}, "
            f"observed {observed}"
        )
    return {
        "kind": "diagnostic_target_head_precision_gate",
        "passed": True,
        "observed_parameter_dtypes": observed,
        "formal_benchmark_eligible": body_dtype == head_dtype == "bfloat16",
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--prefix-lengths", default="0,16,32")
    parser.add_argument("--rows", default="1,12,24,46")
    parser.add_argument("--multi-slots", type=int, default=4)
    parser.add_argument("--attention-backends", default="default,math")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    parser.add_argument(
        "--target-verification-dtype", choices=("bfloat16", "float16", "float32"),
        help=(
            "optionally convert only Target after prefixes and proposal trees "
            "are frozen; Draft remains at the loaded dtype"
        ),
    )
    parser.add_argument(
        "--target-head-dtype", choices=("bfloat16", "float32"),
        help="optionally convert only the Target LM head after probes are frozen",
    )
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument("--audit-descendants", action="store_true",
                        help="compare every tree node against serial ancestor-path AR")
    parser.add_argument(
        "--disable-bf16-reduced-precision-reduction", action="store_true",
        help=(
            "force full FP32 accumulation for BF16 GEMM reductions while "
            "retaining BF16 model parameters and layer outputs"
        ),
    )
    args = parser.parse_args()
    prefix_lengths = parse_positive_csv(args.prefix_lengths, allow_zero=True)
    rows = parse_positive_csv(args.rows)
    backends = tuple(item for item in args.attention_backends.split(",") if item)
    if (
        args.prompts < 1 or args.multi_slots < 2
        or args.prompts % args.multi_slots
        or any(name not in {"default", "math"} for name in backends)
        or max(rows) > 46
        or args.timing_repeats < 0
    ):
        raise ValueError("Invalid probe shape")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite posterior-path audit")

    cfg = dict(load_config(args.config.resolve())["model"])
    if args.dtype is not None:
        cfg["dtype"] = args.dtype
    if (
        args.target_head_dtype is not None
        and args.target_verification_dtype is not None
        and args.target_verification_dtype != cfg["dtype"]
    ):
        raise ValueError("Target-head and whole-Target precision overrides conflict")
    if args.disable_bf16_reduced_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    allocation_gate("cuda:0")
    engine, tokenizer = load_models(cfg, "cuda:0")
    runtime_gate = diagnostic_model_gate(engine, cfg["dtype"])
    engine.set_target_verification_backend("eager")
    prompts, identities = prepared_prompts(
        tokenizer, cfg, engine.device, args.data_dir,
        args.dataset, args.offset, args.prompts,
    )

    probes = []
    with attention_backend("default"):
        # Prefix-major ordering makes every packed multi-request group contain
        # distinct natural prompts at the same generation depth.
        for extra in prefix_lengths:
            for prompt, identity in zip(prompts, identities):
                prefix, generated = frozen_prefix(engine, prompt, extra)
                tree = proposal_tree(engine, prompt, prefix, generated, max(rows))
                probes.append({
                    "prompt": prompt,
                    "identity": identity,
                    "prefix": prefix,
                    "prefix_extra_tokens": extra,
                    "prefix_length": int(prefix.shape[1]),
                    "generated": generated,
                    "root": generated[-1],
                    "tree": tree,
                })
                print(
                    json.dumps({"prepared": len(probes), "identity": identity,
                                "prefix_extra_tokens": extra}), flush=True,
                )

    target_verification_dtype = args.target_verification_dtype or cfg["dtype"]
    if target_verification_dtype != cfg["dtype"]:
        engine.target.to(dtype=getattr(torch, target_verification_dtype))
        torch.cuda.empty_cache()
    if args.target_head_dtype is not None:
        detach_target_output_head(engine, args.target_head_dtype)
        torch.cuda.empty_cache()
        target_precision_gate = target_head_precision_gate(
            engine, target_verification_dtype, args.target_head_dtype,
        )
    else:
        target_precision_gate = target_verification_precision_gate(
            engine, target_verification_dtype,
        )
    timing = None
    if args.timing_repeats:
        with attention_backend("default"):
            timing = benchmark_packed_wave(
                engine, probes[:args.multi_slots], max(rows), args.timing_repeats,
            )

    records = []
    for backend in backends:
        with attention_backend(backend):
            for start in range(0, len(probes), args.multi_slots):
                group = probes[start:start + args.multi_slots]
                if len(group) != args.multi_slots:
                    raise RuntimeError("Incomplete packed multi-request group")
                cached = []
                for probe in group:
                    cached.append(probe | {"cache": base_cache(engine, probe["prefix"])})
                for row_count in rows:
                    items = [item | {"tree": local_tree(item["tree"], row_count)} for item in cached]
                    multi_logits = packed_tree_logits(
                        engine, items, all_nodes=args.audit_descendants, check_structure=True,
                    )
                    for item, packed_multi in zip(items, multi_logits):
                        ar = (causal_tree_logits(
                            engine, item["cache"], item["prefix_length"], item["root"], item["tree"],
                        ) if args.audit_descendants else causal_ar_logits(
                            engine, item["cache"], item["prefix_length"], item["root"],
                        ))
                        unpacked = unpacked_tree_logits(
                            engine, item["cache"], item["prefix_length"],
                            item["root"], item["tree"], all_nodes=args.audit_descendants,
                        )
                        packed_single = packed_tree_logits(
                            engine, [item], all_nodes=args.audit_descendants, check_structure=True,
                        )[0]
                        node_rows = (
                            zip(ar, unpacked, packed_single, packed_multi)
                            if args.audit_descendants
                            else [(ar, unpacked, packed_single, packed_multi)]
                        )
                        for node, (ar, unpacked, packed_single, packed_multi) in enumerate(node_rows):
                            comparisons = {
                                "ar_vs_unpacked_tree": distribution_gap(ar, unpacked),
                                "ar_vs_packed_single": distribution_gap(ar, packed_single),
                                "ar_vs_packed_multi": distribution_gap(ar, packed_multi),
                                "unpacked_vs_packed_single": distribution_gap(unpacked, packed_single),
                                "packed_single_vs_packed_multi": distribution_gap(packed_single, packed_multi),
                            }
                            records.append({
                                "tree_node": node,
                                "ancestor_path": ancestor_path(item["tree"], node),
                                "identity": item["identity"],
                                "prefix_extra_tokens": item["prefix_extra_tokens"],
                                "prefix_length": item["prefix_length"],
                                "root_token": item["root"],
                                "rows": row_count,
                                "multi_slots": args.multi_slots,
                                "attention_backend": backend,
                                "logits": {
                                    "ar": logit_summary(ar),
                                    "unpacked_tree": logit_summary(unpacked),
                                    "packed_single": logit_summary(packed_single),
                                    "packed_multi": logit_summary(packed_multi),
                                },
                                "comparisons": comparisons,
                                "prefix_cache_clone_byte_equal": True,
                                "position_ids_semantically_matched": True,
                                "packed_root_attention_support_checked": True,
                            })
                    print(
                        json.dumps({"backend": backend, "rows": row_count,
                                    "records": len(records)}), flush=True,
                    )

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "src/gbv_experiments/engine.py",
        ROOT / "src/gbv_experiments/continuous_tree_block_decode.py",
        ROOT / "scripts/audit_ours_greedy_same_prefix_h20.py",
    ]
    report = {
        "kind": "same_prefix_target_posterior_path_audit",
        "model": cfg,
        "precision": parameter_dtypes(engine.target),
        "runtime_gate": runtime_gate,
        "target_verification_precision_gate": target_precision_gate,
        "packed_wave_timing": timing,
        "attention_backends": {
            "default": "PyTorch SDPA dispatcher",
            "math": "math SDPA with BF16 inputs and FP32 attention intermediates",
        },
        "cuda_matmul": {
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "scope": {
            "dataset": args.dataset,
            "offset": args.offset,
            "prompts": args.prompts,
            "prefix_extra_tokens": list(prefix_lengths),
            "tree_rows": list(rows),
            "multi_slots": args.multi_slots,
            "loaded_model_dtype": cfg["dtype"],
            "target_verification_dtype": target_verification_dtype,
            "target_head_dtype": (
                args.target_head_dtype or target_verification_dtype
            ),
            "draft_dtype": str(next(engine.draft.parameters()).dtype),
            "draft_evaluated_after_target_conversion": False,
            "temperature": 1.0,
            "node_scope": "all" if args.audit_descendants else "root",
            "all_node_mask_position_kv_checks": True,
            "claim": "numerical localization only; not end-to-end sequence-law certification",
        },
        "frozen_prefixes": [
            {
                "identity": probe["identity"],
                "prefix_extra_tokens": probe["prefix_extra_tokens"],
                "generated_tokens_through_root": probe["generated"],
                "tree": {
                    "tokens": list(probe["tree"].tokens),
                    "parents": list(probe["tree"].parents),
                    "depths": list(probe["tree"].depths),
                },
            }
            for probe in probes
        ],
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "records": records,
        "aggregate": aggregate(records),
        "strict_sequence_law_certified": False,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps({
        "records": len(records),
        "all_logits_finite": True,
        "prefix_cache_clone_byte_equal": True,
        "packed_root_attention_support_checked": True,
        "strict_sequence_law_certified": False,
    }, indent=2) + "\n")
    print(json.dumps({"complete": True, "records": len(records),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
