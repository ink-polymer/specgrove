#!/usr/bin/env python3
"""Localize same-prefix path drift layer by layer on known BF16 flip cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from audit_bf16_posterior_paths_h20 import (
    as_block,
    attention_backend,
    base_cache,
    distribution_gap,
    frozen_prefix,
    local_tree,
    project_target_logits,
    proposal_tree,
)
from audit_ours_greedy_same_prefix_h20 import clone_cache
from benchmark_continuous_tree_block_e2e import prepared_prompts
from gbv_experiments.config import load_config
from gbv_experiments.continuous_tree_block_decode import (
    ContinuousDecodeRequest,
    _pack_cache_sequence,
    _packed_block_inputs,
)
from gbv_experiments.engine import load_models
from gbv_experiments.terminal_formal import allocation_gate, model_gate


def parse_cases(value: str) -> tuple[tuple[int, int], ...]:
    try:
        cases = tuple(
            tuple(int(part) for part in item.split(":"))
            for item in value.split(",") if item
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected offset:extra pairs") from error
    if (
        not cases or any(len(case) != 2 or min(case) < 0 for case in cases)
        or len(set(cases)) != len(cases)
    ):
        raise argparse.ArgumentTypeError("Expected unique nonnegative offset:extra pairs")
    return cases


def hidden_gap(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    left = reference.detach().float().cpu()
    right = candidate.detach().float().cpu()
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Expected equal hidden vectors")
    delta = (left - right).abs()
    denominator = float(left.norm())
    cosine = torch.nn.functional.cosine_similarity(left, right, dim=0)
    return {
        "byte_equal": bool(torch.equal(left, right)),
        "max_error": float(delta.max()),
        "mean_error": float(delta.mean()),
        "rms_error": float(delta.square().mean().sqrt()),
        "relative_l2_error": float((left - right).norm()) / denominator if denominator else None,
        "cosine_similarity": float(cosine),
    }


def root_trace(output, row: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    if output.hidden_states is None:
        raise RuntimeError("Target did not return hidden states")
    states = [hidden[0, row] for hidden in output.hidden_states]
    final = output.last_hidden_state[0, row]
    return states, final


def forward_paths(engine, items, slot: int, rows: int) -> dict:
    local_items = [item | {"tree": local_tree(item["tree"], rows)} for item in items]
    selected = local_items[slot]
    root = selected["root"]
    prefix_length = selected["prefix_length"]
    ids = torch.tensor([[root]], dtype=torch.long, device=engine.device)
    positions = torch.tensor([[prefix_length]], dtype=torch.long, device=engine.device)
    ar_output = engine.target_hidden_forward(
        ids, clone_cache(engine, selected["cache"]), positions=positions,
    )

    tree = selected["tree"]
    tree_ids = torch.tensor(
        [[root] + list(tree.tokens)], dtype=torch.long, device=engine.device,
    )
    tree_positions = (
        torch.tensor([tree.depths], dtype=torch.long, device=engine.device)
        + prefix_length
    )
    tree_mask = tree.mask(
        prefix_length, next(engine.target.parameters()).dtype, engine.device,
    )
    unpacked_output = engine.target_hidden_forward(
        tree_ids, clone_cache(engine, selected["cache"]),
        positions=tree_positions, mask=tree_mask,
    )

    def packed_output(packed_items):
        states, blocks, caches = [], [], []
        for index, item in enumerate(packed_items):
            state = ContinuousDecodeRequest(index, item["prompt"], 17 + index)
            state.target_cache = clone_cache(engine, item["cache"])
            state.round_prefix_len = item["prefix_length"]
            state.generated = [item["root"]]
            state.tree = item["tree"]
            states.append(state)
            blocks.append(as_block(item["tree"]))
            caches.append(state.target_cache)
        packed, lengths, offsets, total = _pack_cache_sequence(
            caches, engine.cache_factory, engine.device,
        )
        packed_ids, packed_positions, mask, queries = _packed_block_inputs(
            states, blocks, lengths, offsets, total,
            next(engine.target.parameters()).dtype, engine.device,
        )
        output = engine.target_hidden_forward(
            packed_ids, packed, positions=packed_positions, mask=mask,
        )
        return output, queries

    single_output, single_queries = packed_output([selected])
    multi_output, multi_queries = packed_output(local_items)
    return {
        "ar": root_trace(ar_output, 0),
        "unpacked_tree": root_trace(unpacked_output, 0),
        "packed_single": root_trace(single_output, single_queries[0]),
        "packed_multi": root_trace(multi_output, multi_queries[slot]),
    }


def summarize_layerwise(comparisons: list[dict]) -> list[dict]:
    keys = sorted({
        (row["attention_backend"], row["rows"], row["comparison"])
        for row in comparisons
    })
    summary = []
    for backend, rows, comparison in keys:
        selected = [
            row for row in comparisons
            if row["attention_backend"] == backend
            and row["rows"] == rows and row["comparison"] == comparison
        ]
        nonzero = [row["layer"] for row in selected if not row["gap"]["byte_equal"]]
        summary.append({
            "attention_backend": backend,
            "rows": rows,
            "comparison": comparison,
            "cases": len({row["case"] for row in selected}),
            "first_nonzero_layer": min(nonzero) if nonzero else None,
            "maximum_hidden_error": max(row["gap"]["max_error"] for row in selected),
            "maximum_relative_l2_error": max(
                row["gap"]["relative_l2_error"] or 0.0 for row in selected
            ),
        })
    return summary


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--cases", type=parse_cases, required=True)
    parser.add_argument("--rows", default="1,12,46")
    parser.add_argument("--multi-slots", type=int, default=8)
    parser.add_argument("--attention-backends", default="default,math")
    args = parser.parse_args()
    rows = tuple(int(item) for item in args.rows.split(",") if item)
    backends = tuple(item for item in args.attention_backends.split(",") if item)
    if (
        args.output.exists() or args.multi_slots < 2
        or not rows or min(rows) < 1 or max(rows) > 46
        or any(name not in {"default", "math"} for name in backends)
    ):
        raise ValueError("Invalid layerwise audit controls")

    cfg = dict(load_config(args.config.resolve())["model"])
    if cfg.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("Layerwise audit is pinned to the formal BF16 model")
    allocation_gate("cuda:0")
    engine, tokenizer = load_models(cfg, "cuda:0")
    model_gate(engine)
    engine.set_target_verification_backend("eager")

    records = []
    projection_records = []
    comparisons = (
        ("ar", "unpacked_tree"),
        ("ar", "packed_single"),
        ("ar", "packed_multi"),
        ("unpacked_tree", "packed_single"),
        ("packed_single", "packed_multi"),
    )
    for case_offset, extra in args.cases:
        group_offset = case_offset - case_offset % args.multi_slots
        slot = case_offset - group_offset
        prompts, identities = prepared_prompts(
            tokenizer, cfg, engine.device, args.data_dir,
            args.dataset, group_offset, args.multi_slots,
        )
        probes = []
        with attention_backend("default"):
            for prompt, identity in zip(prompts, identities):
                prefix, generated = frozen_prefix(engine, prompt, extra)
                probes.append({
                    "prompt": prompt,
                    "identity": identity,
                    "prefix": prefix,
                    "prefix_length": int(prefix.shape[1]),
                    "root": generated[-1],
                    "tree": proposal_tree(engine, prompt, prefix, generated, max(rows)),
                })
        if probes[slot]["identity"]["source_id"] is None:
            raise RuntimeError("Missing case identity")
        for backend in backends:
            with attention_backend(backend):
                cached = [probe | {"cache": base_cache(engine, probe["prefix"])} for probe in probes]
                for row_count in rows:
                    traces = forward_paths(engine, cached, slot, row_count)
                    case_name = f"{case_offset}:{extra}:{probes[slot]['identity']['source_id']}"
                    for reference, candidate in comparisons:
                        left_states, left_final = traces[reference]
                        right_states, right_final = traces[candidate]
                        if len(left_states) != len(right_states):
                            raise RuntimeError("Hidden-state depth mismatch")
                        for layer, (left, right) in enumerate(zip(left_states, right_states)):
                            records.append({
                                "case": case_name,
                                "attention_backend": backend,
                                "rows": row_count,
                                "comparison": f"{reference}_vs_{candidate}",
                                "layer": layer,
                                "gap": hidden_gap(left, right),
                            })
                        left_logits = project_target_logits(engine, left_final.unsqueeze(0))[0].float()
                        right_logits = project_target_logits(engine, right_final.unsqueeze(0))[0].float()
                        projection_records.append({
                            "case": case_name,
                            "attention_backend": backend,
                            "rows": row_count,
                            "comparison": f"{reference}_vs_{candidate}",
                            "gap": distribution_gap(left_logits, right_logits),
                        })
                    print(json.dumps({
                        "case": case_name, "backend": backend, "rows": row_count,
                    }), flush=True)

    source_paths = [Path(__file__).resolve(), ROOT / "scripts/audit_bf16_posterior_paths_h20.py"]
    report = {
        "kind": "bf16_same_prefix_layerwise_path_audit",
        "model": cfg,
        "scope": {
            "dataset": args.dataset,
            "cases": [list(case) for case in args.cases],
            "rows": list(rows),
            "multi_slots": args.multi_slots,
            "attention_backends": list(backends),
            "claim": "diagnostic localization only",
        },
        "layerwise_records": records,
        "projection_records": projection_records,
        "aggregate": summarize_layerwise(records),
        "source_hashes": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "strict_sequence_law_certified": False,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps({
        "complete": True,
        "cases": len(args.cases),
        "layerwise_records": len(records),
        "projection_records": len(projection_records),
    }, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
