#!/usr/bin/env python3
"""Matched common-entry reference-verifier audit, distinct from old tuning.

The decoder's ancestral_reference branch is shared by all three methods.
Existing production verifier implementations and defaults remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
DD = "ddtree_full46"
DF = "dflash_r16"
OURS = "ours_global_hetero_11_23_45_h20_curve384_tierfit_a8_c1_propfp32"
NAMES = (DD, DF, OURS)


def reference_tree_verify(parents, tokens, values, generator=None, **_kwargs):
    from gbv_experiments.sampling import tree_verify_ancestral_batched
    return tree_verify_ancestral_batched(
        parents, tokens, values, generator, validate=False,
    )


def reference_chain_verify(path, values, generator=None):
    nodes, _tokens, bonus = reference_tree_verify(
        [-1] + list(range(path.numel())), path.tolist(), values, generator,
    )
    return len(nodes), bonus


def matched_specs(temperature, curve):
    from tune_cross_request_architectures import method_specs
    source = method_specs(temperature)
    specs = {}
    for name in NAMES:
        original = source[name]
        decode = dict(original["decode"])
        decode.update(
            verifier="greedy" if temperature == 0 else "ancestral_reference",
            proposal_probability_dtype="float32",
            reuse_request_draft_cache=False,
        )
        if name == OURS:
            decode["global_tree_row_cost_curve"] = tuple(curve)
        specs[name] = {
            "family": original["family"],
            "variant": replace(
                original["variant"], probability_dtype="float32",
                reuse_draft_cache=False,
            ),
            "decode": decode,
        }
    return specs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def first_difference(left, right):
    for position, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"position": position, "method_token": a, "reference_token": b}
    if len(left) != len(right):
        return {"position": min(len(left), len(right)), "lengths": [len(left), len(right)]}
    return None


@torch.inference_mode()
def calibration(engine, tokenizer, cfg):
    from gbv_experiments.conversation import encode_messages
    from gbv_experiments.continuous_tree_block_decode import (
        ContinuousDecodeRequest, _pack_cache_sequence, _packed_block_inputs,
    )
    from gbv_experiments.tree import Tree
    from gbv_experiments.tree_block_skip import local_tree_block

    states = []
    for index in range(32):
        text = (f"Calibration only: explain binary search invariant {index}. " * 20)
        ids = encode_messages(tokenizer, [{"role": "user", "content": text}], cfg, str(engine.device))[:, -64:]
        cache = engine.cache_factory()
        output = engine.target_forward(ids, cache, hidden=False, last_only=True)
        state = ContinuousDecodeRequest(index, ids, index)
        state.target_cache = cache
        state.generated = [int(output.logits[0, -1].argmax())]
        state.round_prefix_len = 64
        states.append(state)
    samples = {}
    cases = {}
    for rows in (12, 46, 96, 193, 384):
        counts = [12] * (rows // 12) + ([rows % 12] if rows % 12 else [])
        selected = states[:len(counts)]
        blocks = []
        for state, count in zip(selected, counts):
            tree = Tree(
                tokens=[int(state.generated[0])] * (count - 1),
                parents=[-1] + list(range(count - 1)),
                depths=list(range(count)), path_nodes=[],
            )
            state.tree = tree
            blocks.append(local_tree_block(tree.parents, tree.tokens, tree.depths, 0, count))
        packed, lengths, offsets, old = _pack_cache_sequence(
            [s.target_cache for s in selected], engine.cache_factory, engine.device,
        )
        ids, positions, mask, _ = _packed_block_inputs(
            selected, blocks, lengths, offsets, old,
            next(engine.target.parameters()).dtype, engine.device,
        )
        cases[rows] = (packed, old, ids, positions, mask)
        samples[rows] = []
    for repeat in range(15):
        order = list(cases)
        order = order[repeat % len(order):] + order[:repeat % len(order)]
        for rows in order:
            packed, old, ids, positions, mask = cases[rows]
            packed.crop(old)
            engine.sync()
            started = time.perf_counter()
            output = engine.target_hidden_forward(ids, packed, positions=positions, mask=mask)
            logits = engine.target.get_output_embeddings()(output.last_hidden_state)
            values = logits.float().softmax(-1)
            engine.sync()
            elapsed = 1000 * (time.perf_counter() - started)
            if repeat >= 5:
                samples[rows].append(elapsed)
            del output, logits, values
    # Only a monotonic measurement envelope, no evaluation-token tuning.
    curve = []
    for rows in sorted(samples):
        median = statistics.median(samples[rows])
        curve.append((rows, max(median, curve[-1][1] if curve else median)))
    return {
        "curve": curve, "samples_ms": samples,
        "scope": "independent synthetic 64-token prefixes, <=12-row chains; Target+head+FP32 softmax",
        "warmup": 5, "repeats": 10,
        "warning": "row-only proxy; not a universal model for arbitrary history length/tree topology",
    }


@torch.inference_mode()
def isolation_audit(engine, tokenizer, cfg):
    from gbv_experiments.conversation import encode_messages
    from gbv_experiments.continuous_tree_block_decode import (
        ContinuousDecodeRequest, _pack_cache_sequence, _packed_block_inputs,
    )
    from gbv_experiments.tree import Tree
    from gbv_experiments.tree_block_skip import local_tree_block

    prompts = [encode_messages(tokenizer, [{"role": "user", "content":
        f"Isolation test {i}: explain a sort invariant. " * 30}], cfg, str(engine.device))[:, -(64 + i * 7):]
        for i in range(4)]

    def execute(poison):
        states, blocks = [], []
        for i, (ids, rows) in enumerate(zip(prompts, (12, 24, 46, 12))):
            if poison and i == 3:
                ids = torch.roll(ids, 7, 1)
            cache = engine.cache_factory()
            engine.target_forward(ids, cache, hidden=False, last_only=True)
            state = ContinuousDecodeRequest(i, ids, i)
            state.target_cache = cache
            state.generated = [1 if not poison or i != 3 else 2]
            state.round_prefix_len = int(ids.shape[1])
            token_offset = 100 if poison and i == 3 else 0
            tree = Tree(tokens=[token_offset + token for token in range(1, rows)],
                parents=[-1] + [0] * (rows - 1), depths=[0] + [1] * (rows - 1), path_nodes=[])
            state.tree = tree
            states.append(state)
            blocks.append(local_tree_block(tree.parents, tree.tokens, tree.depths, 0, rows))
        cache, lengths, offsets, old = _pack_cache_sequence(
            [s.target_cache for s in states], engine.cache_factory, engine.device,
        )
        ids, positions, mask, queries = _packed_block_inputs(
            states, blocks, lengths, offsets, old,
            next(engine.target.parameters()).dtype, engine.device,
        )
        output = engine.target_hidden_forward(ids, cache, positions=positions, mask=mask)
        return engine.target.get_output_embeddings()(output.last_hidden_state).float(), queries

    baseline, offsets = execute(False)
    poisoned, second_offsets = execute(True)
    assert offsets == second_offsets
    split = offsets[-1]
    error = float((baseline[:, :split] - poisoned[:, :split]).abs().max())
    change = float((baseline[:, split:] - poisoned[:, split:]).abs().max())
    report = dict(rows=[12, 24, 46, 12], prefix_lengths=[64, 71, 78, 85],
        unaffected_max_absolute_error=error, poisoned_max_absolute_change=change,
        passed=error == 0 and change > 0)
    if not report["passed"]:
        raise RuntimeError(f"Heterogeneous request isolation failed: {report}")
    return report


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seeds", default="17,29")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party/ddtree_pinned"))
    from ddtree import ddtree_generate
    from dflash import dflash_generate
    from gbv_experiments.config import load_config
    from gbv_experiments.engine import load_models
    from gbv_experiments.runner import stop_token_ids
    import gbv_experiments.continuous_tree_block_decode as decoder
    from gbv_experiments.terminal_formal import allocation_gate, model_gate
    from benchmark_continuous_tree_block_e2e import prepared_prompts, summarize
    from tune_cross_request_architectures import median_summary, analyze, generate_batched_target_ar

    if args.repeats < 2 or args.max_new_tokens < 16:
        raise ValueError("Requires >=2 repeats and >=16 tokens")
    cfg = load_config(args.config.resolve())["model"]
    seeds = tuple(int(x) for x in args.seeds.split(","))
    allocation_gate("cuda:0")
    engine, tokenizer = load_models(cfg, "cuda:0")
    model_gate(engine)
    engine.set_target_verification_backend("eager")
    stops = stop_token_ids(engine, tokenizer)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__).resolve(), ROOT / "src/gbv_experiments/continuous_tree_block_decode.py",
            ROOT / "src/gbv_experiments/sampling.py", ROOT / "src/gbv_experiments/tree.py",
            ROOT / "third_party/ddtree_pinned/ddtree.py", ROOT / "third_party/ddtree_pinned/dflash.py"]}
    write_json(args.output / "source_hashes.json", sources)
    write_json(args.output / "request_isolation.json", isolation_audit(engine, tokenizer, cfg))
    cost = calibration(engine, tokenizer, cfg)
    write_json(args.output / "calibration.json", cost)
    print("calibration", cost["curve"], flush=True)
    all_prompts, all_ids = prepared_prompts(tokenizer, cfg, engine.device, args.data_dir, "gsm8k", 0, 32)
    warm = all_prompts[:2]
    native_rows = []
    for temperature in (0, 1):
        def native_run(name, prompt, seed, cap):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            engine.sync()
            started = time.perf_counter()
            kwargs = dict(model=engine.draft, target=engine.target, input_ids=prompt,
                mask_token_id=engine.draft.mask_token_id, max_new_tokens=cap,
                block_size=16, stop_token_ids=stops, temperature=temperature)
            result = ddtree_generate(**kwargs, tree_budget=45) if name == "native_ddtree" else dflash_generate(**kwargs)
            engine.sync()
            wall_ms = 1000 * (time.perf_counter() - started)
            tokens = result.output_ids[0, result.num_input_tokens:].tolist()
            return {"wall_ms": wall_ms, "output_tokens": len(tokens),
                "target_calls": result.decode_rounds, "outputs": tokens}
        for name in ("native_ddtree", "native_dflash"):
            native_run(name, warm[0], 17, 4)
        for index in range(4):
            for seed in seeds:
                request_seed = seed * 1_000_003 + index * 104_729
                samples = {n: [] for n in ("native_ddtree", "native_dflash")}
                for repeat in range(args.repeats):
                    order = list(samples)
                    if (repeat + index) % 2:
                        order.reverse()
                    for name in order:
                        samples[name].append(native_run(name, all_prompts[index], request_seed, args.max_new_tokens))
                native_rows.append(dict(temperature=temperature, identity=all_ids[index],
                    seed=seed, samples=samples))
                write_json(args.output / "native_c1.json", native_rows)
        for budget in (96, 193, 384):
            output = args.output / f"t{temperature}_r{budget}"
            if (output / "analysis.json").is_file():
                print("skip complete", output.name, flush=True)
                continue
            specs = matched_specs(temperature, cost["curve"])
            manifest = dict(kind="matched_reference_global_tree_audit", model=cfg,
                temperature=temperature, row_budget=budget, concurrencies=[1, 4, 8, 16, 32],
                seeds=seeds, repeats=args.repeats, max_new_tokens=args.max_new_tokens,
                dataset="gsm8k", c1_independent_prompts=4,
                actual_verifier="greedy" if temperature == 0 else "ancestral_all_rows_common_reference",
                verifier_entry_point="tree_verify_ancestral_batched via common ancestral_reference branch",
                proposal_probability_dtype="float32", target_probability_dtype="float32",
                actual_draft_kv_cache=False, persistent_request_target_cache=True,
                layout="packed_sequence", timing="end_to_end_including_prefill_first_draft_and_sync",
                calibration=cost, source_hashes=sources,
                methods={n: {"variant": s["variant"].__dict__, "decode": s["decode"]} for n, s in specs.items()},
                exact_token_certified=False, task_quality_evaluated=False)
            write_json(output / "manifest.json", manifest)

            def execute(name, prompts, request_seeds, cap):
                spec = specs[name]
                return decoder.generate_continuous_tree_blocks(
                    engine, prompts, spec["variant"], max_new_tokens=cap,
                    stop_ids=stops, seeds=request_seeds,
                    slot_capacity=budget // spec["decode"]["row_cap"],
                    row_budget=budget, layout="packed_sequence",
                    persistent_request_target_cache=True, **spec["decode"],
                )
            for name in NAMES:
                execute(name, warm, [17, 29], 4)
            rows = []
            with (output / "results.jsonl").open("w") as stream:
                ordinal = 0
                for concurrency in (1, 4, 8, 16, 32):
                    groups = [[i] for i in range(4)] if concurrency == 1 else [list(range(concurrency))]
                    for indices in groups:
                        prompts = [all_prompts[i] for i in indices]
                        for seed in seeds:
                            request_seeds = [seed * 1_000_003 + i * 104_729 for i in indices]
                            runs, outputs = {n: [] for n in NAMES}, {n: [] for n in NAMES}
                            for repeat in range(args.repeats):
                                shift = (ordinal + repeat) % len(NAMES)
                                for name in NAMES[shift:] + NAMES[:shift]:
                                    result = execute(name, prompts, request_seeds, args.max_new_tokens)
                                    runs[name].append(summarize(result))
                                    outputs[name].append(result.outputs)
                            ar = None
                            if temperature == 0:
                                ar = generate_batched_target_ar(engine, tokenizer, prompts,
                                    max_new_tokens=args.max_new_tokens, stop_ids=stops,
                                    seeds=request_seeds, target_temperature=0).outputs
                            row = dict(dataset="gsm8k", concurrency=concurrency, seed=seed,
                                prompt_identities=[all_ids[i] for i in indices],
                                methods={n: median_summary(runs[n]) for n in NAMES}, samples=runs, outputs=outputs,
                                within_method_repeatable={n: all(v == outputs[n][0] for v in outputs[n]) for n in NAMES},
                                greedy_ar_outputs=ar,
                                first_difference_from_greedy_ar={n: [first_difference(a, b) for a, b in zip(outputs[n][0], ar)]
                                    if ar is not None else None for n in NAMES})
                            if not all(row["within_method_repeatable"].values()):
                                raise RuntimeError("Non-repeatable method in audit cell")
                            rows.append(row)
                            stream.write(json.dumps(row, default=str) + "\n")
                            stream.flush()
                            print(output.name, "C", concurrency, "seed", seed, "cell", len(rows), "/16", flush=True)
                            ordinal += 1
            analysis = analyze(rows, specs, (1, 4, 8, 16, 32))
            analysis["all_methods_repeatable"] = True
            analysis["paired_cells"] = len(rows)
            analysis["greedy_ar_exact_requests"] = {
                name: sum(d is None for row in rows for d in (row["first_difference_from_greedy_ar"][name] or []))
                for name in NAMES}
            analysis["greedy_ar_total_requests"] = sum(len(row["prompt_identities"]) for row in rows) if temperature == 0 else 0
            write_json(output / "analysis.json", analysis)
            print("complete", output.name, analysis["combined"]["ours_vs_ddtree"],
                analysis["combined"]["ours_vs_dflash"], flush=True)


if __name__ == "__main__":
    main()
