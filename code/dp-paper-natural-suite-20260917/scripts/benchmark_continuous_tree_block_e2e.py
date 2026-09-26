#!/usr/bin/env python3
"""Run a real padded-KV continuous DDTree full46/cap24 H20 pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import torch

from gbv_experiments.config import Variant, load_config
from gbv_experiments.continuous_tree_block_decode import (
    generate_continuous_tree_blocks,
)
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.terminal_formal import allocation_gate, model_gate, telemetry


TOPICS = (
    "binary search invariants", "numerical cancellation", "TCP congestion",
    "red black trees", "virtual memory", "database indexes", "causal attention",
    "dynamic programming", "garbage collection", "the CAP theorem",
    "mutexes and semaphores", "hash collisions", "the central limit theorem",
    "authentication and authorization", "quicksort worst cases", "bias variance",
)


def prompts(tokenizer, cfg, device, count, minimum_tokens):
    result = []
    repeats = max(12, (minimum_tokens + 23) // 24)
    for index in range(count):
        text = (f"Explain {TOPICS[index]} carefully with an example and a failure case. "
                * repeats)
        ids = encode_messages(
            tokenizer, [{"role": "user", "content": text}], cfg, str(device),
        )
        if ids.shape[1] < minimum_tokens:
            raise RuntimeError("Prompt construction did not reach the requested length")
        result.append(ids[:, -minimum_tokens:].contiguous())
    return result


def prepared_prompts(tokenizer, cfg, device, data_dir, dataset, offset, count):
    path = data_dir / f"{dataset}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = rows[offset:offset + count]
    if len(selected) != count:
        raise ValueError("Prepared-data slice is incomplete")
    encoded = [
        encode_messages(
            tokenizer,
            [{"role": "user", "content": row["prompt"]}],
            cfg,
            str(device),
        )
        for row in selected
    ]
    identities = [
        {
            "dataset": row["dataset"],
            "source_id": row["source_id"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_tokens": int(ids.shape[1]),
        }
        for row, ids in zip(selected, encoded)
    ]
    return encoded, identities


def summarize(result):
    continuation = sum(
        round_["continuation_blocks"]
        for request in result.request_rounds for round_ in request
    )
    rounds = sum(len(request) for request in result.request_rounds)
    return {
        "wall_ms": result.wall_ms,
        "output_tokens": result.output_tokens,
        "tokens_per_second": 1000 * result.output_tokens / result.wall_ms,
        "physical_target_calls": result.physical_target_calls,
        "physical_target_calls_per_output_token": (
            result.physical_target_calls / result.output_tokens
        ),
        "physical_target_rows": result.physical_target_rows,
        "useful_target_rows": result.useful_target_rows,
        "padded_target_rows": result.padded_target_rows,
        "padding_fraction": result.padded_target_rows / result.physical_target_rows,
        "logical_rounds": rounds,
        "continuation_blocks": continuation,
        "continuation_probability": continuation / rounds if rounds else 0.0,
        "gated_continuations": result.gated_continuations,
        "effective_row_cap_counts": dict(result.effective_row_cap_counts),
        "effective_tree_budget_counts": dict(
            result.effective_tree_budget_counts
        ),
        "stage_ms": {
            "prefill": result.prefill_ms,
            "proposal": result.proposal_ms,
            "target": result.target_ms,
            "kv_pack": result.pack_ms,
            "select_commit": result.select_commit_ms,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--candidate-cap", type=int, default=24)
    parser.add_argument("--target-row-budget", type=int, default=193)
    parser.add_argument(
        "--layout", choices=("padded", "packed_sequence"), default="padded",
    )
    parser.add_argument(
        "--strict-shape-reference", action="store_true",
        help=(
            "run both methods at padded [full slots, 46 rows]; this audits "
            "shape-induced BF16 drift and is not a performance configuration"
        ),
    )
    args = parser.parse_args()
    using_data = args.data_dir is not None or args.dataset is not None
    if (args.output.exists() or args.requests < 2
            or (not using_data and args.requests > len(TOPICS))
            or args.max_new_tokens < 4 or args.prefix_tokens < 16
            or args.repeats < 2 or args.offset < 0
            or args.candidate_cap < 1 or args.candidate_cap > 46
            or args.target_row_budget < 46
            or (args.strict_shape_reference and args.layout != "padded")
            or using_data != (args.data_dir is not None and args.dataset is not None)):
        raise ValueError("Invalid controls or output already exists")

    cfg = load_config(args.config.resolve())["model"]
    allocation_gate(args.device)
    engine, tokenizer = load_models(cfg, args.device)
    model_gate(engine)
    if "H20" not in torch.cuda.get_device_name(engine.device):
        raise RuntimeError("This pilot is pinned to NVIDIA H20")
    if using_data:
        request_prompts, prompt_identities = prepared_prompts(
            tokenizer, cfg, engine.device, args.data_dir.resolve(),
            args.dataset, args.offset, args.requests,
        )
    else:
        request_prompts = prompts(
            tokenizer, cfg, engine.device, args.requests, args.prefix_tokens,
        )
        prompt_identities = [
            {"dataset": "synthetic", "source_id": str(index),
             "prompt_sha256": None, "prompt_tokens": int(ids.shape[1])}
            for index, ids in enumerate(request_prompts)
        ]
    seeds = [args.seed + 104729 * index for index in range(args.requests)]
    variant = Variant(
        name="continuous_ddtree_t1",
        method="ddtree",
        paths=1,
        length=15,
        temperature=1.0,
        draft_temperature=1.0,
        probability_dtype="float64",
        tree_budget=45,
    )
    full_slots = args.target_row_budget // 46
    candidate_slots = args.target_row_budget // args.candidate_cap
    if args.strict_shape_reference:
        candidate_slots = full_slots

    stop_ids = stop_token_ids(engine, tokenizer)

    def execute(method):
        return generate_continuous_tree_blocks(
            engine, request_prompts, variant,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
            seeds=seeds,
            row_cap=46 if method == "full46" else args.candidate_cap,
            slot_capacity=full_slots if method == "full46" else candidate_slots,
            layout=args.layout,
            row_budget=args.target_row_budget,
            physical_row_width=46 if args.strict_shape_reference else None,
        )

    # Warm both physical shapes before timing.  The first model invocation can
    # otherwise make whichever method runs first look several times slower.
    warm_prompts = request_prompts[:2]
    warm_seeds = seeds[:2]
    for row_cap, slots in (
            (args.candidate_cap, candidate_slots), (46, full_slots)):
        generate_continuous_tree_blocks(
            engine, warm_prompts, variant,
            max_new_tokens=4,
            stop_ids=stop_ids,
            seeds=warm_seeds,
            row_cap=row_cap,
            slot_capacity=slots,
            layout=args.layout,
            row_budget=args.target_row_budget,
            physical_row_width=46 if args.strict_shape_reference else None,
        )

    runs = {"full46": [], "cap24": []}
    outputs = {"full46": [], "cap24": []}
    for repeat in range(args.repeats):
        order = ("full46", "cap24") if repeat % 2 == 0 else ("cap24", "full46")
        for method in order:
            result = execute(method)
            runs[method].append(summarize(result))
            outputs[method].append(result.outputs)

    def median_summary(method):
        rows = runs[method]
        representative = dict(rows[0])
        for field in (
                "wall_ms", "tokens_per_second", "physical_target_calls",
                "physical_target_calls_per_output_token", "physical_target_rows",
                "useful_target_rows", "padded_target_rows", "padding_fraction",
                "logical_rounds", "continuation_blocks", "continuation_probability"):
            representative[field] = statistics.median(row[field] for row in rows)
        representative["stage_ms"] = {
            field: statistics.median(row["stage_ms"][field] for row in rows)
            for field in rows[0]["stage_ms"]
        }
        return representative

    full_summary = median_summary("full46")
    cap_summary = median_summary("cap24")
    report = {
        "kind": "continuous_tree_block_h20_pilot",
        "formal_complete": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "model": cfg["target"],
            "draft": cfg["draft"],
            "temperature": 1.0,
            "draft_temperature": 1.0,
            "target_backend": "eager BF16 SDPA",
            "cuda_graph": False,
            "different_prompts": True,
            "prepared_dataset": args.dataset,
            "integrated_decode_loop": True,
            "kv_layout": args.layout,
            "full_tree_rows": 46,
            "candidate_cap": args.candidate_cap,
            "target_row_budget": args.target_row_budget,
            "full46_target_slots": full_slots,
            "candidate_target_slots": candidate_slots,
            "strict_shape_reference": args.strict_shape_reference,
        },
        "controls": vars(args) | {"config": str(args.config), "output": str(args.output)},
        "prompt_identities": prompt_identities,
        "full46": full_summary,
        "cap24": cap_summary,
        "comparisons": {
            "aggregate_throughput_speedup": (
                cap_summary["tokens_per_second"] / full_summary["tokens_per_second"]
            ),
            "target_stage_speedup": (
                full_summary["stage_ms"]["target"] / cap_summary["stage_ms"]["target"]
            ),
            "physical_target_call_reduction": (
                1 - cap_summary["physical_target_calls"]
                / full_summary["physical_target_calls"]
            ),
            "within_method_outputs_repeatable": {
                method: all(value == values[0] for value in values)
                for method, values in outputs.items()
            },
            "cap24_vs_full46_outputs_exactly_equal_by_repeat": [
                outputs["cap24"][index] == outputs["full46"][index]
                for index in range(args.repeats)
            ],
            "finite_precision_interpretation": (
                "Output equality is diagnostic only. Different BF16 execution shapes may "
                "produce different computed Target conditionals even though the cascade "
                "is exact for fixed conditionals."
            ),
        },
        "repeat_summaries": runs,
        "telemetry": telemetry(args.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
