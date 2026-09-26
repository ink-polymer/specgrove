#!/usr/bin/env python3
"""Matched scheduler baselines and real Poisson open-loop serving on one H20.

The TETRIS/ECHO-style labels are explicit backend adaptations: every method
uses the same DFlash proposal, nested DDTree candidates, packed verifier,
precision, row cap, prompts, and request RNG.  They are not presented as a
reproduction of the original vLLM/SGLang systems.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_continuous_tree_block_e2e import prepared_prompts, summarize
from gbv_experiments.config import load_config
from gbv_experiments.continuous_tree_block_decode import (
    generate_continuous_tree_blocks,
)
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from paper_dp_allocators import control
from rerun_audited_global_tree_matrix_h20 import OURS, matched_specs
from run_natural_paper_phase import load_natural


METHODS = ("dp", "tetris_style", "echo_style")
SEEDS = (17, 29, 43)
SCHEDULER_CELLS = ((193, 8), (193, 16), (384, 16), (384, 32))
ARRIVAL_RATES = (1.0, 2.0, 4.0)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = [
        Path(__file__), ROOT / "scripts/paper_dp_allocators.py",
        ROOT / "src/gbv_experiments/continuous_tree_block_decode.py",
        ROOT / "src/gbv_experiments/tree.py",
        ROOT / "src/gbv_experiments/sampling.py",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def jain(values):
    values = [float(value) for value in values]
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator else 0.0


def request_seed(seed, identity):
    suffix = int(hashlib.sha256(
        str(identity["source_id"]).encode()
    ).hexdigest()[:12], 16) % 1_000_000_007
    return seed * 1_000_003 + suffix


def poisson_arrivals(count, rate, seed):
    rng = random.Random(seed * 104_729 + int(rate * 1000))
    arrivals = [0.0]
    for _ in range(1, count):
        arrivals.append(arrivals[-1] + rng.expovariate(rate))
    return tuple(arrivals)


def clean_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def planning_summary(trace):
    values = [float(row["planning_ms"]) for row in trace]
    return {
        "waves": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.mean(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95) or 0.0,
        "max_ms": max(values, default=0.0),
    }


def run_method(engine, prompts, seeds, stops, spec, method, *, row_budget,
               max_new_tokens, echo_thresholds, arrivals=None,
               max_active=None):
    trace = []
    decode = dict(spec["decode"])
    if arrivals is not None:
        # Dynamic admission currently uses request-local caches and repacks at
        # every wave.  This setting is identical for all compared schedulers.
        decode["persistent_request_target_cache"] = False
    else:
        decode["persistent_request_target_cache"] = True
    with control(method, trace, echo_thresholds=echo_thresholds):
        result = generate_continuous_tree_blocks(
            engine, prompts, spec["variant"], max_new_tokens=max_new_tokens,
            stop_ids=stops, seeds=seeds,
            slot_capacity=row_budget // decode["row_cap"],
            row_budget=row_budget, layout="packed_sequence",
            arrival_times_s=arrivals, max_active_requests=max_active,
            **decode,
        )
    return result, trace


def calibrate_echo(engine, tokenizer, cfg, data_dir, stops, spec):
    prompts, identities = prepared_prompts(
        tokenizer, cfg, engine.device, data_dir, "gsm8k", 96, 16,
    )
    all_confidences = [[], []]
    for first in range(0, len(prompts), 8):
        batch = prompts[first:first + 8]
        ids = identities[first:first + 8]
        seeds = [request_seed(911, identity) for identity in ids]
        result, trace = run_method(
            engine, batch, seeds, stops, spec, "echo_style",
            row_budget=193, max_new_tokens=64,
            echo_thresholds=(0.0, 0.0),
        )
        for wave in trace:
            for request in wave.get("gate_confidences", []):
                for tier, value in enumerate(request):
                    all_confidences[tier].append(float(value))
        del result
        clean_cuda()
    thresholds = tuple(percentile(values, 0.5) for values in all_confidences)
    if any(value is None or value < 0 for value in thresholds):
        raise RuntimeError("ECHO-style disjoint calibration failed")
    return {
        "dataset": "gsm8k",
        "prompt_slice": [96, 112],
        "target_labels_used": False,
        "rule": "median draft-only marginal covered mass per added row",
        "thresholds": thresholds,
        "samples_per_gate": [len(values) for values in all_confidences],
    }


def run_scheduler_matrix(args, engine, tokenizer, cfg, stops, spec,
                         echo_calibration):
    prompts, identities = load_natural(
        tokenizer, cfg, args.data_dir, "mixed_natural",
        args.scheduler_requests,
    )
    prompts = [prompt.to(engine.device) for prompt in prompts]
    cells = SCHEDULER_CELLS if not args.quick else ((193, 8),)
    seeds_to_run = SEEDS if not args.quick else (17,)
    groups = args.output / "scheduler_baselines" / "groups"
    for row_budget, concurrency in cells:
        for seed in seeds_to_run:
            for first in range(0, len(prompts), concurrency):
                batch = prompts[first:first + concurrency]
                batch_ids = identities[first:first + concurrency]
                batch_seeds = [request_seed(seed, identity)
                               for identity in batch_ids]
                identity = {
                    "row_budget": row_budget, "concurrency": concurrency,
                    "seed": seed, "first": first,
                    "source_ids": [row["source_id"] for row in batch_ids],
                }
                key = hashlib.sha256(json.dumps(
                    identity, sort_keys=True
                ).encode()).hexdigest()[:24]
                path = groups / f"{key}.json"
                if path.exists():
                    continue
                order = list(METHODS)
                random.Random(seed * 10_007 + first + row_budget).shuffle(order)
                record = {"identity": identity, "method_order": order,
                          "runs": []}
                for method in order:
                    result, trace = run_method(
                        engine, batch, batch_seeds, stops, spec, method,
                        row_budget=row_budget, max_new_tokens=args.max_new_tokens,
                        echo_thresholds=echo_calibration["thresholds"],
                    )
                    record["runs"].append({
                        "method": method, "summary": summarize(result),
                        "planning": planning_summary(trace),
                        "allocator_trace": trace,
                        "outputs": result.outputs,
                    })
                    del result
                    clean_cuda()
                write_json(path, record)
                print(json.dumps({"event": "scheduler_group_complete",
                                  **identity}), flush=True)


def open_loop_summary(result, trace, offered_rate):
    metrics = list(result.request_metrics)
    output_tokens = sum(row["output_tokens"] for row in metrics)
    service_rates = [
        1000 * row["output_tokens"] / max(row["e2e_ms"], 1e-9)
        for row in metrics
    ]
    summary = {
        "requests": len(metrics), "offered_rate_rps": offered_rate,
        "wall_ms": result.wall_ms,
        "completed_rps": 1000 * len(metrics) / result.wall_ms,
        "tokens_per_second": 1000 * output_tokens / result.wall_ms,
        "output_tokens": output_tokens,
        "jain_request_service_rate": jain(service_rates),
        "planning": planning_summary(trace),
    }
    for field in ("queue_ms", "ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[field] for row in metrics]
        summary[field] = {
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    return summary


def run_open_loop(args, engine, tokenizer, cfg, stops, spec,
                  echo_calibration):
    prompts, identities = load_natural(
        tokenizer, cfg, args.data_dir, "mixed_natural",
        args.open_loop_requests,
    )
    prompts = [prompt.to(engine.device) for prompt in prompts]
    rates = ARRIVAL_RATES if not args.quick else (4.0,)
    seeds_to_run = SEEDS if not args.quick else (17,)
    output = args.output / "open_loop"
    for rate in rates:
        for seed in seeds_to_run:
            arrivals = poisson_arrivals(len(prompts), rate, seed)
            batch_seeds = [request_seed(seed, identity)
                           for identity in identities]
            order = list(METHODS)
            random.Random(seed * 65_537 + int(rate * 1000)).shuffle(order)
            for method in order:
                path = output / (
                    f"rate{rate:g}_seed{seed}_{method}.json"
                )
                if path.exists():
                    continue
                result, trace = run_method(
                    engine, prompts, batch_seeds, stops, spec, method,
                    row_budget=384, max_new_tokens=args.max_new_tokens,
                    echo_thresholds=echo_calibration["thresholds"],
                    arrivals=arrivals, max_active=32,
                )
                record = {
                    "method": method, "seed": seed,
                    "arrival_process": "Poisson",
                    "offered_rate_rps": rate,
                    "arrival_times_s": arrivals,
                    "source_ids": [row["source_id"] for row in identities],
                    "summary": open_loop_summary(result, trace, rate),
                    "request_metrics": result.request_metrics,
                    "allocator_trace": trace,
                    "outputs": result.outputs,
                }
                write_json(path, record)
                print(json.dumps({
                    "event": "open_loop_run_complete", "method": method,
                    "rate": rate, "seed": seed,
                    "tokens_per_second": record["summary"]["tokens_per_second"],
                    "ttft_p95_ms": record["summary"]["ttft_ms"]["p95"],
                }), flush=True)
                del result
                clean_cuda()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3_4b", "qwen3_8b"),
                        required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("scheduler", "open_loop", "all"),
                        default="all")
    parser.add_argument("--scheduler-requests", type=int, default=128)
    parser.add_argument("--open-loop-requests", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20260922)
    cfg = load_config(args.config)["model"]
    engine, tokenizer = load_models(cfg, "cuda:0")
    engine.set_target_verification_backend("eager")
    stops = stop_token_ids(engine, tokenizer)
    curve = tuple(map(tuple, json.loads(
        args.calibration.read_text()
    )["curve"]))
    spec = matched_specs(1.0, curve)[OURS]
    # Warm kernels before calibration or measurement.
    warm, warm_ids = prepared_prompts(
        tokenizer, cfg, engine.device, args.data_dir, "gsm8k", 120, 2,
    )
    warm_seeds = [request_seed(123, identity) for identity in warm_ids]
    result, _ = run_method(
        engine, warm, warm_seeds, stops, spec, "dp", row_budget=193,
        max_new_tokens=4, echo_thresholds=(0.0, 0.0),
    )
    del result
    clean_cuda()
    echo_calibration = calibrate_echo(
        engine, tokenizer, cfg, args.data_dir, stops, spec,
    )
    write_json(args.output / "echo_style_calibration.json", echo_calibration)
    write_json(args.output / "manifest.json", {
        "kind": "matched_scheduler_and_real_poisson_open_loop_serving",
        "model": cfg, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "source_hashes": source_hashes(),
        "methods": METHODS, "scheduler_cells": SCHEDULER_CELLS,
        "arrival_rates_rps": ARRIVAL_RATES,
        "echo_style_calibration": echo_calibration,
        "matched_backend": (
            "DFlash proposal + nested DDTree tiers + packed eager BF16/SDPA; "
            "FP32 proposal/posterior; no CUDA graph"
        ),
        "baseline_scope": (
            "TETRIS/ECHO-style adaptations over identical DDTree tiers; "
            "not original vLLM/SGLang system reproductions"
        ),
        "open_loop_cache": (
            "request-local cache repacking at each wave for every method; "
            "dynamic arrivals do not use the closed-batch persistent arena"
        ),
    })
    if args.phase in ("scheduler", "all"):
        run_scheduler_matrix(
            args, engine, tokenizer, cfg, stops, spec, echo_calibration,
        )
    if args.phase in ("open_loop", "all"):
        run_open_loop(
            args, engine, tokenizer, cfg, stops, spec, echo_calibration,
        )
    write_json(args.output / "complete.json", {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": args.phase, "quick": args.quick,
    })


if __name__ == "__main__":
    main()
