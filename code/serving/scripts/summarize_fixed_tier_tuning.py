#!/usr/bin/env python3
"""Select fixed DDTree tiers on a disjoint prompt split and report holdout results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


FIXED = {"fixed_tier_11": 11, "fixed_tier_23": 23, "fixed_tier_45": 45}


def throughput(runs):
    tokens = sum(run["summary"]["output_tokens"] for run in runs)
    seconds = sum(run["summary"]["wall_ms"] for run in runs) / 1000.0
    if tokens <= 0 or seconds <= 0:
        raise ValueError("Nonpositive token/time aggregate")
    return tokens / seconds


def paired_gain(groups, fixed_method):
    return 100.0 * (throughput([g["dp"] for g in groups]) /
                    throughput([g[fixed_method] for g in groups]) - 1.0)


def cluster_bootstrap(groups, fixed_method, samples, seed):
    clusters = defaultdict(list)
    for group in groups:
        case = group["case"]
        clusters[(case["dataset"], case["first"])].append(group)
    units = list(clusters.values())
    if not units:
        raise ValueError("No holdout clusters")
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        draw = [units[rng.randrange(len(units))] for _ in units]
        values.append(paired_gain([g for cluster in draw for g in cluster], fixed_method))
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return [lo, hi], len(units)


def load_groups(root):
    complete = json.loads((root / "complete.json").read_text())
    paths = sorted((root / "groups").glob("*.json"))
    if len(paths) != complete["groups"]:
        raise ValueError("Group count does not match complete.json")
    groups = []
    for path in paths:
        raw = json.loads(path.read_text())
        case = raw["case"]
        if case["phase"] != "fixed_tier_tuning" or case["requests"] != 128:
            raise ValueError(f"Unexpected case in {path}")
        runs = {run["method"]: run for run in raw["runs"]}
        if set(runs) != set(case["methods"]):
            raise ValueError(f"Method mismatch in {path}")
        for method, budget in FIXED.items():
            if method not in runs:
                continue
            summary = runs[method]["summary"]
            counts = {int(k): int(v) for k, v in
                      summary["effective_tree_budget_counts"].items()}
            if set(counts) != {budget}:
                raise ValueError(f"{method} did not remain fixed in {path}")
            if any(trace["rows"] > case["budget"] or
                   set(trace["allocation"]) != {budget}
                   for trace in runs[method]["allocation_trace"]):
                raise ValueError(f"{method} allocation trace invalid in {path}")
        groups.append({"case": case, **runs})
    return groups, complete


def aggregate_record(groups, fixed_method, bootstrap_samples, seed):
    dp_tps = throughput([g["dp"] for g in groups])
    fixed_tps = throughput([g[fixed_method] for g in groups])
    interval, clusters = cluster_bootstrap(
        groups, fixed_method, bootstrap_samples, seed)
    return {
        "selected_fixed_method": fixed_method,
        "selected_budget": FIXED[fixed_method],
        "dp_tokens_per_second": dp_tps,
        "fixed_tokens_per_second": fixed_tps,
        "dp_gain_percent": 100.0 * (dp_tps / fixed_tps - 1.0),
        "paired_prompt_cluster_95_interval_percent": interval,
        "prompt_clusters": clusters,
        "groups": len(groups),
    }


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-md", required=True, type=Path)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = ap.parse_args()
    groups, complete = load_groups(args.input)
    tune = [g for g in groups if g["case"]["first"] < 64]
    holdout = [g for g in groups if g["case"]["first"] >= 64]
    if len(tune) != len(holdout):
        raise ValueError("Expected equal first64/last64 prompt splits")

    shapes = sorted({(g["case"]["budget"], g["case"]["concurrency"])
                     for g in groups})
    report = {
        "protocol": {
            "selection": "maximize aggregate tokens/s over first 64 prompts of every dataset and all prespecified seeds; one fixed B per (model,R,C)",
            "evaluation": "last 64 prompts; selected B frozen before holdout aggregation",
            "post_hoc_oracle": "reported only as a diagnostic upper envelope and never as the formal comparator",
            "bootstrap": "paired prompt-identity clusters with all seeds kept inside each cluster",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "complete": complete,
        "shapes": {},
    }
    lines = [
        "# Tuned fixed-tier DDTree holdout",
        "",
        "A single fixed tier is selected per `(R,C)` on the first 64 prompt identities from each task, pooled across the four tasks and all prespecified seeds. The selected tier is then frozen and evaluated on the remaining 64 prompt identities. The post-hoc oracle is diagnostic only.",
        "",
        "| R | C | selected B | held-out DP tok/s | held-out fixed tok/s | DP gain | paired 95% interval | holdout clusters |",
        "|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for index, shape in enumerate(shapes):
        row_budget, concurrency = shape
        tune_shape = [g for g in tune if (g["case"]["budget"],
                     g["case"]["concurrency"]) == shape]
        hold_shape = [g for g in holdout if (g["case"]["budget"],
                     g["case"]["concurrency"]) == shape]
        candidates = sorted(set(FIXED).intersection(tune_shape[0]),
                            key=lambda method: FIXED[method])
        tuning_tps = {method: throughput([g[method] for g in tune_shape])
                      for method in candidates}
        selected = max(candidates, key=lambda method: (tuning_tps[method],
                                                       -FIXED[method]))
        heldout_tps = {method: throughput([g[method] for g in hold_shape])
                       for method in candidates}
        oracle = max(candidates, key=lambda method: heldout_tps[method])
        overall = aggregate_record(
            hold_shape, selected, args.bootstrap_samples, 20260922 + index)
        per_dataset = {
            dataset: aggregate_record(
                [g for g in hold_shape if g["case"]["dataset"] == dataset],
                selected, args.bootstrap_samples, 20261022 + index * 10 + j)
            for j, dataset in enumerate(sorted({g["case"]["dataset"]
                                                 for g in hold_shape}))
        }
        key = f"R{row_budget}_C{concurrency}"
        report["shapes"][key] = {
            "tuning_tokens_per_second": tuning_tps,
            "selected_fixed_method": selected,
            "selected_budget": FIXED[selected],
            "heldout": overall,
            "heldout_by_dataset": per_dataset,
            "diagnostic_holdout_oracle": {
                "method": oracle,
                "budget": FIXED[oracle],
                "tokens_per_second": heldout_tps[oracle],
                "dp_gain_percent": paired_gain(hold_shape, oracle),
            },
        }
        lo, hi = overall["paired_prompt_cluster_95_interval_percent"]
        lines.append(
            f"| {row_budget} | {concurrency} | {FIXED[selected]} | "
            f"{overall['dp_tokens_per_second']:.1f} | "
            f"{overall['fixed_tokens_per_second']:.1f} | "
            f"{overall['dp_gain_percent']:+.1f}% | [{lo:+.1f}, {hi:+.1f}] | "
            f"{overall['prompt_clusters']} |")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
