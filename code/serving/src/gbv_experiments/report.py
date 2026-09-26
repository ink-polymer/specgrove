from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .common import digest, read_jsonl, write_json
from .data import evaluation_coverage, evaluation_policy
from .runner import key


def expected_keys(manifest):
    return {(e["variant"]["name"], dataset, source_id, seed)
            for e in manifest["variants"] for dataset, source_id, _ in manifest["prompt_ids"]
            for seed in manifest["seeds"]}


def validate_results(manifest, records, scores=None, allow_partial=False):
    prompt_keys = [(d, i) for d, i, _ in manifest["prompt_ids"]]
    if len(set(prompt_keys)) != len(prompt_keys):
        raise ValueError("Duplicate prompt IDs in run manifest")
    if manifest["coverage"] == "fixed_evaluation_subset":
        policy = evaluation_policy(manifest["dataset_names"], manifest.get("evaluation"))
        if evaluation_coverage(policy) != "fixed_evaluation_subset":
            raise ValueError("Missing fixed-subset evaluation policy")
        counts = dict(Counter(d for d, _, _ in manifest["prompt_ids"]))
        if counts != policy["counts"]:
            raise ValueError("Run prompt counts do not match the fixed evaluation subset")
        data_manifest = manifest.get("data_manifest", {})
        if data_manifest.get("evaluation") != policy or data_manifest.get("coverage") != manifest["coverage"]:
            raise ValueError("Run and prepared data selection policies do not match")
        selected = [(name, ident) for name in manifest["dataset_names"]
                    for ident in data_manifest["datasets"][name]["selected_source_ids"]]
        if prompt_keys != selected:
            raise ValueError("Run prompts do not match the prepared sample IDs and order")
    expected = expected_keys(manifest)
    actual = {key(r) for r in records}
    if len(actual) != len(records) or actual - expected:
        raise ValueError("Duplicate or unexpected result keys")
    if not allow_partial and (actual != expected or manifest["coverage"] not in {"full_evaluation_split", "fixed_evaluation_subset"}):
        raise ValueError(f"Formal report requires every evaluation record: {len(actual)}/{len(expected)}")
    if (actual == expected
            and manifest.get("method_order", {}).get("policy") == "balanced_rotation"):
        names = {entry["variant"]["name"] for entry in manifest["variants"]}
        positions = defaultdict(lambda: {name:[] for name in names})
        scheduled = defaultdict(list)
        for record in records:
            name = record["variant"]
            position = record.get("method_execution_position")
            ordinal = record.get("method_order_ordinal")
            if not isinstance(position, int) or not isinstance(ordinal, int):
                raise ValueError("Missing balanced method-order evidence")
            positions[record["dataset"]][name].append(position)
            scheduled[record["dataset"], record["source_id"], record["seed"]].append(
                (name, position, ordinal)
            )
        for group in scheduled.values():
            if ({name for name, _, _ in group} != names
                    or {position for _, position, _ in group} != set(range(len(names)))
                    or len({ordinal for _, _, ordinal in group}) != 1):
                raise ValueError("Incomplete or inconsistent balanced method rotation")
        for dataset, by_name in positions.items():
            for values in by_name.values():
                counts = [values.count(position) for position in range(len(names))]
                if max(counts) - min(counts) > 1:
                    raise ValueError(
                        f"Method execution positions are not balanced within dataset {dataset}"
                    )
    hashes = {(d, i): h for d, i, h in manifest["prompt_ids"]}
    for r in records:
        if r["run_id"] != manifest["run_id"] or r["prompt_sha256"] != hashes[r["dataset"], r["source_id"]]:
            raise ValueError("Run/prompt hash mismatch")
        expected_turns = manifest.get("dataset_turn_counts", {}).get(r["dataset"], 1)
        if r.get("turn_count", 1) != expected_turns:
            raise ValueError("Missing conversation turn")
        turns = r.get("turn_results", [r])
        if len(turns) != expected_turns:
            raise ValueError("Missing conversation turn results")
        if len(r["generated_token_ids"]) != r["generated_tokens"] or r["decode_tokens"] != r["generated_tokens"] - expected_turns:
            raise ValueError("Generated token count mismatch")
        if any(t["generated_tokens"] < 1 or t["generated_tokens"] > manifest["max_new_tokens"] for t in turns):
            raise ValueError("Invalid generation length")
        if expected_turns > 1:
            if [t["turn_index"] for t in turns] != list(range(1, expected_turns + 1)):
                raise ValueError("Invalid conversation turn order")
            for field in ("generated_tokens", "decode_tokens", "prefill_ms", "decode_ms", "e2e_ms"):
                if not np.isclose(r[field], sum(t[field] for t in turns)):
                    raise ValueError("Conversation metric aggregation mismatch")
            if r["generated_token_ids"] != [x for t in turns for x in t["generated_token_ids"]]:
                raise ValueError("Conversation token aggregation mismatch")
            if r["text"] != json.dumps([t["text"] for t in turns], ensure_ascii=False, sort_keys=True, separators=(",", ":")):
                raise ValueError("Conversation text aggregation mismatch")
        if any(t["decode_tokens"] != t["generated_tokens"] - 1 or len(t["generated_token_ids"]) != t["generated_tokens"] for t in turns):
            raise ValueError("Per-turn token count mismatch")
        for timed in [r, *turns]:
            times = [timed[field] for field in ("prefill_ms", "decode_ms", "e2e_ms")]
            if not np.isfinite(times).all() or min(times) < 0 or not np.isclose(times[2], times[0] + times[1], atol=1e-5):
                raise ValueError("Invalid timer decomposition")
    if scores is not None:
        score_keys = {key(r) for r in scores}
        if len(score_keys) != len(scores) or score_keys - actual or (not allow_partial and score_keys != actual):
            raise ValueError("Missing, duplicate, or unexpected scores")
        lookup = {key(r): r for r in records}
        for score in scores:
            if score["run_id"] != manifest["run_id"] or score["prediction_sha256"] != digest(lookup[key(score)]["text"]):
                raise ValueError("Stale quality scores")
            if score["dataset"] == "mt-bench":
                if score.get("metric") != "not_scored" or score.get("passed") is not None:
                    raise ValueError("MT-Bench quality uses separate external judgments, not pass/fail")
            elif not isinstance(score["passed"], bool):
                raise ValueError("Objective evaluation is missing a boolean score")
    return {"expected": len(expected), "actual": len(actual), "missing": len(expected - actual),
            "coverage": manifest["coverage"], "evaluation": manifest.get("evaluation", {"protocol": "full"}),
            "complete": actual == expected}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def speedup(rows, baseline):
    # Aggregate time per generated decode token; N-1 matches decode timing.
    t = sum(r["decode_ms"] for r in rows)
    n = sum(r["decode_tokens"] for r in rows)
    bt = sum(r["decode_ms"] for r in baseline)
    bn = sum(r["decode_tokens"] for r in baseline)
    return ratio(bt * n, bn * t)


def clustered_ci(rows, baseline, samples=1000, seed=0):
    """Paired prompt bootstrap: all seeds of a question stay in one cluster."""
    if samples < 1:
        return [None, None]
    grouped = defaultdict(lambda: np.zeros(4))
    for r, b in zip(rows, baseline):
        if (r["source_id"], r["seed"]) != (b["source_id"], b["seed"]):
            raise ValueError("Bootstrap pairs are misaligned")
        grouped[r["source_id"]] += [r["decode_ms"], r["decode_tokens"], b["decode_ms"], b["decode_tokens"]]
    if len(grouped) < 2:
        return [None, None]
    x = np.stack(list(grouped.values()))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        t, n, bt, bn = x[rng.integers(0, len(x), len(x))].sum(0)
        value = ratio(bt * n, bn * t)
        if value is not None:
            estimates.append(value)
    return np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else [None, None]


def summarize(manifest, records, scores=None, bootstrap=1000):
    variants = {e["variant"]["name"]: e for e in manifest["variants"]}
    baseline_names = {e["variant"]["temperature"]: name for name, e in variants.items()
                      if e["variant"]["method"] == "target"}
    lookup = {key(r): r for r in records}
    score_lookup = {key(r): r for r in scores or []}
    grouped = defaultdict(list)
    for r in records:
        grouped[r["dataset"], r["variant"]].append(r)
    table = []
    for (dataset, name), rows in sorted(grouped.items()):
        entry = variants[name]
        v = entry["variant"]
        rows.sort(key=lambda r: (r["source_id"], r["seed"]))
        baseline_name = baseline_names.get(v["temperature"])
        pairs = [(r, lookup[(baseline_name, dataset, r["source_id"], r["seed"])]) for r in rows
                 if (baseline_name, dataset, r["source_id"], r["seed"]) in lookup]
        paired_rows, baselines = zip(*pairs) if pairs else ([], [])
        intervals = clustered_ci(paired_rows, baselines, bootstrap)
        rounds = [item for r in rows for item in r["rounds"]]
        def scored(r):
            return key(r) in score_lookup and score_lookup[key(r)]["passed"] is not None
        quality = [score_lookup[key(r)]["passed"] for r in rows if scored(r)]
        quality_clusters = defaultdict(list)
        for r, b in pairs:
            if scored(r) and scored(b):
                quality_clusters[r["source_id"]].append(float(score_lookup[key(r)]["passed"]) - float(score_lookup[key(b)]["passed"]))
        differences = np.array([np.mean(x) for x in quality_clusters.values()])
        quality_ci = [None, None]
        if len(differences) >= 2 and bootstrap > 0:
            rng = np.random.default_rng(0)
            estimates = [float(differences[rng.integers(0, len(differences), len(differences))].mean()) for _ in range(bootstrap)]
            quality_ci = np.quantile(estimates, [.025, .975]).tolist()
        per_seed = {str(seed): float(np.mean([score_lookup[key(r)]["passed"] for r in rows
                    if r["seed"] == seed and scored(r)]))
                    for seed in manifest["seeds"] if any(r["seed"] == seed and scored(r) for r in rows)}
        turns = [t for r in rows for t in r.get("turn_results", [r])]
        row = {"dataset": dataset, "variant": name, "method": v["method"],
               "groups": ";".join(entry["groups"]), "temperature": v["temperature"],
               "paths": v["paths"], "length": v["length"], "samples": len(rows),
               "unique_prompts": len({r["source_id"] for r in rows}),
               "generation_turns": len(turns),
               "generated_tokens": sum(r["generated_tokens"] for r in rows),
               "decode_tps": ratio(1000 * sum(r["decode_tokens"] for r in rows), sum(r["decode_ms"] for r in rows)),
               "e2e_tps": ratio(1000 * sum(r["generated_tokens"] for r in rows), sum(r["e2e_ms"] for r in rows)),
               "speedup_vs_ar": speedup(paired_rows, baselines),
               "baseline_status": "paired" if len(pairs) == len(rows) else "missing_or_partial",
               "speedup_ci_low": intervals[0], "speedup_ci_high": intervals[1],
               "matched_baseline_samples": len(pairs),
               "ttft_mean_ms": float(np.mean([t["prefill_ms"] for t in turns])),
               "latency_p50_ms": float(np.quantile([r["e2e_ms"] for r in rows], .5)),
               "latency_p95_ms": float(np.quantile([r["e2e_ms"] for r in rows], .95)),
               "mean_generation_length": float(np.mean([r["generated_tokens"] for r in rows])),
               "length_cap_fraction": float(np.mean([t["finish_reason"] == "length" for t in turns])),
               "accepted_per_verify": ratio(sum(x["accepted_draft_tokens"] for x in rounds), len(rounds)),
               "acceptance_rate": ratio(sum(x["accepted_draft_tokens"] for x in rounds),
                                         sum(x.get("proposed_tokens", 0) for x in rounds)),
               "committed_per_verify": ratio(sum(x["committed_tokens"] for x in rounds), len(rounds)),
               "mean_tree_nodes": float(np.mean([x["tree_nodes"] for x in rounds])) if rounds else None,
               "target_forward_calls": sum(r["target_forward_calls"] for r in rows),
               "draft_forward_calls": sum(r["draft_forward_calls"] for r in rows),
               "peak_allocated_gib": max((r["peak_allocated_bytes"] or 0) for r in rows) / 1024**3,
               "quality": float(np.mean(quality)) if quality else None,
               "quality_status": "external_judge_report_separate" if dataset == "mt-bench" else ("scored" if quality else "not_scored"),
               "quality_delta_vs_ar": float(differences.mean()) if len(differences) else None,
               "quality_delta_ci_low": quality_ci[0], "quality_delta_ci_high": quality_ci[1],
               "scored_samples": len(quality), "quality_per_seed": json.dumps(per_seed, sort_keys=True),
               "greedy_token_match_fraction": float(np.mean([[t["generated_token_ids"] for t in r.get("turn_results", [r])] == [t["generated_token_ids"] for t in b.get("turn_results", [b])]
                    for r, b in pairs])) if pairs and v["temperature"] == 0 else None}
        table.append(row)
    return table


def report(run_dir: Path, output: Path, bootstrap=1000, allow_partial=False, performance_only=False, plots=True):
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    records = read_jsonl(run_dir / "results.jsonl")
    scores = None if performance_only else read_jsonl(run_dir / "scores.jsonl")
    coverage = validate_results(manifest, records, scores, allow_partial)
    table = summarize(manifest, records, scores, bootstrap)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", {"run_id": manifest["run_id"], "coverage": coverage,
               "performance_only": performance_only, "bootstrap_clusters": "source_id_with_all_seeds",
               "bootstrap_samples": bootstrap, "rows": table})
    if not table:
        raise ValueError("No results to report")
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    def fmt(x):
        return "--" if x is None else f"{x:.3f}"
    lines = [r"\begin{tabular}{llrrrr}", r"Dataset & Method & Decode tok/s & Speedup & Quality & N \\", r"\hline"]
    for row in table:
        name = row["variant"].replace("_", r"\_")
        lines.append(f"{row['dataset']} & {name} & {fmt(row['decode_tps'])} & {fmt(row['speedup_vs_ar'])} & {fmt(row['quality'])} & {row['samples']} " + r"\\")
    lines.append(r"\end{tabular}")
    (output / "table.tex").write_text("\n".join(lines) + "\n")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = sorted({g for r in table for g in r["groups"].split(";")})
        for dataset in manifest["dataset_names"]:
            for group in groups:
                part = [r for r in table if r["dataset"] == dataset and group in r["groups"].split(";")]
                if not part:
                    continue
                fig, ax = plt.subplots(figsize=(9, max(3, .5 * len(part) + 1)))
                y = np.arange(len(part))
                paired = all(r["speedup_vs_ar"] is not None and r["baseline_status"] == "paired" for r in part)
                field = "speedup_vs_ar" if paired else "decode_tps"
                ax.barh(y, [r[field] or 0 for r in part], color="#3179aa")
                if paired:
                    for i, r in enumerate(part):
                        if r["speedup_ci_low"] is not None:
                            ax.hlines(i, r["speedup_ci_low"], r["speedup_ci_high"], color="black", linewidth=1.5)
                ax.set_yticks(y, [r["variant"] for r in part])
                if paired:
                    ax.axvline(1, color="black", linewidth=.8, linestyle="--")
                label = "Decode speedup vs. matched target AR (95% paired CI)" if paired else "Decode tokens/s (AR comparison unavailable)"
                ax.set(xlabel=label, title=f"{dataset}: {group}")
                ax.invert_yaxis()
                fig.tight_layout()
                for ext in ("pdf", "png"):
                    fig.savefig(output / f"{dataset}_{group}.{ext}", dpi=200)
                plt.close(fig)
    return coverage
