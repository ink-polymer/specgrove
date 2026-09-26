"""Frozen design, provenance and paired statistics for tree-verification studies."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import re

import numpy as np

from .common import ROOT, digest, file_hash, write_json
from .config import Variant, build_variants, load_config
from .data import DATASETS, DDTREE_COUNTS, evaluation_policy
from .fairness import assert_architecture_only_pair


NAMES = ("target_t1", "dflash_match", "ddtree", "tm_full", "tm_serial_prefix", "tm_dense_exit")
METHODS = ("target", "dflash", "ddtree", "ddtree_terminal_block",
           "ddtree_terminal_serial", "ddtree_terminal_dense")
DATASET_NAMES = ("gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench", "mt-bench")


def variants(study=None):
    if study is not None:
        return [Variant(**entry["variant"]) for entry in study["config"]["explicit_variants"]]
    base = Variant(name="ddtree", method="ddtree", paths=1, length=15,
                   temperature=1., draft_temperature=1., tree_budget=45,
                   probability_dtype="float64")
    return [replace(base, name=name, method=method) for name, method in zip(NAMES, METHODS)]


def method_names(study=None):
    return tuple(v.name for v in variants(study))


def primary_candidate(study):
    if study["spec"]["study"] in {"diffusion_tree_verification", "diffusion_scaffold_verification"}:
        return "diffusion_full"
    if study["spec"]["study"] == "atom_tree_verification":
        return "atom_full"
    return "protected_full" if study["spec"]["study"] == "protected_tree_verification" else "tm_full"


def study_sources():
    paths = list((ROOT / "src/gbv_experiments").glob("*.py"))
    paths += list((ROOT / "tests/gbv_paper").glob("*.py"))
    paths += list((ROOT / "third_party/ddtree_official").rglob("*.py"))
    paths += list((ROOT / "third_party/livecodebench_official").glob("*.py"))
    paths += [ROOT / "scripts/run_terminal_mass_formal.sh",
              ROOT / "scripts/terminal-mass-formal.sbatch",
              ROOT / "scripts/protected-tree-pilot.sbatch",
              ROOT / "docs/TERMINAL_MASS_FORMAL_PROTOCOL.md",
              ROOT / "docs/PROTECTED_TREE_BV.md"]
    paths += [ROOT / "docs/ATOM_TREE_BV.md", ROOT / "scripts/atom-tree-pilot.sbatch"]
    paths += [ROOT / "docs/DIFFUSION_TREE_BV.md", ROOT / "docs/DIFFUSION_TREE_THEORY_AUDIT.md",
              ROOT / "docs/DIFFUSION_SCAFFOLD_BV.md",
              ROOT / "docs/DIFFUSION_CORE_STOPPING.md",
              ROOT / "scripts/benchmark_diffusion_scaffold.py",
              ROOT / "scripts/diffusion-tree-pilot.sbatch",
              ROOT / "scripts/run_diffusion_tree.sh", ROOT / "scripts/diffusion-tree-formal.sbatch"]
    paths += list((ROOT / "configs").glob("diffusion_tree_*.json"))
    paths += list((ROOT / "configs").glob("diffusion_scaffold_*.json"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def load_study(path: Path):
    spec = json.loads(path.read_text())
    required = {"schema", "study", "models", "seeds", "timing_repeats", "max_new_tokens",
                "warmup_tokens", "bootstrap_samples", "bootstrap_seed", "gpu_family",
                "cpu_threads", "diagnostic_tokens", "capture_states_per_prompt",
                "replay_warmup", "replay_iterations", "replay_repeats", "primary_endpoint",
                "success_rule", "mtbench_quality", "claim_scope"}
    scaffold_study = spec.get("study") == "diffusion_scaffold_verification"
    diffusion_study = spec.get("study") == "diffusion_tree_verification" or scaffold_study
    if diffusion_study:
        required.add("temperature")
        if type(spec.get("temperature")) not in (int, float) or spec["temperature"] not in (0.3, 0.6, 1.0):
            raise ValueError("Diffusion formal study requires preregistered T=0.3, 0.6 or 1.0")
    if set(spec) != required or spec["schema"] != 1 or spec["study"] not in {
            "terminal_mass_tree_verification", "protected_tree_verification", "atom_tree_verification",
            "diffusion_tree_verification", "diffusion_scaffold_verification"}:
        raise ValueError("Unrecognized formal study schema")
    fixed = {"seeds": [17, 29, 43], "max_new_tokens": 2048,
             "primary_endpoint": "equal_dataset_geomean_decode_speedup_vs_ddtree",
             "success_rule": "complete_evidence_and_95pct_CI_lower_bound_gt_1",
             "mtbench_quality": "not_claimed_no_external_judge",
             "claim_scope": "controlled_shared_engine_single_model_architecture_comparison"}
    if spec["study"] == "atom_tree_verification" or diffusion_study:
        fixed["success_rule"] = "complete_evidence_and_both_baseline_95pct_CI_lower_bounds_gt_1"
    if any(spec[k] != value for k, value in fixed.items()):
        raise ValueError("Formal endpoints/seeds/length/claim scope changed")
    for field, minimum in {"timing_repeats": 3, "bootstrap_samples": 1000,
                           "warmup_tokens": 16, "cpu_threads": 1, "diagnostic_tokens": 16,
                           "capture_states_per_prompt": 1, "replay_warmup": 1,
                           "replay_iterations": 10, "replay_repeats": 3}.items():
        if type(spec[field]) is not int or spec[field] < minimum:
            raise ValueError(f"Insufficient {field}")
    if type(spec["bootstrap_seed"]) is not int or spec["gpu_family"] not in {"H200", "GH200"}:
        raise ValueError("Invalid bootstrap seed or GPU family")
    # Each registration is one immutable model pair, never a pooled model claim.
    if not isinstance(spec["models"], list) or len(spec["models"]) != 1:
        raise ValueError("This registered protocol requires exactly one model pair")
    entry = spec["models"][0]
    if set(entry) != {"id", "config"} or not re.fullmatch(r"[a-z0-9_]+", entry["id"]):
        raise ValueError("Invalid model entry")
    source_path = (path.parent / entry["config"]).resolve()
    original = load_config(source_path)
    if original["datasets"] != list(DATASET_NAMES):
        raise ValueError("Must reuse the previous seven datasets in their existing order")
    policy = evaluation_policy(original["datasets"], original.get("evaluation"))
    if policy != {"protocol": "ddtree_counts", "sample_seed": 0,
                  "counts": {name: DDTREE_COUNTS[name] for name in DATASET_NAMES}}:
        raise ValueError("Must reuse the previous fixed sample selection")
    for role in ("target", "draft"):
        if not re.fullmatch(r"[a-f0-9]{40}", original["model"].get(role + "_revision", "")):
            raise ValueError("Formal checkpoints require immutable commit revisions")
    methods = variants()
    if diffusion_study:
        methods = [replace(v, temperature=spec["temperature"], draft_temperature=spec["temperature"])
                   for v in methods[:3]]
        methods[0] = replace(methods[0], name="target_sample")
        baseline = methods[2]
        methods += [replace(baseline, name=name, method=method, paths=count)
                    for name, method, count in (
                        ("diffusion_full", "diffusion_tree_bv", 3),
                        ("diffusion_aligned", "diffusion_tree_bv_aligned", 3),
                        ("diffusion_no_pool", "diffusion_tree_bv_no_pool", 3),
                        ("diffusion_ancestral", "diffusion_tree_ancestral", 3),
                        ("diffusion_unmerged", "diffusion_tree_bv_unmerged", 3),
                        ("diffusion_single", "diffusion_tree_bv", 1),
                        ("atom_control", "atom_tree_bv", 3),
                        ("gbv", "gbv", 3))]
        if scaffold_study:
            # Separate study identity; never relabel the previous random trie.
            methods = methods[:3] + [replace(baseline, name=name, method=method, paths=count)
                                    for name, method, count in (
                                        ("diffusion_full", "diffusion_scaffold_bv", 1),
                                        ("diffusion_old", "diffusion_tree_bv", 3),
                                        ("diffusion_two_paths", "diffusion_scaffold_bv", 2),
                                        ("diffusion_no_fill", "diffusion_scaffold_no_fill", 1),
                                        ("diffusion_no_recycle", "diffusion_scaffold_no_recycle", 1),
                                        ("diffusion_ancestral_recycle", "diffusion_scaffold_ancestral", 1),
                                        ("diffusion_single", "diffusion_tree_bv", 1))]
    if spec["study"] == "atom_tree_verification":
        baseline = methods[2]
        methods = methods[:3] + [
            replace(baseline, name=name, method=method, paths=3)
            for name, method in (
                ("atom_full", "atom_tree_bv"),
                ("atom_fixed", "atom_tree_bv_fixed"),
                ("atom_aligned", "atom_tree_bv_aligned"),
                ("atom_no_pool", "atom_tree_bv_no_pool"),
                ("atom_ancestral", "atom_tree_ancestral"),
                ("protected_control", "root_protected_bv"),
                ("gbv", "gbv"),
            )
        ]
    if spec["study"] == "protected_tree_verification":
        methods = [Variant(**item["variant"]) for item in build_variants(original)]
        expected = {"target_t1": "target", "dflash_match": "dflash", "ddtree": "ddtree",
                    "rm_full": "root_marginal_bv", "dflash_bv": "bv", "gbv": "gbv",
                    "rm_serial_ref": "root_marginal_bv_ref", "rm_token": "root_marginal_token",
                    "rm_early_root": "root_early_bv", "rm_shared_ddtree": "root_shared_ddtree"}
        if {v.name: v.method for v in methods} != expected:
            raise ValueError("Protected study requires the frozen RM controls and ablations")
        baseline = next(v for v in methods if v.name == "ddtree")
        methods.insert(3, replace(baseline, name="protected_full", method="root_protected_bv", paths=3))
        if any(v.length != 15 or v.tree_budget != 45 or v.temperature != 1.
               or v.draft_temperature != 1. or v.probability_dtype != "float64"
               or v.paths != (3 if v.method in {"root_protected_bv", "root_marginal_bv",
                    "root_marginal_bv_ref", "root_marginal_token", "root_early_bv",
                    "root_shared_ddtree", "gbv"} else 1) for v in methods):
            raise ValueError("Protected study requires K=3, L=15, B=45 and matched T=1 FP64 settings")
    for method in methods[1:]:
        assert_architecture_only_pair(methods[2], method, original["model"],
                                      positive_temperature_study=diffusion_study)
    cfg = {key: original[key] for key in ("model", "datasets", "scoring", "evaluation")}
    cfg.update({key: spec[key] for key in ("seeds", "max_new_tokens", "warmup_tokens", "bootstrap_samples")})
    cfg["explicit_variants"] = [{"variant": v.to_dict(), "groups": ["main" if i < 4 else "ablation"]}
                                for i, v in enumerate(methods)]
    return {"spec": spec, "config": cfg, "model_id": entry["id"],
            "parent_config_sha256": file_hash(source_path)}


def plan(study):
    cfg, spec = study["config"], study["spec"]
    counts = evaluation_policy(cfg["datasets"], cfg["evaluation"])["counts"]
    questions = sum(counts.values())
    turns = sum(counts[name] * DATASETS[name].turns for name in counts)
    multiplier = len(variants(study)) * len(spec["seeds"]) * spec["timing_repeats"]
    return {"study": spec["study"], "model_id": study["model_id"], "datasets": counts,
            "source_rows": questions, "turns_per_method_seed_repeat": turns,
            "variants": [v.to_dict() for v in variants(study)], "seeds": spec["seeds"],
            "timing_repeats": spec["timing_repeats"], "max_new_tokens": spec["max_new_tokens"],
            "expected_records": questions * multiplier, "expected_generations": turns * multiplier,
            "primary_endpoint": spec["primary_endpoint"], "success_rule": spec["success_rule"],
            "temperature": methods_temperature(study),
            "formal_complete": False, "gpu_executed": False}


def methods_temperature(study):
    return sorted({v.temperature for v in variants(study)})


def freeze_document(path, value):
    """Never replace a prior registration or a completed group with new content."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen artifact changed: {path}; start a new study directory")
    else:
        write_json(path, value)
        import os
        with path.open("rb") as stream:
            os.fsync(stream.fileno())


def gpu_matches(name, family):
    # H200 is a substring of GH200; substring checks mislabel Grace Hopper.
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(family) + r"(?![A-Za-z0-9])", name) is not None


def paired_order(seed, repeat, study=None):
    # Cycling a fixed permutation balances method position over repeats.
    import random
    order = list(method_names(study))
    random.Random(seed).shuffle(order)
    shift = repeat % len(order)
    return order[shift:] + order[:shift]


def check_group(group, run_id, dataset, source_id, seed, repeat, study=None):
    rows = group.get("records", [])
    names = method_names(study)
    if (group.get("run_id") != run_id or group.get("repeat") != repeat
            or group.get("group_key") != [dataset, source_id, seed]
            or len(rows) != len(names) or {r["variant"] for r in rows} != set(names)):
        raise ValueError("Incomplete or foreign paired timing group")
    if any((r["run_id"], r["dataset"], r["source_id"], r["seed"]) !=
           (run_id, dataset, source_id, seed) for r in rows):
        raise ValueError("Paired group record identity mismatch")
    if group.get("records_sha256") != digest(rows):
        raise ValueError("Paired timing group content changed")
    return rows


def compare(records, candidate, baseline, datasets, samples=10000, seed=0, metric="decode"):
    """Dataset-stratified paired prompt bootstrap; seeds/repeats stay clustered."""
    if metric not in {"decode", "e2e"} or samples < 2:
        raise ValueError("Invalid comparison settings")
    index = {}
    for row in records:
        if row["variant"] not in {candidate, baseline}:
            continue
        key = (row["variant"], row["dataset"], row["source_id"], row["seed"], row["repeat"])
        if key in index:
            raise ValueError("Duplicate comparison record")
        index[key] = row
    left = {key[1:]: row for key, row in index.items() if key[0] == candidate}
    right = {key[1:]: row for key, row in index.items() if key[0] == baseline}
    if not left or set(left) != set(right):
        raise ValueError("Missing matched baseline/candidate records")
    groups = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
    time_field, token_field = ("decode_ms", "decode_tokens") if metric == "decode" else ("e2e_ms", "generated_tokens")
    for key, row in left.items():
        other = right[key]
        values = np.array([row[time_field], row[token_field], other[time_field], other[token_field]], dtype=float)
        if not np.isfinite(values).all() or min(values) < 0 or values[0] <= 0 or values[2] <= 0:
            raise ValueError("Invalid timing data")
        groups[key[0]][key[1]] += values
    if set(groups) != set(datasets):
        raise ValueError("Dataset coverage missing in comparison")
    rng = np.random.default_rng(seed)
    point_estimates, bootstrap_estimates, per_dataset, totals = [], [], {}, []
    for dataset in datasets:
        data = np.array(list(groups[dataset].values()))
        if len(data) < 2:
            raise ValueError("Need at least two prompt clusters per dataset")
        total = data.sum(0)
        if min(total) <= 0:
            raise ValueError("Zero aggregate tokens or time")
        point = total[1] * total[2] / (total[0] * total[3])
        draws = np.stack([data[rng.integers(0, len(data), len(data))].sum(0) for _ in range(samples)])
        if (draws[:, [1, 3]] <= 0).any():
            raise ValueError("Degenerate bootstrap has zero tokens")
        estimates = draws[:, 1] * draws[:, 2] / (draws[:, 0] * draws[:, 3])
        per_dataset[dataset] = {"speedup": float(point), "ci95": np.quantile(estimates, [.025, .975]).tolist(),
                                "prompt_clusters": len(data),
                                "candidate_tps": float(1000 * total[1] / total[0]),
                                "baseline_tps": float(1000 * total[3] / total[2])}
        point_estimates.append(point)
        bootstrap_estimates.append(estimates)
        totals.append(total)
    geomean = float(np.exp(np.log(point_estimates).mean()))
    boot = np.exp(np.log(np.stack(bootstrap_estimates)).mean(0))
    total = np.sum(totals, axis=0)
    return {"candidate": candidate, "baseline": baseline, "metric": metric,
            "dataset_geomean_speedup": geomean, "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "pooled_speedup": float(total[1] * total[2] / (total[0] * total[3])),
            "matched_pairs": len(left), "per_dataset": per_dataset,
            "bootstrap_clusters": "(dataset, source_id), all seeds and repeats retained",
            "secondary_intervals": "descriptive; no multiple-testing claim"}
