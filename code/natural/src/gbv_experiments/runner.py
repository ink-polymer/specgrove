from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import traceback
from contextlib import contextmanager

from .common import canonical, digest, file_hash, prompt_seed, source_hashes, write_json
from .config import Variant, build_variants, select_variants
from .data import DATASETS, evaluation_coverage, evaluation_policy, load_prepared
from .runtime_model import (MODEL_PARAMETERS_SCHEMA,
                            RUNTIME_READY_GATE_SCHEMA,
                            enforce_runtime_model_gate,
                            is_registered_same_tree_t1,
                            runtime_model_expectations,
                            validate_model_parameters_evidence)


def key(record):
    return (record["variant"], record["dataset"], record["source_id"], record["seed"])


def resume_records(path: Path, run_id: str) -> dict:
    records = {}
    if not path.exists():
        return records
    # Only an unterminated final write is recoverable; other corruption is fatal.
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                stream.truncate(start)
                break
            row = json.loads(raw)
            if row["run_id"] != run_id or key(row) in records:
                raise ValueError("Resume result run ID mismatch or duplicate key")
            records[key(row)] = row
    return records


def make_plan(cfg, groups=None, only_variants=None):
    entries = select_variants(build_variants(cfg, groups), only_variants)
    policy = evaluation_policy(cfg["datasets"], cfg.get("evaluation"))
    counts = policy["counts"]
    turn_counts = {name: counts[name] * DATASETS[name].turns for name in cfg["datasets"]}
    plan = {"coverage": evaluation_coverage(policy), "evaluation": policy, "datasets": counts,
            "model": cfg["model"], "only_variants": only_variants,
            "source_counts": {name: DATASETS[name].expected for name in cfg["datasets"]},
            "seeds": cfg["seeds"], "max_new_tokens": cfg["max_new_tokens"],
            "method_order": cfg.get("method_order", {"policy": "seeded_shuffle", "seed": 0}),
            "variants": entries, "variant_count": len(entries),
            "user_turns": turn_counts,
            "expected_records": sum(counts.values()) * len(cfg["seeds"]) * len(entries),
            "expected_generations": sum(turn_counts.values()) * len(cfg["seeds"]) * len(entries)}
    if "target_verification_backends" in cfg:
        plan["target_verification_backends"] = verification_backend_map(
            cfg, entries,
        )
    return plan


def verification_backend_map(cfg, entries) -> dict[str, str]:
    """Validate per-variant Target tree executors for registered runs."""
    raw = cfg.get("target_verification_backends", {})
    if not isinstance(raw, dict):
        raise ValueError("target_verification_backends must be an object")
    names = {entry["variant"]["name"] for entry in entries}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"Target verification backend names are unknown: {unknown}")
    allowed = {
        "eager",
        "torch_compile_inductor_dynamic",
        "torch_compile_inductor_default_dynamic",
        "torch_compile_inductor_reduce_overhead_static",
        "torch_compile_inductor_full_target_dynamic",
    }
    invalid = {name:value for name, value in raw.items() if value not in allowed}
    if invalid:
        raise ValueError(f"Invalid Target verification backends: {invalid}")
    return {name:raw.get(name, "eager") for name in sorted(names)}


def scheduled_variants(entries, method_order, ordinal, per_prompt_seed):
    """Schedule timed methods reproducibly, with optional exact position balance."""
    policy = method_order.get("policy", "seeded_shuffle")
    order = entries.copy()
    if policy == "seeded_shuffle":
        random.Random(per_prompt_seed).shuffle(order)
        return order
    if policy != "balanced_rotation":
        raise ValueError("Unknown method order policy")
    random.Random(method_order["seed"]).shuffle(order)
    offset = ordinal % len(order)
    return order[offset:] + order[:offset]


def dataset_local_schedule(rows):
    """Return each row's ordinal and row count within its own dataset."""
    counts = {}
    ordinals = []
    for row in rows:
        dataset = row["dataset"]
        ordinal = counts.get(dataset, 0)
        ordinals.append(ordinal)
        counts[dataset] = ordinal + 1
    return ordinals, counts


def model_identity(cfg):
    from huggingface_hub import HfApi
    result = dict(cfg)
    for field in ("target", "draft"):
        path = Path(cfg[field]).expanduser()
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"})
            result[field] = str(path.resolve())
            result[field + "_local_hashes"] = {str(p.relative_to(path)): file_hash(p) for p in files}
        else:
            revision = cfg.get(field + "_revision")
            if not revision or len(revision) != 40:
                result[field + "_revision"] = HfApi().model_info(cfg[field], revision=revision).sha
    return result


def stop_token_ids(engine, tokenizer):
    """Return the EOS token IDs used by both preflight and formal runs."""
    value = engine.target.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    return [value] if isinstance(value, int) else list(value or [])


@contextmanager
def output_lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def run(cfg, data_dir: Path, output: Path, device: str, groups=None, smoke=False, profile=False, only_variants=None):
    with output_lock(output):
        return _run(cfg, data_dir, output, device, groups, smoke, profile, only_variants)


def _run(cfg, data_dir: Path, output: Path, device: str, groups=None, smoke=False, profile=False, only_variants=None):
    import torch
    from .engine import load_models
    from .conversation import encode_messages, generate_conversation

    policy = evaluation_policy(cfg["datasets"], cfg.get("evaluation"))
    data_manifest, rows = load_prepared(data_dir, cfg["datasets"], policy)
    entries = build_variants(cfg, groups)
    active_entries = select_variants(entries, only_variants)
    verification_backends = verification_backend_map(cfg, entries)
    if smoke:
        rows = [next(r for r in rows if r["dataset"] == name) for name in cfg["datasets"]]
        if "smoke" not in output.name.lower():
            raise ValueError("Smoke output directory must contain 'smoke'")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Formal inference needs CUDA; use the CPU tests for local verification")
    model = model_identity(cfg["model"])
    registered_same_tree_t1 = is_registered_same_tree_t1(cfg)
    runtime_expectations = runtime_model_expectations(
        cfg["model"], strict_same_tree_t1=registered_same_tree_t1,
        device=str(torch.device(device)) if registered_same_tree_t1 else None,
    )
    versions = {}
    for pkg in ("torch", "transformers", "datasets", "huggingface-hub", "numpy"):
        versions[pkg] = importlib.metadata.version(pkg)
    device_properties = torch.cuda.get_device_properties(torch.device(device))
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        driver = "unavailable"
    spec = {"schema": 3, "model": model, "variants": entries, "seeds": cfg["seeds"],
            "max_new_tokens": min(16, cfg["max_new_tokens"]) if smoke else cfg["max_new_tokens"],
            "method_order": cfg.get("method_order", {"policy": "seeded_shuffle", "seed": 0}),
            "coverage": "smoke" if smoke else evaluation_coverage(policy), "evaluation": policy, "profile": profile,
            "scoring": cfg.get("scoring", {}), "bootstrap_samples": cfg.get("bootstrap_samples", 1000),
            "data_manifest": data_manifest, "dataset_names": cfg["datasets"],
            "dataset_turn_counts": {name: DATASETS[name].turns for name in cfg["datasets"]},
            "prompt_ids": [[r["dataset"], r["source_id"], r["prompt_sha256"]] for r in rows],
            "source_hashes": source_hashes(), "versions": versions,
            "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(device)),
            "gpu_uuid":str(getattr(device_properties, "uuid", "unknown")),
            "gpu_memory_bytes": device_properties.total_memory, "driver": driver,
            "torch_cpu_threads": torch.get_num_threads(),
            "expected_records": len(rows) * len(cfg["seeds"]) * len(entries),
            "expected_generations": sum(DATASETS[r["dataset"]].turns for r in rows) * len(cfg["seeds"]) * len(entries)}
    if "target_verification_backends" in cfg:
        spec["target_verification_backends"] = verification_backends
    run_id = digest(spec)
    manifest = {"run_id": run_id, **spec}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["run_id"] != run_id:
            raise ValueError("Run configuration/data/code/environment changed; use a new output directory")
    else:
        if (output / "results.jsonl").exists():
            raise ValueError("Existing results have no run manifest")
        write_json(manifest_path, manifest)
    completed = resume_records(output / "results.jsonl", run_id)
    model_parameters_path = output / "model_parameters.json"
    stored_model_parameters = None
    if model_parameters_path.exists():
        stored_model_parameters = json.loads(model_parameters_path.read_text())
        validate_model_parameters_evidence(
            stored_model_parameters, run_id=run_id,
            expected=runtime_expectations, require_ready=bool(completed),
        )
    elif completed:
        raise RuntimeError(
            "Existing result rows have no model_parameters.json runtime gate"
        )
    valid = {(e["variant"]["name"], r["dataset"], r["source_id"], seed)
             for e in entries for r in rows for seed in cfg["seeds"]}
    active_names = {e["variant"]["name"] for e in active_entries}
    active_keys = {k for k in valid if k[0] in active_names}
    if set(completed) - valid:
        raise ValueError("Resume file contains unexpected experiments")
    def mark_completion():
        full_complete = set(completed) == valid
        if full_complete:
            write_json(output / "completed.json", {"run_id": run_id, "records": len(completed),
                       "generations": sum(r["turn_count"] for r in completed.values())})
        if only_variants is not None:
            write_json(output / f"stage_completed_{digest(sorted(active_names))[:12]}.json", {
                "run_id": run_id, "variants": sorted(active_names), "records": len(active_keys),
                "generations": sum(completed[k]["turn_count"] for k in active_keys),
                "stage_complete": True, "full_experiment_complete": full_complete})

    if active_keys <= set(completed):
        mark_completion()
        print(f"Requested variants already complete: {len(active_keys)} question/conversation records")
        return
    engine, tokenizer = load_models(model, device)
    loaded_runtime_identity = enforce_runtime_model_gate(
        engine, runtime_expectations,
        tokenizer=tokenizer if registered_same_tree_t1 else None,
    )
    model_parameters = {
        "schema": MODEL_PARAMETERS_SCHEMA,
        "run_id": run_id,
        "target_parameters": sum(p.numel() for p in engine.target.parameters()),
        "draft_parameters": sum(p.numel() for p in engine.draft.parameters()),
        "trainable_parameters": sum(p.numel() for module in (engine.target, engine.draft) for p in module.parameters() if p.requires_grad),
        "draft_block_size": engine.draft.block_size,
        "target_feature_layers": engine.draft.target_layer_ids,
        "runtime_model_gate": {
            "schema": RUNTIME_READY_GATE_SCHEMA,
            "status": "loaded_not_ready",
            "expectations": runtime_expectations,
            "identity_after_load": loaded_runtime_identity,
            "identity_before_timing": None,
            "passed_after_load": True,
            "passed_before_timing": False,
            "identities_match": False,
        },
    }
    ready_model_parameters = model_parameters | {
        "runtime_model_gate": model_parameters["runtime_model_gate"] | {
            "status": "ready_for_timing",
            "identity_before_timing": loaded_runtime_identity,
            "identities_match": True,
            "passed_before_timing": True,
        }
    }
    if stored_model_parameters is not None:
        if canonical(stored_model_parameters) not in {
                canonical(model_parameters), canonical(ready_model_parameters)}:
            raise ValueError("Runtime model identity changed since the prior formal stage")
    else:
        write_json(model_parameters_path, model_parameters)
    stop_ids = stop_token_ids(engine, tokenizer)

    warmup = encode_messages(tokenizer, [{"role": "user", "content": "Compute 1 + 1."}], model, device)
    for entry in active_entries:
        engine.set_target_verification_backend(
            verification_backends[entry["variant"]["name"]]
        )
        engine.generate(warmup, Variant(**entry["variant"]), cfg.get("warmup_tokens", 16),
                        stop_ids, seed=0, profile=profile)
    timing_runtime_identity = enforce_runtime_model_gate(
        engine, runtime_expectations,
        tokenizer=tokenizer if registered_same_tree_t1 else None,
    )
    if timing_runtime_identity != loaded_runtime_identity:
        raise RuntimeError("Runtime model identity changed between load and formal timing")
    model_parameters = model_parameters | {
        "runtime_model_gate": model_parameters["runtime_model_gate"] | {
            "status": "ready_for_timing",
            "identity_before_timing": timing_runtime_identity,
            "identities_match": True,
            "passed_before_timing": True,
        }
    }
    validate_model_parameters_evidence(
        model_parameters, run_id=run_id,
        expected=runtime_expectations, require_ready=True,
    )
    write_json(model_parameters_path, model_parameters)
    row_ordinals, dataset_counts = dataset_local_schedule(rows)
    with (output / "results.jsonl").open("a", encoding="utf-8") as stream:
        for seed_index, seed in enumerate(cfg["seeds"]):
            for row_index, row in enumerate(rows):
                per_prompt_seed = prompt_seed(seed, row["dataset"], row["source_id"])
                # Start each dataset at ordinal zero and continue its rotation
                # across seeds.  A global ordinal can be balanced overall while
                # systematically placing one method first inside a dataset.
                ordinal = (seed_index * dataset_counts[row["dataset"]]
                           + row_ordinals[row_index])
                order = scheduled_variants(
                    entries, spec["method_order"], ordinal, per_prompt_seed
                )
                for execution_position, entry in enumerate(order):
                    if entry["variant"]["name"] not in active_names:
                        continue
                    v = Variant(**entry["variant"])
                    ident = (v.name, row["dataset"], row["source_id"], seed)
                    if ident in completed:
                        continue
                    try:
                        engine.set_target_verification_backend(
                            verification_backends[v.name]
                        )
                        result = generate_conversation(engine, tokenizer, row, v, spec["max_new_tokens"], stop_ids,
                                                       per_prompt_seed, model, profile)
                        record = {"run_id": run_id, "variant": v.name, "dataset": row["dataset"],
                                  "source_id": row["source_id"], "prompt_sha256": row["prompt_sha256"],
                                  "seed": seed, "sampling_seed": per_prompt_seed,
                                  **result}
                        if spec["method_order"]["policy"] == "balanced_rotation":
                            record["method_order_ordinal"] = ordinal
                            record["method_execution_position"] = execution_position
                        stream.write(canonical(record) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        completed[ident] = record
                        print(f"[{len(completed)}/{len(valid)}] {v.name} {row['dataset']}/{row['source_id']} seed={seed}", flush=True)
                    except Exception:
                        with (output / "errors.jsonl").open("a") as errors:
                            errors.write(canonical({"experiment": ident, "traceback": traceback.format_exc()}) + "\n")
                        raise
    mark_completion()
