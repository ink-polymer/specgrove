"""Staged, resumable formal experiment. No GPU results can be created offline."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback
import uuid
import xml.etree.ElementTree as ET

from .common import ROOT, canonical, digest, file_hash, prompt_seed, read_jsonl, write_json
from .data import DATASETS, load_prepared
from .runner import output_lock, stop_token_ids
from .runtime_model import enforce_runtime_model_gate
from .terminal_protocol import (check_group, compare, freeze_document, gpu_matches,
                                load_study, method_names, paired_order, plan,
                                primary_candidate, study_sources, variants)


UNIT_FILES = ["tests/gbv_paper/test_terminal_mass.py", "tests/gbv_paper/test_terminal_formal.py",
              "tests/gbv_paper/test_probability_law.py", "tests/gbv_paper/test_engine.py",
              "tests/gbv_paper/test_fairness.py", "tests/gbv_paper/test_evaluation_selection.py",
              "tests/gbv_paper/test_data_and_reporting.py",
              "tests/gbv_paper/test_protected_tree_bv.py",
              "tests/gbv_paper/test_root_marginalized_bv.py",
              "tests/gbv_paper/test_atom_tree_bv.py",
              "tests/gbv_paper/test_diffusion_tree_bv.py",
              "tests/gbv_paper/test_diffusion_preflight.py",
              "tests/gbv_paper/test_diffusion_theory_comparison.py",
              "tests/gbv_paper/test_diffusion_scaffold.py",
              "tests/gbv_paper/test_diffusion_core_stopping.py",
              "tests/gbv_paper/test_root_marginal_adversarial.py",
              "tests/gbv_paper/test_root_marginal_protocol.py"]
DIAGNOSTIC_PROMPTS = ["Compute 19 + 23. Give a brief explanation.",
                      "Write a Python function that reverses a list.",
                      "Explain why the sum of two even integers is even."]


def read(path):
    return json.loads(path.read_text())


def registration(study, data_dir):
    cfg = study["config"]
    manifest, rows = load_prepared(data_dir, cfg["datasets"], cfg["evaluation"])
    import re
    if any(not re.fullmatch(r"[a-f0-9]{40}", entry.get("revision", ""))
           for entry in manifest["datasets"].values()):
        raise ValueError("Dataset revisions must be locked before registration")
    body = {"study": study, "plan": plan(study), "sources": study_sources(),
            "data_manifest": manifest,
            "prompt_ids": [[r["dataset"], r["source_id"], r["prompt_sha256"]] for r in rows]}
    return {"registration_id": digest(body), **body}, rows


def registered(study, data_dir, output):
    expected, rows = registration(study, data_dir)
    if read(output / "registration.json") != expected:
        raise ValueError("Registered code, configuration, or data changed; create a new study")
    return expected, rows


def freeze(study, data_dir, output):
    from .audit import audit
    value, _ = registration(study, data_dir)
    with output_lock(output):
        freeze_document(output / "registration.json", value)
        result = audit(study["config"], data_dir, [], output / "data_audit.json")
        write_json(output / "data_gate.json", {"registration_id": value["registration_id"],
                   "passed": result["checks_passed"],
                   "artifacts": {"data_audit.json": file_hash(output / "data_audit.json")}})
    return value["plan"]


def gate(output, name, registration_id):
    value = read(output / f"{name}.json")
    if value.get("registration_id") != registration_id or value.get("passed") is not True:
        raise ValueError(f"Missing, stale, or failed {name}")
    for relative, expected in value.get("artifacts", {}).items():
        if file_hash(output / relative) != expected:
            raise ValueError(f"Changed gate evidence: {relative}")
    return value


def unit_gate(study, data_dir, output):
    import sys
    reg, _ = registered(study, data_dir, output)
    with output_lock(output):
        xml_path = output / "unit_tests.xml"
        result = subprocess.run([sys.executable, "-m", "pytest", *UNIT_FILES,
                                 f"--junitxml={xml_path}", "-q"], cwd=ROOT)
        suites = ET.parse(xml_path).getroot().iter("testsuite") if xml_path.exists() else []
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
        value = {"registration_id": reg["registration_id"], "selection": UNIT_FILES,
                 "passed": result.returncode == 0 and counts["tests"] > 0
                           and not any(counts[k] for k in ("failures", "errors", "skipped")),
                 "counts": counts, "artifacts": {"unit_tests.xml": file_hash(xml_path)} if xml_path.exists() else {}}
        write_json(output / "unit_gate.json", value)
        if not value["passed"]:
            raise RuntimeError("Formal unit gate failed or skipped tests")
        return value


def runtime(study, device):
    import torch
    spec = study["spec"]
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("This stage requires the allocated CUDA GPU")
    torch.cuda.set_device(device)
    torch.set_num_threads(spec["cpu_threads"])
    name = torch.cuda.get_device_name(device)
    if not gpu_matches(name, spec["gpu_family"]):
        raise RuntimeError(f"GPU mismatch: registered {spec['gpu_family']}, actual {name}; do not relabel GH200 as H200")
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    props = torch.cuda.get_device_properties(device)
    return {"gpu": name, "gpu_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor], "driver": driver,
            "cuda": torch.version.cuda, "python": platform.python_version(),
            "architecture": platform.machine(), "cpu_threads": torch.get_num_threads(),
            "versions": {key: importlib.metadata.version(key) for key in
                         ("torch", "transformers", "datasets", "huggingface-hub", "numpy")}}


def telemetry(device):
    import torch
    props = torch.cuda.get_device_properties(device)
    # No environment dump: job identity and performance metadata only.
    value = {"utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
             "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "gpu_uuid": str(getattr(props, "uuid", "unavailable"))}
    query = "uuid,name,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"],
                            capture_output=True, text=True)
    value["nvidia_smi"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return value


def allocation_gate(device):
    """Refuse detectable concurrent compute processes on the assigned GPU."""
    import torch
    identity = str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
    if not identity:
        raise RuntimeError("Cannot resolve the assigned GPU UUID for contention checks")
    normalized = identity.lower().removeprefix("gpu-")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("GPU process audit unavailable")
    foreign = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, gpu_uuid = [part.strip() for part in line.split(",", 1)]
        if gpu_uuid.lower().removeprefix("gpu-") == normalized and int(pid) != os.getpid():
            foreign.append(int(pid))
    if foreign:
        raise RuntimeError(f"Concurrent compute on the measured GPU: {foreign}")


def model_gate(engine):
    return enforce_runtime_model_gate(engine)


def load_frozen_engine(study, device):
    from .engine import load_models
    engine, tokenizer = load_models(study["config"]["model"], device)
    model_gate(engine)
    return engine, tokenizer


def capture_states(study, engine, ids, seed, *, limit=None, tokens=None):
    """Diagnostic-only states, including the actual Q for shared-suffix BV."""
    spec = study["spec"]
    limit = spec["capture_states_per_prompt"] if limit is None else limit
    tokens = spec["diagnostic_tokens"] if tokens is None else tokens
    by_name = {v.name: v for v in variants(study)}
    states = []

    def observe(parents, proposed, all_p):
        if len(states) < limit:
            states.append({"kind": "probability_tree", "parents": list(parents),
                           "tokens": list(proposed), "all_p": all_p.detach().cpu().clone()})

    engine.generate(ids, by_name["ddtree"], tokens, [], seed=seed, tree_observer=observe)
    if primary_candidate(study) == "diffusion_full":
        from . import diffusion_tree_bv as diffusion
        from .sampling import probabilities
        captured = []
        scaffold_round = 0
        is_scaffold = study["spec"]["study"] == "diffusion_scaffold_verification"

        def observe_diffusion(tree, proposal, logits, temperature):
            if len(captured) < limit:
                kind = ("scaffold_diffusion_trie" if study["spec"]["study"] == "diffusion_scaffold_verification"
                        else "one_step_diffusion_trie")
                captured.append({"kind": kind, "parents": tree.parents,
                                 "tokens": tree.tokens, "depths": tree.depths, "path_nodes": tree.path_nodes,
                                 "logits": logits.detach().cpu().clone(), "temperature": temperature,
                                 "all_p": probabilities(logits, temperature).detach().cpu().clone(),
                                 **diffusion.snapshot(proposal)})

        def observe_scaffold(tree, proposal, draft_logits, nodes, selected, bonus):
            from .diffusion_preflight import scaffold_contract
            nonlocal scaffold_round
            variant = by_name["diffusion_full"]
            greedy = draft_logits.argmax(-1).tolist()
            check = scaffold_contract(tree, proposal, variant, greedy, nodes, selected,
                                      bonus, draft_logits.shape[-1])
            if scaffold_round < len(captured):
                captured[scaffold_round]["scaffold_witness"] = {
                    "variant": variant.to_dict(), "greedy_tokens": greedy,
                    "accepted_nodes": list(nodes), "accepted_tokens": list(selected),
                    "bonus_token": bonus, "vocab": draft_logits.shape[-1], "checks": check}
            scaffold_round += 1

        engine.generate(ids, by_name["diffusion_full"], tokens, [], seed=seed,
                        diffusion_observer=observe_diffusion, scaffold_observer=observe_scaffold)
        if not captured:
            raise RuntimeError("No diffusion transition and labelled trie captured")
        if is_scaffold and any("scaffold_witness" not in state for state in captured):
            raise RuntimeError("Missing scaffold selection/greedy witnesses")
        states.extend(captured)
    if primary_candidate(study) == "atom_full":
        from .sampling import probabilities
        atoms = []

        def observe_atom(parents, proposed, proposal, logits, temperature):
            if len(atoms) < limit:
                state = {"kind": "latent_atoms", "parents": list(parents), "tokens": list(proposed),
                         "logits": logits.detach().cpu().clone(), "temperature": temperature,
                         "all_p": probabilities(logits, temperature).detach().cpu().clone()}
                state.update({"atom_" + key: value.detach().cpu().clone()
                              for key, value in vars(proposal).items()})
                atoms.append(state)

        engine.generate(ids, by_name["atom_full"], tokens, [], seed=seed, atom_observer=observe_atom)
        if not atoms:
            raise RuntimeError("No joint atom proposal captured")
        states.extend(atoms)
    if primary_candidate(study) == "protected_full":
        shared = []

        def observe_shared(parents, proposed, paths, all_p, q):
            if len(shared) < limit:
                shared.append({"kind": "shared_suffix", "parents": list(parents),
                               "tokens": list(proposed), "paths": paths.detach().cpu().clone(),
                               "all_p": all_p.detach().cpu().clone(), "q": q.detach().cpu().clone()})

        engine.generate(ids, by_name["rm_shared_ddtree"], tokens, [], seed=seed,
                        shared_suffix_observer=observe_shared)
        if not shared:
            raise RuntimeError("No shared-suffix proposal captured")
        states.extend(shared)
    if not states:
        raise RuntimeError("No verified state captured")
    return states


def replay_functions(state):
    """Same-state verifier calls; tensor transfer and tree construction excluded."""
    from . import protected_tree_bv, root_marginalized_bv as rm, sampling
    from .fused_tree_sampling import (tree_verify_ancestral_fused,
                                      tree_verify_ancestral_fused_parallel)
    parents, tokens, p = state["parents"], state["tokens"], state["all_p"]
    def ancestral(generator):
        return sampling.tree_verify_ancestral_batched(
            parents, tokens, p, generator, validate=False
        )
    if state.get("kind") in {"one_step_diffusion_trie", "scaffold_diffusion_trie"}:
        from . import diffusion_tree_bv as diffusion
        from .tree import Tree
        proposal = diffusion.restore(state)
        proposal.validate(p.shape[-1])
        tree = Tree(tokens, parents, state["depths"], state["path_nodes"])
        args = state["logits"], tree, proposal, state["temperature"]
        if state["kind"] == "scaffold_diffusion_trie":
            from .config import Variant
            from .diffusion_preflight import scaffold_contract
            witness = state.get("scaffold_witness")
            if witness is None or witness["vocab"] != p.shape[-1]:
                raise ValueError("Missing or invalid scaffold replay witness; recapture the state")
            # Validate the saved evidence instead of trusting its boolean flags.
            scaffold_contract(tree, proposal, Variant(**witness["variant"]), witness["greedy_tokens"],
                              witness["accepted_nodes"], witness["accepted_tokens"],
                              witness["bonus_token"], witness["vocab"])
            return {"diffusion_full": lambda g: diffusion.verify_scaffold_logits(*args, g, validate=False),
                    "diffusion_ancestral_recycle": lambda g: diffusion.verify_scaffold_logits(
                        *args, g, continuation="ancestral", validate=False),
                    "diffusion_no_recycle": lambda g: diffusion.verify_scaffold_logits(
                        *args, g, recycle=False, validate=False), "scaffold_ancestral": ancestral}
        return {"diffusion_full": lambda g: diffusion.verify_logits(*args, g, validate=False),
                "diffusion_no_pool": lambda g: diffusion.verify_logits(*args, g, pool=False, validate=False),
                "diffusion_ancestral": ancestral}
    if state.get("kind") == "latent_atoms":
        from . import atom_tree_bv as atom
        proposal = atom.AtomProposal(**{key: state["atom_" + key]
                                       for key in ("roots", "tokens", "slots", "source", "draws")})
        proposal.validate(p.shape[-1])
        k, m, _ = proposal.tokens.shape
        logits = state["logits"]
        args = (logits[0], logits[1:].view(k, m + 1, -1), proposal, state["temperature"])
        return {"atom_full": lambda g: atom.verify_logits(*args, g, validate=False),
                "atom_no_pool": lambda g: atom.verify_logits(*args, g, pool=False, validate=False),
                "atom_ancestral": ancestral}
    if state.get("kind", "probability_tree") != "shared_suffix":
        return {
            "ddtree": ancestral,
            "tm_full": lambda g: sampling.tree_block_verify_terminal_mass(parents, tokens, p, g, validate=False),
            "tm_serial_prefix": lambda g: sampling.tree_block_verify_terminal_mass(
                parents, tokens, p, g, validate=False, prefix_mode="serial"),
            "tm_dense_exit": lambda g: sampling.tree_block_verify_terminal_mass(
                parents, tokens, p, g, validate=False, exit_mode="dense"),
            "tm_complement": lambda g: sampling.tree_block_verify_terminal_mass(
                parents, tokens, p, g, validate=False, exit_mode="complement"),
            "tm_joint": lambda g: sampling.tree_block_verify_terminal_mass(
                parents, tokens, p, g, validate=False, exit_mode="joint"),
            "fused_ancestral": lambda g: tree_verify_ancestral_fused(
                parents, tokens, p, g, validate=False),
            "fused_parallel": lambda g: tree_verify_ancestral_fused_parallel(
                parents, tokens, p, g, validate=False),
        }
    paths, q = state["paths"], state["q"]
    args = (paths[:, 0], p[0], paths[0, 1:], p[1:].view(*paths.shape, -1), q[1:])
    rm._validate(*args)
    return {
        "rm_shared_ddtree": ancestral,
        "protected_full": lambda g: protected_tree_bv.verify(*args, g, validate=False),
        "rm_full": lambda g: rm.verify_tensorized(*args, g, validate=False),
        "rm_serial_ref": lambda g: rm.verify(*args, g, validate=False),
        "rm_early_root": lambda g: rm.verify_early_root(*args, g, validate=False),
        "rm_token": lambda g: rm.verify_tensorized(*args, g, validate=False, rule="token"),
    }


_VERIFIER_IMPLEMENTATIONS = {
    "ddtree": ("src/gbv_experiments/sampling.py",
               "gbv_experiments.sampling.tree_verify_ancestral_batched", {}),
    "tm_full": ("src/gbv_experiments/sampling.py",
                "gbv_experiments.sampling.tree_block_verify_terminal_mass", {}),
    "tm_dense_exit": ("src/gbv_experiments/sampling.py",
                      "gbv_experiments.sampling.tree_block_verify_terminal_mass",
                      {"exit_mode": "dense"}),
    "tm_complement": ("src/gbv_experiments/sampling.py",
                      "gbv_experiments.sampling.tree_block_verify_terminal_mass",
                      {"exit_mode": "complement"}),
    "tm_joint": ("src/gbv_experiments/sampling.py",
                 "gbv_experiments.sampling.tree_block_verify_terminal_mass",
                 {"exit_mode": "joint"}),
    "fused_ancestral": ("src/gbv_experiments/fused_tree_sampling.py",
                        "gbv_experiments.fused_tree_sampling.tree_verify_ancestral_fused", {}),
    "fused_parallel": ("src/gbv_experiments/fused_tree_sampling.py",
                       "gbv_experiments.fused_tree_sampling.tree_verify_ancestral_fused_parallel", {}),
}


def verifier_implementation_manifest(method):
    """Return the auditable callable, options and source hash for a replay method."""
    from .common import ROOT, file_hash

    try:
        relative, callable_name, options = _VERIFIER_IMPLEMENTATIONS[method]
    except KeyError as exc:
        raise ValueError(f"Unknown verifier implementation: {method}") from exc
    return {
        "callable": callable_name,
        "options": options,
        "source_file": relative,
        "source_sha256": file_hash(ROOT / relative),
    }


def diagnostics(study, data_dir, output, device):
    import gc
    import torch
    from .conversation import encode_messages
    from .preflight import check_model
    reg, _ = registered(study, data_dir, output)
    ident, spec, cfg = reg["registration_id"], study["spec"], study["config"]
    gate(output, "unit_gate", ident)
    env = runtime(study, device)
    allocation_gate(device)
    freeze_document(output / "runtime.json", env)
    with output_lock(output):
        # This process backend executes only preflight-owned evaluator fixtures.
        # Generated benchmark programs are scored separately with Docker.
        if primary_candidate(study) == "diffusion_full":
            from .diffusion_preflight import check_diffusion_model
            try:
                checked = check_diffusion_model(cfg, output / "checkpoint_checks.json", device)
            except Exception as exc:
                write_json(output / "diagnostics_gate.json", {"registration_id": ident, "passed": False,
                           "reason": "Positive-temperature checkpoint check raised an exception",
                           "error": {"type": type(exc).__name__, "message": str(exc)}})
                raise
        else:
            checked = check_model(cfg, output / "checkpoint_checks.json", device, "process")
        if not checked["passed"] or checked.get("greedy_exact_passed", True) is False:
            write_json(output / "diagnostics_gate.json", {"registration_id": ident, "passed": False,
                       "reason": "Registered checkpoint correctness gate failed"})
            raise RuntimeError("Inspect checkpoint checks before formal timing")
        gc.collect()
        torch.cuda.empty_cache()
        engine, tokenizer = load_frozen_engine(study, device)
        stops = stop_token_ids(engine, tokenizer)
        captures = output / "captures"
        captures.mkdir(exist_ok=True)
        index, stage_rows, replay_rows = [], [], []
        for prompt_index, prompt in enumerate(DIAGNOSTIC_PROMPTS):
            ids = encode_messages(tokenizer, [{"role": "user", "content": prompt}], cfg["model"], device)
            captured = capture_states(study, engine, ids, 105 + prompt_index)
            for state_index, state in enumerate(captured):
                relative = f"captures/p{prompt_index}-r{state_index}.pt"
                torch.save(state, output / relative)
                index.append({"file": relative, "sha256": file_hash(output / relative),
                              "prompt_sha256": digest(prompt), "prompt_index": prompt_index,
                              "kind": state["kind"], "nodes": len(state["parents"]),
                              "vocabulary": state["all_p"].shape[-1]})
                gpu_state = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                             for key, value in state.items()}
                functions = replay_functions(gpu_state)
                functions = {name: fn for name, fn in functions.items() if name in method_names(study)}
                for name, function in functions.items():
                    generator = torch.Generator(device=device).manual_seed(1)
                    for _ in range(spec["replay_warmup"]):
                        function(generator)
                for repeat in range(spec["replay_repeats"]):
                    order = list(functions)
                    random.Random(prompt_index * 1000 + state_index * 10 + repeat).shuffle(order)
                    for name in order:
                        generator = torch.Generator(device=device).manual_seed(105 + repeat)
                        torch.cuda.reset_peak_memory_stats(device)
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        for _ in range(spec["replay_iterations"]):
                            functions[name](generator)
                        torch.cuda.synchronize(device)
                        elapsed = (time.perf_counter() - started) * 1000
                        replay_rows.append({"state": relative, "state_sha256": index[-1]["sha256"],
                                            "method": name, "repeat": repeat,
                                            "iterations": spec["replay_iterations"], "total_ms": elapsed,
                                            "ms_per_call": elapsed / spec["replay_iterations"],
                                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
                del functions, function, gpu_state
            for variant in variants(study):
                engine.generate(ids, variant, spec["warmup_tokens"], [], seed=0)
                result = engine.generate(ids, variant, spec["diagnostic_tokens"], stops,
                                         seed=105 + prompt_index, profile=True)
                stage_rows.append({"prompt_index": prompt_index, "variant": variant.name, **result})
        write_json(output / "capture_index.json", index)
        write_json(output / "verifier_replay.json", {"rows": replay_rows, "primary_timing": False})
        write_json(output / "stage_profiles.json", {"rows": stage_rows, "primary_timing": False})
        artifacts = ["checkpoint_checks.json", "capture_index.json", "verifier_replay.json", "stage_profiles.json"]
        result = {"registration_id": ident, "passed": True, "runtime": env,
                  "artifacts": {name: file_hash(output / name) for name in artifacts},
                  "same_state_latency_only": "replay cannot establish end-to-end speedup"}
        write_json(output / "diagnostics_gate.json", result)
        return result


def run_repeat(study, data_dir, output, device, repeat):
    from .conversation import encode_messages, generate_conversation
    from .report import validate_results
    reg, data = registered(study, data_dir, output)
    ident, spec, cfg = reg["registration_id"], study["spec"], study["config"]
    if repeat not in range(spec["timing_repeats"]):
        raise ValueError("Unregistered timing repeat")
    gate(output, "unit_gate", ident)
    gate(output, "data_gate", ident)
    checked = gate(output, "diagnostics_gate", ident)
    env = runtime(study, device)
    allocation_gate(device)
    if env != read(output / "runtime.json") or checked["runtime"] != env:
        raise ValueError("Runtime changed since GPU correctness checks")
    run_dir = output / f"repeat_{repeat}"
    with output_lock(output), output_lock(run_dir):
        body = {"schema": 4, "registration_id": ident, "repeat": repeat, "model": cfg["model"],
                "variants": cfg["explicit_variants"], "seeds": spec["seeds"],
                "max_new_tokens": spec["max_new_tokens"], "coverage": reg["data_manifest"]["coverage"],
                "evaluation": cfg["evaluation"], "profile": False, "runtime": env,
                "data_manifest": reg["data_manifest"], "dataset_names": cfg["datasets"],
                "dataset_turn_counts": {name: DATASETS[name].turns for name in cfg["datasets"]},
                "prompt_ids": reg["prompt_ids"], "scoring": cfg["scoring"],
                "bootstrap_samples": spec["bootstrap_samples"]}
        run_id = digest(body)
        manifest = {"run_id": run_id, **body}
        freeze_document(run_dir / "run_manifest.json", manifest)
        groups_dir = run_dir / "groups"
        groups_dir.mkdir(exist_ok=True)
        groups = [(row, seed) for row in data for seed in spec["seeds"]]
        random.Random(spec["bootstrap_seed"] + repeat).shuffle(groups)
        expected_files = {digest([r["dataset"], r["source_id"], seed]) + ".json" for r, seed in groups}
        if {p.name for p in groups_dir.glob("*.json")} - expected_files:
            raise ValueError("Unexpected paired group files")
        completed = {}
        for row, seed in groups:
            filename = digest([row["dataset"], row["source_id"], seed]) + ".json"
            path = groups_dir / filename
            if path.exists():
                group = read(path)
                check_group(group, run_id, row["dataset"], row["source_id"], seed, repeat, study)
                completed[filename] = group
        engine = tokenizer = None
        if len(completed) != len(groups):
            engine, tokenizer = load_frozen_engine(study, device)
            stops = stop_token_ids(engine, tokenizer)
            warmup = encode_messages(tokenizer, [{"role": "user", "content": "Compute 1 + 1."}], cfg["model"], device)
            for variant in variants(study):
                engine.generate(warmup, variant, spec["warmup_tokens"], [], seed=0)
        by_name = {v.name: v for v in variants(study)}
        attempt_id = uuid.uuid4().hex
        for row, seed in groups:
            filename = digest([row["dataset"], row["source_id"], seed]) + ".json"
            if filename in completed:
                continue
            per_seed = prompt_seed(seed, row["dataset"], row["source_id"])
            order = paired_order(per_seed, repeat, study)
            group = {"run_id": run_id, "repeat": repeat, "group_key": [row["dataset"], row["source_id"], seed],
                     "attempt_id": attempt_id, "order": order, "before": telemetry(device), "records": []}
            try:
                allocation_gate(device)
                for name in order:
                    model_gate(engine)
                    result = generate_conversation(engine, tokenizer, row, by_name[name], spec["max_new_tokens"],
                                                   stops, per_seed, cfg["model"], profile=False)
                    if any(r["tree_nodes"] > 45 for r in result["rounds"]):
                        raise RuntimeError("Tree budget exceeded")
                    record = {"run_id": run_id, "variant": name, "dataset": row["dataset"],
                              "source_id": row["source_id"], "prompt_sha256": row["prompt_sha256"],
                              "seed": seed, "sampling_seed": per_seed, **result}
                    group["records"].append(record)
                group["after"] = telemetry(device)
                allocation_gate(device)
                group["records_sha256"] = digest(group["records"])
                validate_results(manifest, group["records"], allow_partial=True)
                freeze_document(groups_dir / filename, group)
                completed[filename] = group
                write_json(run_dir / "progress.json", {"run_id": run_id, "completed_groups": len(completed),
                           "expected_groups": len(groups), "last_group": group["group_key"]})
                print(f"repeat={repeat} completed_groups={len(completed)}/{len(groups)}", flush=True)
            except BaseException:
                # Completed groups remain immutable; the interrupted group is
                # recorded as an attempt and rerun in full on the next launch.
                group["traceback"] = traceback.format_exc()
                write_json(run_dir / "attempts" / f"{attempt_id}-{filename}", group)
                raise
        rows = [r for row, seed in groups
                for r in completed[digest([row["dataset"], row["source_id"], seed]) + ".json"]["records"]]
        validate_results(manifest, rows)
        result_path = run_dir / "results.jsonl"
        payload = "".join(canonical(row) + "\n" for row in rows)
        if result_path.exists() and result_path.read_text() != payload:
            raise ValueError("Export differs from immutable paired groups")
        if not result_path.exists():
            temporary = result_path.with_suffix(".jsonl.tmp")
            with temporary.open("w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(result_path)
        value = {"run_id": run_id, "records": len(rows), "groups": len(groups),
                 "results_sha256": file_hash(result_path), "profile": False}
        freeze_document(run_dir / "completed.json", value)
        return value


def score(study, data_dir, output):
    from .scoring import score_run, validate_gold
    reg, _ = registered(study, data_dir, output)
    cfg = study["config"]
    # One prescribed generation repeat supplies quality samples. Repeated
    # timings with the same seed are not independent quality observations.
    settings = cfg["scoring"]
    gold = validate_gold(data_dir, cfg["datasets"], output / "gold_audit.json", "docker",
                         settings["timeout_seconds"], cfg["evaluation"])
    score_run(output / "repeat_0", data_dir, "docker", settings["workers"],
              settings["timeout_seconds"], settings["lcb_timeout_seconds"])
    files = ["gold_audit.json", "repeat_0/scores.jsonl", "repeat_0/scoring_manifest.json"]
    value = {"registration_id": reg["registration_id"], "passed": gold["passed"],
             "artifacts": {name: file_hash(output / name) for name in files},
             "quality_repeat": 0, "mtbench_quality": "not_scored_no_quality_claim"}
    write_json(output / "scoring_gate.json", value)
    return value


def final_report(study, data_dir, output):
    from .report import summarize, validate_results
    reg, _ = registered(study, data_dir, output)
    ident, spec, cfg = reg["registration_id"], study["spec"], study["config"]
    for name in ("unit_gate", "data_gate", "diagnostics_gate", "scoring_gate"):
        gate(output, name, ident)
    for capture in read(output / "capture_index.json"):
        if file_hash(output / capture["file"]) != capture["sha256"]:
            raise ValueError("Captured GPU state changed")
    all_rows, audit_files, quality, repeat_metrics = [], {}, None, {}
    for repeat in range(spec["timing_repeats"]):
        run_dir = output / f"repeat_{repeat}"
        manifest = read(run_dir / "run_manifest.json")
        completed = read(run_dir / "completed.json")
        if (manifest["registration_id"] != ident or manifest["profile"] is not False
                or manifest["repeat"] != repeat or manifest["runtime"] != read(output / "runtime.json")
                or completed["run_id"] != manifest["run_id"]
                or completed["results_sha256"] != file_hash(run_dir / "results.jsonl")):
            raise ValueError("Stale, profiled, or changed formal results")
        body = {k: v for k, v in manifest.items() if k != "run_id"}
        if digest(body) != manifest["run_id"]:
            raise ValueError("Run manifest was modified")
        frozen_fields = {"model": cfg["model"], "variants": cfg["explicit_variants"],
                         "seeds": spec["seeds"], "max_new_tokens": spec["max_new_tokens"],
                         "evaluation": cfg["evaluation"], "data_manifest": reg["data_manifest"],
                         "dataset_names": cfg["datasets"], "prompt_ids": reg["prompt_ids"]}
        if any(manifest.get(key) != value for key, value in frozen_fields.items()):
            raise ValueError("Run is inconsistent with the registered matrix")
        rows = read_jsonl(run_dir / "results.jsonl")
        scores = read_jsonl(run_dir / "scores.jsonl") if repeat == 0 else None
        validate_results(manifest, rows, scores)
        # Exported results must still be backed by the original complete groups.
        persisted = []
        for group_path in sorted((run_dir / "groups").glob("*.json")):
            group = read(group_path)
            persisted.extend(check_group(group, manifest["run_id"], *group["group_key"], repeat, study))
            audit_files[str(group_path.relative_to(output))] = file_hash(group_path)
        if sorted(map(canonical, persisted)) != sorted(map(canonical, rows)):
            raise ValueError("Paired group/export mismatch")
        all_rows.extend({**row, "repeat": repeat} for row in rows)
        repeat_metrics[str(repeat)] = summarize(manifest, rows, bootstrap=0)
        for filename in ("run_manifest.json", "completed.json", "results.jsonl"):
            audit_files[f"repeat_{repeat}/{filename}"] = file_hash(run_dir / filename)
        if repeat == 0:
            quality = summarize(manifest, rows, scores, spec["bootstrap_samples"])
    comparisons = {}
    candidate = primary_candidate(study)
    for baseline in method_names(study):
        if baseline == candidate:
            continue
        for metric in ("decode", "e2e"):
            comparisons[f"{candidate}_vs_{baseline}_{metric}"] = compare(
                all_rows, candidate, baseline, cfg["datasets"], spec["bootstrap_samples"],
                spec["bootstrap_seed"], metric)
    primary = comparisons[f"{candidate}_vs_ddtree_decode"]
    baseline_gates = {"ddtree": primary["ci95"][0] > 1}
    if candidate in {"atom_full", "diffusion_full"}:
        baseline_gates["dflash_match"] = comparisons[f"{candidate}_vs_dflash_match_decode"]["ci95"][0] > 1
    achieved = all(baseline_gates.values())
    first_outputs = {(r["variant"], r["dataset"], r["source_id"], r["seed"]): r["generated_token_ids"]
                     for r in all_rows if r["repeat"] == 0}
    reproducibility = {}
    for repeat in range(1, spec["timing_repeats"]):
        repeated = [r for r in all_rows if r["repeat"] == repeat]
        equal = sum(r["generated_token_ids"] == first_outputs[
                    r["variant"], r["dataset"], r["source_id"], r["seed"]] for r in repeated)
        reproducibility[str(repeat)] = {"records": len(repeated), "same_tokens_as_repeat_0": equal,
                                       "note": "within-method repeatability; not cross-sampler equality"}
    result = {"registration_id": ident, "formal_complete": True,
              "target_achieved": achieved, "baseline_speed_gates": baseline_gates,
              "decision": ("faster_than_both_baselines" if candidate in {"atom_full", "diffusion_full"} else "faster_than_ddtree")
                          if achieved else "speed_target_not_demonstrated",
              "primary": primary, "comparisons": comparisons, "quality_and_metrics": quality,
              "timing_metrics_by_repeat": repeat_metrics, "token_repeatability": reproducibility,
              "scope": spec["claim_scope"], "mtbench_quality": spec["mtbench_quality"],
              "repeats": spec["timing_repeats"], "records": len(all_rows),
              "limitations": ["Exactness is a real-arithmetic theorem, not certified bitwise equivalence",
                              "Repeated timing seeds are not extra quality samples",
                              "GPU allocation sessions are recorded; cluster CI is conditional on measured sessions",
                              "Controlled shared-engine comparison is not a reproduction of all official benchmark settings",
                              "Secondary intervals and ablations are descriptive; no claim of universal speedup"]}
    if candidate == "diffusion_full":
        result.update({"temperature": spec["temperature"], "denoising_steps": 1,
                       "novelty_established": False, "research_goal_achieved": False,
                       "target_achieved_means": "fixed-budget shared-engine speed gate only",
                       "theory_document": "docs/DIFFUSION_TREE_BV.md",
                       "theory_comparison_document": "docs/DIFFUSION_TREE_THEORY_AUDIT.md",
                       "universal_acceptance_dominance": "disproved_by_counterexample",
                       "conditional_acceptance_advantage": "proved_under_unverified_distribution_conditions"})
        if spec["study"] == "diffusion_scaffold_verification":
            result.update({"theory_document": "docs/DIFFUSION_SCAFFOLD_BV.md",
                           "theory_comparison_document": "docs/DIFFUSION_SCAFFOLD_BV.md",
                           "universal_acceptance_dominance": "disproved_vs_ddtree_by_scaffold_counterexample",
                           "dflash_acceptance_dominance": "proved_at_fixed_context_under_exact_oracle",
                           "conditional_acceptance_advantage": "proved_examples_not_certified_on_frozen_models"})
    for name in ("registration.json", "runtime.json", "unit_gate.json", "data_gate.json",
                 "diagnostics_gate.json", "scoring_gate.json"):
        audit_files[name] = file_hash(output / name)
    for name in ("diagnostics_gate", "scoring_gate", "unit_gate", "data_gate"):
        audit_files.update(read(output / f"{name}.json").get("artifacts", {}))
    audit_files.update({item["file"]: item["sha256"] for item in read(output / "capture_index.json")})
    with output_lock(output):
        write_json(output / "formal_report.json", result)
        table = []
        for comparison in comparisons.values():
            for dataset, values in comparison["per_dataset"].items():
                table.append({"candidate": comparison["candidate"], "baseline": comparison["baseline"],
                              "metric": comparison["metric"], "dataset": dataset,
                              "speedup": values["speedup"], "ci_low": values["ci95"][0],
                              "ci_high": values["ci95"][1], "candidate_tps": values["candidate_tps"],
                              "baseline_tps": values["baseline_tps"]})
        with (output / "comparisons.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        lines = [f"# {spec['study']} registered experiment", "",
                 f"Decision: {result['decision']}. Complete records: {len(all_rows)}.", "",
                 "Primary: equal-dataset geometric mean of decode speedups vs DDTree.", "",
                 "| Baseline | Metric | Geomean speedup | 95% paired CI |",
                 "| --- | --- | ---: | --- |"]
        for comparison in comparisons.values():
            low, high = comparison["ci95"]
            lines.append(f"| {comparison['baseline']} | {comparison['metric']} | "
                         f"{comparison['dataset_geomean_speedup']:.4f} | [{low:.4f}, {high:.4f}] |")
        if candidate == "diffusion_full":
            lines.extend(["", f"Target and draft temperature: {spec['temperature']}; one denoising step.",
                          "Passing the speed gate does not establish novelty or complete the research goal."])
            if spec["study"] == "diffusion_scaffold_verification":
                lines.extend(["The scaffold has a fixed-context DFlash acceptance floor, not a speed guarantee.",
                              "Universal DDTree dominance and frozen-model distribution assumptions remain unverified."])
            else:
                lines.extend(["Universal acceptance dominance is disproved; conditional advantage requires",
                              "distribution assumptions not certified by this timing report."])
        lines.extend(["", "Secondary intervals are descriptive. See comparisons.csv for every dataset,",
                      "formal_report.json for timing repeats and objective quality, and artifact_index.json",
                      "for provenance. MT-Bench quality is unscored. This is a controlled shared-engine",
                      "single-model experiment, not a claim of universal speedup or publication novelty.", ""])
        (output / "report.md").write_text("\n".join(lines))
        audit_files["formal_report.json"] = file_hash(output / "formal_report.json")
        for name in ("comparisons.csv", "report.md"):
            audit_files[name] = file_hash(output / name)
        write_json(output / "artifact_index.json", {"registration_id": ident, "sha256": audit_files})
    return result


def status(study, output):
    phases = {name: (output / file).exists() for name, file in
              (("registered", "registration.json"), ("unit_tests", "unit_gate.json"),
               ("diagnostics", "diagnostics_gate.json"), ("scoring", "scoring_gate.json"),
               ("report", "formal_report.json"))}
    repeats = {str(i): read(output / f"repeat_{i}/progress.json")
               if (output / f"repeat_{i}/progress.json").exists() else {"completed_groups": 0}
               for i in range(study["spec"]["timing_repeats"])}
    return {"artifacts_present_not_validated": phases, "repeats": repeats,
            "note": "Only the report stage validates all evidence and marks formal completion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "freeze", "test", "diagnostics", "run", "score", "report", "status"])
    parser.add_argument("--study", type=Path, default=ROOT / "configs/terminal_mass_formal.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets/gbv_paper_ddtree_counts")
    parser.add_argument("--output", type=Path, help="Default: outputs/<study filename without extension>")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    study = load_study(args.study)
    args.output = (args.output or ROOT / "outputs" / args.study.stem).resolve()
    args.data_dir = args.data_dir.resolve()
    if args.stage == "plan":
        value = plan(study)
    elif args.stage == "status":
        value = status(study, args.output)
    elif args.stage == "freeze":
        value = freeze(study, args.data_dir, args.output)
    elif args.stage == "test":
        value = unit_gate(study, args.data_dir, args.output)
    elif args.stage == "diagnostics":
        value = diagnostics(study, args.data_dir, args.output, args.device)
    elif args.stage == "run":
        value = run_repeat(study, args.data_dir, args.output, args.device, args.repeat)
    elif args.stage == "score":
        value = score(study, args.data_dir, args.output)
    else:
        value = final_report(study, args.data_dir, args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
