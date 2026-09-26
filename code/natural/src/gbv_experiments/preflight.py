"""Production-checkpoint checks executed on the benchmark GPU before long runs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import subprocess

from .common import ROOT, digest, file_hash, source_hashes, write_json
from .config import Variant, build_variants
from .runtime_model import (enforce_runtime_model_gate, is_registered_same_tree_t1,
                            runtime_model_expectations)


BF16_NUMERICAL_LOGIT_ERROR_LIMIT = 0.5
SAME_TREE_WITNESS_PROMPT = "Compute 19 + 23. Give a brief explanation."
SAME_TREE_WITNESS_SEED = 205
SAME_TREE_VARIANT_NAMES = (
    "ddtree_t1p0", "tree_block_verification_t1p0",
)
SAME_TREE_VERIFIER_EXPECTATIONS = {
    "ddtree_t1p0":{
        "module":"gbv_experiments.sampling",
        "qualname":"tree_verify_ancestral_batched",
        "callable":"gbv_experiments.sampling.tree_verify_ancestral_batched",
        "source_file":"src/gbv_experiments/sampling.py",
    },
    "tree_block_verification_t1p0":{
        "module":"gbv_experiments.fused_tree_sampling",
        "qualname":"tree_verify_ancestral_fused_scan",
        "callable":(
            "gbv_experiments.fused_tree_sampling."
            "tree_verify_ancestral_fused_scan"
        ),
        "source_file":"src/gbv_experiments/fused_tree_sampling.py",
    },
}


def _tensor_sha256(value) -> str:
    """Hash a detached CPU tensor together with its numerical representation."""
    tensor = value.detach().cpu().contiguous()
    payload = hashlib.sha256()
    payload.update(str(tensor.dtype).encode())
    payload.update(str(tuple(tensor.shape)).encode())
    payload.update(tensor.numpy().tobytes())
    return payload.hexdigest()


def _same_tree_runtime_witness(engine, tokenizer, model, variants, stop_ids,
                               device) -> dict | None:
    """Bind equal real T=1 inputs to the verifier callables that consumed them."""
    by_name = {variant.name: variant for variant in variants}
    names = SAME_TREE_VARIANT_NAMES
    if not set(names) <= set(by_name):
        return None

    import torch

    ids = tokenizer.apply_chat_template(
        [{"role":"user", "content":SAME_TREE_WITNESS_PROMPT}],
        tokenize=True, add_generation_prompt=True,
        enable_thinking=bool(model.get("enable_thinking", False)),
        return_tensors="pt",
    ).to(device)
    captured = {}
    verifier_events = {name:[] for name in names}
    for name in names:
        def observe(parents, tree_tokens, all_p, witness_name=name):
            if witness_name not in captured:
                captured[witness_name] = (
                    list(parents), list(tree_tokens),
                    all_p.detach().cpu().clone(),
                )

        def observe_verifier(identity, witness_name=name):
            verifier_events[witness_name].append(identity)

        engine.generate(
            ids, by_name[name], 32, stop_ids,
            seed=SAME_TREE_WITNESS_SEED, tree_observer=observe,
            verifier_observer=observe_verifier,
        )

    target_vocab_size = engine.target.config.vocab_size
    verifier_observer_calls = {
        name:len(verifier_events[name]) for name in names
    }
    if set(captured) != set(names):
        return {
            "passed":False,
            "failure":"a registered method did not expose its first probability tree",
            "captured_variants":sorted(captured),
            "target_vocab_size":target_vocab_size,
            "verifier_observer_calls":verifier_observer_calls,
        }
    if any(count != 1 for count in verifier_observer_calls.values()):
        return {
            "passed":False,
            "failure":"a registered method did not execute exactly one observed verifier",
            "captured_variants":sorted(captured),
            "target_vocab_size":target_vocab_size,
            "verifier_observer_calls":verifier_observer_calls,
        }
    baseline = captured[names[0]]
    candidate = captured[names[1]]
    parents_equal = baseline[0] == candidate[0]
    tokens_equal = baseline[1] == candidate[1]
    probabilities_equal = torch.equal(baseline[2], candidate[2])

    def plain_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def valid_tree_state(name, state) -> bool:
        parents, tree_tokens, probabilities = state
        if (not plain_int(target_vocab_size) or target_vocab_size < 1
                or len(parents) != by_name[name].tree_budget + 1
                or len(tree_tokens) != len(parents) - 1
                or list(probabilities.shape) != [len(parents), target_vocab_size]
                or str(probabilities.dtype) != "torch.float64"
                or not parents or not plain_int(parents[0]) or parents[0] != -1
                or any(not plain_int(parent) or parent < 0 or parent >= node
                       for node, parent in enumerate(parents[1:], 1))
                or any(not plain_int(token)
                       or token < 0 or token >= target_vocab_size
                       for token in tree_tokens)
                or len(set(zip(parents[1:], tree_tokens))) != len(tree_tokens)):
            return False
        depths = [0] * len(parents)
        for node, parent in enumerate(parents[1:], 1):
            depths[node] = depths[parent] + 1
        return max(depths) <= by_name[name].length

    def verifier_output_matches(state, output) -> bool:
        if (not isinstance(output, dict)
                or set(output) != {
                    "accepted_nodes", "accepted_tokens", "bonus_token",
                }):
            return False
        nodes = output.get("accepted_nodes")
        output_tokens = output.get("accepted_tokens")
        bonus = output.get("bonus_token")
        if (not isinstance(nodes, list) or not isinstance(output_tokens, list)
                or len(nodes) != len(output_tokens)
                or not plain_int(bonus)
                or bonus < 0 or bonus >= target_vocab_size):
            return False
        parents, tree_tokens, _ = state
        previous = 0
        for node, token in zip(nodes, output_tokens):
            if (not plain_int(node) or not 0 < node < len(parents)
                    or not plain_int(token)
                    or parents[node] != previous
                    or tree_tokens[node - 1] != token):
                return False
            previous = node
        child_edges = {
            (parent, tree_tokens[node - 1])
            for node, parent in enumerate(parents[1:], 1)
        }
        return (previous, bonus) not in child_edges

    def valid_generator_before(value) -> bool:
        return (
            isinstance(value, dict)
            and set(value) == {"initial_seed", "device", "state_sha256"}
            and plain_int(value.get("initial_seed"))
            and value["initial_seed"] == SAME_TREE_WITNESS_SEED
            and value.get("device") == str(device)
            and isinstance(value.get("state_sha256"), str)
            and len(value["state_sha256"]) == 64
            and all(character in "0123456789abcdef"
                    for character in value["state_sha256"])
        )

    def verifier_identity_matches(name, state, actual) -> bool:
        expected = SAME_TREE_VERIFIER_EXPECTATIONS[name]
        expected_input = {
            "tree_sha256":digest([state[0], state[1]]),
            "probabilities_sha256":_tensor_sha256(state[2]),
            "probability_shape":list(state[2].shape),
            "probability_dtype":str(state[2].dtype),
        }
        actual_input = actual.get("input", {}) if isinstance(actual, dict) else {}
        actual_output = actual.get("output", {}) if isinstance(actual, dict) else {}
        return (
            isinstance(actual, dict)
            and set(actual) == {
                "module", "qualname", "callable", "source_file",
                "source_sha256", "observed_call_index", "input",
                "input_sha256", "output", "output_sha256",
            }
            and all(actual.get(field) == value
                    for field, value in expected.items())
            and actual.get("source_sha256")
               == file_hash(ROOT / expected["source_file"])
            and plain_int(actual.get("observed_call_index"))
            and actual["observed_call_index"] == 1
            and isinstance(actual_input, dict)
            and set(actual_input) == {
                *expected_input, "generator_before", "validate",
            }
            and all(actual_input.get(field) == value
                    for field, value in expected_input.items())
            and valid_generator_before(actual_input.get("generator_before"))
            and actual_input.get("validate") is False
            and actual.get("input_sha256") == digest(actual_input)
            and verifier_output_matches(state, actual_output)
            and actual.get("output_sha256") == digest(actual_output)
        )

    tree_states_valid = all(
        valid_tree_state(name, captured[name]) for name in names
    )
    verifier_routes_correct = all(
        verifier_identity_matches(name, captured[name], verifier_events[name][0])
        for name in names
    )
    def observed_generator_before(name):
        event = verifier_events[name][0]
        event_input = event.get("input") if isinstance(event, dict) else None
        return (
            event_input.get("generator_before")
            if isinstance(event_input, dict) else None
        )

    baseline_generator_before = observed_generator_before(names[0])
    candidate_generator_before = observed_generator_before(names[1])
    verifier_generator_inputs_equal = (
        baseline_generator_before is not None
        and baseline_generator_before == candidate_generator_before
    )
    probability_shape = list(baseline[2].shape)
    probability_vocab_matches_target = (
        plain_int(target_vocab_size) and target_vocab_size > 0
        and len(probability_shape) == 2
        and probability_shape[1] == target_vocab_size
    )

    def identity(name, state):
        return {
            "name":name,
            "method":by_name[name].method,
            "variant":by_name[name].to_dict(),
            "parents":state[0],
            "tree_tokens":state[1],
            "tree_sha256":digest([state[0], state[1]]),
            "probabilities_sha256":_tensor_sha256(state[2]),
            "probability_shape":list(state[2].shape),
            "probability_dtype":str(state[2].dtype),
            "target_vocab_size":target_vocab_size,
            "verifier_observer_calls":len(verifier_events[name]),
            "verifier":verifier_events[name][0],
        }

    return {
        "passed":(
            parents_equal and tokens_equal and probabilities_equal
            and tree_states_valid and verifier_routes_correct
            and verifier_generator_inputs_equal
            and probability_vocab_matches_target
        ),
        "prompt_sha256":digest(SAME_TREE_WITNESS_PROMPT),
        "seed":SAME_TREE_WITNESS_SEED,
        "parents_equal":parents_equal,
        "tree_tokens_equal":tokens_equal,
        "target_probabilities_equal":probabilities_equal,
        "tree_states_valid":tree_states_valid,
        "verifier_routes_correct":verifier_routes_correct,
        "verifier_generator_inputs_equal":verifier_generator_inputs_equal,
        "verifier_observer_calls":verifier_observer_calls,
        "verifier_observation_policy":"first_successful_call_only",
        "expected_verifier_observer_calls_per_method":1,
        "target_vocab_size":target_vocab_size,
        "probability_vocab_matches_target":probability_vocab_matches_target,
        "probability_shape":probability_shape,
        "probability_dtype":str(baseline[2].dtype),
        "baseline":identity(names[0], baseline),
        "candidate":identity(names[1], candidate),
    }


def classify_greedy_mismatch(mismatch, reference_trace, *, dtype,
                             max_logit_error_limit=BF16_NUMERICAL_LOGIT_ERROR_LIMIT):
    """Allow only a stable, bounded BF16 near-tie; all unexplained mismatches fail closed."""
    index = mismatch["first_index"]
    reference_point = next((point for point in reference_trace
                            if point["generated_index"] == index), None)
    audit_point = None
    for round_ in mismatch.get("greedy_logit_audit", []):
        offset = index - round_["generated_start_index"]
        if 0 <= offset < len(round_["tree_token_ids"]):
            audit_point = {
                "tree_token_id": round_["tree_token_ids"][offset],
                "sequential_token_id": round_["sequential_token_ids"][offset],
                "max_absolute_logit_error": round_["max_absolute_logit_error"][offset],
                "tree_top1_margin": round_["tree_top1_margin"][offset],
                "sequential_top1_margin": round_["sequential_top1_margin"][offset],
            }
            break

    evidence = {"reference_trace": reference_point, "tree_sequential_audit": audit_point}
    conditions = {
        "bf16_checkpoint": dtype == "bfloat16",
        "different_tokens": mismatch.get("reference_token_id") is not None
                            and mismatch.get("result_token_id") is not None
                            and mismatch["reference_token_id"] != mismatch["result_token_id"],
        "reference_repeat_stable": bool(mismatch.get("reference_repeat_equal")),
        "candidate_repeat_stable": bool(mismatch.get("result_audit_repeat_equal")),
        "reference_trace_present": reference_point is not None,
        "tree_audit_present": audit_point is not None,
    }
    if reference_point is not None:
        conditions["reference_trace_matches_output"] = (
            reference_point["token_id"] == mismatch.get("reference_token_id")
        )
    if audit_point is not None:
        error = float(audit_point["max_absolute_logit_error"])
        ambiguity_bound = 2.0 * error
        values = (error, float(reference_point["top1_margin"]) if reference_point else math.inf,
                  float(audit_point["tree_top1_margin"]),
                  float(audit_point["sequential_top1_margin"]))
        evidence["ambiguity_margin_bound"] = ambiguity_bound
        evidence["max_logit_error_limit"] = max_logit_error_limit
        conditions.update({
            "tree_trace_matches_output": audit_point["tree_token_id"] == mismatch.get("result_token_id"),
            "finite_nonzero_error": all(math.isfinite(value) for value in values) and error > 0,
            "bounded_logit_error": error <= max_logit_error_limit,
            "reference_is_near_tie": reference_point is not None
                                     and float(reference_point["top1_margin"]) <= ambiguity_bound,
            "tree_is_near_tie": float(audit_point["tree_top1_margin"]) <= ambiguity_bound,
            "sequential_replay_is_near_tie": (
                float(audit_point["sequential_top1_margin"]) <= ambiguity_bound
            ),
        })
    passed = all(conditions.values())
    return {
        "classification": "bounded_bf16_near_tie" if passed else "unexplained_greedy_mismatch",
        "gate_passed": passed,
        "conditions": conditions,
        "failed_conditions": [name for name, value in conditions.items() if not value],
        "evidence": evidence,
    }


def check_environment(cfg, code_backend="docker", device="cuda:0"):
    import torch
    import math_verify  # noqa: F401
    from transformers import Qwen3ForCausalLM  # noqa: F401
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(device)
    if cfg["model"].get("dtype", "bfloat16") == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The GPU does not support BF16")
    image_id = None
    process_identity = {
        "process_python":None, "process_python_resolved":None,
        "process_python_sha256":None, "process_python_runtime":None,
        "process_pyvenv_cfg_sha256":None,
    }
    if code_backend == "docker":
        image_id = subprocess.check_output(["docker", "image", "inspect", "gbv-code-eval:py311",
                                           "--format", "{{.Id}}"], text=True).strip()
        subprocess.run(["docker", "run", "--rm", "--network=none", "gbv-code-eval:py311",
                        "python", "-c", "import numpy, sympy; print('code evaluator ready')"], check=True)
    elif code_backend == "process":
        from .scoring import process_python_identity
        process_identity = process_python_identity()
    else:
        raise ValueError("Unknown code scoring backend")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "huggingface-hub", "datasets", "numpy", "math-verify")}
    evaluator_checks = []
    if "livecodebench" in cfg["datasets"]:
        from .lcb import evaluate_lcb
        # The chosen production backend must execute both LCB calling conventions.
        for functional in (False, True):
            evaluation = {"fn_name": "add" if functional else None, "tests": [
                {"input": "1\n2" if functional else "1 2\n", "output": "3"},
                {"input": "7\n9" if functional else "7 9\n", "output": "16"}]}
            for correct in (False, True):
                expression = "a+b" if correct else "3"
                code = ("class Solution:\n    def add(self,a,b): return " + expression if functional else
                        "import sys\na,b=map(int,sys.stdin.buffer.read().split())\nprint(" + expression + ")")
                result = evaluate_lcb(code, evaluation, backend=code_backend, timeout=2)
                if result["passed"] is not correct:
                    raise RuntimeError(f"LiveCodeBench evaluator preflight failed: {result}")
                evaluator_checks.append({"functional": functional, "expected_pass": correct, "result": result})
    properties = torch.cuda.get_device_properties(device)
    return {"versions": versions, "python": platform.python_version(), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_uuid":str(getattr(properties, "uuid", "unknown")),
            "total_memory_bytes": properties.total_memory,
            "code_backend": code_backend, "code_image_id": image_id,
            **process_identity, "lcb_evaluator_checks": evaluator_checks}


def check_model(cfg, output: Path, device="cuda:0", code_backend="docker", only_variants=None):
    import torch
    from .engine import load_models
    from .runner import model_identity, stop_token_ids
    from .tree import sampled_tree, compact_cache

    environment = check_environment(cfg, code_backend, device)
    model = model_identity(cfg["model"])
    engine, tokenizer = load_models(model, device)
    registered_same_tree_t1 = is_registered_same_tree_t1(cfg)
    runtime_expectations = runtime_model_expectations(
        cfg["model"], strict_same_tree_t1=registered_same_tree_t1,
        device=str(torch.device(device)) if registered_same_tree_t1 else None,
    )
    runtime_identity_after_load = enforce_runtime_model_gate(
        engine, runtime_expectations,
        tokenizer=tokenizer if registered_same_tree_t1 else None,
    )
    stop_ids = stop_token_ids(engine, tokenizer)
    prompts = ["Compute 19 + 23. Give a brief explanation.",
               "Write a Python function that reverses a list.",
               "Explain why the sum of two even integers is even."]
    from .config import select_variants
    variants = [Variant(**e["variant"]) for e in select_variants(build_variants(cfg), only_variants)]
    # Keep every structural variation; set T=0 solely for the equality gate.
    unique = {}
    for v in variants:
        v = replace(v, temperature=0., draft_temperature=1.)
        unique[tuple((k, value) for k, value in v.to_dict().items() if k != "name")] = v
    checks = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=bool(model.get("enable_thinking", False)), return_tensors="pt").to(device)
            reference = engine.generate(
                ids, Variant(name="ar", method="target", paths=1, temperature=0),
                64, stop_ids, seed=105, audit_greedy=True
            )
            for v in unique.values():
                result = engine.generate(ids, v, 64, stop_ids, seed=105)
                reference_ids = reference["generated_token_ids"]
                result_ids = result["generated_token_ids"]
                equal = result_ids == reference_ids
                check = {"prompt_sha256": digest(prompt), "variant": v.to_dict(),
                         "greedy_equal": equal, "gate_passed": equal}
                if not equal:
                    common = min(len(reference_ids), len(result_ids))
                    first_mismatch = next((i for i in range(common) if reference_ids[i] != result_ids[i]), common)
                    repeated_reference = engine.generate(
                        ids, Variant(name="ar_repeat", method="target", paths=1, temperature=0),
                        64, stop_ids, seed=105, audit_greedy=True
                    )
                    audited_result = engine.generate(ids, v, 64, stop_ids, seed=105, audit_greedy=True)
                    mismatch = {
                        "first_index": first_mismatch,
                        "reference_token_id": reference_ids[first_mismatch] if first_mismatch < len(reference_ids) else None,
                        "result_token_id": result_ids[first_mismatch] if first_mismatch < len(result_ids) else None,
                        "reference_token_ids": reference_ids,
                        "result_token_ids": result_ids,
                        "reference_finish_reason": reference["finish_reason"],
                        "result_finish_reason": result["finish_reason"],
                        "repeated_reference_token_ids": repeated_reference["generated_token_ids"],
                        "reference_repeat_equal": repeated_reference["generated_token_ids"] == reference_ids,
                        "audited_result_token_ids": audited_result["generated_token_ids"],
                        "result_audit_repeat_equal": audited_result["generated_token_ids"] == result_ids,
                        "result_rounds": result["rounds"],
                        "greedy_logit_audit": audited_result["greedy_audit"],
                    }
                    diagnosis = classify_greedy_mismatch(
                        mismatch, reference["target_greedy_trace"], dtype=model.get("dtype", "bfloat16")
                    )
                    mismatch["numerical_diagnosis"] = diagnosis
                    check["mismatch"] = mismatch
                    check["numerical_ambiguity"] = diagnosis["gate_passed"]
                    check["gate_passed"] = diagnosis["gate_passed"]
                checks.append(check)
            # Verify actual checkpoint logits and KV compaction on a branched tree.
            future = reference["generated_token_ids"][:4]
            paths = torch.tensor([future[1:4], [future[1], future[3], future[2]]], device=device)
            tree = sampled_tree(paths)
            prefix_length = ids.shape[1]
            cache = engine.cache_factory()
            engine.target_forward(ids, cache, hidden=False)
            tree_out = engine.target_forward(torch.tensor([[future[0]] + tree.tokens], device=device), cache,
                hidden=False, positions=(torch.tensor(tree.depths, device=device) + prefix_length)[None],
                mask=tree.mask(prefix_length, next(engine.target.parameters()).dtype, device))
            for path, nodes in zip(paths, tree.path_nodes):
                sequence = torch.cat([ids, torch.tensor([[future[0]]], device=device), path[None]], 1)
                sequential = engine.target(sequence).logits[:, prefix_length:]
                actual = tree_out.logits[:, [0] + nodes]
                equal = bool((actual.argmax(-1) == sequential.argmax(-1)).all())
                checks.append({"check": "tree_argmax", "passed": equal,
                               "max_absolute_logit_error": float((actual - sequential).abs().max())})
            keep = [0] + tree.path_nodes[1][:2]
            compact_cache(cache, prefix_length, keep, device)
            token = future[-1]
            actual = engine.target_forward(torch.tensor([[token]], device=device), cache, hidden=False).logits[:, -1]
            sequence = torch.cat([ids, torch.tensor([[future[0], int(paths[1, 0]), int(paths[1, 1]), token]], device=device)], 1)
            sequential = engine.target(sequence).logits[:, -1]
            checks.append({"check": "compacted_cache_argmax", "passed": bool((actual.argmax(-1) == sequential.argmax(-1)).all()),
                           "max_absolute_logit_error": float((actual - sequential).abs().max())})
        same_tree_runtime_witness = _same_tree_runtime_witness(
            engine, tokenizer, model, variants, stop_ids, device,
        )
    runtime_identity_after_preflight = enforce_runtime_model_gate(
        engine, runtime_expectations,
        tokenizer=tokenizer if registered_same_tree_t1 else None,
    )
    runtime_identities_match = (
        runtime_identity_after_preflight == runtime_identity_after_load
    )
    if not runtime_identities_match:
        raise RuntimeError("Runtime model identity changed during GPU preflight")
    generation_checks = [check for check in checks if "greedy_equal" in check]
    passed = (
        all(c.get("gate_passed", c.get("passed", False)) for c in checks)
        and (same_tree_runtime_witness is None
             or same_tree_runtime_witness.get("passed") is True)
    )
    greedy_exact_passed = all(check["greedy_equal"] for check in generation_checks)
    numerical_ambiguities = sum(bool(check.get("numerical_ambiguity")) for check in generation_checks)
    result = {"passed": passed, "greedy_exact_passed": greedy_exact_passed,
              "numerical_ambiguities": numerical_ambiguities,
              "numerical_policy": {
                  "dtype": "bfloat16_only",
                  "max_absolute_logit_error": BF16_NUMERICAL_LOGIT_ERROR_LIMIT,
                  "required": "stable repeated outputs and top-1 margins <= 2 * measured error",
              },
              "scope": "checkpoint structural and bounded-numerical smoke gate, not full benchmark results",
              "environment": environment, "model": model, "stop_token_ids": stop_ids,
              "source_hashes": source_hashes(), "checks": checks,
              "runtime_model_gate":{
                  "schema":1,
                  "expectations":runtime_expectations,
                  "identity_after_load":runtime_identity_after_load,
                  "identity_after_preflight":runtime_identity_after_preflight,
                  "passed_after_load":True,
                  "passed_after_preflight":True,
                  "identities_match":runtime_identities_match,
              },
              "same_tree_runtime_witness":same_tree_runtime_witness}
    write_json(output, result)
    if not passed:
        raise RuntimeError(f"GPU correctness gate failed; inspect {output}. Do not launch formal timing runs.")
    return result
