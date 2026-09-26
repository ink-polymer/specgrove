"""Positive-temperature checkpoint checks for the one-step diffusion study."""
from dataclasses import replace
import math

import torch

from .common import digest, write_json
from .config import Variant, DIFFUSION_SCAFFOLD_METHODS, DIFFUSION_LAW_METHODS as DIFFUSION_TREE_METHODS
from .sampling import probabilities


MAX_NODE_TV = 1e-3  # Numerical smoke threshold, not an exactness theorem.


def numerical_probe(device):
    """Compare the full-depth diffusion flow to CPU FP64 with the SAME witness."""
    from . import diffusion_tree_bv as diffusion, atom_tree_bv as transport
    errors, cases = [], 0
    g = torch.Generator().manual_seed(20260907)
    for length in (1, 2, 15, 129):
        logits = torch.randn(length, 17, generator=g, dtype=torch.float64)
        law = diffusion.DiffusionBlockLaw.from_logits(
            logits, torch.tensor([0] + [16] * length), .6, 16)
        for k in (1, 3):
            proposal = diffusion.propose(law, k, g)
            proposal.validate(17)
            state = diffusion.snapshot(proposal)
            remote = diffusion.restore({key: value.to(device) if isinstance(value, torch.Tensor) else value
                                        for key, value in state.items()})
            remote.validate(17)
            p = torch.randn(k, length, 17, generator=g, dtype=torch.float64).softmax(-1)
            p[..., 0] = 0
            p /= p.sum(-1, keepdim=True)
            support = p.gather(-1, proposal.tokens)
            alpha = p.new_full((k,), 1 / k)
            for pool in (False, True):
                reference = transport.plan(alpha, support, proposal, pool=pool)
                actual = transport.plan(alpha.to(device), support.to(device), remote, pool=pool)
                for key, value in vars(reference).items():
                    measured = getattr(actual, key).cpu()
                    torch.testing.assert_close(measured, value, rtol=1e-10, atol=1e-12)
                    errors.append(float((measured - value).abs().max()))
                if not bool(torch.isfinite(actual.scores).all()
                            & (actual.scores >= actual.floors - 1e-12).all()
                            & (actual.scores.sum(0) <= 1 + 1e-12).all()
                            & (actual.flow.sum(0) <= remote.source + 1e-12).all()
                            & (actual.residual_mass >= 0).all()):
                    raise AssertionError("Diffusion flow numerical invariant failed")
                cases += 1
    return {"passed": True, "device": str(device), "cases": cases,
            "max_absolute_error": max(errors), "stochastic_first_token": True,
            "scope": "same-witness full-depth CPU/device FP64 parity, not RNG unbiasedness or a proof"}


def _check_probabilities(p, label):
    if (p.ndim != 2 or min(p.shape) < 1 or not p.is_floating_point()
            or not bool(torch.isfinite(p).all() & (p >= 0).all()
                        & torch.isclose(p.sum(-1), torch.ones_like(p.sum(-1)),
                                        rtol=0, atol=1e-10).all())):
        raise FloatingPointError(f"{label}: invalid or nonfinite probability rows")


def _checked_probabilities(logits, temperature, label):
    # -inf entries are valid target-zero tokens; NaN, +inf and all--inf rows
    # are not valid logits. Check every row, including unselected tree leaves.
    if (logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point()
            or bool(torch.isnan(logits).any() | torch.isposinf(logits).any()
                    | ~torch.isfinite(logits).any(-1).all())):
        raise FloatingPointError(f"{label}: invalid or nonfinite target logits")
    p = probabilities(logits, temperature, torch.float64)
    _check_probabilities(p, label)
    return p


def _node_tv(actual, reference):
    _check_probabilities(actual, "captured target")
    _check_probabilities(reference, "cached AR reference")
    if actual.shape != reference.shape:
        raise ValueError("Captured/reference probability shape mismatch")
    error = float(0.5 * (actual - reference).abs().sum())
    if not math.isfinite(error):
        raise FloatingPointError("Nonfinite target total variation")
    return error


def _tree_errors(engine, ids, prefix, parents, tokens, actual, temperature):
    """Check every node against cached AR, visiting each node once in DFS order.

    Cropping this independent sequential cache back to a parent avoids repeating
    long prompt prefills for each DDTree leaf. It never uses compact_cache.
    """
    if (not parents or parents[0] != -1 or len(parents) != len(tokens) + 1
            or actual.shape[0] != len(parents)
            or any(parent < 0 or parent >= node for node, parent in enumerate(parents[1:], 1))):
        raise ValueError("Invalid captured target tree")
    cache = engine.cache_factory()
    output = engine.target_forward(ids, cache, hidden=False, last_only=True)
    reference = _checked_probabilities(output.logits[0, -1:], temperature, "reference prefill")
    for token in prefix:
        output = engine.target_forward(torch.tensor([[token]], device=ids.device),
                                       cache, hidden=False, last_only=True)
        reference = _checked_probabilities(output.logits[0, -1:], temperature, "reference prefix")
    errors = [_node_tv(actual[:1], reference)]
    children = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        children[parent].append(node)
    stack = [(node, 1) for node in reversed(children[0])]
    base_length = ids.shape[1] + len(prefix)
    while stack:
        node, depth = stack.pop()
        cache.crop(base_length + depth - 1)
        output = engine.target_forward(torch.tensor([[tokens[node - 1]]], device=ids.device),
                                       cache, hidden=False, last_only=True)
        reference = _checked_probabilities(output.logits[0, -1:], temperature, "reference tree node")
        errors.append(_node_tv(actual[node:node + 1], reference))
        stack.extend((child, depth + 1) for child in reversed(children[node]))
    return errors


def scaffold_contract(tree, proposal, variant, greedy, nodes, tokens, bonus, vocab):
    """Independently check observed premises, not just self-reported round flags.

    Greedy must come from the ORIGINAL draft logits' argmax, not from top-k
    support ordering or a second invocation of the scaffold builder. Selection
    is checked before EOS/cap truncation. This finite witness is not a proof of
    distributional correctness or of coverage at all future model contexts.
    """
    from .tree import sampled_tree

    variant.validate()
    proposal.validate(vocab)
    if variant.method not in DIFFUSION_SCAFFOLD_METHODS:
        raise ValueError("Scaffold contract requires a scaffold variant")
    labelled_length = (
        variant.diffusion_spur_length
        if variant.method == "diffusion_core_spur_bv"
        else variant.length
    )
    if (len(tree.tokens) > variant.tree_budget or len(tree.parents) != len(tree.tokens) + 1
            or len(tree.depths) != len(tree.parents) or tree.parents[0] != -1 or tree.depths[0] != 0
            or proposal.slots.shape[:2] != (variant.paths, labelled_length)):
        raise ValueError("Scaffold contract: invalid budget, topology or labelled shape")
    children = {}
    for node, (parent, token) in enumerate(zip(tree.parents[1:], tree.tokens), 1):
        if (not 0 <= parent < node or not 0 <= token < vocab or (parent, token) in children
                or tree.depths[node] != tree.depths[parent] + 1 or tree.depths[node] > variant.length):
            raise ValueError("Scaffold contract: invalid tree edge")
        children[parent, token] = node
    base = sampled_tree(proposal.paths())
    count = len(base.parents)
    if (tree.parents[:count] != base.parents or tree.tokens[:count - 1] != base.tokens
            or tree.depths[:count] != base.depths or tree.path_nodes != base.path_nodes):
        raise ValueError("Scaffold contract: retained labelled samples changed")
    if len(greedy) != variant.length:
        raise ValueError("Scaffold contract: missing original greedy tokens")
    parent, greedy_nodes = 0, []
    if variant.method != "diffusion_core_spur_bv":
        for token in greedy:
            if (parent, token) not in children:
                raise ValueError("Scaffold contract: original greedy path is missing")
            parent = children[parent, token]
            greedy_nodes.append(parent)
    if len(nodes) != len(tokens):
        raise ValueError("Scaffold contract: selected node/token length mismatch")
    parent = 0
    for node, token in zip(nodes, tokens):
        if children.get((parent, token)) != node:
            raise ValueError("Scaffold contract: selected tokens are not a tree path")
        parent = node
    if not 0 <= bonus < vocab:
        raise ValueError("Scaffold contract: invalid bonus token")
    recycle = variant.method != "diffusion_scaffold_no_recycle"
    if recycle and (parent, bonus) in children:
        raise ValueError("Scaffold contract: continuation stopped before maximal exit")
    return {"greedy_prefix_nodes": greedy_nodes, "budget_checked": True,
            "labelled_paths_checked": True, "selected_path_checked": True,
            "maximal_exit_checked": recycle}


@torch.inference_mode()
def probe(engine, ids, variant, *, tokens=32, seed=105, tv_limit=MAX_NODE_TV):
    variant.validate()
    if variant.temperature <= 0 or not math.isfinite(tv_limit) or not 0 <= tv_limit <= 1:
        raise ValueError("Checkpoint probe requires positive temperature and a finite TV limit")
    captured, ar_rows, diffusion_metadata, scaffold_checks = [], [], [], []

    def observe_tree(parents, proposed, p):
        _check_probabilities(p, "captured tree")  # Also validate rounds beyond the capture limit.
        if len(captured) < 3:
            captured.append((list(parents), list(proposed), p.detach().clone()))

    def observe_ar(offset, logits):
        p = _checked_probabilities(logits, variant.temperature, "captured AR/prefill")
        if len(ar_rows) < 3:
            ar_rows.append((offset, p.detach().clone()))

    def observe_diffusion(tree, proposal, logits, temperature):
        proposal.validate(logits.shape[-1])
        _checked_probabilities(logits, temperature, "captured diffusion logits")
        if len(diffusion_metadata) < 3:
            diffusion_metadata.append({"anchor": int(proposal.law.noise_ids[0]),
                                       "retained_diffusion_block_mass": float(proposal.law.retained_block_mass()),
                                       "labelled_path_nodes": sum(map(len, tree.path_nodes))})

    def observe_scaffold(tree, proposal, draft_logits, nodes, selected, bonus):
        # Check EVERY round, even after the expensive AR-row capture limit.
        scaffold_checks.append(scaffold_contract(
            tree, proposal, variant, draft_logits.argmax(-1).tolist(), nodes, selected,
            bonus, draft_logits.shape[-1]))

    result = engine.generate(ids, variant, tokens, [], seed=seed, tree_observer=observe_tree,
                             ar_observer=observe_ar, diffusion_observer=observe_diffusion,
                             scaffold_observer=observe_scaffold)
    repeat = engine.generate(ids, variant, tokens, [], seed=seed)
    is_target = variant.method == "target"
    expected_captures = 0 if is_target else min(3, len(result["rounds"]))
    expected_ar = min(3, len(result["generated_token_ids"])) if is_target else 1
    if (variant.method in DIFFUSION_SCAFFOLD_METHODS
            and len(scaffold_checks) != len(result["rounds"])):
        raise RuntimeError("Positive-temperature checkpoint probe has missing scaffold witnesses")
    if (len(captured) != expected_captures or len(ar_rows) != expected_ar
            or (not is_target and not captured)
            or (variant.method in DIFFUSION_TREE_METHODS and len(diffusion_metadata) != len(captured))):
        raise RuntimeError("Positive-temperature checkpoint probe has missing target/proposal captures")
    errors, ar_checks, round_checks, generated_offset = [], [], [], 1
    for offset, actual in ar_rows:
        error = _tree_errors(engine, ids, result["generated_token_ids"][:offset],
                             [-1], [], actual, variant.temperature)[0]
        errors.append(error)
        ar_checks.append({"generated_prefix_tokens": offset, "max_node_total_variation": error})
    for round_index, (parents, proposed, actual) in enumerate(captured):
        prefix = result["generated_token_ids"][:generated_offset]
        metadata = dict(diffusion_metadata[round_index]) if diffusion_metadata else {}
        if metadata and prefix[-1] != metadata.pop("anchor"):
            raise AssertionError("Captured diffusion anchor does not match the committed prefix")
        current_errors = _tree_errors(engine, ids, prefix, parents, proposed, actual, variant.temperature)
        errors.extend(current_errors)
        round_checks.append({"round": round_index, "generated_prefix_tokens": generated_offset,
                             "max_node_total_variation": max(current_errors),
                             "tree_nodes": len(proposed), **metadata})
        if scaffold_checks:
            round_checks[-1]["scaffold_contract"] = scaffold_checks[round_index]
        generated_offset += result["rounds"][round_index]["committed_tokens"]
    # _node_tv rejects nonfinite errors before any Python max() aggregation.
    expected_draft_calls = 0 if is_target else len(result["rounds"])
    expected_target_calls = len(result["generated_token_ids"]) if is_target else 1 + len(result["rounds"])
    call_counts_passed = (result["draft_forward_calls"] == expected_draft_calls
                          and result["target_forward_calls"] == expected_target_calls)
    repeatable = result["generated_token_ids"] == repeat["generated_token_ids"]
    return {"passed": max(errors) <= tv_limit and repeatable and call_counts_passed,
            "temperature": variant.temperature, "variant": variant.to_dict(),
            "max_node_total_variation": max(errors), "node_tv_limit": tv_limit,
            "finite_probabilities_checked": True, "same_seed_repeatable": repeatable,
            "forward_call_counts_passed": call_counts_passed,
            "one_denoising_call_per_round": None if is_target else result["draft_forward_calls"] == expected_draft_calls,
            "captured_rounds": len(captured), "round_checks": round_checks, "ar_checks": ar_checks,
            "scaffold_rounds_checked": len(scaffold_checks),
            "target_reference": "original prompt prefill then cached autoregressive token calls",
            "scope": "numerical integration evidence; finite precision is not exact real arithmetic"}


def check_diffusion_model(cfg, output, device="cuda:0"):
    from .engine import load_models
    from .preflight import check_environment
    from .conversation import encode_messages
    result = {"passed": False, "checks": [], "temperature_scope": "positive_only",
              "scope": "all registered methods: target conditionals, repeatability and diffusion law; no bitwise lossless claim"}
    try:
        variants = [Variant(**e["variant"]) for e in cfg["explicit_variants"]]
        candidates = [v for v in variants if v.method in DIFFUSION_TREE_METHODS]
        if (not candidates or any(v.temperature <= 0 for v in variants)
                or len({v.temperature for v in variants}) != 1
                or len({v.name for v in variants}) != len(variants)):
            raise ValueError("All registered methods require the same positive temperature and unique names")
        for v in variants:
            v.validate()
        result["required_variants"] = [v.name for v in variants]
        result["environment"] = check_environment(cfg, "process", device)
        result["numerical_probe"] = numerical_probe(device)
        if not result["numerical_probe"]["passed"]:
            raise RuntimeError("Diffusion numerical probe failed")
        engine, tokenizer = load_models(cfg["model"], device)
        for prompt in ("Compute 19 + 23.", "Write a function that reverses a list."):
            ids = encode_messages(tokenizer, [{"role": "user", "content": prompt}], cfg["model"], device)
            # Every registered baseline/control is checked, then the diffusion
            # candidate also exercises rebuilding its draft cache.
            for variant in variants + [replace(candidates[0], reuse_draft_cache=False)]:
                check = {"prompt_sha256": digest(prompt), "variant": variant.to_dict(), "passed": False}
                result["checks"].append(check)
                try:
                    check.update(probe(engine, ids, variant))
                except Exception as exc:
                    check["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    raise
                if not check["passed"]:
                    raise RuntimeError(f"Positive-temperature correctness check failed for {variant.name}")
        result["passed"] = True
    except Exception as exc:
        # Replace stale successful evidence even when failure happens before a
        # verifier returns (e.g. a NaN in an unselected leaf or CUDA failure).
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output, result)
        raise RuntimeError(f"Diffusion checkpoint check failed; inspect {output}") from exc
    write_json(output, result)
    return result
