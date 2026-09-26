#!/usr/bin/env python3
"""Diagnose selected Ours/AR divergences with byte-identical teacher-prefix KV.

These isolated prefix probes do not certify all real decoding histories.
Run only after timed matrix generation has completed on the GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark_continuous_tree_block_e2e import prepared_prompts
from gbv_experiments.config import Variant, load_config
from gbv_experiments.continuous_tree_block_decode import (
    ContinuousDecodeRequest, _propose_many, _prefix_tree,
    _PersistentRequestCachePool, tree_verify_greedy_block,
)
from gbv_experiments.engine import load_models
from gbv_experiments.terminal_formal import allocation_gate, model_gate
from gbv_experiments.tree import Tree

OURS = "ours_global_hetero_11_23_45_h20_curve384_tierfit_a8_c1_propfp32"


def select_cases(matrix):
    rows = [json.loads(line) for line in (matrix / "t0_r193/results.jsonl").read_text().splitlines()]
    cases = []
    seen = set()
    for concurrency in (1, 8, 32, 4, 16):
        for row in rows:
            if row["concurrency"] != concurrency or row["seed"] != 17:
                continue
            for slot, difference in enumerate(row["first_difference_from_greedy_ar"][OURS]):
                if difference is None or difference["position"] == 0:
                    continue
                if difference["position"] >= min(len(row["greedy_ar_outputs"][slot]), len(row["outputs"][OURS][0][slot])):
                    continue
                identity = row["prompt_identities"][slot]
                if identity["prompt_sha256"] in seen:
                    continue
                seen.add(identity["prompt_sha256"])
                cases.append(dict(identity=identity, concurrency=concurrency,
                    difference=difference, ar=row["greedy_ar_outputs"][slot],
                    ours=row["outputs"][OURS][0][slot]))
                break
            if len(cases) >= 3:
                return cases
    return cases


def same_bytes(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def clone_cache(engine, cache):
    result = engine.cache_factory()
    for index, layer in enumerate(cache.layers):
        result.update(layer.keys.clone(), layer.values.clone(), index)
    for original, copied in zip(cache.layers, result.layers):
        assert same_bytes(original.keys, copied.keys)
        assert same_bytes(original.values, copied.values)
    return result


def summary(logits):
    values, indices = logits.topk(5)
    return dict(argmax=int(logits.argmax()), top_tokens=indices.tolist(),
        top_logits=values.tolist(), top_margin=float(values[0] - values[1]))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())["model"]
    allocation_gate("cuda:0")
    engine, tokenizer = load_models(cfg, "cuda:0")
    model_gate(engine)
    engine.set_target_verification_backend("eager")
    prompts, identities = prepared_prompts(tokenizer, cfg, engine.device, args.data_dir, "gsm8k", 0, 32)
    lookup = {identity["prompt_sha256"]: prompt for identity, prompt in zip(identities, prompts)}
    reports = []
    for case in select_cases(args.matrix):
        position = case["difference"]["position"]
        prompt = lookup[case["identity"]["prompt_sha256"]]
        prefix_tokens = torch.tensor([case["ar"][:position - 1]], dtype=torch.long, device=engine.device)
        prefix = torch.cat((prompt, prefix_tokens), dim=1)
        base = engine.cache_factory()
        initial = engine.target_forward(prefix, base, hidden=True, last_only=True)
        root = int(case["ar"][position - 1])
        state = ContinuousDecodeRequest(0, prompt, 17)
        state.target_cache = base
        state.full_features = engine.features(initial.hidden_states)
        state.generated = list(case["ar"][:position])
        variant = Variant(name="teacher_prefix_probe", method="ddtree", length=15,
            paths=1, temperature=0., draft_temperature=1., probability_dtype="float32", tree_budget=45)
        _propose_many(engine, [state], variant, proposal_probability_dtype="float32")
        tree = state.tree
        root_ids = torch.tensor([[root]], device=engine.device)
        root_positions = torch.tensor([[prefix.shape[1]]], device=engine.device)
        causal = engine.target_forward(root_ids, clone_cache(engine, base), hidden=False,
            positions=root_positions, last_only=True).logits[0, -1].float()
        results = {"causal_root1": summary(causal)}
        compaction_checks = []
        for rows in (1, 12, 24, 46):
            local = Tree(tokens=[], parents=[-1], depths=[0], path_nodes=[]) if rows == 1 else _prefix_tree(tree, rows - 1)
            ids = torch.tensor([[root] + local.tokens], device=engine.device)
            positions = torch.tensor([local.depths], device=engine.device) + prefix.shape[1]
            mask = local.mask(prefix.shape[1], next(engine.target.parameters()).dtype, engine.device)
            queried_cache = clone_cache(engine, base)
            output = engine.target_hidden_forward(ids, queried_cache, positions=positions, mask=mask)
            all_logits = engine.target.get_output_embeddings()(output.last_hidden_state)[0]
            logits = all_logits[0].float()
            results[f"masked_tree{rows}"] = summary(logits) | {"max_absolute_error_vs_causal_root1": float((logits - causal).abs().max())}
            nodes, _tokens, _bonus = tree_verify_greedy_block(local.parents, local.tokens, all_logits)
            keep = [0] + nodes
            pool = _PersistentRequestCachePool([base], prefix.shape[1] + 47)
            pool.append_packed_sequence(queried_cache, pool.caches, prefix.shape[1], (0,), (keep,), engine.device)
            indices = torch.tensor(keep, device=engine.device) + prefix.shape[1]
            for old, queried, compacted in zip(base.layers, queried_cache.layers, pool.caches[0].layers):
                expected_keys = torch.cat((old.keys, queried.keys.index_select(-2, indices)), dim=-2)
                expected_values = torch.cat((old.values, queried.values.index_select(-2, indices)), dim=-2)
                assert same_bytes(expected_keys, compacted.keys)
                assert same_bytes(expected_values, compacted.values)
            compaction_checks.append({"query_rows": rows, "kept_rows": keep, "all_layers_kv_byte_equal": True})
        reports.append(dict(identity=case["identity"], concurrency=case["concurrency"], position=position,
            actual_ours_token=case["ours"][position], actual_batched_ar_token=case["ar"][position],
            prefix_kv_byte_identical=True, forwards=results,
            real_gpu_kv_compaction_checks=compaction_checks,
            observed_query_mask_root_flip=any(r["argmax"] != results["causal_root1"]["argmax"] for n, r in results.items() if n != "causal_root1")))
    report = dict(model=cfg, scope="selected first divergences, teacher-forced identical prefix KV; not actual-history certification",
        cases=reports, task_quality_evaluated=False,
        warning="A flip isolates finite-precision query/mask/forward effects in this context; no flip is inconclusive about the original mismatch.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(reports), "root_flips": sum(c["observed_query_mask_root_flip"] for c in reports)}, indent=2))


if __name__ == "__main__":
    main()
