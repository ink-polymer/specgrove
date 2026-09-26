#!/usr/bin/env python3
"""Actual-history structural/numerical audit of frozen shared-budget decoder.

All instrumentation is reverted on exit. Audit wall times are NOT benchmarks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
from rerun_audited_global_tree_matrix_h20 import (
    OURS, first_difference, matched_specs, write_json,
)
from audit_ours_greedy_same_prefix_h20 import clone_cache, same_bytes, summary


def snapshot(cache):
    return [(l.keys.clone(), l.values.clone()) for l in cache.layers]


def restore_cache(engine, layers):
    cache = engine.cache_factory()
    for i, (k, v) in enumerate(layers):
        cache.update(k.clone(), v.clone(), i)
    return cache


def probability_gap(left, right):
    a = left.float().softmax(-1).double()
    b = right.float().softmax(-1).double()
    a = a / a.sum(); b = b / b.sum()
    return dict(total_variation=float((a - b).abs().sum() / 2),
        max_probability_error=float((a - b).abs().max()),
        max_logit_error=float((left.float() - right.float()).abs().max()),
        argmax_equal=int(left.argmax()) == int(right.argmax()),
        logits_byte_equal=same_bytes(left.float(), right.float()))


@torch.inference_mode()
def canonical_ar(engine, prompt, stops, cap=64):
    cache = engine.cache_factory()
    out = engine.target_forward(prompt, cache, hidden=False, last_only=True)
    logits = out.logits[0, -1]
    tokens, logit_rows = [], []
    for _ in range(cap):
        logit_rows.append(logits.float().cpu())
        token = int(logits.argmax()); tokens.append(token)
        if token in stops or len(tokens) == cap:
            break
        ids = torch.tensor([[token]], device=engine.device)
        positions = torch.tensor([[cache.get_seq_length()]], device=engine.device)
        out = engine.target_forward(ids, cache, hidden=False, positions=positions, last_only=True)
        logits = out.logits[0, -1]
    return tokens, logit_rows


class ActualHistoryAudit:
    def __init__(self, engine, decoder, prompts, references, temperature, budget):
        self.engine, self.decoder = engine, decoder
        self.prompts, self.references = prompts, references
        self.temperature, self.budget = temperature, budget
        self.wave = None
        self.states, self.features, self.shadows = {}, {}, {}
        self.probed, self.first_divergent = set(), set()
        self.probes, self.prefills = [], []
        self.stats = dict(waves=0, request_blocks=0, masks_checked=0,
                          packed_layers_checked=0, compacted_request_layers_checked=0,
                          feature_histories_checked=0, rng_probe_checks=0,
                          forward_prefix_layers_checked=0)

    def select(self, *args, **kwargs):
        states, blocks = self.old_select(*args, **kwargs)
        if not states:
            return states, blocks
        rows = sum(len(b.nodes) for b in blocks)
        assert rows <= self.budget
        for state in states:
            i = state.request_id; self.states[i] = state
            assert state.block_root == 0  # Frozen policy has no continuations.
            assert state.target_cache.get_seq_length() == state.input_ids.shape[1] + len(state.generated) - 1
            if i in self.features:
                assert same_bytes(state.full_features, self.features[i])
                self.stats["feature_histories_checked"] += 1
        self.wave = dict(states=states, blocks=blocks, rows=rows)
        self.stats["waves"] += 1; self.stats["request_blocks"] += len(states)
        return states, blocks

    def pack(self, caches, factory, device):
        w = self.wave
        assert w is not None
        assert all(a is b.target_cache for a, b in zip(caches, w["states"]))
        w["old"] = [snapshot(c) for c in caches]
        w["old_features"] = [s.full_features.clone() for s in w["states"]]
        packed, lengths, offsets, total = self.old_pack(caches, factory, device)
        for layer_i, layer in enumerate(packed.layers):
            keys = torch.cat([c[layer_i][0] for c in w["old"]], dim=-2)
            values = torch.cat([c[layer_i][1] for c in w["old"]], dim=-2)
            assert same_bytes(keys, layer.keys) and same_bytes(values, layer.values)
            self.stats["packed_layers_checked"] += 1
        w.update(packed=packed, lengths=lengths, offsets=offsets, old_total=total)
        return packed, lengths, offsets, total

    def inputs(self, states, blocks, lengths, offsets, total, dtype, device):
        ids, positions, mask, queries = self.old_inputs(states, blocks, lengths, offsets, total, dtype, device)
        w = self.wave
        assert not bool(torch.isnan(mask).any())
        expected = torch.zeros((w["rows"], total + w["rows"]), dtype=torch.bool, device=device)
        for state, block, length, offset, query in zip(states, blocks, lengths, offsets, queries):
            assert state.round_prefix_len == length
            expected_ids = [state.generated[-1]] + list(block.tokens)
            assert ids[0, query:query + len(block.nodes)].tolist() == expected_ids
            for node in range(len(block.nodes)):
                ancestors = []; ancestor = node
                while ancestor >= 0:
                    ancestors.append(ancestor); ancestor = block.parents[ancestor]
                expected[query + node, offset:offset + length] = True
                expected[query + node, [total + query + n for n in ancestors]] = True
                assert int(positions[0, query + node]) == length + len(ancestors) - 1
            self.stats["masks_checked"] += 1
        assert torch.equal(torch.isfinite(mask[0, 0]), expected)
        assert bool((mask[0, 0][expected] == 0).all())
        assert bool(torch.isneginf(mask[0, 0][~expected]).all())
        w.update(ids=ids, positions=positions, queries=queries)
        return ids, positions, mask, queries

    def hidden(self, ids, cache, **kwargs):
        w = self.wave
        assert w is not None and cache is w["packed"]
        output = self.old_hidden(ids, cache, **kwargs)
        logits = self.engine.target.get_output_embeddings()(output.last_hidden_state)[0]
        p = logits.float().softmax(-1)
        assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(p).all())
        assert bool((p >= 0).all()) and bool(((p.sum(-1) - 1).abs() < 1e-5).all())
        assert cache.get_seq_length() == w["old_total"] + w["rows"]
        for layer_i, layer in enumerate(cache.layers):
            keys = torch.cat([c[layer_i][0] for c in w["old"]], dim=-2)
            values = torch.cat([c[layer_i][1] for c in w["old"]], dim=-2)
            assert same_bytes(keys, layer.keys[..., :w["old_total"], :])
            assert same_bytes(values, layer.values[..., :w["old_total"], :])
            self.stats["forward_prefix_layers_checked"] += 1
        w.update(output=output, logits=logits)
        return output

    def forward(self, ids, cache, *, hidden, **kwargs):
        result = self.old_forward(ids, cache, hidden=hidden, **kwargs)
        if hidden:
            # Frozen decoder only uses this entry for per-request prefill.
            i = len(self.prefills)
            reference = self.references[i][1][0].to(self.engine.device)
            logits = result.logits[0, -1].float()
            self.prefills.append(dict(request_id=i, gap=probability_gap(logits, reference)))
        return result

    def replay_path(self, old, state, block, node):
        path = []; cursor = node
        while cursor:
            path.append(block.tokens[cursor - 1]); cursor = block.parents[cursor]
        tokens = [state.generated[-1]] + list(reversed(path))
        cache = restore_cache(self.engine, old)
        for token in tokens:
            ids = torch.tensor([[token]], device=self.engine.device)
            positions = torch.tensor([[cache.get_seq_length()]], device=self.engine.device)
            out = self.engine.target_forward(ids, cache, hidden=False, positions=positions, last_only=True)
        return out.logits[0, -1].float(), tokens

    def shadow_root(self, state):
        i = state.request_id
        if i not in self.shadows:
            cache = self.engine.cache_factory()
            self.engine.target_forward(state.input_ids, cache, hidden=False, last_only=True)
            self.shadows[i] = (cache, [])
        cache, history = self.shadows[i]
        past = list(state.generated[:-1])
        assert past[:len(history)] == history
        for token in past[len(history):]:
            ids = torch.tensor([[token]], device=self.engine.device)
            positions = torch.tensor([[cache.get_seq_length()]], device=self.engine.device)
            self.engine.target_forward(ids, cache, hidden=False, positions=positions, last_only=True)
            history.append(token)
        assert cache.get_seq_length() == state.round_prefix_len
        ids = torch.tensor([[state.generated[-1]]], device=self.engine.device)
        positions = torch.tensor([[cache.get_seq_length()]], device=self.engine.device)
        out = self.engine.target_forward(ids, clone_cache(self.engine, cache), hidden=False,
                                        positions=positions, last_only=True)
        return out.logits[0, -1].float(), cache

    def numeric_probes(self, slot, keep):
        w = self.wave; state, block = w["states"][slot], w["blocks"][slot]
        i, generated = state.request_id, list(state.generated)
        query, old = w["queries"][slot], w["old"][slot]
        nodes = []
        reference_tokens, reference_logits = self.references[i]
        if self.temperature == 0 and i not in self.first_divergent and generated == reference_tokens[:len(generated)]:
            emitted = [block.tokens[n - 1] for n in keep[1:]] + [int(w["logits"][query + keep[-1]].argmax())]
            for offset, token in enumerate(emitted[:64 - len(generated)]):
                index = len(generated) + offset
                if index >= len(reference_tokens):
                    break
                if token != reference_tokens[index]:
                    nodes.append((keep[offset], "actual_first_divergence", index))
                    self.first_divergent.add(i); break
        bucket = len(generated) // 16
        selected = i in {0, len(self.prompts) - 1} and (i, bucket) not in self.probed
        if selected:
            self.probed.add((i, bucket)); nodes.append((0, "generation_bucket_root", None))
        for node, reason, canonical_index in nodes:
            before_rng = state.generator.get_state().clone()
            actual = w["logits"][query + node].float()
            replay, path = self.replay_path(old, state, block, node)
            record = dict(request_id=i, generated_tokens=len(generated), reason=reason,
                local_node=node, wave_query_rows=w["rows"], block_rows=len(block.nodes), path_tokens=path,
                actual=summary(actual), same_actual_kv_sequential=summary(replay),
                gap_same_actual_kv=probability_gap(actual, replay))
            if canonical_index is not None:
                canonical = reference_logits[canonical_index].to(self.engine.device)
                record.update(canonical_index=canonical_index, canonical=summary(canonical),
                    gap_vs_canonical_ar=probability_gap(actual, canonical),
                    replay_vs_canonical_ar=probability_gap(replay, canonical))
            if selected and node == 0:
                teacher, shadow = self.shadow_root(state)
                max_key_error, max_value_error, byte_equal = 0., 0., True
                for (k, v), l in zip(old, shadow.layers):
                    byte_equal = byte_equal and same_bytes(k, l.keys) and same_bytes(v, l.values)
                    max_key_error = max(max_key_error, float((k.float() - l.keys.float()).abs().max()))
                    max_value_error = max(max_value_error, float((v.float() - l.values.float()).abs().max()))
                record.update(teacher_forced_sequential=summary(teacher),
                    gap_vs_teacher_forced_sequential=probability_gap(actual, teacher),
                    actual_kv_vs_teacher=dict(byte_equal=byte_equal,
                        max_key_error=max_key_error, max_value_error=max_value_error))
            assert torch.equal(before_rng, state.generator.get_state())
            self.stats["rng_probe_checks"] += 1; self.probes.append(record)

    def append(self, pool, cache, caches, old_total, queries, keep_rows, device):
        w = self.wave
        expected = []
        for slot, (state, block, query, keep) in enumerate(zip(w["states"], w["blocks"], queries, keep_rows)):
            assert list(keep)[0] == 0
            assert all(block.parents[child] == parent for parent, child in zip(keep, keep[1:]))
            indices = torch.tensor([old_total + query + n for n in keep], device=device)
            layers = []
            for (old_k, old_v), layer in zip(w["old"][slot], cache.layers):
                layers.append((torch.cat((old_k, layer.keys.index_select(-2, indices)), dim=-2),
                               torch.cat((old_v, layer.values.index_select(-2, indices)), dim=-2)))
            expected.append(layers)
            feature_indices = torch.tensor([query + n for n in keep], device=device)
            additions = torch.cat([w["output"].hidden_states[layer + 1].index_select(1, feature_indices)
                                   for layer in self.engine.draft.target_layer_ids], dim=-1)
            self.features[state.request_id] = torch.cat((w["old_features"][slot], additions), dim=1)
        self.old_append(pool, cache, caches, old_total, queries, keep_rows, device)
        for actual_cache, layers in zip(caches, expected):
            for actual, (k, v) in zip(actual_cache.layers, layers):
                assert same_bytes(actual.keys, k) and same_bytes(actual.values, v)
                self.stats["compacted_request_layers_checked"] += 1
        for slot, keep in enumerate(keep_rows):
            self.numeric_probes(slot, keep)

    @contextmanager
    def instrument(self):
        d, e = self.decoder, self.engine
        self.old_select, self.old_pack, self.old_inputs = d._select_ready, d._pack_cache_sequence, d._packed_block_inputs
        self.old_append = d._PersistentRequestCachePool.append_packed_sequence
        self.old_hidden, self.old_forward = e.target_hidden_forward, e.target_forward
        d._select_ready, d._pack_cache_sequence, d._packed_block_inputs = self.select, self.pack, self.inputs
        audit = self
        def appended(pool, *args, **kwargs):
            return audit.append(pool, *args, **kwargs)
        d._PersistentRequestCachePool.append_packed_sequence = appended
        e.target_hidden_forward, e.target_forward = self.hidden, self.forward
        try:
            yield self
            assert len(self.prefills) == len(self.prompts)
            for i, expected in self.features.items():
                assert same_bytes(self.states[i].full_features, expected)
                self.stats["feature_histories_checked"] += 1
        finally:
            d._select_ready, d._pack_cache_sequence, d._packed_block_inputs = self.old_select, self.old_pack, self.old_inputs
            d._PersistentRequestCachePool.append_packed_sequence = self.old_append
            e.target_hidden_forward, e.target_forward = self.old_hidden, self.old_forward


@torch.inference_mode()
def sampler_gpu_check():
    from gbv_experiments.sampling import sample
    # Complete binary depth-2 tree: first three output tokens have AR law.
    p0 = torch.tensor([.25, .0, .75, 1., .5, .25, .75], device="cuda:0")
    values = torch.stack((p0, 1 - p0), dim=-1)
    count = 100000
    generator = torch.Generator(device="cuda:0").manual_seed(98173)
    draws = sample(values.repeat(count, 1), generator).reshape(count, 7)
    indices = torch.arange(count, device="cuda:0")
    first = draws[:, 0]; second_node = 1 + first
    second = draws[indices, second_node]; third_node = 3 + 2 * first + second
    third = draws[indices, third_node]
    bins = 4 * first + 2 * second + third
    frequencies = torch.bincount(bins, minlength=8).double().cpu() / count
    expected = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                p = float(values[0, a]) * float(values[1 + a, b]) * float(values[3 + 2 * a + b, c])
                expected.append(p)
    zscores = [abs(float(f) - p) / (max(p * (1 - p), 1e-30) / count) ** .5
               for f, p in zip(frequencies, expected)]
    assert all(z <= 7 for z in zscores)
    return dict(passed=True, draws=count, frequencies=frequencies.tolist(), expected=expected,
                maximum_binomial_z=max(zscores), scope="fixed seven-row binary tree; sampler smoke, not real-model sequence certification")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output / "results.jsonl").exists():
        raise FileExistsError("Refusing to overwrite actual-history audit")
    sys.path.insert(0, str(ROOT / "third_party/ddtree_pinned"))
    from gbv_experiments.config import load_config
    from gbv_experiments.engine import load_models
    from gbv_experiments.runner import stop_token_ids
    from gbv_experiments.terminal_formal import allocation_gate, model_gate
    import gbv_experiments.continuous_tree_block_decode as decoder
    from benchmark_continuous_tree_block_e2e import prepared_prompts
    cfg = load_config(args.config)["model"]
    allocation_gate("cuda:0")
    engine, tokenizer = load_models(cfg, "cuda:0"); model_gate(engine)
    engine.set_target_verification_backend("eager")
    stops = stop_token_ids(engine, tokenizer)
    prompts, identities = prepared_prompts(tokenizer, cfg, engine.device, args.data_dir, "gsm8k", 96, 32)
    cost = json.loads(args.calibration.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__).resolve(), ROOT / "src/gbv_experiments/continuous_tree_block_decode.py",
        ROOT / "src/gbv_experiments/sampling.py", ROOT / "src/gbv_experiments/engine.py",
        ROOT / "scripts/rerun_audited_global_tree_matrix_h20.py",
        ROOT / "scripts/audit_ours_greedy_same_prefix_h20.py"]
    manifest = dict(model=cfg, kind="actual_history_correctness_audit", temperatures=[0, 1],
        budgets=[96, 193, 384], concurrencies=[1, 4, 8, 16, 32], seeds={"0": 419, "1": 911},
        prompt_offset=96, prompt_count=32, max_new_tokens=64,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        calibration=cost, canonical_reference="per-request SDPA prefill then one causal query per token; identical model weights",
        audit_timing_is_benchmark=False, probe_scope="all actual cache/mask/feature writes; two requests across generation buckets plus observed T0 first divergences",
        numerical_probe_reference="byte-identical actual old KV, sequential ancestor replay; canonical/teacher-forced one-query AR",
        formal_real_model_sequence_law_certified=False, task_accuracy_evaluated=False)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "sampler_gpu.json", sampler_gpu_check())
    references = []
    for i, prompt in enumerate(prompts):
        references.append(canonical_ar(engine, prompt, stops))
        print("canonical AR", i + 1, "/32", flush=True)
    write_json(args.output / "canonical_ar.json", [dict(identity=identity, output=ref[0])
        for identity, ref in zip(identities, references)])
    rows = []
    with (args.output / "results.jsonl").open("x") as stream:
        for temperature in (0, 1):
            spec = matched_specs(temperature, cost["curve"])[OURS]
            for budget in (96, 193, 384):
                for concurrency in (1, 4, 8, 16, 32):
                    groups = [[i] for i in range(4)] if concurrency == 1 else [list(range(concurrency))]
                    for indices in groups:
                        selected = [prompts[i] for i in indices]
                        refs = [references[i] for i in indices]
                        seeds = [(419 if temperature == 0 else 911) * 1_000_003 + (96 + i) * 104_729 for i in indices]
                        audit = ActualHistoryAudit(engine, decoder, selected, refs, temperature, budget)
                        with audit.instrument():
                            result = decoder.generate_continuous_tree_blocks(engine, selected, spec["variant"],
                                max_new_tokens=64, stop_ids=stops, seeds=seeds, row_budget=budget,
                                slot_capacity=budget // 12, layout="packed_sequence",
                                persistent_request_target_cache=True, **spec["decode"])
                        # Probes must not alter decoding decisions or consume RNG.
                        uninstrumented = decoder.generate_continuous_tree_blocks(engine, selected, spec["variant"],
                            max_new_tokens=64, stop_ids=stops, seeds=seeds, row_budget=budget,
                            slot_capacity=budget // 12, layout="packed_sequence",
                            persistent_request_target_cache=True, **spec["decode"])
                        assert result.outputs == uninstrumented.outputs
                        row = dict(temperature=temperature, row_budget=budget, concurrency=concurrency,
                            identities=[identities[i] for i in indices], outputs=result.outputs,
                            request_seeds=seeds, structural_checks_passed=True, instrumentation_noninterference=True,
                            stats=audit.stats, prefills=audit.prefills, probes=audit.probes,
                            physical_target_calls=result.physical_target_calls,
                            first_difference_from_canonical_ar=[first_difference(list(out), ref[0])
                                for out, ref in zip(result.outputs, refs)] if temperature == 0 else None)
                        rows.append(row); stream.write(json.dumps(row) + "\n"); stream.flush()
                        print("audit T", temperature, "R", budget, "C", concurrency, "case",len(rows), "/48",
                              "checks", audit.stats, "probes",len(audit.probes), flush=True)
    write_json(args.output / "complete.json", dict(cases=len(rows), summary_cells=30,
        structural_checks_passed=True, formal_real_model_sequence_law_certified=False,
        greedy_exact_requests=sum(d is None for r in rows if r["temperature"] == 0
            for d in r["first_difference_from_canonical_ar"]),
        greedy_total_requests=sum(len(r["outputs"]) for r in rows if r["temperature"] == 0)))
    print("actual-history audit complete", flush=True)


if __name__ == "__main__":
    main()
