from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "audited_global_tree_matrix", ROOT / "scripts/rerun_audited_global_tree_matrix_h20.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_matched_specs_remove_precision_cache_and_fused_verifier_asymmetry():
    curve = ((12, 1.0), (46, 2.0), (96, 3.0), (193, 4.0), (384, 5.0))
    for temperature in (0, 1):
        specs = audit.matched_specs(temperature, curve)
        for spec in specs.values():
            assert spec["variant"].probability_dtype == "float32"
            assert spec["variant"].reuse_draft_cache is False
            assert spec["decode"]["proposal_probability_dtype"] == "float32"
            assert spec["decode"]["reuse_request_draft_cache"] is False
            assert spec["decode"]["verifier"] == ("greedy" if temperature == 0 else "ancestral_reference")
        assert specs[audit.OURS]["decode"]["global_tree_row_cost_curve"] == curve


def test_chain_and_tree_reference_share_exact_rng_and_decision():
    values = torch.tensor([[.6, .4], [.25, .75], [.9, .1]], dtype=torch.float32)
    path = torch.tensor([0, 1])
    for seed in range(50):
        a = torch.Generator().manual_seed(seed)
        b = torch.Generator().manual_seed(seed)
        nodes, tokens, bonus = audit.reference_tree_verify([-1, 0, 1], [0, 1], values, a)
        accepted, chain_bonus = audit.reference_chain_verify(path, values, b)
        assert accepted == len(nodes)
        assert chain_bonus == bonus
        assert torch.equal(a.get_state(), b.get_state())


def test_reference_tree_exit_law_exhaustively(monkeypatch):
    import gbv_experiments.sampling as sampling
    parents, tokens = [-1, 0, 0, 1], [0, 1, 1]
    values = torch.tensor([[.6, .4], [.25, .75], [.9, .1], [.3, .7]], dtype=torch.float64)
    actual = {}
    for posterior in itertools.product(range(2), repeat=4):
        mass = 1.0
        for row, token in enumerate(posterior):
            mass *= float(values[row, token])
        monkeypatch.setattr(sampling, "sample", lambda *_args, p=posterior: torch.tensor(p))
        _, accepted, bonus = audit.reference_tree_verify(parents, tokens, values)
        sequence = tuple(accepted + [bonus])
        actual[sequence] = actual.get(sequence, 0.0) + mass
    expected = {(0, 0): .6 * .25, (0, 1, 0): .6 * .75 * .3,
        (0, 1, 1): .6 * .75 * .7, (1, 0): .4 * .9, (1, 1): .4 * .1}
    assert actual == pytest.approx(expected)
    assert sum(actual.values()) == pytest.approx(1.0)


def test_expanded_global_final_round_exhausts_old_pool_but_not_fixed_pool():
    from gbv_experiments.continuous_tree_block_decode import (
        _PersistentRequestCachePool, _persistent_cache_capacity,
    )
    # Same already-committed history and final accepted query rows for both.
    prefix = torch.arange(162, dtype=torch.float32).reshape(1, 1, 162, 1)
    original = SimpleNamespace(layers=[SimpleNamespace(keys=prefix, values=-prefix)],
        get_seq_length=lambda: 162)
    all_keys = torch.arange(178, dtype=torch.float32).reshape(1, 1, 178, 1)
    packed = SimpleNamespace(layers=[SimpleNamespace(keys=all_keys, values=-all_keys)])
    old = _PersistentRequestCachePool([original], 177)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        old.append_packed_sequence(packed, old.caches, 162, (0,), (tuple(range(16)),), "cpu")
    capacity = _persistent_cache_capacity(100, 64, row_cap=12, length=15, global_options=(11, 23, 45))
    fixed = _PersistentRequestCachePool([original], capacity)
    fixed.append_packed_sequence(packed, fixed.caches, 162, (0,), (tuple(range(16)),), "cpu")
    assert fixed.caches[0].get_seq_length() == 178
    assert torch.equal(fixed.caches[0].layers[0].keys, all_keys)


def test_global_allocator_matches_bruteforce_with_heterogeneous_utilities():
    from gbv_experiments.continuous_tree_block_decode import (
        _allocate_global_tree_budgets, _global_wave_cost,
    )
    rng = random.Random(20260916)
    options = (11, 23, 45)
    curve = ((12, 24.), (46, 24.5), (96, 25.), (193, 28.), (384, 40.))
    for _ in range(30):
        count = rng.randrange(1, 5)
        states = [SimpleNamespace(generated=[0] * rng.randrange(1, 64), age=rng.randrange(5))
            for _ in range(count)]
        utilities = [sorted(1 + rng.random() * 5 for _ in options) for _ in states]
        row_budget = rng.randrange(count * 12, 194)
        leading = max(len(s.generated) for s in states)
        maximum_age = max(s.age for s in states)
        weights = [1 + (leading - len(s.generated)) / 64 + s.age / max(1, maximum_age) for s in states]
        feasible = []
        for allocation in itertools.product(options, repeat=count):
            rows = sum(b + 1 for b in allocation)
            if rows <= row_budget:
                reward = sum(w * u[options.index(b)] for w, u, b in zip(weights, utilities, allocation))
                feasible.append((reward / _global_wave_cost(rows, None, curve), -rows,
                    tuple(-b for b in allocation), allocation))
        expected = max(feasible)[-1]
        actual = _allocate_global_tree_budgets(states, options, utilities,
            row_budget=row_budget, fixed_row_equivalent=None, criticality_weight=1,
            max_new_tokens=64, row_cost_curve=curve)
        assert actual == expected


class ToyCache:
    def __init__(self):
        self.layers = []

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2] if self.layers else 0

    def update(self, keys, values, layer_index):
        if layer_index == len(self.layers):
            self.layers.append(SimpleNamespace(keys=keys, values=values))
        else:
            layer = self.layers[layer_index]
            layer.keys = torch.cat((layer.keys, keys), dim=-2)
            layer.values = torch.cat((layer.values, values), dim=-2)


class ToyTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 16)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(16))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return torch.nn.Identity()


class ToyDraft:
    block_size = 16
    mask_token_id = 0
    target_layer_ids = [0]

    def __call__(self, *, noise_embedding, **_kwargs):
        anchors = noise_embedding[:, 0].argmax(-1)
        predictions = (anchors[:, None] + torch.arange(16)[None]) % 16
        return torch.nn.functional.one_hot(predictions, 16).float() * 30


class ToyEngine:
    device = torch.device("cpu")
    cache_factory = ToyCache

    def __init__(self):
        self.target = ToyTarget()
        self.draft = ToyDraft()
        self.query_rows = []

    def sync(self):
        pass

    def features(self, hidden_states):
        return hidden_states[1]

    def target_hidden_forward(self, ids, cache, **_kwargs):
        self.query_rows.append(int(ids.shape[1]))
        keys = ids[:, None, :, None].float()
        cache.update(keys, -keys, 0)
        values = torch.nn.functional.one_hot((ids + 1) % 16, 16).float() * 30
        return SimpleNamespace(last_hidden_state=values, hidden_states=(values, values))

    def target_forward(self, ids, cache, *, last_only=False, **kwargs):
        output = self.target_hidden_forward(ids, cache, **kwargs)
        output.logits = output.last_hidden_state[:, -1:] if last_only else output.last_hidden_state
        return output


@pytest.mark.parametrize("budget", [96, 193, 384])
@pytest.mark.parametrize("requests", [1, 16])
def test_full_global_scheduler_greedy_outputs_and_cache_feature_alignment(budget, requests):
    import gbv_experiments.continuous_tree_block_decode as decoder
    specs = audit.matched_specs(0, ((12, 1.), (46, 1.), (96, 1.), (193, 1.), (384, 1.)))
    spec = specs[audit.OURS]
    prompts = [torch.tensor([[3, 4, 5 + i % 3]]) for i in range(requests)]
    engine = ToyEngine()
    result = decoder.generate_continuous_tree_blocks(
        engine, prompts, spec["variant"], max_new_tokens=32,
        stop_ids=[], seeds=list(range(requests)), row_budget=budget,
        slot_capacity=budget // 12, layout="packed_sequence",
        persistent_request_target_cache=True, **spec["decode"],
    )
    expected = tuple(tuple((int(p[0, -1]) + 1 + i) % 16 for i in range(32)) for p in prompts)
    assert result.outputs == expected
    assert result.physical_target_rows == result.useful_target_rows
    assert result.padded_target_rows == 0
    assert max(engine.query_rows) <= budget


@pytest.mark.parametrize("name", audit.NAMES)
def test_full_matched_decoders_truncate_at_first_eos_not_later_verified_tokens(name):
    import gbv_experiments.continuous_tree_block_decode as decoder
    spec = audit.matched_specs(0, ((12, 1.), (46, 1.), (96, 1.), (193, 1.), (384, 1.)))[name]
    result = decoder.generate_continuous_tree_blocks(
        ToyEngine(), [torch.tensor([[3, 4, 5]])], spec["variant"], max_new_tokens=32,
        stop_ids=[9], seeds=[17], row_budget=193,
        slot_capacity=193 // spec["decode"]["row_cap"], layout="packed_sequence",
        persistent_request_target_cache=True, **spec["decode"],
    )
    assert result.outputs == ((6, 7, 8, 9),)


@pytest.mark.parametrize("name", audit.NAMES)
def test_t1_all_methods_execute_common_reference_branch_not_old_special_cases(name, monkeypatch):
    import gbv_experiments.continuous_tree_block_decode as decoder
    spec = audit.matched_specs(1, ((12, 1.), (46, 1.), (96, 1.), (193, 1.), (384, 1.)))[name]
    reference = decoder.tree_verify_ancestral_batched
    observed = []

    def spy(parents, tokens, values, generator, **kwargs):
        observed.append((len(parents), values.dtype))
        return reference(parents, tokens, values, generator, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Old special-case verifier was used")

    monkeypatch.setattr(decoder, "tree_verify_ancestral_batched", spy)
    monkeypatch.setattr(decoder, "matching_verify", forbidden)
    monkeypatch.setattr(decoder, "tree_block_verify_terminal_mass", forbidden)
    result = decoder.generate_continuous_tree_blocks(
        ToyEngine(), [torch.tensor([[3, 4, 5]])], spec["variant"], max_new_tokens=32,
        stop_ids=[], seeds=[17], row_budget=193,
        slot_capacity=193 // spec["decode"]["row_cap"], layout="packed_sequence",
        persistent_request_target_cache=True, **spec["decode"],
    )
    assert observed and all(dtype == torch.float32 for _, dtype in observed)
    assert result.outputs == (tuple((6 + i) % 16 for i in range(32)),)
