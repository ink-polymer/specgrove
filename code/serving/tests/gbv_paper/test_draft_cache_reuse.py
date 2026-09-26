"""Request-local incremental Draft cache tests, independent of GPU kernels."""
from types import SimpleNamespace

import pytest
import torch

from gbv_experiments.config import Variant
from gbv_experiments.continuous_tree_block_decode import ContinuousDecodeRequest, _propose_many


class Cache:
    def __init__(self):
        self.layers = []

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2] if self.layers else 0

    def update(self, keys, values, index):
        if index == len(self.layers):
            self.layers.append(SimpleNamespace(keys=keys, values=values))
        else:
            layer = self.layers[index]
            layer.keys = torch.cat((layer.keys, keys), dim=-2)
            layer.values = torch.cat((layer.values, values), dim=-2)


class Draft:
    block_size = 4
    mask_token_id = 7

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        features = kwargs['target_hidden']
        cache = kwargs['past_key_values']
        noise = torch.full((features.shape[0], self.block_size, 8), -999.0)
        keys = torch.cat((features, noise), dim=1).unsqueeze(1)
        cache.update(keys, -keys, 0)
        hidden = torch.zeros((features.shape[0], self.block_size, 8))
        hidden[:, 1, 1] = 4
        hidden[:, 2, 2] = 4
        hidden[:, 3, 3] = 4
        return hidden


def engine():
    embedding = torch.nn.Embedding(8, 8)
    return SimpleNamespace(device=torch.device('cpu'), cache_factory=Cache,
        draft=Draft(), target=SimpleNamespace(
            get_input_embeddings=lambda: embedding,
            get_output_embeddings=lambda: torch.nn.Identity()))


def request(index, length):
    state = ContinuousDecodeRequest(index, torch.zeros((1, length), dtype=torch.long), 17)
    state.target_cache = SimpleNamespace(get_seq_length=lambda: length)
    state.draft_cache = Cache()
    state.draft_update = torch.full((1, length, 8), float(index + 1))
    state.generated = [1]
    return state


@pytest.mark.parametrize('method', ['ddtree', 'dflash'])
def test_ragged_cache_retains_only_context_not_noise_and_updates_incrementally(method):
    model = engine()
    states = [request(0, 3), request(1, 5)]
    variant = Variant(name='cache_test', method=method, paths=1, length=3,
        temperature=1.0, probability_dtype='float32', tree_budget=3)
    _propose_many(model, states, variant, reuse_request_draft_cache=True)
    assert [s.draft_cache.get_seq_length() for s in states] == [3, 5]
    assert all(s.draft_update is None for s in states)
    for state in states:
        assert torch.all(state.draft_cache.layers[0].keys == state.request_id + 1)
    if method == 'dflash':
        assert all(s.tree.tokens == [1, 2, 3] for s in states)
        assert all(s.tree.parents == [-1, 0, 1, 2] for s in states)

    # Reordering, different accepted lengths and right padding must not leak
    # another request's prefix or retain speculative noise in the next round.
    states.reverse()
    for state, update in zip(states, [2, 1]):
        prefix = state.draft_cache.get_seq_length() + update
        state.target_cache = SimpleNamespace(get_seq_length=lambda p=prefix: p)
        state.draft_update = torch.full((1, update, 8), float(state.request_id + 11))
    _propose_many(model, states, variant, reuse_request_draft_cache=True)
    call = model.draft.calls[-1]
    assert call['target_hidden'].shape[1] == 2  # Not the seven-token full prefix.
    assert call['use_cache'] is True
    assert torch.isneginf(call['attention_mask'][1, 0, :, 3:5]).all()
    assert torch.isneginf(call['attention_mask'][1, 0, :, 6:7]).all()
    for state, old, new in zip(states, [5, 3], [2, 1]):
        keys = state.draft_cache.layers[0].keys
        assert keys.shape[-2] == old + new
        assert torch.all(keys[:, :, :old] == state.request_id + 1)
        assert torch.all(keys[:, :, old:] == state.request_id + 11)


def test_cached_dp_still_selects_feasible_heterogeneous_budgets():
    model = engine()
    states = [request(0, 3), request(1, 5)]
    variant = Variant(name='dp_cache_test', method='ddtree', paths=1, length=3,
        temperature=1.0, probability_dtype='float32', tree_budget=3)
    budgets = _propose_many(model, states, variant, reuse_request_draft_cache=True,
        global_tree_budget_options=(1, 3), global_tree_row_budget=8,
        global_tree_fixed_row_equivalent=1.0, max_new_tokens=64)
    assert all(b in (1, 3) for b in budgets)
    assert sum(b + 1 for b in budgets) <= 8
    assert all(len(s.tree.tokens) == b for s, b in zip(states, budgets))


def test_cache_prefix_mismatch_is_rejected():
    model = engine()
    state = request(0, 3)
    state.draft_update = torch.zeros((1, 2, 8))
    variant = Variant(name='cache_bad', method='dflash', paths=1, length=3,
        temperature=1.0, tree_budget=3)
    with pytest.raises(RuntimeError, match='feature update disagree'):
        _propose_many(model, [state], variant, reuse_request_draft_cache=True)
