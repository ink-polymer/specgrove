import torch
from types import SimpleNamespace

from gbv_experiments.config import Variant
from gbv_experiments.continuous_tree_block_decode import (
    ContinuousDecodeRequest,
    _packed_block_inputs,
    _padded_block_inputs,
    _propose_many,
    _continue_existing_tree,
    _effective_row_cap,
    _effective_tree_budget,
    _effective_tree_budgets,
    _active_global_tree_budget_options,
    _globally_feasible_budget_options,
    _admit_global_proposal_states,
    _allocate_global_tree_budgets,
    _draft_tree_expected_tokens,
    _global_wave_cost,
    _prefix_tree,
    _adaptive_expansion_eligible,
    _PersistentRequestCachePool,
    _persistent_cache_capacity,
    _pack_cache_sequence,
    _split_compacted_sequence_caches,
    tree_verify_greedy_block,
)
from gbv_experiments.tree import Tree, probability_tree
from gbv_experiments.tree_block_skip import local_tree_block


def test_global_persistent_capacity_covers_expanded_tree_before_truncation():
    capacity = _persistent_cache_capacity(
        100, 64, row_cap=12, length=15, global_options=(11, 23, 45),
    )
    assert capacity == 211
    # A final round can retain anchor + fifteen accepted nodes before the
    # output is clipped. The old base-cap allocation (177) was too small.
    assert 100 + 62 + 16 <= capacity
    assert 100 + 62 + 16 > 100 + 64 + 12 + 1


def test_persistent_capacity_keeps_fixed_ddtree_allocation_unchanged():
    assert _persistent_cache_capacity(
        100, 64, row_cap=46, length=15,
    ) == 211


def test_greedy_tree_block_returns_internal_path_and_first_exit_token():
    parents = [-1, 0, 0, 1]
    tokens = [4, 5, 6]
    logits = torch.zeros((4, 8))
    logits[0, 4] = 2
    logits[1, 6] = 2
    logits[3, 7] = 2
    nodes, accepted, bonus = tree_verify_greedy_block(
        parents, tokens, logits,
    )
    assert nodes == [1, 3]
    assert accepted == [4, 6]
    assert bonus == 7


def test_greedy_tree_block_exits_at_root_when_argmax_child_is_absent():
    logits = torch.zeros((2, 8))
    logits[0, 3] = 2
    nodes, accepted, bonus = tree_verify_greedy_block(
        [-1, 0], [4], logits,
    )
    assert nodes == []
    assert accepted == []
    assert bonus == 3


class State:
    def __init__(self, tree, cache_length, generated, block_root=0, prefix=7):
        self.tree = tree
        self.target_cache = Cache(cache_length)
        self.generated = generated
        self.block_root = block_root
        self.round_prefix_len = prefix


class Cache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class ResultCache:
    def __init__(self):
        self.layers = []

    def update(self, keys, values, layer_index):
        assert layer_index == len(self.layers)
        self.layers.append(SimpleNamespace(keys=keys, values=values))


class FakeDraft:
    block_size = 16
    mask_token_id = 7

    def __call__(self, *, target_hidden, noise_embedding, **_kwargs):
        batch = target_hidden.shape[0]
        hidden = torch.zeros((batch, 16, 8))
        for slot in range(batch):
            for depth in range(1, 16):
                hidden[slot, depth, (slot + depth) % 8] = 1
        return hidden


class FakeTarget:
    def __init__(self):
        self.embedding = torch.nn.Embedding(8, 8)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return torch.nn.Identity()


class FakeEngine:
    device = torch.device("cpu")

    def __init__(self):
        self.draft = FakeDraft()
        self.target = FakeTarget()


def test_padded_block_mask_preserves_each_tree_visibility_and_cache_length():
    tree = Tree(
        tokens=[11, 12, 13, 14],
        parents=[-1, 0, 0, 1, 3],
        depths=[0, 1, 1, 2, 3],
        path_nodes=[],
    )
    block = local_tree_block(tree.parents, tree.tokens, tree.depths, 0, 4)
    states = [State(tree, 7, [9]), State(tree, 5, [9])]
    ids, positions, mask = _padded_block_inputs(
        states, [block, block], 6, 7, torch.float64, torch.device("cpu"),
    )
    assert ids.shape == positions.shape == (2, 6)
    assert mask.shape == (2, 1, 6, 13)
    assert torch.isfinite(mask[0, 0, :4, :7]).all()
    assert torch.isfinite(mask[1, 0, :4, :5]).all()
    assert torch.isneginf(mask[1, 0, :4, 5:7]).all()
    expected = tree.visibility()[:4, :4]
    assert torch.isfinite(mask[0, 0, :4, 7:11]).eq(expected).all()
    assert torch.isfinite(mask[1, 0, :4, 7:11]).eq(expected).all()
    assert torch.isfinite(mask[:, 0, 4:, 0]).all()


def test_continuation_positions_retain_global_tree_depths():
    tree = Tree(
        tokens=[11, 12, 13, 14],
        parents=[-1, 0, 1, 2, 3],
        depths=[0, 1, 2, 3, 4],
        path_nodes=[],
    )
    block = local_tree_block(tree.parents, tree.tokens, tree.depths, 2, 3)
    state = State(tree, 9, [8], block_root=2, prefix=7)
    _, positions, _ = _padded_block_inputs(
        [state], [block], 3, 9, torch.float64, torch.device("cpu"),
    )
    assert positions.tolist() == [[9, 10, 11]]


def test_packed_sequence_mask_is_strictly_request_local():
    tree = Tree(
        tokens=[11, 12, 13, 14],
        parents=[-1, 0, 0, 1, 3],
        depths=[0, 1, 1, 2, 3],
        path_nodes=[],
    )
    block = local_tree_block(tree.parents, tree.tokens, tree.depths, 0, 4)
    states = [State(tree, 7, [9]), State(tree, 5, [10])]
    ids, positions, mask, query_offsets = _packed_block_inputs(
        states, [block, block], (7, 5), (0, 7), 12,
        torch.float64, torch.device("cpu"),
    )
    assert ids.shape == positions.shape == (1, 8)
    assert query_offsets == (0, 4)
    assert mask.shape == (1, 1, 8, 20)
    assert torch.isfinite(mask[0, 0, :4, :7]).all()
    assert torch.isneginf(mask[0, 0, :4, 7:12]).all()
    assert torch.isneginf(mask[0, 0, :4, 16:20]).all()
    assert torch.isneginf(mask[0, 0, 4:, :7]).all()
    assert torch.isfinite(mask[0, 0, 4:, 7:12]).all()
    assert torch.isneginf(mask[0, 0, 4:, 12:16]).all()


def test_bulk_packed_cache_split_matches_request_local_rows():
    # Packed old caches occupy [0:5], followed by five query rows.  Request 0
    # owns old [0:2] and keeps query rows [0,2]; request 1 owns old [2:5] and
    # keeps local query row [1].
    keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1)
    packed = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=-keys)])
    results = _split_compacted_sequence_caches(
        packed,
        ResultCache,
        cache_offsets=(0, 2),
        old_lengths=(2, 3),
        packed_old_length=5,
        query_offsets=(0, 3),
        keep_query_rows=((0, 2), (1,)),
        device=torch.device("cpu"),
    )
    assert len(results) == 2
    assert results[0].layers[0].keys.flatten().tolist() == [0, 1, 5, 7]
    assert results[1].layers[0].keys.flatten().tolist() == [2, 3, 4, 9]
    assert results[0].layers[0].values.flatten().tolist() == [0, -1, -5, -7]
    assert results[1].layers[0].values.flatten().tolist() == [-2, -3, -4, -9]


def test_persistent_cache_pool_appends_only_kept_query_rows():
    first_keys = torch.tensor([1, 2], dtype=torch.float32).reshape(1, 1, 2, 1)
    second_keys = torch.tensor([3, 4, 5], dtype=torch.float32).reshape(1, 1, 3, 1)
    source = [
        SimpleNamespace(layers=[SimpleNamespace(
            keys=first_keys, values=-first_keys,
        )], get_seq_length=lambda: 2),
        SimpleNamespace(layers=[SimpleNamespace(
            keys=second_keys, values=-second_keys,
        )], get_seq_length=lambda: 3),
    ]
    pool = _PersistentRequestCachePool(source, capacity=8)
    # Five packed prefix rows followed by request-local query spans [5:8]
    # and [8:10].  Keep query rows [0,2] for request 0 and [1] for request 1.
    packed_keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1)
    packed = SimpleNamespace(layers=[SimpleNamespace(
        keys=packed_keys, values=-packed_keys,
    )])
    pool.append_packed_sequence(
        packed, pool.caches, packed_old_length=5,
        query_offsets=(0, 3), keep_query_rows=((0, 2), (1,)),
        device=torch.device("cpu"),
    )
    assert [cache.get_seq_length() for cache in pool.caches] == [4, 4]
    assert pool.caches[0].layers[0].keys.flatten().tolist() == [1, 2, 5, 7]
    assert pool.caches[1].layers[0].keys.flatten().tolist() == [3, 4, 5, 9]
    assert pool.caches[0].layers[0].values.flatten().tolist() == [-1, -2, -5, -7]
    assert pool.caches[1].layers[0].values.flatten().tolist() == [-3, -4, -5, -9]


def test_persistent_cache_pool_packs_request_prefixes_in_requested_order():
    first_keys = torch.tensor([1, 2], dtype=torch.float32).reshape(1, 1, 2, 1)
    second_keys = torch.tensor([3, 4, 5], dtype=torch.float32).reshape(1, 1, 3, 1)
    source = [
        SimpleNamespace(layers=[SimpleNamespace(
            keys=first_keys, values=-first_keys,
        )], get_seq_length=lambda: 2),
        SimpleNamespace(layers=[SimpleNamespace(
            keys=second_keys, values=-second_keys,
        )], get_seq_length=lambda: 3),
    ]
    pool = _PersistentRequestCachePool(source, capacity=8)
    packed, lengths, offsets, total = _pack_cache_sequence(
        tuple(reversed(pool.caches)), ResultCache, torch.device("cpu"),
    )
    assert lengths == (3, 2)
    assert offsets == (0, 3)
    assert total == 5
    assert packed.layers[0].keys.flatten().tolist() == [3, 4, 5, 1, 2]
    assert packed.layers[0].values.flatten().tolist() == [-3, -4, -5, -1, -2]


def test_continuous_dflash_proposal_uses_parallel_greedy_path():
    states = []
    for request_id in range(2):
        state = ContinuousDecodeRequest(
            request_id=request_id,
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            seed=request_id,
            target_cache=Cache(3),
            full_features=torch.zeros((1, 3, 8)),
            generated=[request_id],
        )
        states.append(state)
    variant = Variant(
        name="dflash_t1", method="dflash", paths=1, length=15,
        temperature=1.0, tree_budget=45, probability_dtype="float64",
    )
    _propose_many(FakeEngine(), states, variant)
    assert states[0].tree.tokens == [depth % 8 for depth in range(1, 16)]
    assert states[1].tree.tokens == [(depth + 1) % 8 for depth in range(1, 16)]
    assert states[0].tree.parents == [-1] + list(range(15))


def test_continuous_block_verify_samples_and_retains_diffusion_proposal_law():
    def request():
        return ContinuousDecodeRequest(
            request_id=0,
            input_ids=torch.zeros((1, 3), dtype=torch.long),
            seed=17,
            generator=torch.Generator().manual_seed(17),
            target_cache=Cache(3),
            full_features=torch.zeros((1, 3, 8)),
            generated=[0],
        )

    variant = Variant(
        name="dflash_block_verify_t1", method="bv", paths=1, length=15,
        temperature=1.0, draft_temperature=1.0, tree_budget=45,
        probability_dtype="float64",
    )
    engine = FakeEngine()
    first, second = request(), request()
    _propose_many(engine, [first], variant)
    _propose_many(engine, [second], variant)

    assert first.tree == second.tree
    assert first.tree.parents == [-1] + list(range(15))
    assert len(first.tree.tokens) == 15
    assert first.proposal_probabilities.shape == (15, 8)
    assert first.proposal_probabilities.dtype == torch.float64
    assert torch.allclose(
        first.proposal_probabilities.sum(-1), torch.ones(15, dtype=torch.float64),
    )
    selected = first.proposal_probabilities.gather(
        1, torch.tensor(first.tree.tokens)[:, None],
    )
    assert bool((selected > 0).all())


def test_adaptive_cap_restores_full_tree_only_at_low_occupancy():
    assert _effective_row_cap(16, 46, 4, 16, 24, 8) == 16
    assert _effective_row_cap(16, 46, 4, 9, 24, 8) == 16
    assert _effective_row_cap(16, 46, 4, 8, 24, 8) == 24
    assert _effective_row_cap(16, 46, 4, 5, 24, 8) == 24
    assert _effective_row_cap(16, 46, 4, 4, 24, 8) == 46
    assert _effective_row_cap(16, 46, 4, 1, 24, 8) == 46
    assert _effective_row_cap(24, None, None, 1) == 24


def test_continuation_gate_can_rebuild_at_load_and_bound_tree_reuse():
    controls = {
        "continuation_ready_threshold": 4,
        "max_continuation_blocks": 1,
    }
    assert not _continue_existing_tree(16, 1, **controls)
    assert _continue_existing_tree(4, 1, **controls)
    assert not _continue_existing_tree(4, 2, **controls)
    assert not _continue_existing_tree(
        1, 1, continuation_ready_threshold=0,
        max_continuation_blocks=None,
    )
    assert _continue_existing_tree(
        16, 1, continuation_ready_threshold=None,
        max_continuation_blocks=None,
    )


def test_tree_budget_expands_only_for_the_low_occupancy_tail():
    assert _effective_tree_budget(45, 11, 45, 4, 16) == 11
    assert _effective_tree_budget(45, 11, 45, 4, 5) == 11
    assert _effective_tree_budget(45, 11, 45, 4, 4) == 45
    assert _effective_tree_budget(45, 11, 45, 4, 1) == 45
    assert _effective_tree_budget(45, None, None, None, 16) == 45


def test_tail_expansion_can_require_drain_from_higher_load():
    assert _adaptive_expansion_eligible(16, (4, 4), True)
    assert _adaptive_expansion_eligible(8, (4, 4), True)
    assert not _adaptive_expansion_eligible(4, (4, 4), True)
    assert _adaptive_expansion_eligible(4, (4, 4), False)


def test_tail_tree_budget_targets_requests_with_most_remaining_work():
    states = [
        SimpleNamespace(request_id=0, generated=[0] * 40, age=0),
        SimpleNamespace(request_id=1, generated=[0] * 20, age=0),
        SimpleNamespace(request_id=2, generated=[0] * 30, age=0),
        SimpleNamespace(request_id=3, generated=[0] * 20, age=0),
    ]
    assert _effective_tree_budgets(
        states, 45, 11, 45, 4, 4, 2,
    ) == (11, 45, 11, 45)
    assert _effective_tree_budgets(
        states, 45, 11, 45, 4, 5, 2,
    ) == (11, 11, 11, 11)
    assert _effective_tree_budgets(
        states, 45, 11, 45, 4, 4, None,
    ) == (45, 45, 45, 45)


def test_draft_tree_expected_tokens_uses_prefix_reach_mass():
    q = torch.tensor([
        [0.75, 0.25],
        [0.50, 0.50],
    ], dtype=torch.float64)
    tree = probability_tree(q, 2)
    expected = 1.0 + sum(
        torch.exp(torch.tensor(tree.draft_prefix_log_masses[1:])).tolist()
    )
    assert abs(_draft_tree_expected_tokens(tree) - expected) < 1e-7


def test_canonical_ddtree_budget_is_prefix_of_larger_tree():
    generator = torch.Generator().manual_seed(17)
    q = torch.softmax(torch.randn((4, 32), generator=generator), -1).double()
    full = probability_tree(q, 12)
    prefix = _prefix_tree(full, 5)
    direct = probability_tree(q, 5)
    assert prefix.tokens == direct.tokens
    assert prefix.parents == direct.parents
    assert prefix.depths == direct.depths
    assert prefix.draft_prefix_log_masses == direct.draft_prefix_log_masses


def test_global_tree_budget_allocator_can_leave_rows_unused():
    states = [
        SimpleNamespace(generated=[0], age=0),
        SimpleNamespace(generated=[0], age=0),
    ]
    assert _allocate_global_tree_budgets(
        states, (1, 3), ((2.0, 2.1), (2.0, 2.1)),
        row_budget=8, fixed_row_equivalent=1.0,
        criticality_weight=0.0, max_new_tokens=8,
    ) == (1, 1)


def test_global_tree_budget_allocator_spends_rows_on_best_request():
    states = [
        SimpleNamespace(generated=[0], age=0),
        SimpleNamespace(generated=[0], age=0),
    ]
    assert _allocate_global_tree_budgets(
        states, (1, 3), ((2.0, 4.0), (2.0, 2.1)),
        row_budget=6, fixed_row_equivalent=32.0,
        criticality_weight=0.0, max_new_tokens=8,
    ) == (3, 1)


def test_global_tree_budget_allocator_prioritizes_lagging_request():
    states = [
        SimpleNamespace(generated=[0] * 4, age=0),
        SimpleNamespace(generated=[0] * 2, age=0),
    ]
    assert _allocate_global_tree_budgets(
        states, (1, 3), ((2.0, 3.0), (2.0, 3.0)),
        row_budget=6, fixed_row_equivalent=32.0,
        criticality_weight=2.0, max_new_tokens=8,
    ) == (1, 3)


def test_measured_global_wave_cost_interpolates_and_changes_allocation():
    curve = ((2, 1.0), (4, 1.0))
    assert _global_wave_cost(3, None, curve) == 1.0
    state = SimpleNamespace(generated=[0], age=0)
    assert _allocate_global_tree_budgets(
        [state], (1, 3), ((2.0, 3.0),),
        row_budget=4, fixed_row_equivalent=None,
        criticality_weight=0.0, max_new_tokens=8,
        row_cost_curve=curve,
    ) == (3,)


def test_global_admission_prioritizes_lag_then_age_without_starvation():
    states = [
        SimpleNamespace(request_id=0, generated=[0] * 4, age=9),
        SimpleNamespace(request_id=1, generated=[0] * 2, age=0),
        SimpleNamespace(request_id=2, generated=[0] * 2, age=3),
        SimpleNamespace(request_id=3, generated=[0] * 3, age=20),
    ]
    admitted = _admit_global_proposal_states(
        states, row_budget=4, minimum_tree_budget=1,
    )
    assert tuple(state.request_id for state in admitted) == (1, 2)


def test_global_tier_gate_keeps_smaller_feasible_expansions():
    assert _globally_feasible_budget_options(
        (11, 23, 45), request_count=4, row_budget=193,
        require_tier_fit=True,
    ) == (11, 23, 45)
    assert _globally_feasible_budget_options(
        (11, 23, 45), request_count=8, row_budget=193,
        require_tier_fit=True,
    ) == (11, 23)
    assert _globally_feasible_budget_options(
        (11, 23, 45), request_count=16, row_budget=193,
        require_tier_fit=True,
    ) == (11,)


def test_global_active_gate_uses_small_trees_above_limit():
    assert _active_global_tree_budget_options(
        (11, 23, 45), active_count=9, expansion_active_limit=8,
    ) == (11,)
    assert _active_global_tree_budget_options(
        (11, 23, 45), active_count=8, expansion_active_limit=8,
    ) == (11, 23, 45)
