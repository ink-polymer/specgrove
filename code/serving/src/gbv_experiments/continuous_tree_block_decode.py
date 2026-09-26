"""Reference continuous-batch decoder for exact DDTree block cascades.

This is deliberately a padded/ragged-KV reference implementation.  It keeps
request RNGs, caches, positions, proposal trees, and cascade state separate,
while coalescing ready Target blocks into one physical forward.  A production
packed-varlen implementation can replace the cache pack/unpack functions after
it passes the same request-local correctness gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from .config import Variant
from .fused_tree_sampling import (
    tree_verify_ancestral_fused_scan,
    tree_verify_ancestral_lazy_softmax_fused_scan,
    tree_verify_ancestral_lazy_projection_fused_scan,
    tree_verify_ancestral_lazy_projection_fused_scan_batched,
    tree_verify_ancestral_lazy_projection_sparse_exit_batched,
    tree_verify_ancestral_logits_fused_scan,
    tree_verify_ancestral_logits_fused_scan_batched,
    tree_verify_ancestral_logits_sparse_exit_batched,
    tree_verify_ancestral_same_draw_fused,
    tree_verify_ancestral_sparse_exit_fused_scan,
    tree_verify_ancestral_sparse_exit_fused_scan_batched,
)
from .sampling import (
    block_verify_batched,
    matching_verify,
    probabilities,
    sample,
    tree_block_verify_terminal_mass,
    tree_verify_ancestral_batched,
)
from .tree import Tree, probability_tree, sampled_tree
from .tree_block_skip import TreeBlock, child_for_edge, local_tree_block


def tree_verify_greedy_block(parents, tokens, target_values):
    """Follow the Target argmax path through one request-local tree block.

    ``target_values[node]`` may contain logits or probabilities; only its
    argmax is observed.  Returning the first token outside the local block as
    ``bonus`` lets the caller either commit it or continue into the omitted
    part of the original tree without changing T=0 DDTree semantics.
    """
    if (
        len(parents) != len(tokens) + 1
        or target_values.ndim != 2
        or target_values.shape[0] != len(parents)
        or not parents
        or int(parents[0]) != -1
    ):
        raise ValueError("Tree block and Target rows disagree")
    node = 0
    accepted_nodes = []
    accepted_tokens = []
    while True:
        bonus = int(target_values[node].argmax().item())
        child = child_for_edge(parents, tokens, node, bonus)
        if child is None:
            return accepted_nodes, accepted_tokens, bonus
        accepted_nodes.append(child)
        accepted_tokens.append(bonus)
        node = child


@dataclass
class ContinuousDecodeRequest:
    request_id: int
    input_ids: torch.Tensor
    seed: int
    generator: torch.Generator | None = None
    target_cache: object | None = None
    draft_cache: object | None = None
    draft_update: torch.Tensor | None = None
    full_features: torch.Tensor | None = None
    generated: list[int] = field(default_factory=list)
    tree: Tree | None = None
    proposal_tree_budget: int | None = None
    proposal_probabilities: torch.Tensor | None = None
    round_prefix_len: int = 0
    block_root: int = 0
    accepted_nodes: list[int] = field(default_factory=list)
    accepted_tokens: list[int] = field(default_factory=list)
    feature_chunks: list[torch.Tensor] | None = None
    round_block_rows: list[int] = field(default_factory=list)
    rounds: list[dict] = field(default_factory=list)
    done: bool = False
    age: int = 0

    @property
    def continuation(self) -> bool:
        return self.tree is not None and self.block_root != 0


@dataclass(frozen=True)
class ContinuousDecodeResult:
    outputs: tuple[tuple[int, ...], ...]
    request_rounds: tuple[tuple[dict, ...], ...]
    wall_ms: float
    prefill_ms: float
    proposal_ms: float
    target_ms: float
    pack_ms: float
    select_commit_ms: float
    physical_target_calls: int
    physical_target_rows: int
    useful_target_rows: int
    padded_target_rows: int
    effective_row_cap_counts: tuple[tuple[int, int], ...]
    effective_tree_budget_counts: tuple[tuple[int, int], ...]
    gated_continuations: int
    arrival_times_s: tuple[float, ...] = ()
    request_metrics: tuple[dict, ...] = ()

    @property
    def output_tokens(self) -> int:
        return sum(len(output) for output in self.outputs)


class _PersistentRequestCacheLayer:
    """A request-local view into an append-only, multi-request KV arena."""

    def __init__(self, cache: "_PersistentRequestCache", layer_index: int):
        self._cache = cache
        self._layer_index = layer_index

    @property
    def keys(self):
        pool = self._cache.pool
        start = self._cache.slot * pool.capacity
        end = start + pool.lengths[self._cache.slot]
        return pool.key_storage[self._layer_index][
            :, start:end
        ].unsqueeze(0)

    @property
    def values(self):
        pool = self._cache.pool
        start = self._cache.slot * pool.capacity
        end = start + pool.lengths[self._cache.slot]
        return pool.value_storage[self._layer_index][
            :, start:end
        ].unsqueeze(0)


class _PersistentRequestCache:
    """Cache-compatible request view whose history is never recopied."""

    def __init__(self, pool: "_PersistentRequestCachePool", slot: int):
        self.pool = pool
        self.slot = slot
        self.layers = [
            _PersistentRequestCacheLayer(self, layer_index)
            for layer_index in range(len(pool.key_storage))
        ]

    def get_seq_length(self):
        return self.pool.lengths[self.slot]


class _PersistentRequestCachePool:
    """Preallocated Target KV pages shared by independent requests.

    The Target backend still receives one packed DynamicCache per physical
    wave.  The important first step is that demultiplexing no longer copies
    every request's complete prefix after every verification: only newly kept
    query rows are scattered back into this arena.
    """

    def __init__(self, caches: Sequence, capacity: int):
        if not caches:
            raise ValueError("At least one cache is required")
        lengths = [_cache_length(cache) for cache in caches]
        if capacity < max(lengths):
            raise ValueError("Persistent cache capacity is below the prefix")
        layer_count = len(caches[0].layers)
        if any(len(cache.layers) != layer_count for cache in caches):
            raise ValueError("Target caches have different layer counts")
        self.capacity = int(capacity)
        self.lengths = list(lengths)
        self.key_storage = []
        self.value_storage = []
        for layer_index in range(layer_count):
            example = caches[0].layers[layer_index]
            if example.keys.shape[0] != 1:
                raise ValueError("Expected one request per source cache")
            key_shape = (
                example.keys.shape[1], len(caches) * self.capacity,
                example.keys.shape[-1],
            )
            keys = torch.empty(
                key_shape, dtype=example.keys.dtype, device=example.keys.device,
            )
            values = torch.empty(
                key_shape, dtype=example.values.dtype,
                device=example.values.device,
            )
            for slot, (cache, length) in enumerate(zip(caches, lengths)):
                source = cache.layers[layer_index]
                start = slot * self.capacity
                keys[:, start:start + length].copy_(source.keys[0])
                values[:, start:start + length].copy_(source.values[0])
            self.key_storage.append(keys)
            self.value_storage.append(values)
        self.caches = tuple(
            _PersistentRequestCache(self, slot) for slot in range(len(caches))
        )

    def append_packed_sequence(
            self, packed, caches: Sequence[_PersistentRequestCache],
            packed_old_length: int, query_offsets: Sequence[int],
            keep_query_rows: Sequence[Sequence[int]], device):
        """Scatter only newly accepted query KV rows back to persistent pages."""
        if not (
            len(caches) == len(query_offsets) == len(keep_query_rows)
            and caches
        ):
            raise ValueError("Persistent cache append metadata disagree")
        source_indices = []
        destination_indices = []
        increments = []
        for cache, query_offset, rows in zip(
                caches, query_offsets, keep_query_rows):
            if cache.pool is not self:
                raise ValueError("Persistent cache belongs to another pool")
            slot = cache.slot
            old_length = self.lengths[slot]
            increment = len(rows)
            if old_length + increment > self.capacity:
                raise RuntimeError("Persistent request KV capacity exhausted")
            source_indices.extend(
                packed_old_length + query_offset + int(row) for row in rows
            )
            destination_indices.extend(
                slot * self.capacity + old_length + index
                for index in range(increment)
            )
            increments.append((slot, increment))
        source = torch.tensor(
            source_indices, dtype=torch.long, device=device,
        )
        destination = torch.tensor(
            destination_indices, dtype=torch.long, device=device,
        )
        for layer_index, layer in enumerate(packed.layers):
            kept_keys = layer.keys[0].index_select(-2, source)
            kept_values = layer.values[0].index_select(-2, source)
            self.key_storage[layer_index].index_copy_(
                -2, destination, kept_keys,
            )
            self.value_storage[layer_index].index_copy_(
                -2, destination, kept_values,
            )
        for slot, increment in increments:
            self.lengths[slot] += increment


def _cache_length(cache) -> int:
    return int(cache.get_seq_length())


def _persistent_cache_capacity(
        maximum_prompt: int, max_new_tokens: int, *,
        row_cap: int, length: int,
        low_occupancy_row_cap: int | None = None,
        mid_occupancy_row_cap: int | None = None,
        global_options: Sequence[int] | None = None) -> int:
    # KV rows are retained before EOS/output truncation.  A global B45 tree
    # may execute 46 rows even though the admission cap is only 12 rows.
    maximum_block = max(
        row_cap, length + 1,
        low_occupancy_row_cap or row_cap,
        mid_occupancy_row_cap or row_cap,
        (max(global_options) + 1) if global_options else row_cap,
    )
    return maximum_prompt + max_new_tokens + maximum_block + 1


def _pad_cache_batch(caches: Sequence, cache_factory, device):
    """Right-pad independent DynamicCaches and concatenate their batch rows."""
    if not caches:
        raise ValueError("At least one cache is required")
    lengths = tuple(_cache_length(cache) for cache in caches)
    maximum = max(lengths)
    layer_count = len(caches[0].layers)
    if any(len(cache.layers) != layer_count for cache in caches):
        raise ValueError("Target caches have different layer counts")
    packed = cache_factory()
    for layer_index in range(layer_count):
        keys = []
        values = []
        for cache, length in zip(caches, lengths):
            layer = cache.layers[layer_index]
            if layer.keys.shape[0] != 1 or layer.keys.shape[-2] != length:
                raise ValueError("Expected one request per source cache")
            padding = maximum - length
            keys.append(F.pad(layer.keys, (0, 0, 0, padding)))
            values.append(F.pad(layer.values, (0, 0, 0, padding)))
        packed.update(
            torch.cat(keys, dim=0).to(device),
            torch.cat(values, dim=0).to(device),
            layer_index,
        )
    return packed, lengths, maximum


def _pack_cache_sequence(caches: Sequence, cache_factory, device):
    """Concatenate request caches along sequence, for block-diagonal attention."""
    if not caches:
        raise ValueError("At least one cache is required")
    lengths = tuple(_cache_length(cache) for cache in caches)
    offsets = []
    total = 0
    for length in lengths:
        offsets.append(total)
        total += length
    layer_count = len(caches[0].layers)
    if any(len(cache.layers) != layer_count for cache in caches):
        raise ValueError("Target caches have different layer counts")
    packed = cache_factory()
    if (
        all(isinstance(cache, _PersistentRequestCache) for cache in caches)
        and all(cache.pool is caches[0].pool for cache in caches)
    ):
        # A single gather from the arena avoids concatenating several strided
        # request views.  The resulting DynamicCache remains byte-for-byte the
        # same logical prefix presented to the Target model.
        pool = caches[0].pool
        flat_indices = []
        for cache, length in zip(caches, lengths):
            start = cache.slot * pool.capacity
            flat_indices.extend(range(start, start + length))
        index = torch.tensor(flat_indices, dtype=torch.long, device=device)
        for layer_index in range(layer_count):
            packed.update(
                pool.key_storage[layer_index].index_select(
                    -2, index,
                ).unsqueeze(0),
                pool.value_storage[layer_index].index_select(
                    -2, index,
                ).unsqueeze(0),
                layer_index,
            )
        return packed, lengths, tuple(offsets), total
    for layer_index in range(layer_count):
        keys = [cache.layers[layer_index].keys for cache in caches]
        values = [cache.layers[layer_index].values for cache in caches]
        if any(value.shape[0] != 1 for value in keys):
            raise ValueError("Expected one request per source cache")
        packed.update(
            torch.cat(keys, dim=-2).to(device),
            torch.cat(values, dim=-2).to(device),
            layer_index,
        )
    return packed, lengths, tuple(offsets), total


def _split_compacted_cache(
        packed, cache_factory, slot: int, old_length: int,
        packed_old_length: int, keep_query_rows: Sequence[int], device):
    """Recover one request cache, dropping right padding and rejected rows."""
    old = torch.arange(old_length, dtype=torch.long, device=device)
    new = torch.tensor(
        [packed_old_length + int(row) for row in keep_query_rows],
        dtype=torch.long,
        device=device,
    )
    indices = torch.cat((old, new))
    result = cache_factory()
    for layer_index, layer in enumerate(packed.layers):
        result.update(
            layer.keys[slot:slot + 1].index_select(-2, indices),
            layer.values[slot:slot + 1].index_select(-2, indices),
            layer_index,
        )
    return result


def _split_compacted_sequence_cache(
        packed, cache_factory, cache_offset: int, old_length: int,
        packed_old_length: int, query_offset: int,
        keep_query_rows: Sequence[int], device):
    old = torch.arange(
        cache_offset, cache_offset + old_length,
        dtype=torch.long, device=device,
    )
    new = torch.tensor(
        [packed_old_length + query_offset + int(row)
         for row in keep_query_rows],
        dtype=torch.long, device=device,
    )
    indices = torch.cat((old, new))
    result = cache_factory()
    for layer_index, layer in enumerate(packed.layers):
        result.update(
            layer.keys.index_select(-2, indices),
            layer.values.index_select(-2, indices),
            layer_index,
        )
    return result


def _split_compacted_sequence_caches(
        packed, cache_factory, cache_offsets: Sequence[int],
        old_lengths: Sequence[int], packed_old_length: int,
        query_offsets: Sequence[int],
        keep_query_rows: Sequence[Sequence[int]], device):
    """Demultiplex all packed request caches with one gather per layer.

    The scalar implementation issued one key and one value ``index_select``
    per layer *and per request*.  cap24 deliberately admits more requests per
    wave, so that launch pattern erased part of the Target saving.  Gathering
    the concatenated request indices once keeps exactly the same cache rows and
    request isolation while reducing the launch count from O(layers*requests)
    to O(layers).
    """
    count = len(cache_offsets)
    if not (
        count == len(old_lengths) == len(query_offsets) == len(keep_query_rows)
        and count > 0
    ):
        raise ValueError("Packed cache split metadata disagree")
    flat_indices: list[int] = []
    boundaries = [0]
    for cache_offset, old_length, query_offset, rows in zip(
            cache_offsets, old_lengths, query_offsets, keep_query_rows):
        flat_indices.extend(range(cache_offset, cache_offset + old_length))
        flat_indices.extend(
            packed_old_length + query_offset + int(row) for row in rows
        )
        boundaries.append(len(flat_indices))
    indices = torch.tensor(flat_indices, dtype=torch.long, device=device)
    results = [cache_factory() for _ in range(count)]
    for layer_index, layer in enumerate(packed.layers):
        keys = layer.keys.index_select(-2, indices)
        values = layer.values.index_select(-2, indices)
        for slot, result in enumerate(results):
            start, end = boundaries[slot:slot + 2]
            result.update(
                keys[..., start:end, :],
                values[..., start:end, :],
                layer_index,
            )
    return tuple(results)


def _padded_block_inputs(
        states: Sequence[ContinuousDecodeRequest], blocks: Sequence[TreeBlock],
        width: int, packed_cache_length: int, dtype, device):
    """Compile request-local blocks into one fixed-width batch."""
    ids = torch.zeros((len(states), width), dtype=torch.long, device=device)
    positions = torch.zeros_like(ids)
    mask = torch.full(
        (len(states), 1, width, packed_cache_length + width),
        float("-inf"), dtype=dtype, device=device,
    )
    for slot, (state, block) in enumerate(zip(states, blocks)):
        if state.tree is None or len(block.nodes) > width:
            raise ValueError("Invalid ready block")
        old_length = _cache_length(state.target_cache)
        rows = len(block.nodes)
        root_token = (
            state.generated[-1]
            if state.block_root == 0
            else state.tree.tokens[state.block_root - 1]
        )
        ids[slot, :rows] = torch.tensor(
            [root_token] + list(block.tokens), dtype=torch.long, device=device,
        )
        if rows < width:
            ids[slot, rows:] = root_token
        positions[slot, :rows] = (
            torch.tensor(block.depths, dtype=torch.long, device=device)
            + state.round_prefix_len
        )
        positions[slot, rows:] = state.round_prefix_len
        mask[slot, 0, :rows, :old_length] = 0
        local = Tree(
            tokens=list(block.tokens),
            parents=list(block.parents),
            depths=list(block.depths),
            path_nodes=[],
        ).visibility(device)
        mask[slot, 0, :rows, packed_cache_length:packed_cache_length + rows].masked_fill_(
            local, 0,
        )
        # Padding rows are discarded, but must have a finite attention row so
        # that an all-masked SDPA query cannot create NaNs in the shared call.
        if rows < width:
            mask[slot, 0, rows:, 0] = 0
    return ids, positions, mask


def _packed_block_inputs(
        states: Sequence[ContinuousDecodeRequest], blocks: Sequence[TreeBlock],
        cache_lengths: Sequence[int], cache_offsets: Sequence[int],
        packed_cache_length: int, dtype, device):
    """Pack variable-size request blocks into one block-diagonal sequence."""
    block_rows = tuple(len(block.nodes) for block in blocks)
    query_offsets = []
    total_queries = 0
    for rows in block_rows:
        query_offsets.append(total_queries)
        total_queries += rows
    ids = torch.empty((1, total_queries), dtype=torch.long, device=device)
    positions = torch.empty_like(ids)
    mask = torch.full(
        (1, 1, total_queries, packed_cache_length + total_queries),
        float("-inf"), dtype=dtype, device=device,
    )
    for state, block, old_length, cache_offset, query_offset in zip(
            states, blocks, cache_lengths, cache_offsets, query_offsets):
        rows = len(block.nodes)
        query_slice = slice(query_offset, query_offset + rows)
        root_token = (
            state.generated[-1]
            if state.block_root == 0
            else state.tree.tokens[state.block_root - 1]
        )
        ids[0, query_slice] = torch.tensor(
            [root_token] + list(block.tokens), dtype=torch.long, device=device,
        )
        positions[0, query_slice] = (
            torch.tensor(block.depths, dtype=torch.long, device=device)
            + state.round_prefix_len
        )
        mask[
            0, 0, query_slice, cache_offset:cache_offset + old_length
        ] = 0
        visibility = Tree(
            tokens=list(block.tokens), parents=list(block.parents),
            depths=list(block.depths), path_nodes=[],
        ).visibility(device)
        query_keys = mask[
            0, 0, query_slice,
            packed_cache_length + query_offset:
            packed_cache_length + query_offset + rows,
        ]
        query_keys.masked_fill_(visibility, 0)
    return ids, positions, mask, tuple(query_offsets)


def _probability_tree_for_variant(q, budget: int, variant: Variant) -> Tree:
    return probability_tree(
        q, budget,
        depth_reward=variant.tree_depth_reward,
        depth_reward_schedule=variant.tree_depth_reward_schedule,
        rank_caps=variant.tree_rank_caps,
        adaptive_min_budget=variant.tree_adaptive_min_budget,
        confidence_threshold=variant.tree_adaptive_confidence_threshold,
    )


def _draft_tree_expected_tokens(tree: Tree) -> float:
    """Draft-only proxy for one tree's accepted tokens plus its bonus token."""
    masses = tree.draft_prefix_log_masses
    if masses is None or len(masses) != len(tree.parents):
        raise ValueError("Global tree budgeting requires DDTree prefix masses")
    value = 1.0 + sum(math.exp(log_mass) for log_mass in masses[1:])
    if not math.isfinite(value) or value < 1.0:
        raise FloatingPointError("Invalid draft-tree expected-token utility")
    return value


def _prefix_tree(tree: Tree, budget: int) -> Tree:
    """Take an ancestor-closed prefix of canonical DDTree best-first order."""
    if budget < 1 or budget > len(tree.tokens):
        raise ValueError("DDTree prefix budget is outside the source tree")
    if any(parent > budget for parent in tree.parents[1:budget + 1]):
        raise RuntimeError("Canonical DDTree prefix is not ancestor closed")
    masses = tree.draft_prefix_log_masses
    return Tree(
        tokens=list(tree.tokens[:budget]),
        parents=list(tree.parents[:budget + 1]),
        depths=list(tree.depths[:budget + 1]),
        path_nodes=[],
        draft_prefix_log_masses=(
            list(masses[:budget + 1]) if masses is not None else None
        ),
    )


def _normalize_global_row_cost_curve(
        curve: Sequence[Sequence[float]] | None,
        ) -> tuple[tuple[int, float], ...] | None:
    """Validate a measured monotone wave-cost curve."""
    if curve is None:
        return None
    try:
        normalized = tuple(
            (int(point[0]), float(point[1]))
            for point in curve
        )
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("Invalid global Target row-cost curve") from error
    if (
        not normalized
        or any(len(point) != 2 for point in curve)
        or any(
            row < 1 or not math.isfinite(cost) or cost <= 0
            for row, cost in normalized
        )
        or any(
            normalized[index][0] >= normalized[index + 1][0]
            or normalized[index][1] > normalized[index + 1][1]
            for index in range(len(normalized) - 1)
        )
    ):
        raise ValueError("Invalid global Target row-cost curve")
    return normalized


def _global_wave_cost(
        rows: int, fixed_row_equivalent: float | None,
        row_cost_curve: Sequence[tuple[int, float]] | None) -> float:
    """Interpolate a measured wave cost, or use the legacy linear proxy."""
    if rows < 1:
        raise ValueError("A Target wave must contain at least one row")
    if row_cost_curve is None:
        if (
            fixed_row_equivalent is None
            or not math.isfinite(fixed_row_equivalent)
            or fixed_row_equivalent <= 0
        ):
            raise ValueError("Missing global Target cost model")
        return fixed_row_equivalent + rows
    curve = tuple(row_cost_curve)
    if rows <= curve[0][0]:
        return curve[0][1]
    for (left_rows, left_cost), (right_rows, right_cost) in zip(
            curve, curve[1:]):
        if rows <= right_rows:
            fraction = (rows - left_rows) / (right_rows - left_rows)
            return left_cost + fraction * (right_cost - left_cost)
    return curve[-1][1]


def _admit_global_proposal_states(
        states: Sequence[ContinuousDecodeRequest], *,
        row_budget: int, minimum_tree_budget: int,
        ) -> tuple[ContinuousDecodeRequest, ...]:
    """Causally choose which requests may build a tree for the next wave."""
    minimum_rows = minimum_tree_budget + 1
    capacity = row_budget // minimum_rows
    if row_budget < minimum_rows or capacity < 1:
        raise ValueError("Global row budget cannot admit one minimum tree")
    if len(states) <= capacity:
        return tuple(states)
    ranked = sorted(
        states,
        key=lambda state: (len(state.generated), -state.age, state.request_id),
    )
    admitted_ids = {
        state.request_id for state in ranked[:capacity]
    }
    return tuple(
        state for state in states if state.request_id in admitted_ids
    )


def _globally_feasible_budget_options(
        budget_options: Sequence[int], *, request_count: int,
        row_budget: int | None, require_tier_fit: bool,
        ) -> tuple[int, ...]:
    """Keep only tiers that could cover every request in the current wave."""
    options = tuple(int(value) for value in budget_options)
    if (
        not options or request_count < 1
        or row_budget is None or row_budget < 1
        or not isinstance(require_tier_fit, bool)
    ):
        raise ValueError("Invalid global tree-budget tier gate")
    if not require_tier_fit:
        return options
    feasible = tuple(
        budget for budget in options
        if request_count * (budget + 1) <= row_budget
    )
    if not feasible:
        raise RuntimeError("No tree-budget tier can cover the admitted wave")
    return feasible


def _active_global_tree_budget_options(
        budget_options: Sequence[int], *, active_count: int,
        expansion_active_limit: int | None,
        ) -> tuple[int, ...]:
    """Disable larger trees while many requests share the Target wave."""
    options = tuple(int(value) for value in budget_options)
    if (
        not options or active_count < 1
        or (expansion_active_limit is not None and expansion_active_limit < 1)
    ):
        raise ValueError("Invalid active-request global tree-budget gate")
    if expansion_active_limit is not None and active_count > expansion_active_limit:
        return (options[0],)
    return options


def _allocate_global_tree_budgets(
        states: Sequence[ContinuousDecodeRequest],
        budget_options: Sequence[int],
        utilities: Sequence[Sequence[float]], *,
        row_budget: int | None,
        fixed_row_equivalent: float | None,
        criticality_weight: float,
        max_new_tokens: int | None,
        row_cost_curve: Sequence[tuple[int, float]] | None = None,
        ) -> tuple[int, ...]:
    """Solve a causal multiple-choice knapsack over one Target wave.

    Each request must receive one tree.  For every reachable total row count,
    dynamic programming retains the allocation with the greatest draft-side
    expected token reward.  The final allocation maximizes expected reward
    divided by either a measured piecewise Target-wave cost or the legacy
    ``fixed_row_equivalent + rows`` surrogate.
    Lagging and older requests can receive additional causal critical-path
    weight; no Target value from the current wave is observed by the policy.
    """
    options = tuple(int(value) for value in budget_options)
    row_cost_curve = _normalize_global_row_cost_curve(row_cost_curve)
    if (
        not states
        or tuple(sorted(set(options))) != options
        or not options
        or options[0] < 1
        or row_budget is None
        or row_budget < len(states) * (options[0] + 1)
        or ((fixed_row_equivalent is None) == (row_cost_curve is None))
        or (fixed_row_equivalent is not None and (
            not math.isfinite(fixed_row_equivalent)
            or fixed_row_equivalent <= 0
        ))
        or not math.isfinite(criticality_weight)
        or criticality_weight < 0
        or max_new_tokens is None
        or max_new_tokens < 1
        or len(utilities) != len(states)
        or any(len(values) != len(options) for values in utilities)
        or any(not math.isfinite(value) or value < 1.0
               for values in utilities for value in values)
    ):
        raise ValueError("Invalid global tree-budget allocation problem")

    progress = tuple(len(state.generated) for state in states)
    leading_progress = max(progress)
    maximum_age = max(state.age for state in states)
    weights = tuple(
        1.0 + criticality_weight * (
            (leading_progress - generated) / max_new_tokens
            + state.age / max(1, maximum_age)
        )
        for state, generated in zip(states, progress)
    )

    # rows -> (weighted reward, allocation).  Keeping one best allocation for
    # every exact row count is sufficient because cost depends only on rows.
    frontier: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for weight, values in zip(weights, utilities):
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used_rows, (reward, allocation) in frontier.items():
            for budget, utility in zip(options, values):
                rows = used_rows + budget + 1
                if rows > row_budget:
                    continue
                candidate = (reward + weight * utility, allocation + (budget,))
                previous = updated.get(rows)
                if (
                    previous is None
                    or candidate[0] > previous[0] + 1e-12
                    or (abs(candidate[0] - previous[0]) <= 1e-12
                        and candidate[1] < previous[1])
                ):
                    updated[rows] = candidate
        frontier = updated
    if not frontier:
        raise RuntimeError("Global tree-budget allocator found no feasible wave")

    _, (_, allocation) = max(
        frontier.items(),
        key=lambda item: (
            item[1][0] / _global_wave_cost(
                item[0], fixed_row_equivalent, row_cost_curve,
            ),
            -item[0],
            tuple(-value for value in item[1][1]),
        ),
    )
    return allocation


def _propose_many(
        engine, states: Sequence[ContinuousDecodeRequest],
        variant: Variant, reuse_request_draft_cache: bool = False,
        tree_budget: int | None = None,
        tree_budgets: Sequence[int] | None = None,
        proposal_probability_dtype: str | None = None,
        global_tree_budget_options: Sequence[int] | None = None,
        global_tree_row_budget: int | None = None,
        global_tree_fixed_row_equivalent: float | None = None,
        global_tree_row_cost_curve: Sequence[tuple[int, float]] | None = None,
        global_tree_require_tier_fit: bool = False,
        global_tree_criticality_weight: float = 0.0,
        max_new_tokens: int | None = None) -> tuple[int, ...]:
    """Run one padded Draft forward for all requests needing a new tree."""
    if not states:
        return ()
    global_budgeting = global_tree_budget_options is not None
    if global_budgeting and (
        variant.method not in {"ddtree", "ddtree_refine"}
        or variant.tree_depth_reward != 0.0
        or variant.tree_depth_reward_schedule is not None
        or variant.tree_rank_caps is not None
        or variant.tree_adaptive_min_budget is not None
    ):
        raise ValueError("Global budgeting requires canonical DDTree proposal masses")
    if (
        (tree_budget is not None and tree_budgets is not None)
        or (global_budgeting and (tree_budget is not None or tree_budgets is not None))
    ):
        raise ValueError("Use either one tree budget or per-request budgets")
    effective_tree_budgets = (
        tuple(int(value) for value in tree_budgets)
        if tree_budgets is not None
        else (variant.tree_budget if tree_budget is None else tree_budget,) * len(states)
    )
    if (
        len(effective_tree_budgets) != len(states)
        or any(
            value < 1 or value > variant.tree_budget
            for value in effective_tree_budgets
        )
    ):
        raise ValueError("Proposal tree budget override is outside the variant budget")

    def select_trees(proposal_rows):
        if not global_budgeting:
            trees = tuple(
                _probability_tree_for_variant(
                    proposal_rows[slot], state_tree_budget, variant,
                )
                for slot, state_tree_budget in enumerate(effective_tree_budgets)
            )
            return trees, effective_tree_budgets
        options = _globally_feasible_budget_options(
            global_tree_budget_options,
            request_count=len(states),
            row_budget=global_tree_row_budget,
            require_tier_fit=global_tree_require_tier_fit,
        )
        full_trees = tuple(
            _probability_tree_for_variant(
                proposal_rows[slot], options[-1], variant,
            )
            for slot in range(len(states))
        )
        candidate_trees = tuple(
            tuple(_prefix_tree(tree, budget) for budget in options)
            for tree in full_trees
        )
        utilities = tuple(
            tuple(_draft_tree_expected_tokens(tree) for tree in choices)
            for choices in candidate_trees
        )
        selected_budgets = _allocate_global_tree_budgets(
            states, options, utilities,
            row_budget=global_tree_row_budget,
            fixed_row_equivalent=global_tree_fixed_row_equivalent,
            criticality_weight=global_tree_criticality_weight,
            max_new_tokens=max_new_tokens,
            row_cost_curve=global_tree_row_cost_curve,
        )
        option_index = {budget: index for index, budget in enumerate(options)}
        trees = tuple(
            candidate_trees[slot][option_index[budget]]
            for slot, budget in enumerate(selected_budgets)
        )
        return trees, selected_budgets
    proposal_dtype = getattr(
        torch, proposal_probability_dtype or variant.probability_dtype,
    )
    if reuse_request_draft_cache:
        if variant.method not in {"ddtree", "dflash"}:
            raise ValueError("Packed persistent Draft cache supports DDTree/DFlash")
        if any(state.draft_cache is None or state.draft_update is None
               for state in states):
            raise RuntimeError("Persistent Draft state is missing")
        probability_dtype = proposal_dtype
        proposal_temperature = (
            variant.tree_proposal_temperature
            or variant.draft_temperature or variant.temperature or 1.0
        )
        packed_cache, cache_lengths, maximum_cache = _pad_cache_batch(
            [state.draft_cache for state in states],
            engine.cache_factory, engine.device,
        )
        update_lengths = tuple(
            int(state.draft_update.shape[1]) for state in states
        )
        prefix_lengths = tuple(
            _cache_length(state.target_cache) for state in states
        )
        if any(cache_len + update_len != prefix_len
               for cache_len, update_len, prefix_len in zip(
                   cache_lengths, update_lengths, prefix_lengths)):
            raise RuntimeError("Draft cache and feature update disagree")
        maximum_update = max(update_lengths)
        feature_width = states[0].draft_update.shape[-1]
        target_hidden = states[0].draft_update.new_zeros(
            (len(states), maximum_update, feature_width)
        )
        block_width = int(engine.draft.block_size)
        noise_ids = torch.full(
            (len(states), block_width), int(engine.draft.mask_token_id),
            dtype=torch.long, device=engine.device,
        )
        noise_ids[:, 0] = torch.tensor(
            [state.generated[-1] for state in states],
            dtype=torch.long, device=engine.device,
        )
        noise = engine.target.get_input_embeddings()(noise_ids)
        positions = torch.zeros(
            (len(states), maximum_update + block_width),
            dtype=torch.long, device=engine.device,
        )
        draft_mask = torch.full(
            (len(states), 1, block_width,
             maximum_cache + maximum_update + block_width),
            float("-inf"), dtype=noise.dtype, device=engine.device,
        )
        for slot, (state, cache_len, update_len, prefix_len, state_tree_budget) in enumerate(zip(
                states, cache_lengths, update_lengths, prefix_lengths,
                effective_tree_budgets)):
            if state.draft_update.shape[-1] != feature_width:
                raise RuntimeError("Draft updates have different feature widths")
            target_hidden[slot, :update_len] = state.draft_update[0]
            positions[slot, :update_len] = torch.arange(
                cache_len, prefix_len, device=engine.device,
            )
            positions[slot, maximum_update:] = torch.arange(
                prefix_len, prefix_len + block_width, device=engine.device,
            )
            draft_mask[slot, 0, :, :cache_len] = 0
            draft_mask[
                slot, 0, :, maximum_cache:maximum_cache + update_len
            ] = 0
            draft_mask[slot, 0, :, maximum_cache + maximum_update:] = 0
        hidden = engine.draft(
            target_hidden=target_hidden,
            noise_embedding=noise,
            position_ids=positions,
            attention_mask=draft_mask,
            past_key_values=packed_cache,
            use_cache=True,
            is_causal=False,
        )
        if not isinstance(hidden, torch.Tensor):
            hidden = hidden.last_hidden_state
        logits = engine.target.get_output_embeddings()(
            hidden[:, 1:variant.length + 1]
        )
        if variant.method == "dflash":
            proposals = None  # Greedy Draft path needs no vocabulary softmax.
        elif variant.tree_proposal_temperature_schedule is not None:
            temperatures = torch.tensor(
                variant.tree_proposal_temperature_schedule,
                dtype=probability_dtype, device=logits.device,
            )
            proposals = torch.softmax(
                logits.to(probability_dtype) / temperatures[None, :, None],
                dim=-1,
            )
        elif variant.tree_proposal_temperature_end is not None:
            temperatures = torch.linspace(
                proposal_temperature, variant.tree_proposal_temperature_end,
                variant.length, dtype=probability_dtype, device=logits.device,
            )
            proposals = torch.softmax(
                logits.to(probability_dtype) / temperatures[None, :, None],
                dim=-1,
            )
        else:
            proposals = probabilities(
                logits, proposal_temperature, probability_dtype,
            )
        if variant.method == "dflash":
            # Keep the baseline's parallel greedy proposal law. Caching must
            # not replace it with a sampled chain or a probability tree.
            paths = logits.argmax(-1)
            selected_trees = tuple(
                sampled_tree(paths[slot:slot + 1])
                for slot in range(len(states))
            )
            selected_budgets = effective_tree_budgets
        else:
            selected_trees, selected_budgets = select_trees(proposals)
        for slot, (state, cache_len, update_len, prefix_len) in enumerate(zip(
                states, cache_lengths, update_lengths, prefix_lengths)):
            state.draft_cache = _split_compacted_cache(
                packed_cache, engine.cache_factory, slot, cache_len,
                maximum_cache, range(update_len), engine.device,
            )
            if _cache_length(state.draft_cache) != prefix_len:
                raise RuntimeError("Draft cache and feature update disagree")
            state.draft_update = None
            state.tree = selected_trees[slot]
            state.proposal_tree_budget = selected_budgets[slot]
            state.proposal_probabilities = None
            state.round_prefix_len = prefix_len
            state.block_root = 0
            state.accepted_nodes.clear()
            state.accepted_tokens.clear()
            state.feature_chunks = None
            state.round_block_rows.clear()
        return selected_budgets
    prefix_lengths = [_cache_length(state.target_cache) for state in states]
    if any(
        state.full_features is None
        or state.full_features.shape[1] != prefix_len
        for state, prefix_len in zip(states, prefix_lengths)
    ):
        raise RuntimeError("Request features and Target cache disagree")
    maximum_prefix = max(prefix_lengths)
    feature_width = states[0].full_features.shape[-1]
    target_hidden = states[0].full_features.new_zeros(
        (len(states), maximum_prefix, feature_width)
    )
    for slot, (state, prefix_len) in enumerate(zip(states, prefix_lengths)):
        target_hidden[slot, :prefix_len] = state.full_features[0]
    block_width = int(engine.draft.block_size)
    noise_ids = torch.full(
        (len(states), block_width), int(engine.draft.mask_token_id),
        dtype=torch.long, device=engine.device,
    )
    noise_ids[:, 0] = torch.tensor(
        [state.generated[-1] for state in states],
        dtype=torch.long, device=engine.device,
    )
    noise = engine.target.get_input_embeddings()(noise_ids)
    positions = torch.zeros(
        (len(states), maximum_prefix + block_width),
        dtype=torch.long, device=engine.device,
    )
    draft_mask = torch.full(
        (len(states), 1, block_width, maximum_prefix + block_width),
        float("-inf"), dtype=noise.dtype, device=engine.device,
    )
    for slot, prefix_len in enumerate(prefix_lengths):
        positions[slot, :prefix_len] = torch.arange(
            prefix_len, device=engine.device,
        )
        positions[slot, maximum_prefix:] = torch.arange(
            prefix_len, prefix_len + block_width, device=engine.device,
        )
        draft_mask[slot, 0, :, :prefix_len] = 0
        draft_mask[slot, 0, :, maximum_prefix:] = 0
    hidden = engine.draft(
        target_hidden=target_hidden,
        noise_embedding=noise,
        position_ids=positions,
        attention_mask=draft_mask,
        past_key_values=None,
        use_cache=False,
        is_causal=False,
    )
    if not isinstance(hidden, torch.Tensor):
        hidden = hidden.last_hidden_state
    logits = engine.target.get_output_embeddings()(
        hidden[:, 1:variant.length + 1]
    )
    if variant.method == "ddtree_refine":
        refined_ids = noise_ids.clone()
        refine_length = min(variant.length, variant.diffusion_spur_length)
        refined_ids[:, 1:refine_length + 1] = logits[
            :, :refine_length
        ].argmax(-1)
        refined_noise = engine.target.get_input_embeddings()(refined_ids)
        refined_hidden = engine.draft(
            target_hidden=target_hidden,
            noise_embedding=refined_noise,
            position_ids=positions,
            attention_mask=draft_mask,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        if not isinstance(refined_hidden, torch.Tensor):
            refined_hidden = refined_hidden.last_hidden_state
        logits = engine.target.get_output_embeddings()(
            refined_hidden[:, 1:variant.length + 1]
        )
    if variant.method in {"ddtree", "ddtree_refine", "bv"}:
        probability_dtype = proposal_dtype
        proposal_temperature = (
            variant.tree_proposal_temperature
            or variant.draft_temperature or variant.temperature or 1.0
        )
        if variant.tree_proposal_temperature_schedule is not None:
            temperatures = torch.tensor(
                variant.tree_proposal_temperature_schedule,
                dtype=probability_dtype, device=logits.device,
            )
            proposals = torch.softmax(
                logits.to(probability_dtype) / temperatures[None, :, None],
                dim=-1,
            )
        elif variant.tree_proposal_temperature_end is not None:
            temperatures = torch.linspace(
                proposal_temperature, variant.tree_proposal_temperature_end,
                variant.length, dtype=probability_dtype, device=logits.device,
            )
            proposals = torch.softmax(
                logits.to(probability_dtype) / temperatures[None, :, None],
                dim=-1,
            )
        else:
            proposals = probabilities(
                logits, proposal_temperature, probability_dtype,
            )
    elif variant.method == "dflash":
        # Match the repository's DFlash baseline: the frozen diffusion Draft
        # contributes its parallel greedy path, while Target sampling at T=1
        # decides the accepted prefix and correction token.
        proposals = logits.argmax(-1)
    else:
        raise ValueError("Continuous proposal batching supports DDTree/DFlash")
    selected_trees = None
    selected_budgets = effective_tree_budgets
    if variant.method in {"ddtree", "ddtree_refine"}:
        selected_trees, selected_budgets = select_trees(proposals)
    for slot, (state, prefix_len) in enumerate(zip(states, prefix_lengths)):
        if variant.method in {"ddtree", "ddtree_refine"}:
            state.tree = selected_trees[slot]
            state.proposal_probabilities = None
        elif variant.method == "bv":
            path = sample(proposals[slot], state.generator)
            state.tree = sampled_tree(path.unsqueeze(0))
            state.proposal_probabilities = proposals[slot]
        else:
            state.tree = sampled_tree(proposals[slot:slot + 1])
            state.proposal_probabilities = None
        state.proposal_tree_budget = selected_budgets[slot]
        state.round_prefix_len = prefix_len
        state.block_root = 0
        state.accepted_nodes.clear()
        state.accepted_tokens.clear()
        state.feature_chunks = None
        state.round_block_rows.clear()
    return selected_budgets


def _select_ready(
        states: Sequence[ContinuousDecodeRequest], row_cap: int,
        slot_capacity: int | None, row_budget: int | None = None,
        use_proposal_tree_budget: bool = False):
    ready = [state for state in states if not state.done and state.tree is not None]
    ready.sort(key=lambda state: (
        0 if state.continuation else 1,
        -state.age,
        state.request_id,
    ))
    chosen = []
    blocks = []
    used_rows = 0
    for state in ready:
        state_row_cap = (
            state.proposal_tree_budget + 1
            if use_proposal_tree_budget
            and state.proposal_tree_budget is not None
            else row_cap
        )
        block = local_tree_block(
            state.tree.parents,
            state.tree.tokens,
            state.tree.depths,
            state.block_root,
            state_row_cap,
        )
        if slot_capacity is not None and len(chosen) >= slot_capacity:
            break
        if row_budget is not None and used_rows + len(block.nodes) > row_budget:
            continue
        chosen.append(state)
        blocks.append(block)
        used_rows += len(block.nodes)
    chosen_ids = {state.request_id for state in chosen}
    for state in ready:
        state.age = 0 if state.request_id in chosen_ids else state.age + 1
    return chosen, blocks


def _effective_row_cap(
        row_cap: int, low_occupancy_row_cap: int | None,
        low_occupancy_threshold: int | None, ready_count: int,
        mid_occupancy_row_cap: int | None = None,
        mid_occupancy_threshold: int | None = None) -> int:
    """Use the full tree when saved rows cannot admit another request."""
    if (
        low_occupancy_row_cap is not None
        and low_occupancy_threshold is not None
        and ready_count <= low_occupancy_threshold
    ):
        return low_occupancy_row_cap
    if (
        mid_occupancy_row_cap is not None
        and mid_occupancy_threshold is not None
        and ready_count <= mid_occupancy_threshold
    ):
        return mid_occupancy_row_cap
    return row_cap


def _continue_existing_tree(
        ready_count: int, blocks_verified: int, *,
        continuation_ready_threshold: int | None,
        max_continuation_blocks: int | None) -> bool:
    """Decide whether a sampled block-exit edge should reuse the old tree.

    Returning false is an exact early stop: the already sampled Target token is
    committed as this round's bonus and the next round builds a fresh tree.
    The controls only choose a stopping time and never inspect future Target
    draws, so they do not change the autoregressive output law.
    """

    if ready_count < 1 or blocks_verified < 1:
        raise ValueError("Continuation gate requires positive runtime counts")
    if (
        continuation_ready_threshold is not None
        and ready_count > continuation_ready_threshold
    ):
        return False
    completed_continuations = blocks_verified - 1
    if (
        max_continuation_blocks is not None
        and completed_continuations >= max_continuation_blocks
    ):
        return False
    return True


def _effective_tree_budget(
        default_budget: int, proposal_tree_budget: int | None,
        low_occupancy_tree_budget: int | None,
        low_occupancy_tree_threshold: int | None,
        active_count: int) -> int:
    """Select tree construction size from causal scheduler occupancy."""

    if (
        low_occupancy_tree_budget is not None
        and low_occupancy_tree_threshold is not None
        and active_count <= low_occupancy_tree_threshold
    ):
        return low_occupancy_tree_budget
    return proposal_tree_budget or default_budget


def _effective_tree_budgets(
        states: Sequence[ContinuousDecodeRequest], default_budget: int,
        proposal_tree_budget: int | None,
        low_occupancy_tree_budget: int | None,
        low_occupancy_tree_threshold: int | None, active_count: int,
        low_occupancy_expansion_slots: int | None) -> tuple[int, ...]:
    """Allocate expensive tail trees only to requests on the critical path.

    The requests with the shortest generated prefixes have the most remaining
    work and are therefore the best candidates to determine the final physical
    Target waves.  Ties are deterministic.  This decision observes only past
    scheduler state, so changing the allocation does not change exactness.
    """

    base_budget = proposal_tree_budget or default_budget
    expanded_budget = _effective_tree_budget(
        default_budget, proposal_tree_budget,
        low_occupancy_tree_budget, low_occupancy_tree_threshold,
        active_count,
    )
    if (
        expanded_budget == base_budget
        or low_occupancy_expansion_slots is None
        or low_occupancy_expansion_slots >= len(states)
    ):
        return (expanded_budget,) * len(states)
    ranked = sorted(
        states,
        key=lambda state: (len(state.generated), -state.age, state.request_id),
    )
    expanded_ids = {
        state.request_id
        for state in ranked[:low_occupancy_expansion_slots]
    }
    return tuple(
        expanded_budget if state.request_id in expanded_ids else base_budget
        for state in states
    )


def _adaptive_expansion_eligible(
        initial_count: int, thresholds: Sequence[int],
        requires_drain: bool) -> bool:
    """Keep low-load arrivals small while allowing a larger drained tail."""

    if initial_count < 1 or any(threshold < 1 for threshold in thresholds):
        raise ValueError("Invalid adaptive expansion occupancy")
    return (
        not requires_drain
        or not thresholds
        or initial_count > max(thresholds)
    )


@torch.inference_mode()
def generate_continuous_tree_blocks(
        engine, prompts: Sequence[torch.Tensor], variant: Variant, *,
        max_new_tokens: int, stop_ids: Sequence[int], seeds: Sequence[int],
        row_cap: int, slot_capacity: int | None = None,
        layout: str = "padded", row_budget: int | None = None,
        physical_row_width: int | None = None,
        low_occupancy_row_cap: int | None = None,
        low_occupancy_threshold: int | None = None,
        mid_occupancy_row_cap: int | None = None,
        mid_occupancy_threshold: int | None = None,
        verifier: str = "terminal_mass",
        reuse_request_draft_cache: bool = False,
        batched_prefill: bool = False,
        persistent_request_target_cache: bool = False,
        continuation_ready_threshold: int | None = None,
        max_continuation_blocks: int | None = None,
        proposal_tree_budget: int | None = None,
        low_occupancy_tree_budget: int | None = None,
        low_occupancy_tree_threshold: int | None = None,
        low_occupancy_expansion_slots: int | None = None,
        proposal_probability_dtype: str | None = None,
        adaptive_expansion_requires_drain: bool = False,
        global_tree_budget_options: Sequence[int] | None = None,
        global_tree_fixed_row_equivalent: float | None = None,
        global_tree_row_cost_curve: Sequence[Sequence[float]] | None = None,
        global_tree_require_tier_fit: bool = False,
        global_tree_expansion_active_limit: int | None = None,
        global_tree_criticality_weight: float = 0.0,
        fixed_padded_slots: bool = True,
        arrival_times_s: Sequence[float] | None = None,
        max_active_requests: int | None = None,
        ) -> ContinuousDecodeResult:
    """Decode independent requests with coalesced Target tree-block waves.

    ``physical_row_width`` is a padded-layout numerical reference control.  It
    lets a logical cap (for example 24 rows) execute inside the same physical
    row shape as a larger baseline (for example 46 rows).  Matching both this
    width and ``slot_capacity`` removes shape-induced BF16 drift, at the cost of
    giving up the smaller dense Target tensor and therefore the fast-path gain.
    The packed layout intentionally rejects this option.
    """
    variant.validate()
    global_options = (
        tuple(int(value) for value in global_tree_budget_options)
        if global_tree_budget_options is not None else None
    )
    global_row_cost_curve = _normalize_global_row_cost_curve(
        global_tree_row_cost_curve,
    )
    global_budgeting = global_options is not None
    open_loop = arrival_times_s is not None
    arrivals = (
        tuple(float(value) for value in arrival_times_s)
        if arrival_times_s is not None else ()
    )
    adaptive_cap = (
        low_occupancy_row_cap is not None
        or low_occupancy_threshold is not None
        or mid_occupancy_row_cap is not None
        or mid_occupancy_threshold is not None
    )
    if (
        variant.method not in {"ddtree", "ddtree_refine", "dflash", "bv"}
        or variant.temperature not in {0.0, 1.0}
        or (variant.method in {"ddtree", "ddtree_refine", "bv"}
            and variant.draft_temperature != 1.0)
        or (variant.method == "dflash"
            and variant.draft_temperature is not None)
        or row_cap < 1
        or verifier not in {
            "terminal_mass", "ancestral_batched", "ancestral_reference", "fused_scan",
            "same_draw_fused", "sparse_exit_fused_scan",
            "lazy_softmax_fused_scan", "direct_logits_fused_scan",
            "batched_direct_logits_fused_scan",
            "batched_sparse_logits_exit",
            "batched_lazy_projection_fused_scan",
            "batched_lazy_projection_sparse_exit",
            "block_verify",
            "greedy",
        }
        or (variant.temperature == 0.0 and (
            variant.method == "bv" or verifier != "greedy"
        ))
        or (variant.temperature == 1.0 and verifier == "greedy")
        or (variant.method == "dflash" and verifier not in (
            ("greedy",) if variant.temperature == 0.0
            else ("terminal_mass", "ancestral_reference")
        ))
        or (variant.method == "bv" and verifier != "block_verify")
        or (variant.method != "bv" and verifier == "block_verify")
        or (verifier == "ancestral_batched" and (
            variant.method not in {"ddtree", "ddtree_refine"}
            or adaptive_cap
            or row_cap != variant.tree_budget + 1
        ))
        or layout not in {"padded", "packed_sequence"}
        or (layout == "padded" and (slot_capacity is None or slot_capacity < 1))
        or (layout == "packed_sequence" and (row_budget is None or row_budget < row_cap))
        or (physical_row_width is not None and (
            layout != "padded" or physical_row_width < row_cap
        ))
        or (adaptive_cap and (
            variant.method not in {"ddtree", "ddtree_refine"}
            or layout != "packed_sequence"
            or low_occupancy_row_cap is None
            or low_occupancy_threshold is None
            or (mid_occupancy_row_cap is None)
               != (mid_occupancy_threshold is None)
            or low_occupancy_row_cap < row_cap
            or low_occupancy_threshold < 1
            or (mid_occupancy_row_cap is not None and (
                mid_occupancy_row_cap < row_cap
                or mid_occupancy_row_cap > low_occupancy_row_cap
                or mid_occupancy_threshold <= low_occupancy_threshold
            ))
            or row_budget is None
            or row_budget < low_occupancy_row_cap
        ))
        or (variant.method == "dflash" and row_cap < variant.length + 1)
        or (variant.method == "bv" and row_cap < variant.length + 1)
        or (reuse_request_draft_cache and variant.method not in {"ddtree", "dflash"})
        or (persistent_request_target_cache and layout != "packed_sequence")
        or (
            continuation_ready_threshold is not None
            and continuation_ready_threshold < 0
        )
        or (
            max_continuation_blocks is not None
            and max_continuation_blocks < 0
        )
        or (
            proposal_tree_budget is not None
            and not 1 <= proposal_tree_budget <= variant.tree_budget
        )
        or (
            (low_occupancy_tree_budget is None)
            != (low_occupancy_tree_threshold is None)
        )
        or (
            low_occupancy_tree_budget is not None
            and (
                proposal_tree_budget is None
                or not proposal_tree_budget <= low_occupancy_tree_budget <= variant.tree_budget
                or low_occupancy_tree_threshold < 1
            )
        )
        or (
            low_occupancy_expansion_slots is not None
            and (
                low_occupancy_tree_budget is None
                or low_occupancy_expansion_slots < 1
                or low_occupancy_expansion_slots
                   > low_occupancy_tree_threshold
            )
        )
        or proposal_probability_dtype not in {None, "float32", "float64"}
        or not isinstance(adaptive_expansion_requires_drain, bool)
        or (global_budgeting and (
            variant.method not in {"ddtree", "ddtree_refine"}
            or layout != "packed_sequence"
            or adaptive_cap
            or proposal_tree_budget is not None
            or low_occupancy_tree_budget is not None
            or low_occupancy_expansion_slots is not None
            or not global_options
            or tuple(sorted(set(global_options))) != global_options
            or global_options[0] < 1
            or global_options[-1] > variant.tree_budget
            or row_cap != global_options[0] + 1
            or row_budget is None
            or global_options[0] + 1 > row_budget
            or continuation_ready_threshold != 0
            or not isinstance(global_tree_require_tier_fit, bool)
            or (
                global_tree_expansion_active_limit is not None
                and global_tree_expansion_active_limit < 1
            )
            or (
                (global_tree_fixed_row_equivalent is None)
                == (global_row_cost_curve is None)
            )
            or (
                global_tree_fixed_row_equivalent is not None
                and (
                    not math.isfinite(global_tree_fixed_row_equivalent)
                    or global_tree_fixed_row_equivalent <= 0
                )
            )
            or (
                global_row_cost_curve is not None
                and (
                    global_row_cost_curve[0][0] > global_options[0] + 1
                    or global_row_cost_curve[-1][0] < row_budget
                )
            )
            or not math.isfinite(global_tree_criticality_weight)
            or global_tree_criticality_weight < 0
        ))
        or (not global_budgeting and (
            global_tree_fixed_row_equivalent is not None
            or global_row_cost_curve is not None
            or global_tree_require_tier_fit
            or global_tree_expansion_active_limit is not None
            or global_tree_criticality_weight != 0.0
        ))
        or not isinstance(fixed_padded_slots, bool)
        or (open_loop and (
            len(arrivals) != len(prompts)
            or any(not math.isfinite(value) or value < 0 for value in arrivals)
            or tuple(sorted(arrivals)) != arrivals
            or max_active_requests is None
            or max_active_requests < 1
            or persistent_request_target_cache
        ))
        or (not open_loop and max_active_requests is not None)
        or len(prompts) != len(seeds)
        or not prompts
    ):
        raise ValueError("Invalid continuous DDTree/DFlash experiment contract")
    if any(prompt.ndim != 2 or prompt.shape[0] != 1 for prompt in prompts):
        raise ValueError("Each prompt must have shape [1, tokens]")
    dtype = getattr(torch, variant.probability_dtype)
    padded_width = physical_row_width or row_cap
    stops = set(int(token) for token in stop_ids)
    states = [ContinuousDecodeRequest(index, prompt, int(seed))
              for index, (prompt, seed) in enumerate(zip(prompts, seeds))]
    engine.sync()
    wall_start = time.perf_counter()
    stage = {"prefill": 0.0, "proposal": 0.0, "target": 0.0,
             "pack": 0.0, "select_commit": 0.0}

    for state in states:
        state.generator = torch.Generator(device=engine.device).manual_seed(
            state.seed
        )
        state.draft_cache = engine.cache_factory()

    def prefill(new_states):
        """Prefill newly admitted requests at an open-loop wave boundary."""
        if not new_states:
            return
        started = time.perf_counter()
        if batched_prefill:
            prompt_lengths = tuple(
                int(state.input_ids.shape[1]) for state in new_states
            )
            maximum_prompt = max(prompt_lengths)
            ids = new_states[0].input_ids.new_zeros(
                (len(new_states), maximum_prompt)
            )
            positions = torch.zeros_like(ids)
            attention = torch.zeros_like(ids, dtype=torch.bool)
            for slot, (state, length) in enumerate(
                    zip(new_states, prompt_lengths)):
                ids[slot, :length] = state.input_ids[0]
                positions[slot, :length] = torch.arange(
                    length, device=engine.device,
                )
                attention[slot, :length] = True
            packed_prefill_cache = engine.cache_factory()
            output = engine.target_hidden_forward(
                ids, packed_prefill_cache, positions=positions, mask=attention,
            )
            last_rows = torch.tensor(
                [length - 1 for length in prompt_lengths],
                dtype=torch.long, device=engine.device,
            )
            last_hidden = output.last_hidden_state[
                torch.arange(len(new_states), device=engine.device), last_rows
            ]
            anchor_logits = engine.target.get_output_embeddings()(last_hidden)
            for slot, (state, length) in enumerate(
                    zip(new_states, prompt_lengths)):
                state.target_cache = _split_compacted_cache(
                    packed_prefill_cache, engine.cache_factory, slot, 0, 0,
                    range(length), engine.device,
                )
                anchor = (
                    int(anchor_logits[slot].argmax().item())
                    if variant.temperature == 0.0
                    else int(sample(
                        probabilities(anchor_logits[slot], 1.0, dtype),
                        state.generator,
                    ))
                )
                state.generated.append(anchor)
                state.full_features = torch.cat([
                    output.hidden_states[layer_index + 1][
                        slot:slot + 1, :length
                    ]
                    for layer_index in engine.draft.target_layer_ids
                ], dim=-1)
                state.draft_update = state.full_features
        else:
            for state in new_states:
                state.target_cache = engine.cache_factory()
                output = engine.target_forward(
                    state.input_ids, state.target_cache,
                    hidden=True, last_only=True,
                )
                anchor = (
                    int(output.logits[0, -1].argmax().item())
                    if variant.temperature == 0.0
                    else int(sample(
                        probabilities(output.logits[0, -1], 1.0, dtype),
                        state.generator,
                    ))
                )
                state.generated.append(anchor)
                state.full_features = engine.features(output.hidden_states)
                state.draft_update = state.full_features
        engine.sync()
        stage["prefill"] += 1000 * (time.perf_counter() - started)

    admitted_at = [None] * len(states)
    first_token_at = [None] * len(states)
    completed_at = [None] * len(states)
    if not open_loop:
        prefill(states)

    persistent_pool = None
    if persistent_request_target_cache:
        started = time.perf_counter()
        persistent_pool = _PersistentRequestCachePool(
            [state.target_cache for state in states],
            capacity=_persistent_cache_capacity(
                max(int(state.input_ids.shape[1]) for state in states),
                max_new_tokens, row_cap=row_cap, length=variant.length,
                low_occupancy_row_cap=low_occupancy_row_cap,
                mid_occupancy_row_cap=mid_occupancy_row_cap,
                global_options=global_options,
            ),
        )
        for state, cache in zip(states, persistent_pool.caches):
            state.target_cache = cache
        engine.sync()
        stage["prefill"] += 1000 * (time.perf_counter() - started)

    target_calls = 0
    physical_rows = 0
    useful_rows = 0
    effective_row_cap_counts: dict[int, int] = {}
    effective_tree_budget_counts: dict[int, int] = {}
    gated_continuations = 0
    expansion_thresholds = tuple(
        value for value in (
            low_occupancy_threshold,
            mid_occupancy_threshold,
            low_occupancy_tree_threshold,
        )
        if value is not None
    )
    expansion_eligible = _adaptive_expansion_eligible(
        len(states), expansion_thresholds,
        adaptive_expansion_requires_drain,
    )
    while not all(state.done for state in states):
        if open_loop:
            now = time.perf_counter() - wall_start
            active = sum(
                admitted_at[state.request_id] is not None and not state.done
                for state in states
            )
            capacity = max_active_requests - active
            ready = [
                state for state, arrival in zip(states, arrivals)
                if admitted_at[state.request_id] is None and arrival <= now
            ]
            newly_admitted = ready[:max(0, capacity)]
            if newly_admitted:
                admission = time.perf_counter() - wall_start
                for state in newly_admitted:
                    admitted_at[state.request_id] = admission
                prefill(newly_admitted)
                token_ready = time.perf_counter() - wall_start
                for state in newly_admitted:
                    first_token_at[state.request_id] = token_ready
                    if (
                        len(state.generated) >= max_new_tokens
                        or state.generated[-1] in stops
                    ):
                        state.done = True
                        completed_at[state.request_id] = token_ready
            active = sum(
                admitted_at[state.request_id] is not None and not state.done
                for state in states
            )
            if active == 0:
                future = [
                    arrival for state, arrival in zip(states, arrivals)
                    if admitted_at[state.request_id] is None
                ]
                if future:
                    time.sleep(max(0.0, min(future) - (
                        time.perf_counter() - wall_start
                    )))
                    continue
                break
        started = time.perf_counter()
        active_count = sum(
            not state.done and (
                not open_loop or admitted_at[state.request_id] is not None
            )
            for state in states
        )
        all_proposal_states = [
            state for state in states
            if (
                not state.done
                and (not open_loop or admitted_at[state.request_id] is not None)
                and state.tree is None
                and len(state.generated) < max_new_tokens
                and state.generated[-1] not in stops
            )
        ]
        if global_budgeting:
            wave_global_options = _active_global_tree_budget_options(
                global_options,
                active_count=active_count,
                expansion_active_limit=global_tree_expansion_active_limit,
            )
            proposal_states = list(_admit_global_proposal_states(
                all_proposal_states,
                row_budget=row_budget,
                minimum_tree_budget=global_options[0],
            ))
            admitted_ids = {state.request_id for state in proposal_states}
            deferred_count = len(all_proposal_states) - len(proposal_states)
            if deferred_count:
                # Budget 0 is telemetry for an admission deferral; it is not
                # passed to DDTree construction or Target verification.
                effective_tree_budget_counts[0] = (
                    effective_tree_budget_counts.get(0, 0) + deferred_count
                )
            for state in all_proposal_states:
                if state.request_id not in admitted_ids:
                    state.age += 1
        else:
            wave_global_options = None
            proposal_states = all_proposal_states
        requested_tree_budgets = (
            None if global_budgeting else _effective_tree_budgets(
                proposal_states, variant.tree_budget, proposal_tree_budget,
                low_occupancy_tree_budget if expansion_eligible else None,
                low_occupancy_tree_threshold if expansion_eligible else None,
                active_count,
                low_occupancy_expansion_slots,
            )
        )
        tree_budgets = _propose_many(
            engine, proposal_states, variant, reuse_request_draft_cache,
            tree_budgets=requested_tree_budgets,
            proposal_probability_dtype=proposal_probability_dtype,
            global_tree_budget_options=wave_global_options,
            global_tree_row_budget=row_budget if global_budgeting else None,
            global_tree_fixed_row_equivalent=(
                global_tree_fixed_row_equivalent if global_budgeting else None
            ),
            global_tree_row_cost_curve=(
                global_row_cost_curve if global_budgeting else None
            ),
            global_tree_require_tier_fit=(
                global_tree_require_tier_fit if global_budgeting else False
            ),
            global_tree_criticality_weight=(
                global_tree_criticality_weight if global_budgeting else 0.0
            ),
            max_new_tokens=max_new_tokens if global_budgeting else None,
        )
        for tree_budget in tree_budgets:
            effective_tree_budget_counts[tree_budget] = (
                effective_tree_budget_counts.get(tree_budget, 0) + 1
            )
        proposal_completions = []
        for state in states:
            if (
                not state.done
                and (not open_loop or admitted_at[state.request_id] is not None)
                and (len(state.generated) >= max_new_tokens
                     or state.generated[-1] in stops)
            ):
                state.done = True
                if open_loop and completed_at[state.request_id] is None:
                    proposal_completions.append(state.request_id)
        engine.sync()
        proposal_completed_at = time.perf_counter() - wall_start
        for request_id in proposal_completions:
            completed_at[request_id] = proposal_completed_at
        stage["proposal"] += 1000 * (time.perf_counter() - started)
        if all(state.done for state in states):
            break
        if open_loop and not any(
                admitted_at[state.request_id] is not None and not state.done
                for state in states):
            continue

        ready_count = sum(
            not state.done and state.tree is not None for state in states
        )
        effective_row_cap = _effective_row_cap(
            row_cap,
            low_occupancy_row_cap if expansion_eligible else None,
            low_occupancy_threshold if expansion_eligible else None,
            ready_count,
            mid_occupancy_row_cap if expansion_eligible else None,
            mid_occupancy_threshold if expansion_eligible else None,
        )
        chosen, blocks = _select_ready(
            states,
            effective_row_cap,
            slot_capacity,
            row_budget if layout == "packed_sequence" else None,
            use_proposal_tree_budget=global_budgeting,
        )
        if not chosen:
            raise RuntimeError("No ready request can make progress")
        effective_row_cap_counts[effective_row_cap] = (
            effective_row_cap_counts.get(effective_row_cap, 0) + 1
        )

        # Pad the final wave with clones.  Its outputs are discarded; the
        # physical Target shape remains fixed for the numerical audit.
        started = time.perf_counter()
        if layout == "padded":
            physical_states = list(chosen)
            physical_blocks = list(blocks)
            if fixed_padded_slots:
                while len(physical_states) < slot_capacity:
                    physical_states.append(chosen[0])
                    physical_blocks.append(blocks[0])
            packed_cache, cache_lengths, maximum_cache = _pad_cache_batch(
                [state.target_cache for state in physical_states],
                engine.cache_factory,
                engine.device,
            )
            cache_offsets = None
            query_offsets = None
            ids, positions, mask = _padded_block_inputs(
                physical_states,
                physical_blocks,
                padded_width,
                maximum_cache,
                next(engine.target.parameters()).dtype,
                engine.device,
            )
        else:
            packed_cache, cache_lengths, cache_offsets, maximum_cache = (
                _pack_cache_sequence(
                    [state.target_cache for state in chosen],
                    engine.cache_factory,
                    engine.device,
                )
            )
            ids, positions, mask, query_offsets = _packed_block_inputs(
                chosen,
                blocks,
                cache_lengths,
                cache_offsets,
                maximum_cache,
                next(engine.target.parameters()).dtype,
                engine.device,
            )
        engine.sync()
        stage["pack"] += 1000 * (time.perf_counter() - started)

        started = time.perf_counter()
        output = engine.target_hidden_forward(
            ids, packed_cache, positions=positions, mask=mask,
        )
        lazy_projection_verifiers = {
            "batched_lazy_projection_fused_scan",
            "batched_lazy_projection_sparse_exit",
        }
        logits_verifiers = {
            "lazy_softmax_fused_scan", "direct_logits_fused_scan",
            "batched_direct_logits_fused_scan",
            "batched_sparse_logits_exit",
        }
        # Probability-space verifiers share one FP64 softmax launch across the
        # physical wave.  The logit-space candidates deliberately defer or
        # fuse normalization and therefore consume the unchanged BF16 logits.
        if verifier in lazy_projection_verifiers:
            wave_values = output.last_hidden_state
        else:
            logits = engine.target.get_output_embeddings()(
                output.last_hidden_state
            )
            if variant.temperature == 0.0:
                wave_values = logits
            elif verifier == "batched_sparse_logits_exit":
                wave_values = logits
                deferred_log_normalizers = torch.logsumexp(
                    logits.to(dtype), dim=-1,
                )
            else:
                wave_values = (
                    logits if verifier in logits_verifiers
                    else probabilities(logits, 1.0, dtype)
                )
        engine.sync()
        stage["target"] += 1000 * (time.perf_counter() - started)
        target_calls += 1
        physical_rows += (
            len(physical_states) * padded_width if layout == "padded"
            else sum(len(block.nodes) for block in blocks)
        )
        useful_rows += sum(len(block.nodes) for block in blocks)

        started = time.perf_counter()
        local_values_by_slot = []
        for slot, block in enumerate(blocks):
            if layout == "padded":
                local_values = wave_values[slot, :len(block.nodes)]
            else:
                query_offset = query_offsets[slot]
                local_values = wave_values[
                    0, query_offset:query_offset + len(block.nodes)
                ]
            local_values_by_slot.append(local_values)
        batched_sparse_decisions = None
        if verifier == "sparse_exit_fused_scan":
            node_counts = {len(block.nodes) for block in blocks}
            depths = {
                max(block.depths) - min(block.depths)
                for block in blocks
            }
            batched_probabilities = local_values_by_slot
            if len(node_counts) == 1 and len(depths) == 1:
                node_count = next(iter(node_counts))
                if layout == "padded":
                    batched_probabilities = wave_values[
                        :len(blocks), :node_count
                    ].contiguous()
                else:
                    batched_probabilities = wave_values[0].view(
                        len(blocks), node_count, wave_values.shape[-1]
                    )
            batched_sparse_decisions = (
                tree_verify_ancestral_sparse_exit_fused_scan_batched(
                    [block.parents for block in blocks],
                    [block.tokens for block in blocks],
                    batched_probabilities,
                    [state.generator for state in chosen],
                    validate=False,
                )
            )
        batched_direct_decisions = None
        if verifier == "batched_direct_logits_fused_scan":
            node_counts = {len(block.nodes) for block in blocks}
            depths = {
                max(block.depths) - min(block.depths)
                for block in blocks
            }
            if len(node_counts) == 1 and len(depths) == 1:
                node_count = next(iter(node_counts))
                if layout == "padded":
                    batched_values = wave_values[
                        :len(blocks), :node_count
                    ].contiguous()
                else:
                    batched_values = wave_values[0].view(
                        len(blocks), node_count, wave_values.shape[-1]
                    )
                batched_direct_decisions = (
                    tree_verify_ancestral_logits_fused_scan_batched(
                        [block.parents for block in blocks],
                        [block.tokens for block in blocks],
                        batched_values, 1.0, dtype,
                        [state.generator for state in chosen],
                        validate=False,
                    )
                )
        batched_sparse_logits_decisions = None
        if verifier == "batched_sparse_logits_exit":
            node_counts = {len(block.nodes) for block in blocks}
            depths = {
                max(block.depths) - min(block.depths)
                for block in blocks
            }
            if len(node_counts) != 1 or len(depths) != 1:
                raise RuntimeError(
                    "Sparse-logit candidate requires equally shaped tree blocks"
                )
            node_count = next(iter(node_counts))
            if layout == "padded":
                batched_values = wave_values[
                    :len(blocks), :node_count
                ].contiguous()
                batched_log_normalizers = deferred_log_normalizers[
                    :len(blocks), :node_count
                ].contiguous()
            else:
                batched_values = wave_values[0].view(
                    len(blocks), node_count, wave_values.shape[-1]
                )
                batched_log_normalizers = deferred_log_normalizers[0].view(
                    len(blocks), node_count
                )
            batched_sparse_logits_decisions = (
                tree_verify_ancestral_logits_sparse_exit_batched(
                    [block.parents for block in blocks],
                    [block.tokens for block in blocks],
                    batched_values, batched_log_normalizers,
                    1.0, dtype, [state.generator for state in chosen],
                    validate=False,
                )
            )
        batched_lazy_projection_decisions = None
        if verifier in lazy_projection_verifiers:
            node_counts = {len(block.nodes) for block in blocks}
            depths = {
                max(block.depths) - min(block.depths)
                for block in blocks
            }
            if len(node_counts) == 1 and len(depths) == 1:
                node_count = next(iter(node_counts))
                if layout == "padded":
                    batched_hidden = wave_values[
                        :len(blocks), :node_count
                    ].contiguous()
                else:
                    batched_hidden = wave_values[0].view(
                        len(blocks), node_count, wave_values.shape[-1]
                    )
                batched_lazy_function = (
                    tree_verify_ancestral_lazy_projection_sparse_exit_batched
                    if verifier == "batched_lazy_projection_sparse_exit"
                    else tree_verify_ancestral_lazy_projection_fused_scan_batched
                )
                batched_lazy_projection_decisions = (
                    batched_lazy_function(
                        [block.parents for block in blocks],
                        [block.tokens for block in blocks],
                        batched_hidden, engine.target.get_output_embeddings(),
                        1.0, dtype,
                        [state.generator for state in chosen],
                        validate=False,
                    )
                )
        decisions = []
        for slot, (state, block, old_length) in enumerate(zip(
                chosen, blocks, cache_lengths)):
            local_values = local_values_by_slot[slot]
            if verifier == "greedy":
                local_nodes, local_tokens, bonus = tree_verify_greedy_block(
                    block.parents, block.tokens, local_values,
                )
            elif verifier == "ancestral_reference":
                # One common probability-space implementation for chains,
                # full trees and global-budget prefixes.  In particular, do
                # not round-trip DFlash's already-host-resident tokens via a
                # GPU tensor just to call the same reference tree walk.
                local_nodes, local_tokens, bonus = tree_verify_ancestral_batched(
                    block.parents, block.tokens, local_values,
                    state.generator, validate=False,
                )
            elif variant.method == "dflash":
                path = torch.tensor(
                    block.tokens, dtype=torch.long, device=engine.device,
                )
                accepted, bonus = matching_verify(
                    path, local_values, state.generator,
                )
                local_nodes = list(range(1, accepted + 1))
                local_tokens = list(block.tokens[:accepted])
            elif variant.method == "bv":
                if (
                    state.block_root != 0
                    or len(block.nodes) != variant.length + 1
                    or state.proposal_probabilities is None
                    or state.proposal_probabilities.shape
                       != (variant.length, local_values.shape[-1])
                ):
                    raise RuntimeError(
                        "BlockVerify requires one complete diffusion proposal block"
                    )
                path = torch.tensor(
                    block.tokens, dtype=torch.long, device=engine.device,
                )
                accepted, bonus = block_verify_batched(
                    path,
                    local_values,
                    state.proposal_probabilities,
                    state.generator,
                )
                local_nodes = list(range(1, accepted + 1))
                local_tokens = list(block.tokens[:accepted])
            elif verifier == "ancestral_batched":
                if len(block.nodes) != variant.tree_budget + 1:
                    raise RuntimeError(
                        "The official ancestral verifier requires the full tree"
                    )
                local_nodes, local_tokens, bonus = tree_verify_ancestral_batched(
                    block.parents,
                    block.tokens,
                    local_values,
                    state.generator,
                    validate=False,
                )
            elif verifier == "fused_scan":
                local_nodes, local_tokens, bonus = (
                    tree_verify_ancestral_fused_scan(
                        block.parents, block.tokens, local_values,
                        state.generator, validate=False,
                    )
                )
            elif verifier == "same_draw_fused":
                local_nodes, local_tokens, bonus = (
                    tree_verify_ancestral_same_draw_fused(
                        block.parents, block.tokens, local_values,
                        state.generator, validate=False,
                    )
                )
            elif verifier == "sparse_exit_fused_scan":
                local_nodes, local_tokens, bonus = batched_sparse_decisions[slot]
            elif verifier == "lazy_softmax_fused_scan":
                local_nodes, local_tokens, bonus, _ = (
                    tree_verify_ancestral_lazy_softmax_fused_scan(
                        block.parents, block.tokens, local_values, 1.0, dtype,
                        state.generator, validate=False,
                    )
                )
            elif verifier == "direct_logits_fused_scan":
                local_nodes, local_tokens, bonus, _ = (
                    tree_verify_ancestral_logits_fused_scan(
                        block.parents, block.tokens, local_values.contiguous(),
                        1.0, dtype, state.generator, validate=False,
                    )
                )
            elif verifier == "batched_direct_logits_fused_scan":
                if batched_direct_decisions is not None:
                    local_nodes, local_tokens, bonus, _ = (
                        batched_direct_decisions[slot]
                    )
                else:
                    local_nodes, local_tokens, bonus, _ = (
                        tree_verify_ancestral_logits_fused_scan(
                            block.parents, block.tokens,
                            local_values.contiguous(), 1.0, dtype,
                            state.generator, validate=False,
                        )
                    )
            elif verifier == "batched_sparse_logits_exit":
                local_nodes, local_tokens, bonus = (
                    batched_sparse_logits_decisions[slot]
                )
            elif verifier in lazy_projection_verifiers:
                if batched_lazy_projection_decisions is not None:
                    local_nodes, local_tokens, bonus, _ = (
                        batched_lazy_projection_decisions[slot]
                    )
                else:
                    if verifier == "batched_lazy_projection_sparse_exit":
                        local_nodes, local_tokens, bonus, _ = (
                            tree_verify_ancestral_lazy_projection_sparse_exit_batched(
                                [block.parents], [block.tokens],
                                local_values.contiguous().unsqueeze(0),
                                engine.target.get_output_embeddings(), 1.0, dtype,
                                [state.generator], validate=False,
                            )[0]
                        )
                    else:
                        local_nodes, local_tokens, bonus, _ = (
                            tree_verify_ancestral_lazy_projection_fused_scan(
                                block.parents, block.tokens,
                                local_values.contiguous(),
                                engine.target.get_output_embeddings(), 1.0, dtype,
                                state.generator, validate=False,
                            )
                        )
            else:
                local_nodes, local_tokens, bonus = (
                    tree_block_verify_terminal_mass(
                        block.parents,
                        block.tokens,
                        local_values,
                        state.generator,
                        validate=False,
                        prefix_mode="batched",
                        exit_mode="internal",
                    )
                )
            mapped_nodes = [block.nodes[node] for node in local_nodes]
            keep_query = [0] + local_nodes
            if layout == "padded":
                state.target_cache = _split_compacted_cache(
                    packed_cache, engine.cache_factory, slot, old_length,
                    maximum_cache, keep_query, engine.device,
                )
            decisions.append((
                slot, state, block, old_length, local_nodes, local_tokens,
                bonus, mapped_nodes, keep_query,
            ))

        if layout == "packed_sequence":
            if persistent_pool is not None:
                persistent_pool.append_packed_sequence(
                    packed_cache,
                    [decision[1].target_cache for decision in decisions],
                    maximum_cache,
                    query_offsets,
                    [decision[-1] for decision in decisions],
                    engine.device,
                )
            else:
                split_caches = _split_compacted_sequence_caches(
                    packed_cache,
                    engine.cache_factory,
                    cache_offsets,
                    cache_lengths,
                    maximum_cache,
                    query_offsets,
                    [decision[-1] for decision in decisions],
                    engine.device,
                )
                for decision, cache in zip(decisions, split_caches):
                    decision[1].target_cache = cache

            flat_keep = []
            feature_boundaries = [0]
            for slot, *_, keep_query in decisions:
                flat_keep.extend(query_offsets[slot] + row for row in keep_query)
                feature_boundaries.append(len(flat_keep))
            feature_index = torch.tensor(
                flat_keep, dtype=torch.long, device=engine.device,
            )
            gathered_feature_layers = [
                output.hidden_states[layer_index + 1].index_select(
                    1, feature_index,
                )
                for layer_index in engine.draft.target_layer_ids
            ]
            gathered_features = torch.cat(gathered_feature_layers, dim=-1)

        commit_completions = []
        for decision_index, decision in enumerate(decisions):
            (
                slot, state, block, old_length, local_nodes, local_tokens,
                bonus, mapped_nodes, keep_query,
            ) = decision
            if layout == "padded":
                keep_index = torch.tensor(
                    keep_query, dtype=torch.long, device=engine.device,
                )
                chosen_features = torch.cat([
                    output.hidden_states[layer_index + 1][
                        slot:slot + 1
                    ].index_select(1, keep_index)
                    for layer_index in engine.draft.target_layer_ids
                ], dim=-1)
            else:
                start, end = feature_boundaries[
                    decision_index:decision_index + 2
                ]
                chosen_features = gathered_features[:, start:end]
            if state.feature_chunks is None:
                state.feature_chunks = [chosen_features]
            else:
                state.feature_chunks.append(chosen_features)
            state.round_block_rows.append(len(block.nodes))
            if state.block_root:
                state.accepted_nodes.append(state.block_root)
                state.accepted_tokens.append(
                    state.tree.tokens[state.block_root - 1]
                )
            state.accepted_nodes.extend(mapped_nodes)
            state.accepted_tokens.extend(int(token) for token in local_tokens)
            terminal = mapped_nodes[-1] if mapped_nodes else state.block_root
            continuation = child_for_edge(
                state.tree.parents,
                state.tree.tokens,
                terminal,
                int(bonus),
            )
            # A BlockVerify correction ends the current coupled block even if
            # its token happens to equal the next drafted token.  Re-entering
            # the already sampled block would apply a second coupling law.
            if variant.method == "bv":
                continuation = None
            if continuation is not None:
                if continuation in block.nodes:
                    raise AssertionError("Cascade exited through a verified edge")
                if _continue_existing_tree(
                    ready_count,
                    len(state.round_block_rows),
                    continuation_ready_threshold=continuation_ready_threshold,
                    max_continuation_blocks=max_continuation_blocks,
                ):
                    state.block_root = continuation
                    continue
                gated_continuations += 1

            appended = state.accepted_tokens + [int(bonus)]
            committed = appended[:max_new_tokens - len(state.generated)]
            for index, token in enumerate(committed):
                if token in stops:
                    committed = committed[:index + 1]
                    break
            state.generated.extend(committed)
            if state.feature_chunks is None:
                raise AssertionError("Completed round has no retained Target state")
            update = torch.cat(state.feature_chunks, dim=1)
            state.full_features = torch.cat((state.full_features, update), dim=1)
            state.draft_update = update
            if state.full_features.shape[1] != _cache_length(state.target_cache):
                raise RuntimeError("Committed features and cache diverged")
            state.rounds.append({
                "committed_tokens": len(committed),
                "accepted_draft_tokens": len(state.accepted_nodes),
                "block_rows": list(state.round_block_rows),
                "continuation_blocks": len(state.round_block_rows) - 1,
                "gated_continuation": continuation is not None,
            })
            state.tree = None
            state.proposal_probabilities = None
            state.feature_chunks = None
            state.age = 0
            if (
                len(state.generated) >= max_new_tokens
                or state.generated[-1] in stops
            ):
                state.done = True
                if open_loop and completed_at[state.request_id] is None:
                    commit_completions.append(state.request_id)
        engine.sync()
        commit_completed_at = time.perf_counter() - wall_start
        for request_id in commit_completions:
            completed_at[request_id] = commit_completed_at
        stage["select_commit"] += 1000 * (time.perf_counter() - started)

    engine.sync()
    wall_ms = 1000 * (time.perf_counter() - wall_start)
    request_metrics = ()
    if open_loop:
        if any(value is None for value in (
                admitted_at + first_token_at + completed_at)):
            raise RuntimeError("Open-loop request timing is incomplete")
        metrics = []
        for state, arrival, admitted, first, completed in zip(
                states, arrivals, admitted_at, first_token_at, completed_at):
            token_count = len(state.generated)
            metrics.append({
                "request_id": state.request_id,
                "arrival_ms": 1000 * arrival,
                "admitted_ms": 1000 * admitted,
                "first_token_ms": 1000 * first,
                "completed_ms": 1000 * completed,
                "queue_ms": 1000 * (admitted - arrival),
                "ttft_ms": 1000 * (first - arrival),
                "e2e_ms": 1000 * (completed - arrival),
                "service_ms": 1000 * (completed - admitted),
                "tpot_ms": (
                    1000 * (completed - first) / (token_count - 1)
                    if token_count > 1 else 0.0
                ),
                "output_tokens": token_count,
            })
        request_metrics = tuple(metrics)
    return ContinuousDecodeResult(
        outputs=tuple(tuple(state.generated) for state in states),
        request_rounds=tuple(tuple(state.rounds) for state in states),
        wall_ms=wall_ms,
        prefill_ms=stage["prefill"],
        proposal_ms=stage["proposal"],
        target_ms=stage["target"],
        pack_ms=stage["pack"],
        select_commit_ms=stage["select_commit"],
        physical_target_calls=target_calls,
        physical_target_rows=physical_rows,
        useful_target_rows=useful_rows,
        padded_target_rows=physical_rows - useful_rows,
        effective_row_cap_counts=tuple(sorted(effective_row_cap_counts.items())),
        effective_tree_budget_counts=tuple(
            sorted(effective_tree_budget_counts.items())
        ),
        gated_continuations=gated_continuations,
        arrival_times_s=arrivals,
        request_metrics=request_metrics,
    )
