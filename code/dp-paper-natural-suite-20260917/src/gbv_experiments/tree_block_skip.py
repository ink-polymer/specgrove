"""Exact tree-block continuation with branch-level skipped verification.

The full proposal tree is partitioned lazily.  A Target call verifies one
ancestor-closed block.  If its terminal draw takes an original-tree edge that
was not present in the block, that sampled child becomes the root of the next
block.  Sibling subtrees can then be skipped because the autoregressive path
has already selected a different branch.

Every visited prefix still uses its complete Target distribution.  Therefore
the cascade has the same output law as a full ancestral DDTree walk; only the
schedule of Target rows changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import torch

from .sampling import tree_block_verify_terminal_mass


def latency_curve_from_anchors(
    maximum_rows: int, anchors: Sequence[tuple[int, float]],
) -> tuple[float, ...]:
    """Linearly interpolate a dense row-latency curve from measurements."""

    if (maximum_rows < 1 or not anchors
            or any(row < 1 or not math.isfinite(value) or value < 0
                   for row, value in anchors)
            or any(left[0] >= right[0]
                   for left, right in zip(anchors, anchors[1:]))
            or anchors[-1][0] < maximum_rows):
        raise ValueError("Invalid Target latency anchors")
    curve = [0.0] * (maximum_rows + 1)
    first_row, first_value = anchors[0]
    for row in range(1, min(first_row, maximum_rows) + 1):
        curve[row] = float(first_value)
    for (left_row, left_value), (right_row, right_value) in zip(
            anchors, anchors[1:]):
        width = right_row - left_row
        for row in range(max(1, left_row), min(maximum_rows, right_row) + 1):
            fraction = (row - left_row) / width
            curve[row] = float(left_value + fraction * (
                right_value - left_value
            ))
    return tuple(curve)


@dataclass(frozen=True)
class TreeBlock:
    """One connected, ancestor-closed block in original node numbering."""

    root: int
    nodes: tuple[int, ...]
    parents: tuple[int, ...]
    tokens: tuple[int, ...]
    depths: tuple[int, ...]


@dataclass(frozen=True)
class CascadeResult:
    accepted_nodes: tuple[int, ...]
    accepted_tokens: tuple[int, ...]
    bonus_token: int
    block_roots: tuple[int, ...]
    block_rows: tuple[int, ...]


@dataclass(frozen=True)
class CostAwareBlockDecision:
    """Branch-aware first block selected by expected Target latency."""

    block: TreeBlock
    first_call_latency_ms: float
    expected_rescue_latency_ms: float
    objective_ms: float
    frontier_nodes: tuple[int, ...]


def _validate_topology(parents: Sequence[int], tokens: Sequence[int]) -> None:
    if (not parents or len(tokens) != len(parents) - 1
            or parents[0] != -1 or any(
                isinstance(parent, bool) or not isinstance(parent, Integral)
                or parent < 0 or parent >= node
                for node, parent in enumerate(parents[1:], 1)
            )):
        raise ValueError("Invalid topologically ordered tree")
    if len(set(zip(parents[1:], tokens))) != len(tokens):
        raise ValueError("A tree parent cannot repeat a child token")


def local_tree_block(
    parents: Sequence[int], tokens: Sequence[int], depths: Sequence[int],
    root: int, row_cap: int,
) -> TreeBlock:
    """Take the earliest ``row_cap`` nodes in ``root``'s remaining subtree.

    Probability-tree ids are best-first Draft prefix order.  Restricting that
    order to a subtree preserves ancestor closure because every ancestor has a
    smaller id.  The root row is included in the cap.
    """

    _validate_topology(parents, tokens)
    if (len(depths) != len(parents) or isinstance(root, bool)
            or not isinstance(root, Integral) or not 0 <= root < len(parents)
            or isinstance(row_cap, bool) or not isinstance(row_cap, Integral)
            or row_cap < 1):
        raise ValueError("Invalid local tree-block controls")
    root = int(root)
    row_cap = int(row_cap)

    def below_root(node: int) -> bool:
        while node > root:
            node = int(parents[node])
        return node == root

    nodes = tuple(
        node for node in range(root, len(parents)) if below_root(node)
    )[:row_cap]
    if not nodes or nodes[0] != root:
        raise AssertionError("Tree block lost its root")
    return tree_block_from_nodes(parents, tokens, depths, root, nodes)


def tree_block_from_nodes(
    parents: Sequence[int], tokens: Sequence[int], depths: Sequence[int],
    root: int, nodes: Sequence[int],
) -> TreeBlock:
    """Build a local block from an arbitrary ancestor-closed node set."""

    _validate_topology(parents, tokens)
    if (len(depths) != len(parents) or not nodes
            or tuple(sorted(set(nodes))) != tuple(nodes)
            or int(nodes[0]) != root):
        raise ValueError("Invalid explicit tree block")
    node_to_row = {int(node): row for row, node in enumerate(nodes)}
    if any(int(parents[node]) not in node_to_row for node in nodes[1:]):
        raise AssertionError("Tree block is not ancestor closed")
    return TreeBlock(
        root=root,
        nodes=tuple(int(node) for node in nodes),
        parents=(-1,) + tuple(
            node_to_row[int(parents[node])] for node in nodes[1:]
        ),
        tokens=tuple(int(tokens[node - 1]) for node in nodes[1:]),
        depths=tuple(int(depths[node]) for node in nodes),
    )


def cost_aware_tree_block(
    parents: Sequence[int], tokens: Sequence[int], depths: Sequence[int],
    draft_prefix_log_masses: Sequence[float],
    latency_by_rows: Sequence[float],
) -> CostAwareBlockDecision:
    """Choose a branch-aware block by tree knapsack, without a row cap.

    For every possible first-block size, a dynamic program finds the
    ancestor-closed node set with minimum expected rescue latency.  Omitting a
    frontier child ``v`` costs

        DraftReach(v | root) * TargetLatency(full_subtree(v)).

    The chosen set minimizes first-call latency plus that rescue cost.  A
    reached omitted child is verified as one full-subtree rescue block, so the
    heuristic affects execution cost only; it never conditions or renormalizes
    the Target draw.
    """

    _validate_topology(parents, tokens)
    count = len(parents)
    if (len(depths) != count
            or len(draft_prefix_log_masses) != count
            or len(latency_by_rows) <= count):
        raise ValueError("Invalid cost-aware tree-block controls")
    root = 0
    if any(not math.isfinite(float(value)) or float(value) < 0
           for value in latency_by_rows[1:count + 1]):
        raise ValueError("Target latency curve must be finite and nonnegative")
    children: list[list[int]] = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        children[int(parent)].append(node)

    subtree_sizes = [1] * count
    for node in range(count - 1, root, -1):
        subtree_sizes[int(parents[node])] += subtree_sizes[node]
    root_log_mass = float(draft_prefix_log_masses[root])
    conditional_mass = [
        math.exp(float(value) - root_log_mass)
        for value in draft_prefix_log_masses
    ]

    # size -> (expected rescue latency, selected nodes).  The zero-size child
    # option means skip that entire branch and retain it as an exact fallback.
    memo: dict[int, dict[int, tuple[float, tuple[int, ...]]]] = {}

    def solve(node: int) -> dict[int, tuple[float, tuple[int, ...]]]:
        cached = memo.get(node)
        if cached is not None:
            return cached
        states: dict[int, tuple[float, tuple[int, ...]]] = {
            1: (0.0, (node,)),
        }
        for child in children[node]:
            skipped = conditional_mass[child] * float(
                latency_by_rows[subtree_sizes[child]]
            )
            options = {0: (skipped, ())} | solve(child)
            merged: dict[int, tuple[float, tuple[int, ...]]] = {}
            for left_size, (left_cost, left_nodes) in states.items():
                for right_size, (right_cost, right_nodes) in options.items():
                    size = left_size + right_size
                    candidate = (left_cost + right_cost,
                                 left_nodes + right_nodes)
                    current = merged.get(size)
                    if current is None or candidate[0] < current[0]:
                        merged[size] = candidate
            states = merged
        memo[node] = states
        return states

    candidates = solve(root)
    size, (rescue_cost, selected) = min(
        candidates.items(),
        key=lambda item: (
            float(latency_by_rows[item[0]]) + item[1][0],
            item[1][0], -item[0],
        ),
    )
    selected = tuple(sorted(selected))
    selected_set = set(selected)
    frontier = tuple(
        child for parent in selected for child in children[parent]
        if child not in selected_set
    )
    first_cost = float(latency_by_rows[size])
    return CostAwareBlockDecision(
        block=tree_block_from_nodes(
            parents, tokens, depths, root, selected,
        ),
        first_call_latency_ms=first_cost,
        expected_rescue_latency_ms=float(rescue_cost),
        objective_ms=first_cost + float(rescue_cost),
        frontier_nodes=frontier,
    )


def child_for_edge(
    parents: Sequence[int], tokens: Sequence[int], parent: int, token: int,
) -> int | None:
    """Return the unique original child reached by ``(parent, token)``."""

    for node in range(parent + 1, len(parents)):
        if int(parents[node]) == parent and int(tokens[node - 1]) == token:
            return node
    return None


def risk_bounded_prefix_cap(
    parents: Sequence[int], draft_prefix_log_masses: Sequence[float],
    risk_budget: float, minimum_rows: int,
) -> int:
    """Smallest canonical prefix whose omitted frontier has low Draft mass.

    The returned count includes root.  This is a cost policy only: an edge on
    the omitted frontier still receives exact continuation if Target samples
    it, so a miscalibrated Draft risk estimate cannot bias the output law.
    """

    if (len(draft_prefix_log_masses) != len(parents)
            or not 0 < risk_budget < 1
            or not 1 <= minimum_rows <= len(parents)):
        raise ValueError("Invalid risk-bounded block controls")
    masses = [math.exp(float(value)) for value in draft_prefix_log_masses]
    children: list[list[int]] = [[] for _ in parents]
    for node, parent in enumerate(parents[1:], 1):
        children[int(parent)].append(node)
    for cap in range(minimum_rows, len(parents) + 1):
        boundary = math.fsum(
            masses[child]
            for parent in range(cap)
            for child in children[parent]
            if child >= cap
        )
        if boundary <= risk_budget:
            return cap
    return len(parents)


def cascade_terminal_verify(
    parents: Sequence[int], tokens: Sequence[int], depths: Sequence[int],
    all_probabilities: torch.Tensor, row_cap: int,
    generator: torch.Generator | None = None, *, validate: bool = True,
) -> CascadeResult:
    """Reference cascade using precomputed Target rows for every full node.

    Production execution obtains only the rows for each reached block.  This
    all-rows reference exists for law tests and offline schedule analysis.
    """

    _validate_topology(parents, tokens)
    if (all_probabilities.ndim != 2
            or all_probabilities.shape[0] != len(parents)):
        raise ValueError("Expected one Target probability row per full node")
    accepted_nodes: list[int] = []
    accepted_tokens: list[int] = []
    block_roots: list[int] = []
    block_rows: list[int] = []
    root = 0
    while True:
        block = local_tree_block(parents, tokens, depths, root, row_cap)
        block_roots.append(root)
        block_rows.append(len(block.nodes))
        index = torch.tensor(
            block.nodes, dtype=torch.long, device=all_probabilities.device,
        )
        local_nodes, local_tokens, bonus = tree_block_verify_terminal_mass(
            block.parents, block.tokens,
            all_probabilities.index_select(0, index), generator,
            validate=validate, prefix_mode="batched", exit_mode="internal",
        )
        mapped = [block.nodes[node] for node in local_nodes]
        if root:
            accepted_nodes.append(root)
            accepted_tokens.append(int(tokens[root - 1]))
        accepted_nodes.extend(mapped)
        accepted_tokens.extend(int(token) for token in local_tokens)
        terminal = mapped[-1] if mapped else root
        child = child_for_edge(parents, tokens, terminal, int(bonus))
        if child is None:
            return CascadeResult(
                tuple(accepted_nodes), tuple(accepted_tokens), int(bonus),
                tuple(block_roots), tuple(block_rows),
            )
        if child in block.nodes:
            raise AssertionError("Terminal draw cannot exit through a block edge")
        root = child


def expected_schedule_cost(
    parents: Sequence[int], block_roots: Sequence[int],
    block_rows: Sequence[int], prefix_masses: Sequence[float],
    latency_by_rows: Sequence[float],
) -> dict[str, float]:
    """Compute exact expected calls/rows/latency for a fixed tree partition."""

    if (len(block_roots) != len(block_rows)
            or len(prefix_masses) != len(parents)
            or not latency_by_rows):
        raise ValueError("Schedule-cost inputs disagree")
    probability = [1.0 if root == 0 else float(prefix_masses[root])
                   for root in block_roots]
    if any(rows < 1 or rows >= len(latency_by_rows) for rows in block_rows):
        raise ValueError("Missing latency for a block row count")
    return {
        "expected_target_calls": sum(probability),
        "expected_target_rows": sum(
            chance * rows for chance, rows in zip(probability, block_rows)
        ),
        "expected_target_latency_ms": sum(
            chance * float(latency_by_rows[rows])
            for chance, rows in zip(probability, block_rows)
        ),
    }
