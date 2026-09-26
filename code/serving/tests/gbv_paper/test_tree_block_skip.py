import collections

import torch

from gbv_experiments.sampling import tree_verify_ancestral_batched
from gbv_experiments.tree_block_skip import (
    cascade_terminal_verify,
    child_for_edge,
    cost_aware_tree_block,
    expected_schedule_cost,
    latency_curve_from_anchors,
    local_tree_block,
    risk_bounded_prefix_cap,
)


PARENTS = [-1, 0, 0, 1, 1, 2, 3, 3]
TOKENS = [1, 2, 3, 4, 5, 6, 7]
DEPTHS = [0, 1, 1, 2, 2, 2, 3, 3]


def test_local_blocks_are_connected_and_ancestor_closed():
    assert local_tree_block(PARENTS, TOKENS, DEPTHS, 0, 4).nodes == (
        0, 1, 2, 3,
    )
    block = local_tree_block(PARENTS, TOKENS, DEPTHS, 3, 8)
    assert block.nodes == (3, 6, 7)
    assert block.parents == (-1, 0, 0)
    assert block.tokens == (6, 7)
    assert block.depths == (2, 3, 3)


def test_cascade_crosses_omitted_child_and_keeps_full_tree_acceptance():
    p = torch.zeros((len(PARENTS), 11), dtype=torch.float64)
    p[:, 10] = 1
    p[0].zero_(); p[0, 1] = 1
    p[1].zero_(); p[1, 3] = 1
    p[3].zero_(); p[3, 7] = 1
    result = cascade_terminal_verify(
        PARENTS, TOKENS, DEPTHS, p, 2,
        torch.Generator().manual_seed(7),
    )
    assert result.accepted_nodes == (1, 3, 7)
    assert result.accepted_tokens == (1, 3, 7)
    assert result.bonus_token == 10
    assert result.block_roots == (0, 3, 7)
    assert result.block_rows == (2, 2, 1)


def test_cascade_and_full_ancestral_have_same_output_distribution():
    p = torch.tensor([
        [.05, .50, .45, 0, 0, 0, 0, 0, 0],
        [.10, 0, 0, .50, .40, 0, 0, 0, 0],
        [.20, 0, 0, 0, 0, .80, 0, 0, 0],
        [.10, 0, 0, 0, 0, 0, .45, .35, .10],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float64)
    full = collections.Counter()
    cascade = collections.Counter()
    for seed in range(20_000):
        generator = torch.Generator().manual_seed(seed)
        nodes, output, bonus = tree_verify_ancestral_batched(
            PARENTS, TOKENS, p, generator,
        )
        full[tuple(output + [bonus])] += 1
        generator = torch.Generator().manual_seed(seed + 100_000)
        result = cascade_terminal_verify(
            PARENTS, TOKENS, DEPTHS, p, 2, generator,
        )
        cascade[result.accepted_tokens + (result.bonus_token,)] += 1
    keys = set(full) | set(cascade)
    total_variation = .5 * sum(
        abs(full[key] - cascade[key]) / 20_000 for key in keys
    )
    assert total_variation < .025


def test_schedule_cost_weights_later_blocks_by_target_prefix_mass():
    result = expected_schedule_cost(
        PARENTS, [0, 3, 7], [2, 2, 1],
        [1., .5, .45, .25, .2, .36, .1125, .0875],
        [0., 2., 3.5],
    )
    assert result == {
        "expected_target_calls": 1.3375,
        "expected_target_rows": 2.5875,
        "expected_target_latency_ms": 4.55,
    }


def test_child_lookup_distinguishes_original_exit_from_skipped_edge():
    assert child_for_edge(PARENTS, TOKENS, 1, 4) == 4
    assert child_for_edge(PARENTS, TOKENS, 1, 8) is None


def test_risk_bounded_cap_is_conditional_but_never_drops_fallback_edges():
    # Cap 5 leaves frontier mass .02 + .005 + .001 = .026; cap 4 is .066.
    log_mass = torch.tensor([1., .5, .2, .1, .04, .02, .005, .001]).log()
    assert risk_bounded_prefix_cap(PARENTS, log_mass, .05, 2) == 5
    assert risk_bounded_prefix_cap(PARENTS, log_mass, .01, 2) == 6


def test_cost_aware_dp_selects_branches_instead_of_a_fixed_prefix():
    masses = torch.tensor([1., .8, .01, .7, .05, .005, .3, .2])
    decision = cost_aware_tree_block(
        PARENTS, TOKENS, DEPTHS, masses.log(),
        [0.] + [10. + rows for rows in range(1, 9)],
    )
    # Keep the valuable 0->1->3 branch but skip earlier-id node 2 and its
    # subtree.  This cannot be represented by a canonical fixed row cap.
    assert decision.block.nodes == (0, 1, 3, 6, 7)
    assert decision.frontier_nodes == (2, 4)
    assert abs(decision.objective_ms - 15.67) < 1e-6


def test_cost_aware_dp_keeps_full_tree_when_an_extra_call_dominates():
    masses = torch.tensor([1., .8, .01, .7, .05, .005, .3, .2])
    decision = cost_aware_tree_block(
        PARENTS, TOKENS, DEPTHS, masses.log(), [0.] + [10.] * 8,
    )
    assert decision.block.nodes == tuple(range(8))
    assert decision.expected_rescue_latency_ms == 0.


def test_latency_curve_interpolates_measured_row_anchors():
    assert latency_curve_from_anchors(5, ((2, 10.), (5, 16.))) == (
        0., 10., 10., 12., 14., 16.,
    )
