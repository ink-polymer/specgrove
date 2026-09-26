from types import SimpleNamespace

from paper_dp_allocators import allocator


def states(n):
    return [SimpleNamespace(request_id=i, generated=[1], age=0) for i in range(n)]


def invoke(policy, utilities, *, rows=48, thresholds=None):
    trace = []
    select = allocator(policy, trace, echo_thresholds=thresholds)
    answer = select(
        states(len(utilities)), (3, 7, 11), utilities,
        row_budget=rows, fixed_row_equivalent=10.0,
        criticality_weight=0.0, max_new_tokens=64,
    )
    return answer, trace


def test_tetris_style_spends_optional_rows_on_highest_marginal_request():
    answer, trace = invoke(
        'tetris_style',
        ((1.0, 1.8, 2.0), (1.0, 1.4, 2.6)),
        rows=16,
    )
    assert answer == (7, 7)
    assert trace[0]['rows'] == 16
    assert trace[0]['planning_ms'] >= 0


def test_echo_style_prioritizes_sparse_gate_pass_before_opportunistic_fill():
    answer, trace = invoke(
        'echo_style',
        ((1.0, 1.8, 1.9), (1.0, 1.1, 2.3)),
        rows=16, thresholds=(0.15, 0.20),
    )
    assert answer == (7, 7)
    assert trace[0]['policy'] == 'echo_style'


def test_style_allocators_are_deterministic_and_capacity_safe():
    utilities = tuple((1.0, 1.2 + i / 100, 1.5 + i / 50)
                      for i in range(8))
    for policy in ('tetris_style', 'echo_style'):
        first, trace = invoke(policy, utilities, rows=96,
                              thresholds=(0.01, 0.01))
        second, _ = invoke(policy, utilities, rows=96,
                           thresholds=(0.01, 0.01))
        assert first == second
        assert sum(value + 1 for value in first) <= 96
        assert trace[0]['requests'] == 8
