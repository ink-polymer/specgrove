"""Rational finite-instance checks; not a formal proof assistant or GPU test."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import random
import sys
from types import SimpleNamespace


def exhaustive(options, utilities, weights, budget, cost):
    feasible = []
    for allocation in product(*options):
        rows = sum(b + 1 for b in allocation)
        if rows <= budget:
            reward = sum(w * u[a.index(b)] for a, u, w, b in
                         zip(options, utilities, weights, allocation))
            feasible.append((reward / cost(rows), rows, allocation, reward))
    return feasible


def rational_dp(options, utilities, weights, budget, cost):
    frontier = {0: (F(0), ())}
    for a, u, w in zip(options, utilities, weights):
        updated = {}
        for rows, (reward, allocation) in frontier.items():
            for b, value in zip(a, u):
                new_rows = rows + b + 1
                if new_rows <= budget:
                    candidate = (reward + w * value, allocation + (b,))
                    old = updated.get(new_rows)
                    if old is None or candidate[0] > old[0] or (
                            candidate[0] == old[0] and candidate[1] < old[1]):
                        updated[new_rows] = candidate
        frontier = updated
    return max((reward / cost(rows), -rows, tuple(-b for b in allocation), allocation)
               for rows, (reward, allocation) in frontier.items())


def check():
    rng = random.Random(20260916)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from gbv_experiments.continuous_tree_block_decode import _allocate_global_tree_budgets
    cases = runtime_cases = weight_checks = error_checks = 0
    for _ in range(1000):
        n = rng.randint(1, 5)
        options = [(1, 3, 5)] * n
        utilities = [tuple(sorted(F(rng.randint(4, 32), 4) for _ in range(3))) for _ in range(n)]
        weights = [F(rng.randint(1, 8), 4) for _ in range(n)]
        budget = rng.randint(2 * n, 6 * n)
        base_cost = F(rng.randint(1, 12), 4)
        cost = lambda rows, c=base_cost: c + rows
        candidates = exhaustive(options, utilities, weights, budget, cost)
        chosen = rational_dp(options, utilities, weights, budget, cost)
        optimum = max(item[0] for item in candidates)
        assert chosen[0] == optimum
        assert chosen[0] >= max(item[0] for item in exhaustive(options, utilities, weights, budget, cost))
        assert rational_dp(options, utilities, weights, budget + 1, cost)[0] >= optimum
        restricted = [a[:2] for a in options]
        assert chosen[0] >= rational_dp(restricted, [u[:2] for u in utilities], weights, budget, cost)[0]
        cases += 1

        # Revealed preference for the ratio objective (not node-count monotonicity).
        boosted = weights.copy(); boosted[0] += F(1, 2)
        new = rational_dp(options, utilities, boosted, budget, cost)
        old_a, new_a = chosen[3], new[3]
        old_s, new_s = sum(b + 1 for b in old_a), sum(b + 1 for b in new_a)
        assert utilities[0][options[0].index(new_a[0])] / cost(new_s) >= \
            utilities[0][options[0].index(old_a[0])] / cost(old_s)
        weight_checks += 1

        # Independent estimated objective with uniformly bounded score perturbations.
        eta = F(1, 4)
        perturbed = [(score + F(rng.randint(-4, 4), 16), score)
                     for score, _rows, _a, _reward in candidates]
        predicted_winner = max(perturbed, key=lambda item: item[0])
        assert predicted_winner[1] >= optimum - 2 * eta
        error_checks += 1

        # Frozen executable compared to rational optimum with its exact current weights.
        if runtime_cases < 500:
            states = [SimpleNamespace(generated=[0] * rng.randint(0, 8), age=rng.randint(0, 4))
                      for _ in range(n)]
            leading = max(len(s.generated) for s in states)
            age = max(1, max(s.age for s in states))
            actual_weights = [1 + F(leading - len(s.generated), 16) + F(s.age, age) for s in states]
            ideal = rational_dp(options, utilities, actual_weights, budget, cost)
            allocation = _allocate_global_tree_budgets(
                states, options[0], [[float(v) for v in u] for u in utilities],
                row_budget=budget, fixed_row_equivalent=float(base_cost),
                criticality_weight=1.0, max_new_tokens=16)
            rows = sum(b + 1 for b in allocation)
            value = sum(w * u[a.index(b)] for w, u, a, b in
                        zip(actual_weights, utilities, options, allocation)) / cost(rows)
            assert abs(float(value - ideal[0])) <= 1e-11
            runtime_cases += 1

    # Bundle-density greedy fails even with decreasing marginal density.
    options = [(11, 23, 45)] * 2
    utility = [(F(11, 5), F(17, 5), F(129, 25)), (F(11, 5), F(169, 50), F(169, 50))]
    optimum = rational_dp(options, utility, [F(1), F(1)], 58, lambda _s: F(1))
    assert optimum[3] == (45, 11) and optimum[0] == F(184, 25)
    assert sum(u[1] for u in utility) == F(339, 50) < optimum[0]

    # A feasible mixed large tree is excluded by tier-fit.
    assert sum(b + 1 for b in (45, 11, 11, 11)) == 82 <= 96
    assert 4 * 46 > 96
    for budget, small, large in [(96, 8, 2), (193, 16, 4), (384, 32, 8)]:
        assert budget // 12 == small and budget // 46 == large

    # Executable tolerance can keep a mathematically worse reward at the same row count.
    states = [SimpleNamespace(generated=[], age=0)] * 2
    almost_equal = [[1.0, 2.0 + 5e-13], [1.0, 2.0]]
    a = _allocate_global_tree_budgets(states, (1, 2), almost_equal,
        row_budget=5, fixed_row_equivalent=1000.0, criticality_weight=0.0, max_new_tokens=1)
    assert a == (1, 2)  # (2, 1) has slightly larger reward but within the tie tolerance.
    assert almost_equal[0][1] + almost_equal[1][0] > almost_equal[0][0] + almost_equal[1][1]
    return {'status': 'passed', 'rational_dp_vs_exhaustive_instances': cases,
            'frozen_allocator_vs_rational_instances': runtime_cases,
            'weight_sensitivity_checks': weight_checks, 'bounded_score_error_checks': error_checks,
            'counterexamples': ['bundle-density greedy', 'tier-fit excludes feasible mix',
                                'floating tie tolerance is not exact real optimality'],
            'scope': 'finite CPU checks; written proofs are in DP_THEORY.md; not machine-formal or GPU evidence'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
