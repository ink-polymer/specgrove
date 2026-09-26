"""Exact finite-state sequence-law tests, not real-model numeric certification."""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gbv_experiments.continuous_tree_block_decode import (
    _admit_global_proposal_states, _allocate_global_tree_budgets,
    _draft_tree_expected_tokens, _globally_feasible_budget_options,
)
from gbv_experiments.sampling import tree_verify_ancestral_batched
from gbv_experiments.tree import probability_tree


def target_p0(request, prefix):
    if not prefix:
        return Fraction(1 if request == 0 else 3, 4)
    # Context-dependent posterior, including exact zero/one support.
    index = (request + len(prefix) + sum(prefix[-2:])) % 5
    return Fraction(index, 4)


def node_path(parents, tokens, node):
    result = []
    while node:
        result.append(tokens[node - 1]); node = parents[node]
    return tuple(reversed(result))


@lru_cache(None)
def round_kernel(request, prefix, parents, tokens, remaining, eos):
    """Integrate every sampled row, including unused/off-path/after-EOS rows."""
    ps = [target_p0(request, prefix + node_path(parents, tokens, node))
          for node in range(len(parents))]
    values = torch.tensor([[float(p), float(1 - p)] for p in ps], dtype=torch.float64)
    mass_by_emission = defaultdict(Fraction)
    for posterior in itertools.product((0, 1), repeat=len(parents)):
        mass = Fraction(1)
        for bit, p0 in zip(posterior, ps):
            mass *= p0 if bit == 0 else 1 - p0
        if not mass:
            continue
        with patch("gbv_experiments.sampling.sample",
                   lambda *_a, p=posterior: torch.tensor(p)):
            _, accepted, bonus = tree_verify_ancestral_batched(parents, tokens, values)
        emitted = tuple((accepted + [bonus])[:remaining])
        if eos is not None and eos in emitted:
            emitted = emitted[:emitted.index(eos) + 1]
        mass_by_emission[emitted] += mass
    assert sum(mass_by_emission.values()) == 1
    return tuple(mass_by_emission.items())


def done(prefix, limit, eos):
    return len(prefix) >= limit or (eos is not None and eos in prefix)


def ar_law(request, limit, eos):
    law = {}
    def visit(prefix, mass):
        if done(prefix, limit, eos):
            law[prefix] = mass; return
        p0 = target_p0(request, prefix)
        for bit, probability in ((0, p0), (1, 1 - p0)):
            if probability:
                visit(prefix + (bit,), mass * probability)
    visit((), Fraction(1))
    return law


def scheduled_joint_law(row_budget, limit, eos, criticality):
    curve = ((2, 1.), (4, 1.1), (6, 1.2), (8, 1.6))
    options = (1, 2, 3)

    @lru_cache(None)
    def advance(prefixes, ages):
        if all(done(p, limit, eos) for p in prefixes):
            return ((prefixes, Fraction(1)),)
        active = [SimpleNamespace(request_id=i, generated=list(p), age=ages[i])
                  for i, p in enumerate(prefixes) if not done(p, limit, eos)]
        admitted = _admit_global_proposal_states(active, row_budget=row_budget, minimum_tree_budget=1)
        tiers = _globally_feasible_budget_options(options, request_count=len(admitted),
            row_budget=row_budget, require_tier_fit=True)
        choices, utilities = [], []
        for state in admitted:
            # Canonical Draft-only trees vary with past prefix; never read
            # current sampled posterior to select budgets or admit requests.
            flip = sum(state.generated) % 2
            q = torch.tensor([[.6, .4], [.7, .3], [.55, .45]], dtype=torch.float64)
            if flip:
                q = q.flip(-1)
            trees = [probability_tree(q, b) for b in tiers]
            choices.append(dict(zip(tiers, trees)))
            utilities.append([_draft_tree_expected_tokens(t) for t in trees])
        allocation = _allocate_global_tree_budgets(admitted, tiers, utilities,
            row_budget=row_budget, fixed_row_equivalent=None,
            criticality_weight=criticality, max_new_tokens=limit, row_cost_curve=curve)
        assert sum(b + 1 for b in allocation) <= row_budget
        kernels = []
        admitted_ids = set()
        for state, b, trees in zip(admitted, allocation, choices):
            i = state.request_id; admitted_ids.add(i); tree = trees[b]
            kernels.append((i, round_kernel(i, prefixes[i], tuple(tree.parents),
                tuple(tree.tokens), limit - len(prefixes[i]), eos)))
        next_ages = tuple(0 if i in admitted_ids else ages[i] + int(not done(p, limit, eos))
                          for i, p in enumerate(prefixes))
        law = defaultdict(Fraction)
        for joint in itertools.product(*(k for _, k in kernels)):
            new = list(prefixes); mass = Fraction(1)
            for (i, _kernel), (emitted, probability) in zip(kernels, joint):
                new[i] += emitted; mass *= probability
            for terminal, conditional in advance(tuple(new), next_ages):
                law[terminal] += mass * conditional
        assert sum(law.values()) == 1
        return tuple(law.items())

    # Real decoder first emits independent Target anchors, then schedules.
    initial = defaultdict(Fraction)
    for anchors in itertools.product((0, 1), repeat=2):
        mass = Fraction(1)
        for i, bit in enumerate(anchors):
            p0 = target_p0(i, ())
            mass *= p0 if bit == 0 else 1 - p0
        for terminal, conditional in advance(tuple((bit,) for bit in anchors), (0, 0)):
            initial[terminal] += mass * conditional
    return dict(initial)


@pytest.mark.parametrize("row_budget", (2, 4, 6, 8))
@pytest.mark.parametrize("limit", (3, 4))
@pytest.mark.parametrize("eos", (None, 1))
@pytest.mark.parametrize("criticality", (0., 1.))
def test_full_joint_sequence_law_exact_under_causal_admission_and_budgeting(row_budget, limit, eos, criticality):
    actual = scheduled_joint_law(row_budget, limit, eos, criticality)
    expected = {(a, b): p * q for a, p in ar_law(0, limit, eos).items()
                for b, q in ar_law(1, limit, eos).items()}
    assert actual == expected  # Exact Fraction equality, not empirical closeness.


def test_repeated_token_on_different_parents_is_legal_but_duplicate_sibling_rejected():
    parents, tokens = (-1, 0, 1, 0), (0, 0, 1)
    round_kernel(0, (), parents, tokens, 4, None)
    with pytest.raises(ValueError, match="repeat a child"):
        tree_verify_ancestral_batched([-1, 0, 0], [0, 0], torch.tensor([[.5, .5]] * 3))


@pytest.mark.parametrize("values", [
    torch.tensor([[float("nan"), 1.]]), torch.tensor([[-1., 2.]]),
    torch.tensor([[0., 0.]]), torch.tensor([[float("inf"), 1.]])])
def test_invalid_target_probability_rows_rejected(values):
    with pytest.raises(FloatingPointError):
        tree_verify_ancestral_batched([-1], [], values)
