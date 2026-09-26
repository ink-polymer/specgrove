"""Same-decoder scheduling controls; never consume a request's Target RNG."""
from contextlib import contextmanager
import heapq
import hashlib
import random
import time

from gbv_experiments import continuous_tree_block_decode as decoder


def weights(states, criticality, limit):
    leading = max(len(s.generated) for s in states)
    age = max(1, max(s.age for s in states))
    return [1 + criticality * ((leading-len(s.generated))/limit + s.age/age) for s in states]


def _tier_marginal(options, values, tier):
    """Draft-only marginal covered mass per added verification row."""
    added_rows = options[tier + 1] - options[tier]
    if added_rows <= 0:
        raise ValueError('Tree tiers must be strictly increasing')
    return (values[tier + 1] - values[tier]) / added_rows


def _tetris_style(options, utilities, row_budget):
    """Greedily select the best cross-request marginal DDTree tier.

    TETRIS ranks draft tokens globally by their draft-side acceptance
    surrogate.  The matched adaptation here keeps the same nested DDTree
    choices as SpecGrove and ranks each ancestor-closed tier increment by its
    added covered mass per row.  It deliberately ignores service weights and
    the measured non-linear cost curve.
    """
    n = len(utilities)
    current = [0] * n
    used = n * (options[0] + 1)
    heap = []
    for request, values in enumerate(utilities):
        if len(options) > 1:
            heapq.heappush(
                heap, (-_tier_marginal(options, values, 0), request, 0),
            )
    while heap:
        _negative_gain, request, tier = heapq.heappop(heap)
        if current[request] != tier:
            continue
        added_rows = options[tier + 1] - options[tier]
        if used + added_rows > row_budget:
            continue
        current[request] += 1
        used += added_rows
        next_tier = current[request]
        if next_tier + 1 < len(options):
            heapq.heappush(
                heap,
                (-_tier_marginal(options, utilities[request], next_tier),
                 request, next_tier),
            )
    return tuple(current)


def _echo_style(options, utilities, row_budget, thresholds):
    """Sparse-gated elastic allocation over matched nested DDTree tiers.

    Each tier boundary is one sparse gate.  Confident requests extend first;
    only after all gates have been visited can leftover capacity widen a
    truncated request.  This is an explicitly named ECHO-style adaptation,
    because DDTree tiers do not expose ECHO's original EAGLE depth/width
    operators.
    """
    if len(thresholds) != len(options) - 1:
        raise ValueError('One ECHO-style threshold is required per tier gate')
    n = len(utilities)
    current = [0] * n
    used = n * (options[0] + 1)

    # Priority 1: global extension at sparse, calibrated tier gates.
    for tier, threshold in enumerate(thresholds):
        candidates = []
        for request, values in enumerate(utilities):
            if current[request] != tier:
                continue
            confidence = _tier_marginal(options, values, tier)
            if confidence >= threshold:
                candidates.append((-confidence, request))
        for _negative_confidence, request in sorted(candidates):
            added_rows = options[tier + 1] - options[tier]
            if used + added_rows <= row_budget:
                current[request] += 1
                used += added_rows

    # Priority 2: opportunistic width/coverage expansion with any remainder.
    heap = []
    for request, values in enumerate(utilities):
        tier = current[request]
        if tier + 1 < len(options):
            heapq.heappush(
                heap, (-_tier_marginal(options, values, tier), request, tier),
            )
    while heap:
        _negative_gain, request, tier = heapq.heappop(heap)
        if current[request] != tier:
            continue
        added_rows = options[tier + 1] - options[tier]
        if used + added_rows > row_budget:
            continue
        current[request] += 1
        used += added_rows
        next_tier = current[request]
        if next_tier + 1 < len(options):
            heapq.heappush(
                heap,
                (-_tier_marginal(options, utilities[request], next_tier),
                 request, next_tier),
            )
    return tuple(current)


def allocator(policy, trace, *, echo_thresholds=None):
    def select(states, options, utilities, *, row_budget, fixed_row_equivalent,
               criticality_weight, max_new_tokens, row_cost_curve=None):
        started = time.perf_counter()
        w = weights(states, criticality_weight, max_new_tokens)
        cost = lambda rows: decoder._global_wave_cost(rows, fixed_row_equivalent, row_cost_curve)
        def score(indices):
            rows = sum(options[k]+1 for k in indices)
            return sum(weight*u[k] for weight,u,k in zip(w,utilities,indices))/cost(rows)
        n = len(states)
        if policy == 'equal':
            feasible = [(k,)*n for k,b in enumerate(options) if n*(b+1) <= row_budget]
            chosen = max(feasible, key=lambda a: (score(a), -sum(options[k]+1 for k in a)))
        elif policy == 'random':
            identity = [(s.request_id,len(s.generated),s.age) for s in states]
            seed = int(hashlib.sha256(repr(identity).encode()).hexdigest()[:16],16)
            rng = random.Random(seed)
            chosen=[]; used=0
            for i in range(n):
                feasible=[k for k,b in enumerate(options) if used+b+1+(n-i-1)*(options[0]+1)<=row_budget]
                k=rng.choice(feasible);chosen.append(k);used+=options[k]+1
            chosen=tuple(chosen)
        elif policy == 'greedy':
            current=[0]*n; visited=[tuple(current)]
            while True:
                used=sum(options[k]+1 for k in current)
                changes=[]
                for i,k in enumerate(current):
                    if k+1<len(options):
                        rows=options[k+1]-options[k]
                        if used+rows<=row_budget:
                            gain=w[i]*(utilities[i][k+1]-utilities[i][k])
                            changes.append((gain/rows,-i,i))
                if not changes:break
                i=max(changes)[2];current[i]+=1;visited.append(tuple(current))
            chosen=max(visited,key=lambda a:(score(a),-sum(options[k]+1 for k in a)))
        elif policy == 'tetris_style':
            chosen = _tetris_style(options, utilities, row_budget)
        elif policy == 'echo_style':
            thresholds = echo_thresholds
            if thresholds is None:
                # Deterministic untuned fallback for smoke tests only.  Formal
                # runs pass thresholds calibrated on disjoint warm-up prompts.
                thresholds = tuple(
                    sorted(_tier_marginal(options, values, tier)
                           for values in utilities)[len(utilities) // 2]
                    for tier in range(len(options) - 1)
                )
            else:
                # The decoder may expose only a prefix of the nested tier set
                # at high occupancy.  Reuse the corresponding calibrated gate
                # prefix so every scheduler sees the same feasible choices.
                thresholds = tuple(thresholds[:len(options) - 1])
            chosen = _echo_style(options, utilities, row_budget, thresholds)
        else:
            raise ValueError(policy)
        answer=tuple(options[k] for k in chosen)
        record = {'requests':n,'rows':sum(b+1 for b in answer),
                  'allocation':answer,'weighted_proxy_ratio':score(chosen),
                  'policy':policy,
                  'planning_ms':1000*(time.perf_counter()-started)}
        if policy == 'echo_style':
            record['gate_confidences'] = [
                [_tier_marginal(options, values, tier)
                 for tier in range(len(options)-1)]
                for values in utilities
            ]
            record['gate_thresholds'] = list(thresholds)
        trace.append(record)
        return answer
    return select


@contextmanager
def control(name, trace, *, echo_thresholds=None):
    original=decoder._allocate_global_tree_budgets
    admit=decoder._admit_global_proposal_states
    dense_policy = name.removesuffix('_dense') if name.endswith('_dense') else name
    if dense_policy in {'equal','greedy','random','tetris_style','echo_style'}:
        decoder._allocate_global_tree_budgets=allocator(
            dense_policy, trace, echo_thresholds=echo_thresholds,
        )
    else:
        def recording(*args,**kwargs):
            started = time.perf_counter()
            answer=original(*args,**kwargs)
            record = {'requests':len(args[0]),'rows':sum(b+1 for b in answer),
                      'allocation':answer,'policy':name,
                      'planning_ms':1000*(time.perf_counter()-started)}
            if name == 'dp_dense':
                states, options, utilities = args[:3]
                record['counterfactual_inputs'] = {
                    'options': list(options),
                    'utilities': [list(map(float, values)) for values in utilities],
                    'weights': weights(
                        states, kwargs['criticality_weight'],
                        kwargs['max_new_tokens']),
                    'row_budget': kwargs['row_budget'],
                    'fixed_row_equivalent': kwargs['fixed_row_equivalent'],
                    'row_cost_curve': kwargs.get('row_cost_curve'),
                }
            trace.append(record)
            return answer
        decoder._allocate_global_tree_budgets=recording
    if name=='age_admission':
        def age_first(states, *, row_budget, minimum_tree_budget):
            capacity=row_budget//(minimum_tree_budget+1)
            ids={s.request_id for s in sorted(states,key=lambda s:(-s.age,s.request_id))[:capacity]}
            return tuple(s for s in states if s.request_id in ids)
        decoder._admit_global_proposal_states=age_first
    try:yield
    finally:
        decoder._allocate_global_tree_budgets=original
        decoder._admit_global_proposal_states=admit
