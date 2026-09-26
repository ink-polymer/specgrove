"""Same-decoder scheduling controls; never consume a request's Target RNG."""
from contextlib import contextmanager
import hashlib
import random

from gbv_experiments import continuous_tree_block_decode as decoder


def weights(states, criticality, limit):
    leading = max(len(s.generated) for s in states)
    age = max(1, max(s.age for s in states))
    return [1 + criticality * ((leading-len(s.generated))/limit + s.age/age) for s in states]


def allocator(policy, trace):
    def select(states, options, utilities, *, row_budget, fixed_row_equivalent,
               criticality_weight, max_new_tokens, row_cost_curve=None):
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
        else:
            raise ValueError(policy)
        answer=tuple(options[k] for k in chosen)
        trace.append({'requests':n,'rows':sum(b+1 for b in answer),'allocation':answer,
                      'weighted_proxy_ratio':score(chosen)})
        return answer
    return select


@contextmanager
def control(name, trace):
    original=decoder._allocate_global_tree_budgets
    admit=decoder._admit_global_proposal_states
    if name in {'equal','greedy','random'}:
        decoder._allocate_global_tree_budgets=allocator(name,trace)
    else:
        def recording(*args,**kwargs):
            answer=original(*args,**kwargs)
            trace.append({'requests':len(args[0]),'rows':sum(b+1 for b in answer),'allocation':answer})
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
