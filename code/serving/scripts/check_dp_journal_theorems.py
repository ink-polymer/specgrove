"""Independent exact-rational finite checks; not a proof assistant or GPU test."""
import argparse
from collections import Counter
from fractions import Fraction as F
from itertools import product
import json
import heapq
from pathlib import Path
import random


def allocations(options,utilities,weights,budget,cost):
    result=[]
    for action in product(*options):
        rows=sum(b+1 for b in action)
        if rows<=budget:
            reward=sum(w*u[b] for w,u,b in zip(weights,utilities,action))
            result.append((action,rows,reward,cost(rows),reward/cost(rows)))
    return result


def frontier(options,utilities,weights,budget,tolerance=F(0),arithmetic_error=None):
    values={0:(F(0),())};comparisons=0
    for layer,(choices,u,w) in enumerate(zip(options,utilities,weights)):
        updated={}
        for rows,(value,prefix) in values.items():
            for b in choices:
                s=rows+b+1
                if s>budget:continue
                comparisons+=1
                score=value+w*u[b]
                if arithmetic_error:score+=arithmetic_error(layer,s,b)
                candidate=(score,prefix+(b,));old=updated.get(s)
                if old is None or score>old[0]+tolerance or (abs(score-old[0])<=tolerance and candidate[1]<old[1]):
                    updated[s]=candidate
        values=updated
    return values,comparisons


def censored_tree_yield(sequence,tree,cap,eos=1):
    prefix=();count=0
    for token in sequence:
        count+=1;prefix=prefix+(token,)
        if token==eos or count==cap or prefix not in tree:return count
    return count


def prefix_probability(prefix,kernel):
    value=F(1)
    for depth,token in enumerate(prefix):value*=kernel(depth,prefix[:depth])[token]
    return value


def run(seed=20260916,instances=500):
    rng=random.Random(seed);counts=Counter()
    for _ in range(instances):
        n=rng.randint(1,4);options=[tuple(sorted(rng.sample(range(1,7),3))) for _i in range(n)]
        utilities=[{b:F(rng.randint(4,40),4) for b in a} for a in options]
        weights=[F(rng.randint(4,12),4) for _i in range(n)]
        minimum=sum(a[0]+1 for a in options);budget=rng.randint(minimum,sum(a[-1]+1 for a in options))
        costs={s:F(rng.randint(1,40),4) for s in range(budget+2)};cost=lambda s:costs[s]
        records=allocations(options,utilities,weights,budget,cost)
        states,_=frontier(options,utilities,weights,budget)
        optimum=max(a[4] for a in records);cmin=min(a[3] for a in records)
        assert max(v[0]/cost(s) for s,v in states.items())==optimum;counts['bellman_vs_exhaustive']+=1
        assert set(states)==set(a[1] for a in records)
        for s,(reward,_action) in states.items():assert reward==max(a[2] for a in records if a[1]==s)
        counts['exact_row_frontier']+=1
        larger,_=frontier(options,utilities,weights,budget+1)
        assert max(v[0]/cost(s) for s,v in larger.items())>=optimum;counts['budget_monotonicity']+=1
        chosen=rng.choice(records);rho=chosen[4]
        phi=max(a[2]-rho*a[3] for a in records)
        assert 0<=optimum-rho<=phi/cmin
        assert max(a[2]-optimum*a[3] for a in records)==0;counts['fractional_residual']+=1
        for nu in [F(-2),F(-1),F(0),F(1),F(2)]:
            separated=sum(max(w*u[b]-nu*(b+1) for b in choices) for choices,u,w in zip(options,utilities,weights))
            bound=separated+max(nu*s-rho*cost(s) for s in states)
            assert bound>=phi and optimum-rho<=bound/cmin
            counts['separated_price_certificate']+=1
        estimate=F(0);steps=0
        while True:
            selected=max(records,key=lambda a:a[2]-estimate*a[3]);residual=selected[2]-estimate*selected[3]
            if residual==0:break
            new=selected[4];assert estimate<new<=optimum;estimate=new;steps+=1
            assert steps<=len(records)
        assert estimate==optimum;counts['finite_fractional_iteration']+=1
        boosted=weights.copy();boosted[0]+=F(1,2)
        old=max(records,key=lambda a:a[4]);new=max(allocations(options,utilities,boosted,budget,cost),key=lambda a:a[4])
        assert utilities[0][new[0][0]]/new[3]>=utilities[0][old[0][0]]/old[3]
        unweighted=allocations(options,utilities,[F(1)]*n,budget,cost);plain=max(a[4] for a in unweighted)
        old_plain=next(a[4] for a in unweighted if a[0]==old[0])
        assert old_plain>=min(weights)/max(weights)*plain;counts['weight_and_unweighted_guarantee']+=1
        eta=F(1,4);perturbed=[(a[4]+F(rng.randint(-4,4),16),a) for a in records]
        predicted,selected=max(perturbed,key=lambda pair:pair[0])
        assert selected[4]>=optimum-2*eta;counts['absolute_prediction_regret']+=1
        others=[value for value,a in perturbed if a!=selected]
        if others and predicted-max(others)>2*eta:
            assert all(selected[4]>a[4] for a in records if a!=selected)
            counts['certified_decision_gap']+=1
        ev,ec=F(1,10),F(1,8);relative=[]
        for a in records:
            vhat=a[2]*(1+F(rng.randint(-10,10),100));chat=a[3]*(1+F(rng.randint(-8,8),64))
            relative.append((vhat/chat,a))
        selected=max(relative,key=lambda pair:pair[0])[1]
        factor=(1-ev)*(1-ec)/((1+ev)*(1+ec))
        assert selected[4]>=factor*optimum;counts['relative_prediction_guarantee']+=1
        h1,h2=F(1),F(5)
        a1=max(records,key=lambda a:a[2]/(h1+a[3]));a2=max(records,key=lambda a:a[2]/(h2+a[3]))
        assert a2[2]>=a1[2] and a2[3]>=a1[3];counts['fixed_overhead_comparative_statics']+=1
        tau=F(1,4);e=F(1,32)
        errors={(i,s,b):F(rng.randint(-1,1),32) for i in range(n) for s in range(budget+1) for b in options[i]}
        tolerant,comparisons=frontier(options,utilities,weights,budget,tau,lambda i,s,b:errors[(i,s,b)])
        by_action={a[0]:a for a in records}
        stored=max((by_action[action][4] for _value,action in tolerant.values()))
        assert stored>=optimum-(2*n*e+comparisons*tau)/cmin
        numeric_s,(_value,numeric_action)=max(tolerant.items(),key=lambda pair:pair[1][0]/cost(pair[0]))
        assert by_action[numeric_action][4]>=optimum-(4*n*e+comparisons*tau)/cmin
        counts['tolerance_and_arithmetic_regret']+=1

        # Dense nested budgets, sparse retained grid, monotone full-row cost.
        dense=[tuple(range(1,7))]*n;grid=[(1,3,6)]*n
        nested=[]
        for _i in range(n):
            value=F(1);u={}
            for b in range(1,7):value+=F(rng.randint(0,4),8);u[b]=value
            nested.append(u)
        monotone=lambda s:F(1)+s
        full=allocations(dense,nested,weights,budget,monotone)
        sparse=allocations(grid,nested,weights,budget,monotone)
        loss=sum(w*max(u[b]-u[max(a for a in g if a<=b)] for b in d) for w,u,g,d in zip(weights,nested,grid,dense))
        assert max(a[4] for a in full)-max(a[4] for a in sparse)<=loss/min(a[3] for a in full)
        counts['sparse_budget_grid_regret']+=1
        plain_full=allocations(dense,nested,[F(1)]*n,budget,monotone)
        plain_sparse=allocations(grid,nested,[F(1)]*n,budget,monotone)
        loss0=sum(max(u[b]-u[max(a for a in g if a<=b)] for b in d) for u,g,d in zip(nested,grid,dense))
        estimated=[]
        for a in sparse:
            vhat=a[2]*(1+F(rng.randint(-10,10),100));chat=a[3]*(1+F(rng.randint(-8,8),64))
            estimated.append((vhat/chat,a))
        solver_error=F(1,16);best_estimate=max(v for v,_a in estimated)
        selected=rng.choice([a for v,a in estimated if v>=best_estimate-solver_error])
        selected_plain=next(a[4] for a in plain_sparse if a[0]==selected[0])
        h0=F(1,8);overhead=F(rng.randint(0,8),64)
        ell=(1-ev)/(1+ec);upper=(1+ev)/(1-ec);chi=ell/upper*min(weights)/max(weights)
        bound=(chi*(max(a[4] for a in plain_full)-loss0/min(a[3] for a in plain_full))-solver_error/(upper*max(weights)))/(1+h0)
        assert selected_plain/(1+overhead)>=bound
        counts['composed_grid_prediction_weight_overhead_bound']+=1

    for _ in range(200):
        depth=3;tree={()}
        for d in range(1,depth+1):
            for prefix in product([0,1],repeat=d):
                if prefix[:-1] in tree and rng.randrange(2):tree.add(prefix)
        masses={prefix:F(rng.randint(1,7),8) for d in range(depth+1) for prefix in product([0,1],repeat=d)}
        p=lambda _d,prefix:(masses[prefix],1-masses[prefix])
        q=lambda _d,_prefix:(F(1,2),F(1,2))
        epsilon=max(abs(m-F(1,2)) for m in masses.values())
        for cap in range(1,5):
            expectations=[]
            for kernel in [p,q]:
                value=sum(prefix_probability(sequence,kernel)*censored_tree_yield(sequence,tree,cap) for sequence in product([0,1],repeat=depth+1))
                formula=sum(prefix_probability(prefix,kernel) for prefix in tree if len(prefix)<=cap-1 and 1 not in prefix)
                assert value==formula and 1<=value<=min(cap,depth+1)
                expectations.append(value);counts['eos_censored_expected_yield']+=1
            k=min(cap,depth+1);delta=sum(1-(1-epsilon)**d for d in range(1,k))
            assert abs(expectations[0]-expectations[1])<=delta;counts['kernel_to_yield_tv_bound']+=1
        sequences=list(product([0,1],repeat=depth+1))
        tv=sum(abs(prefix_probability(y,p)-prefix_probability(y,q)) for y in sequences)/2
        assert tv<=1-(1-epsilon)**(depth+1);counts['full_sequence_tv_bound']+=1

    for _ in range(200):
        probabilities=[F(1,4),F(1,2),F(1,4)]
        rewards=[[F(rng.randint(1,20),4) for _a in range(3)] for _s in range(3)]
        times=[[F(rng.randint(1,20),4) for _a in range(3)] for _s in range(3)]
        def score(action):return sum(p*rewards[s][a] for s,(p,a) in enumerate(zip(probabilities,action)))/sum(p*times[s][a] for s,(p,a) in enumerate(zip(probabilities,action)))
        optimum=max(score(action) for action in product(range(3),repeat=3))
        selected=tuple(max(range(3),key=lambda a:rewards[s][a]-optimum*times[s][a]) for s in range(3))
        assert score(selected)==optimum
        assert sum(p*max(rewards[s][a]-optimum*times[s][a] for a in range(3)) for s,p in enumerate(probabilities))==0
        counts['exogenous_state_average_ratio']+=1
    for _ in range(500):
        n=rng.randint(2,10);m=rng.randint(1,n-1)
        h=sorted([F(rng.randint(-10,30),4) for _i in range(n)],reverse=True)
        mean=sum(h)/n;variance=sum((v-mean)**2 for v in h)/n
        gain=sum(h[:m])-m*mean
        assert gain>=0 and gain**2>=variance*m*(n-m)/(n-1)**2
        assert (gain>0)==(variance>0)
        counts['heterogeneity_variance_gain_bound']+=1
    for _ in range(500):
        n=rng.randint(1,5);weights=[F(rng.randint(4,12),4) for _i in range(n)]
        options=[(-1,1,3)]*n
        utility=[{-1:F(0),1:F(rng.randint(4,20),4),3:F(rng.randint(20,40),4)} for _i in range(n)]
        budget=rng.randint(2,4*n);cost=lambda s:1+F(s*s,32)
        values,_=frontier(options,utility,weights,budget)
        reference=allocations(options,utility,weights,budget,cost)
        assert max(value[0]/cost(s) for s,value in values.items() if s>0)==max(a[4] for a in reference if a[1]>0)
        counts['joint_admission_finite_dp']+=1
        single=[(-1,1)]*n;values,_=frontier(single,utility,weights,budget)
        z=sorted([w*u[1] for w,u in zip(weights,utility)],reverse=True)
        formula=max(sum(z[:m])/cost(2*m) for m in range(1,min(n,budget//2)+1))
        assert formula==max(v[0]/cost(s) for s,v in values.items() if s>0)
        counts['single_tier_admission_sorting']+=1
    for _ in range(500):
        n=rng.randint(1,4);options=[tuple(range(1,7))]*n;weights=[F(rng.randint(4,12),4) for _i in range(n)]
        marginals=[sorted([F(rng.randint(0,8),8) for _b in range(5)],reverse=True) for _i in range(n)]
        utilities=[]
        for gains in marginals:
            value=F(rng.randint(4,16),4);u={1:value}
            for b,gain in enumerate(gains,start=2):value+=gain;u[b]=value
            utilities.append(u)
        budget=rng.randint(2*n,7*n);costs={s:F(rng.randint(1,40),4) for s in range(budget+1)}
        values,_=frontier(options,utilities,weights,budget)
        reference=max(v[0]/costs[s] for s,v in values.items())
        reward=sum(w*u[1] for w,u in zip(weights,utilities));best=reward/costs[2*n]
        queue=[(-w*g[0],i,0) for i,(w,g) in enumerate(zip(weights,marginals))];heapq.heapify(queue)
        for t in range(1,min(budget-2*n,5*n)+1):
            negative,i,offset=heapq.heappop(queue);reward-=negative
            best=max(best,reward/costs[2*n+t])
            if offset+1<len(marginals[i]):heapq.heappush(queue,(-weights[i]*marginals[i][offset+1],i,offset+1))
        assert best==reference;counts['integer_range_concave_heap_vs_dp']+=1
    return {'status':'passed','seed':seed,'random_optimization_instances':instances,'checks':dict(sorted(counts.items())),
            'scope':'Independent exact-Fraction finite arithmetic checks. Written theorem assumptions define proof scope; no proof-assistant or GPU certification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();report=run()
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
