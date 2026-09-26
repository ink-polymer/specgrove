import sys
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_dp_paper_study import target_ar,cases_for
from paper_dp_allocators import allocator
from gbv_experiments.continuous_tree_block_decode import _allocate_global_tree_budgets
from summarize_dp_paper_study import paired_cluster_ratios,paired_ar_ratio,bootstrap_difference
import summarize_dp_paper_study


class Engine:
    device='cpu'
    def sync(self):pass
    def cache_factory(self):return object()
    def target_forward(self,ids,cache,**kwargs):
        logits=torch.tensor([0.8,0.2],dtype=torch.bfloat16).expand(ids.shape[0],ids.shape[1],2)
        return SimpleNamespace(logits=logits)


def test_ar_posterior_dtype_and_repeatability():
    inputs=[torch.tensor([[0,1]]),torch.tensor([[1]])]
    tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=1)
    for temperature in [0,1]:
        a,x=target_ar(Engine(),tokenizer,inputs,[17,29],[1],8,temperature)
        b,y=target_ar(Engine(),tokenizer,inputs,[17,29],[1],8,temperature)
        assert x==y and a['posterior_dtype']=='float32'
        assert a['output_tokens']==sum(map(len,x)) and all(0<len(o)<=8 for o in x)


def test_allocators_same_feasible_set_and_dp_proxy_dominance():
    states=[SimpleNamespace(request_id=i,generated=[0]*i,age=i) for i in range(4)]
    utilities=[(2.0,3.0+i/4,5.0+i/2) for i in range(4)]
    options=(11,23,45)
    kwargs=dict(row_budget=96,fixed_row_equivalent=32.0,criticality_weight=1.0,max_new_tokens=64)
    dp=_allocate_global_tree_budgets(states,options,utilities,**kwargs)
    def score(a):
        w=[1+(3-i)/64+i/3 for i in range(4)]
        return sum(weight*u[options.index(b)] for weight,u,b in zip(w,utilities,a))/(32+sum(b+1 for b in a))
    for name in ['equal','greedy','random']:
        trace=[];a=allocator(name,trace)(states,options,utilities,**kwargs)
        assert all(b in options for b in a) and sum(b+1 for b in a)<=96
        assert score(dp)>=score(a)-1e-12 and len(trace)==1


def test_contract_counts_unique_and_quality_separate():
    assert len(cases_for('main'))==60 and len(cases_for('ablation'))==18
    quality=cases_for('quality_math')
    assert len(quality)==10 and all(c['repeats']==1 and c['requests']==128 for c in quality)


def test_performance_bootstrap_does_not_count_repeats_as_clusters():
    methods={'dp':[],'ddtree':[]}
    for repeat in range(3):
        for name,tokens in [('dp',50),('ddtree',40)]:
            methods[name].append(({'request_ids':['one']},{'summary':{'output_tokens':tokens,'wall_ms':100}}))
    report=paired_cluster_ratios(methods,100)['ddtree']
    assert report['independent_request_group_clusters']==1
    assert report['cluster_bootstrap95_interval'] is None and report['dp_ratio']==1.25


def test_quality_noninferiority_bootstrap_is_prompt_clustered():
    lower,lo,hi=bootstrap_difference([-.01]*128,100)
    assert abs(lower+.01)<1e-12 and abs(lo-hi)<1e-12


def test_ar_speedup_uses_total_tokens_and_exactly_matched_runs():
    ar=[];method=[]
    for repeat,ar_tokens,tokens,wall in [(0,20,40,100),(1,40,20,200)]:
        record={'request_ids':['one'],'prompt_sha256':['hash'],'request_seeds':[17]}
        ar.append((record,{'repeat':repeat,'summary':{'output_tokens':ar_tokens,'wall_ms':100}}))
        method.append((record,{'repeat':repeat,'summary':{'output_tokens':tokens,'wall_ms':wall}}))
    report=paired_ar_ratio(method,ar,100)
    assert abs(report['speedup_vs_ar']-2/3)<1e-12
    assert report['ar_tokens_per_second']==300
    assert report['cluster_bootstrap95_interval'] is None
    wrong=[({'request_ids':['one'],'prompt_sha256':['hash'],'request_seeds':[29]},run) for _record,run in ar]
    assert paired_ar_ratio(method,wrong,100)['speedup_vs_ar'] is None
    assert paired_ar_ratio(method,ar[:1],100)['speedup_vs_ar'] is None


def test_summary_reuses_only_matched_main_ar_and_splits_temperature_budget(tmp_path,monkeypatch):
    case={'dataset':'gsm8k','temperature':1,'budget':193,'concurrency':4,
          'max_new_tokens':256,'prompt_tokens':None}
    record={'identity':{'case':case},'request_ids':['one'],'prompt_sha256':['hash'],
            'request_seeds':[17],'runs':[{'method':'ar','repeat':0,
             'summary':{'output_tokens':20,'wall_ms':100}}]}
    for phase,tokens in [('main',40),('ablation',30)]:
        path=tmp_path/phase/'qwen3_4b'/'groups'/'one.json';path.parent.mkdir(parents=True)
        runs=record['runs'] if phase=='main' else []
        data=dict(record,runs=runs+[{'method':'dp','repeat':0,'summary':{'output_tokens':tokens,'wall_ms':100}}])
        path.write_text(json.dumps(data))
        (path.parent.parent/'contract.json').write_text(json.dumps({'cases':[dict(case,requests=4,repeats=1)]}))
    monkeypatch.setattr(sys,'argv',['summarize','--directory',str(tmp_path)])
    summarize_dp_paper_study.main()
    result=json.loads((tmp_path/'RESULTS.json').read_text())
    cells={(row['cell'][0],row['method']):row for row in result['cells']}
    assert cells[('main','ar')]['ar_comparison']['speedup_vs_ar']==1
    assert cells[('main','dp')]['ar_comparison']['speedup_vs_ar']==2
    assert abs(cells[('ablation','dp')]['ar_comparison']['speedup_vs_ar']-1.5)<1e-12
    assert cells[('ablation','dp')]['ar_reference_source']=='matched main-stage reuse'
    assert all(row['cell_complete'] for row in cells.values())
    assert (tmp_path/'RESULTS_T1_R193.md').exists()


def test_singleton_gate_metrics_do_not_infer_options_from_equal_returned_budgets():
    def record(n,budget,allocation,tier=False,limit=None):
        return ({},{'decode_settings':{'global_tree_budget_options':[11,23,45],
            'global_tree_require_tier_fit':tier,'global_tree_expansion_active_limit':limit},
            'allocator_trace':[{'requests':n,'allocation':allocation}]})
    stats=summarize_dp_paper_study.allocation_gate_stats
    assert stats([record(4,96,[11]*4)],96)['uncertified_waves']==1
    assert stats([record(4,96,[23,11,11,11])],96)['certified_multiple_feasible_allocation_waves']==1
    assert stats([record(8,96,[11]*8)],96)['certified_single_feasible_allocation_waves']==1
    assert stats([record(9,384,[11]*9,limit=8)],384)['certified_single_feasible_allocation_waves']==1
    assert stats([record(5,96,[11]*5,tier=True)],96)['certified_single_feasible_allocation_waves']==1
