"""Summarize only completed groups; grade math separately after GPU runs."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import random
import statistics


def bootstrap_difference(values,draws=10000):
    rng=random.Random(20260916);n=len(values)
    distribution=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(draws))
    return distribution[int(.05*(draws-1))],distribution[int(.025*(draws-1))],distribution[int(.975*(draws-1))]


def paired_cluster_ratios(methods,draws=2000):
    if 'dp' not in methods:return {}
    clusters=defaultdict(lambda:defaultdict(lambda:[0.0,0.0]))
    for name,records in methods.items():
        for record,run in records:
            cluster=tuple(map(str,record['request_ids']))
            clusters[cluster][name][0]+=run['summary']['output_tokens']
            clusters[cluster][name][1]+=run['summary']['wall_ms']
    reports={}
    for name in methods:
        if name=='dp':continue
        paired=[v for v in clusters.values() if 'dp' in v and name in v]
        def ratio(values):
            dt=sum(v['dp'][0] for v in values);dw=sum(v['dp'][1] for v in values)
            bt=sum(v[name][0] for v in values);bw=sum(v[name][1] for v in values)
            return (dt/dw)/(bt/bw)
        value=ratio(paired);n=len(paired);interval=None
        if n>=2:
            rng=random.Random(20260916)
            distribution=sorted(ratio([paired[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
            interval=[distribution[int(.025*(draws-1))],distribution[int(.975*(draws-1))]]
        reports[name]={'dp_ratio':value,'independent_request_group_clusters':n,
            'cluster_bootstrap95_interval':interval,'few_clusters_warning':n<8,
            'scope':'resample request-identity groups, retaining paired methods/all seeds/repeats; timing repeats are not independent clusters',
            'multiple_comparison_significance':'not certified; Holm-adjusted testing requires the confirmation study'}
    return reports


def paired_ar_ratio(records,ar_records,draws=2000):
    """Pair identical prompts/seeds/repeats; report only method versus AR."""
    def identity(record,run):
        return (tuple(map(str,record['request_ids'])),
                tuple(record.get('prompt_sha256',[])),
                tuple(record.get('request_seeds',[])),run['repeat'])
    references={}
    for record,run in ar_records:
        key=identity(record,run)
        if key in references:raise RuntimeError('Ambiguous duplicate AR reference')
        references[key]=(record,run)
    clusters=defaultdict(lambda:[0.0,0.0,0.0,0.0]);paired=0
    for record,run in records:
        baseline=references.get(identity(record,run))
        if baseline is None:continue
        paired+=1;ar=baseline[1]['summary'];method=run['summary']
        values=clusters[tuple(map(str,record['request_ids']))]
        for index,value in enumerate([method['output_tokens'],method['wall_ms'],ar['output_tokens'],ar['wall_ms']]):
            values[index]+=value
    complete=paired==len(records) and paired>0
    report={'reference':'ar','paired_executions':paired,'method_executions':len(records),
        'all_method_executions_paired':complete,'independent_request_group_clusters':len(clusters),
        'few_clusters_warning':len(clusters)<8,'speedup_vs_ar':None,
        'ar_tokens_per_second':None,'cluster_bootstrap95_interval':None,
        'scope':'ratio of paired total-token/total-wall throughput; resample request-identity groups, retaining seeds/repeats',
        'multiple_comparison_significance':'not certified; confirmation study required'}
    if not complete:return report
    values=list(clusters.values())
    def totals(selected):return [sum(v[i] for v in selected) for i in range(4)]
    def ratio(selected):
        mt,mw,at,aw=totals(selected)
        if min(mt,mw,at,aw)<=0:raise RuntimeError('Nonpositive tokens/time cannot define AR speedup')
        return (mt/mw)/(at/aw)
    report['speedup_vs_ar']=ratio(values)
    _mt,_mw,at,aw=totals(values);report['ar_tokens_per_second']=1000*at/aw
    if len(values)>=2:
        rng=random.Random(20260916)
        distribution=sorted(ratio([values[rng.randrange(len(values))] for _ in values]) for _ in range(draws))
        report['cluster_bootstrap95_interval']=[distribution[int(.025*(draws-1))],distribution[int(.975*(draws-1))]]
    return report


def allocation_gate_stats(records,budget):
    """Conservative certificates only: active cohort size itself is not logged."""
    total=single=multiple=unknown=0
    for _record,run in records:
        settings=run.get('decode_settings') or {};options=settings.get('global_tree_budget_options')
        if not options:continue
        options=sorted(options);limit=settings.get('global_tree_expansion_active_limit')
        for trace in run.get('allocator_trace',[]):
            total+=1;n=trace['requests'];minimum=n*(options[0]+1)
            if minimum>budget:raise RuntimeError('Logged admitted minimum exceeds row budget')
            tier=[b for b in options if n*(b+1)<=budget] if settings.get('global_tree_require_tier_fit') else options
            if not tier:raise RuntimeError('No feasible logged minimum tier')
            impossible_upgrade=len(options)==1 or (len(options)>1 and minimum+options[1]-options[0]>budget)
            if len(tier)==1 or impossible_upgrade or (limit is not None and n>limit):single+=1
            elif any(b>options[0] for b in trace['allocation']):multiple+=1
            else:unknown+=1
    return {'logged_allocation_waves':total,'certified_single_feasible_allocation_waves':single,
        'certified_multiple_feasible_allocation_waves':multiple,'uncertified_waves':unknown,
        'certified_no_choice_fraction_lower_bound':single/total if total else None,
        'scope':'uses tier gate, admission-count lower bound on active count, minimum upgrade feasibility; same returned budget alone does not certify no choice'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('/root/autodl-tmp/data/sampling-t1-formal-16c0e91'))
    args=p.parse_args();groups=defaultdict(lambda:defaultdict(list));quality=[];statuses=[];expected=defaultdict(int)
    for phase in ['pilot','main','budget','ablation','temperature0','long_output','context','quality_math']:
        for model in ['qwen3_4b','qwen3_8b']:
            directory=args.directory/phase/model
            statuses.append({'phase':phase,'model':model,'complete':(directory/'complete.json').exists(),
                             'failed':(directory/'failure.json').exists()})
            contract=directory/'contract.json'
            if contract.exists():
                for case in json.loads(contract.read_text())['cases']:
                    key=(phase,model,case['dataset'],case['temperature'],case['budget'],case['concurrency'],case['max_new_tokens'],case['prompt_tokens'])
                    expected[key]+=((case['requests']+case['concurrency']-1)//case['concurrency'])*case['repeats']
            for path in sorted((directory/'groups').glob('*.json')):
                record=json.loads(path.read_text());case=record['identity']['case']
                key=(phase,model,case['dataset'],case['temperature'],case['budget'],case['concurrency'],case['max_new_tokens'],case['prompt_tokens'])
                for run in record['runs']:groups[key][run['method']].append((record,run))
                if phase=='quality_math':quality.append((model,record))
    header=['# AR-normalized H20 batch study: completed executions','',
        'All speedups use matched autoregressive (AR) throughput as denominator; AR=1.00x. Rates use total tokens / total wall time, not mean per-run ratios.',
        'Incomplete stages or missing matched AR are pending, never zero-valued results. Ablations reuse matched main-stage AR on the same model/host/config; they are not within-ablation interleaved AR timings.',
        'No native-serving or strict-lossless certification is implied. Bootstrap intervals are descriptive and may have few independent prompt groups.','',
        '|Phase|Model|Dataset|T|R|C|Output cap|Input control|Method|Cell status|Executions / planned|Tokens|Total ms|tok/s|AR tok/s|Speedup / AR|95% cluster interval|AR source|',
        '|---|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|']
    table=list(header);separate_tables={};machine=[]
    for key,methods in sorted(groups.items(),key=lambda item:repr(item[0])):
        ar_records=methods.get('ar',[]);ar_source='same cell'
        if key[0]=='ablation' and not ar_records:
            ar_records=groups.get(('main',*key[1:]),{}).get('ar',[])
            ar_source='matched main-stage reuse'
        for name,rows in sorted(methods.items()):
            comparison=paired_ar_ratio(rows,ar_records)
            planned=expected.get(key)
            if planned is not None and len(rows)>planned:raise RuntimeError('Measured execution count exceeds preregistered cell')
            complete=planned is not None and len(rows)==planned and comparison['all_method_executions_paired']
            cell_status='complete' if complete else 'partial' if planned is not None else 'unregistered snapshot'
            tokens=sum(run['summary']['output_tokens'] for _record,run in rows)
            wall=sum(run['summary']['wall_ms'] for _record,run in rows)
            rate=1000*tokens/wall
            baseline=comparison['ar_tokens_per_second'];speedup=comparison['speedup_vs_ar']
            interval=comparison['cluster_bootstrap95_interval']
            clusters=comparison['independent_request_group_clusters']
            interval_text=(f'[{interval[0]:.4f}, {interval[1]:.4f}]; n={clusters}'+(' (few)' if clusters<8 else '')) if interval is not None else ('pending AR' if speedup is None else f'insufficient clusters; n={clusters}')
            line='|'+ '|'.join(map(str,[*key,name,cell_status,f'{len(rows)}/{planned if planned is not None else "?"}',tokens,round(wall,3),round(rate,3),
                round(baseline,3) if baseline is not None else 'pending',
                f'{speedup:.4f}x' if speedup is not None else 'pending',
                interval_text,ar_source]))+'|'
            table.append(line);separate_tables.setdefault((key[3],key[4]),list(header)).append(line)
            machine.append({'cell':key,'method':name,'executions':len(rows),'output_tokens':tokens,'wall_ms':wall,
                            'expected_executions':planned,'cell_complete':complete,'cell_status':cell_status,
                            'tokens_per_second':rate,'ar_reference_source':ar_source,'ar_comparison':comparison,
                            'allocation_gate_diagnostics':allocation_gate_stats(rows,key[4])})
    root=args.directory
    (root/'RESULTS.md').write_text('\n'.join(table)+'\n')
    for (temperature,budget),lines in separate_tables.items():
        (root/f'RESULTS_T{temperature}_R{budget}.md').write_text('\n'.join(lines)+'\n')
    (root/'RESULTS.json').write_text(json.dumps({'statuses':statuses,'cells':machine,'speedup_reference':'matched autoregressive',
        'speedup_definition':'(sum_method_tokens/sum_method_wall)/(sum_paired_ar_tokens/sum_paired_ar_wall)',
        'formal_complete':False,
        'missing_prerequisites':['strict BF16 AR certificate','official D-cut/Bastion matched comparison','native dynamic-serving study','safe code quality scoring','larger independent batch groups']},indent=2)+'\n')
    if not quality:return
    from math_verify import LatexExtractionConfig,parse,verify
    answers={}
    for dataset in ['gsm8k','math500']:
        source=[json.loads(line) for line in (args.data_dir/f'{dataset}.jsonl').read_text().split('\n') if line.strip()]
        for row in source:
            answer=row['evaluation']['answer'] if 'evaluation' in row else None
            if answer is None and 'evaluation_ref' in row:
                reference=args.data_dir/row['evaluation_ref']
                answer=json.loads(reference.read_text())['answer']
            if answer is None:raise RuntimeError('Unknown prepared gold-answer schema; never silently score as wrong')
            gold=parse('$'+str(answer)+'$',extraction_config=[LatexExtractionConfig()])
            if not gold or not verify(gold,gold):raise RuntimeError('gold parser self-test failed')
            answers[(dataset,str(row['source_id']))]=gold
    scores=[]
    for model,record in quality:
        dataset=record['identity']['case']['dataset']
        for run in record['runs']:
            for source_id,text in zip(record['request_ids'],run['texts']):
                prediction=parse(text.rsplit('</think>',1)[-1])
                passed=bool(verify(answers[(dataset,str(source_id))],prediction))
                scores.append({'model':model,'dataset':dataset,'source_id':str(source_id),
                               'seed':record['identity']['case']['seed'],'method':run['method'],'passed':passed})
    by_prompt=defaultdict(lambda:defaultdict(list))
    for row in scores:by_prompt[(row['model'],row['dataset'],row['source_id'])][row['method']].append(int(row['passed']))
    reports=[]
    for model in ['qwen3_4b','qwen3_8b']:
        for dataset in ['gsm8k','math500']:
            selected=[v for (m,d,_s),v in by_prompt.items() if m==model and d==dataset]
            if not selected:continue
            if not all(set(v)=={'ar','dflash','ddtree','dp'} and all(len(x)==5 for x in v.values()) for v in selected):
                raise RuntimeError('quality cell incomplete: require five seeds per prompt/method')
            differences=[statistics.mean(v['dp'])-statistics.mean(v['ar']) for v in selected]
            lower,lo,hi=bootstrap_difference(differences)
            reports.append({'model':model,'dataset':dataset,'independent_prompt_clusters':len(selected),
                'method_accuracies':{name:statistics.mean(statistics.mean(v[name]) for v in selected) for name in selected[0]},
                'dp_minus_ar':statistics.mean(differences),'one_sided95_lower':lower,'two_sided95_interval':[lo,hi],
                'noninferiority_margin':0.02,'noninferiority_supported':lower>=-.02,
                'not_strict_distribution_certificate':True})
    (root/'MATH_QUALITY.json').write_text(json.dumps({'grader':'math_verify','version':importlib.metadata.version('math-verify'),
        'reports':reports,'raw_scores':scores,'scope':'task noninferiority only; existing previously used subsets; not strict losslessness'},indent=2)+'\n')


if __name__=='__main__':main()
