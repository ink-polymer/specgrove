"""Memory-lifecycle-safe continuation, preserving immutable core and complete old cells."""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import torch
import run_dp_paper_single_wave as legacy_driver
from paper_single_wave_policy import methods_for_case, partition_cases, policy_contract, method_capacity
from paper_memory_lifecycle import reclaim_completed_decodes, complete_legacy_cells, completed_cell_key, POLICY_VERSION

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rerun_audited_global_tree_matrix_h20 import DD,DF,OURS,matched_specs
from benchmark_continuous_tree_block_e2e import prepared_prompts,summarize
from paper_dp_allocators import control
from gbv_experiments import continuous_tree_block_decode as decoder
from gbv_experiments.config import load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.sampling import probabilities,sample

CORE='3f132a41b659d1306cbfc3fbb348c0f89284d097d69851f0f73d8a6e4518587f'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temporary.replace(path)


def source_hashes():
    paths=list((ROOT/'src/gbv_experiments').glob('*.py'))+[Path(__file__),ROOT/'scripts/paper_dp_allocators.py',ROOT/'scripts/paper_single_wave_policy.py',ROOT/'scripts/run_dp_paper_study.py',ROOT/'scripts/run_dp_paper_single_wave.py',ROOT/'scripts/paper_memory_lifecycle.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@torch.inference_mode()
def target_ar(engine,tokenizer,prompts,seeds,stops,cap,temperature):
    """Eager cached padded AR with BF16 logits and FP32 posterior, no Draft."""
    n=len(prompts);lengths=[p.shape[1] for p in prompts];maximum=max(lengths)
    pad=tokenizer.pad_token_id
    if pad is None:pad=tokenizer.eos_token_id
    if pad is None:pad=stops[0]
    ids=torch.full((n,maximum),int(pad),device=engine.device,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,p in enumerate(prompts):ids[i,-lengths[i]:]=p[0];mask[i,-lengths[i]:]=1
    positions=mask.cumsum(-1).sub(1).clamp_min(0)
    rng=[torch.Generator(device=engine.device).manual_seed(s) for s in seeds]
    generated=[[] for _ in prompts];done=[False]*n;stopset=set(stops)
    engine.sync();started=time.perf_counter();cache=engine.cache_factory();calls=0
    output=engine.target_forward(ids,cache,hidden=False,positions=positions,mask=mask,last_only=True)
    while True:
        logits=output.logits[:,-1]
        for i in range(n):
            if done[i]:continue
            token=int(logits[i].argmax()) if temperature==0 else int(sample(probabilities(logits[i],temperature,torch.float32),rng[i]))
            generated[i].append(token);done[i]=token in stopset or len(generated[i])>=cap
        del output,logits
        if all(done):break
        active=[not d for d in done]
        ids=torch.tensor([[g[-1] if active[i] else int(pad)] for i,g in enumerate(generated)],device=engine.device)
        mask=torch.cat((mask,torch.tensor(active,device=engine.device,dtype=mask.dtype)[:,None]),dim=1)
        positions=torch.tensor([[lengths[i]+len(generated[i])-1 if active[i] else 0] for i in range(n)],device=engine.device)
        output=engine.target_forward(ids,cache,hidden=False,positions=positions,mask=mask,last_only=True);calls+=1
    engine.sync();wall=1000*(time.perf_counter()-started);tokens=sum(map(len,generated))
    return {'wall_ms':wall,'output_tokens':tokens,'tokens_per_second':1000*tokens/wall,
            'physical_target_calls':calls,'posterior_dtype':'float32','ar_layout':'padded_cached'},generated


def methods_for(case):
    return methods_for_case(case)


def cases_for(phase):
    cases=[]
    def add(dataset,cs,budgets,caps,seeds,requests,repeats,temperatures=(1,),prompt_tokens=None):
        for t in temperatures:
            for r in budgets:
                for c in cs:
                    for cap in caps:
                        for seed in seeds:
                            cases.append({'phase':phase,'dataset':dataset,'temperature':t,'budget':r,'concurrency':c,
                                'max_new_tokens':cap,'seed':seed,'requests':max(c,requests),'repeats':repeats,
                                'prompt_tokens':prompt_tokens})
    if phase=='pilot':
        add('gsm8k',[1,4,16,32],[193],[128,512],[17],0,1)
    elif phase=='main':
        for d in ['gsm8k','math500','humaneval','mbpp_sanitized']:
            add(d,[1,4,8,16,32],[193],[256],[17,29,43],32,3)
    elif phase=='budget':
        for d in ['gsm8k','math500']:
            add(d,[4,8,16,32],[96,384],[256],[17,29,43],32,3)
    elif phase=='ablation':
        for d in ['gsm8k','math500']:
            add(d,[4,8,32],[193],[256],[17,29,43],32,3)
    elif phase=='temperature0':
        for d in ['gsm8k','math500']:
            add(d,[1,8,32],[193],[256],[17,29],32,3,temperatures=(0,))
    elif phase=='long_output':
        add('gsm8k',[1,8,32],[193],[512,1024],[17,29,43],16,3)
    elif phase=='context':
        for length in [128,512,2048,8192]:
            add('gsm8k',[1,8,32],[193],[128],[17,29],16,3,prompt_tokens=length)
    elif phase=='quality_math':
        for d in ['gsm8k','math500']:
            add(d,[8],[193],[1024],[17,29,43,67,101],128,1)
    else:raise ValueError(phase)
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['qwen3_4b','qwen3_8b'],required=True)
    p.add_argument('--phase',choices=['pilot','main','budget','ablation','temperature0','long_output','context','quality_math'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--reuse-directory',type=Path,help='Original immutable phase/model results; reuse only exact verified completed groups')
    args=p.parse_args()
    assert hashlib.sha256((ROOT/'src/gbv_experiments/continuous_tree_block_decode.py').read_bytes()).hexdigest()==CORE
    args.output.mkdir(parents=True,exist_ok=True)
    original_contract=cases_for(args.phase)
    contract,excluded_cases=partition_cases(original_contract)
    hashes=source_hashes();legacy_hashes=legacy_driver.source_hashes()
    policy=policy_contract(original_contract)
    reusable_cells,incomplete_cells=complete_legacy_cells(args.reuse_directory)
    write(args.output/'contract.json',{'model':args.model,'phase':args.phase,'cases':contract,'source_hashes':hashes,
        'methods':list(dict.fromkeys(m for case in contract for m in methods_for(case))),
        'excluded_cases':excluded_cases,'single_wave_policy':policy,
        'legacy_reuse_directory':str(args.reuse_directory) if args.reuse_directory else None,
        'legacy_expected_source_hashes':legacy_hashes,
        'kind':'memory_lifecycle_safe_single_target_wave_continuation',
        'memory_lifecycle_policy':{'version':POLICY_VERSION,'cleanup_before_and_after_each_method':True,
            'cleanup_in_decode_wall_ms':False,'maintenance_inclusive_wall_ms_also_recorded':True,
            'unchanged_core':CORE,'reuses_only_complete_legacy_cells':True},
        'legacy_incomplete_cells_not_reused':incomplete_cells,
        'native_serving':False,'strict_AR_certified':False,'sampling':'BF16 SDPA/eager; q32/p32; no Draft KV',
        'calibration':json.loads(args.calibration.read_text()),'quality_margin_pp':2.0,
        'data_history':'Existing fixed evaluation subsets have been used elsewhere in the project; not certified unseen.',
        'code_execution':'Generation only. Never execute generated code as root; code-quality scoring requires a separate sandbox.'})
    if not contract:
        write(args.output/'complete.json',{'completed_groups':0,'skipped_groups':0,'failed_groups':0,'all_cases_excluded':True})
        return
    torch.set_num_threads(1);torch.manual_seed(20260916)
    cfg=load_config(ROOT/f'configs/adaptive_block_{args.model}.json')['model']
    load_started=time.perf_counter();engine,tokenizer=load_models(cfg,'cuda:0')
    engine.set_target_verification_backend('eager');stops=stop_token_ids(engine,tokenizer)
    curve=tuple(map(tuple,json.loads(args.calibration.read_text())['curve']))
    write(args.output/'environment.json',{'torch':torch.__version__,'python':sys.version,
        'gpu':torch.cuda.get_device_name(0),'load_seconds':time.perf_counter()-load_started,
        'model':cfg,'memory_allocated_bytes':torch.cuda.memory_allocated(),'memory_reserved_bytes':torch.cuda.memory_reserved()})
    if args.reuse_directory and (args.reuse_directory/'environment.json').exists():
        previous_environment=json.loads((args.reuse_directory/'environment.json').read_text())
        if previous_environment['model'] != cfg:raise RuntimeError('Legacy model configuration drift')
    cache={};ordinal=0;warmed=set();counts={'completed_groups':0,'skipped_groups':0,'failed_groups':0,'reused_groups':0}
    for case in contract:
        dataset=case['dataset']
        if dataset not in cache:
            path=args.data_dir/f'{dataset}.jsonl'
            rows=[json.loads(line) for line in path.read_text().split('\n') if line.strip()]
            cache[dataset]=(rows,hashlib.sha256(path.read_bytes()).hexdigest())
        rows,data_hash=cache[dataset];c=case['concurrency'];n=case['requests']
        if n>len(rows):raise ValueError('insufficient unique requests')
        offset=64 if args.phase=='pilot' else 0
        if offset+n>len(rows):offset=0
        spec=matched_specs(case['temperature'],curve)
        for method in methods_for(case):
            if method=='ar':continue
            name=DD if method=='ddtree' else DF if method=='dflash' else OURS
            actual_capacity=case['budget']//spec[name]['decode']['row_cap']
            if actual_capacity!=method_capacity(method,case['budget']):raise RuntimeError('Eligibility configuration drift')
        if case['temperature'] not in warmed:
            warm=[encode_messages(tokenizer,[{'role':'user','content':rows[i]['prompt']}],cfg,str(engine.device)) for i in range(2)]
            target_ar(engine,tokenizer,warm,[17,29],stops,4,case['temperature'])
            reclaim_completed_decodes(engine)
            for name in (DD,DF,OURS):
                reclaim_completed_decodes(engine)
                settings=dict(spec[name]['decode'])
                decoder.generate_continuous_tree_blocks(engine,warm,spec[name]['variant'],
                    max_new_tokens=4,stop_ids=stops,seeds=[17,29],slot_capacity=193//settings['row_cap'],
                    row_budget=193,layout='packed_sequence',persistent_request_target_cache=True,**settings)
            reclaim_completed_decodes(engine)
            warmed.add(case['temperature'])
        for group,first in enumerate(range(offset,offset+n,c)):
            selected=rows[first:min(first+c,offset+n)]
            identity={'case':case,'group':group,'first':first,'data_sha256':data_hash}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            path=args.output/'groups'/f'{key}.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['identity']!=identity or old['source_hashes']!=hashes:raise RuntimeError('resume contract drift')
                counts['skipped_groups']+=1;continue
            if args.reuse_directory and completed_cell_key(case) in reusable_cells and (args.reuse_directory/'groups'/path.name).exists():
                previous_path=args.reuse_directory/'groups'/path.name
                previous=json.loads(previous_path.read_text())
                if previous['identity']!=identity or previous['source_hashes']!=legacy_hashes:
                    raise RuntimeError('Legacy identity/source drift; cannot reuse')
                original_names=legacy_driver.methods_for(case)
                actual={(run['method'],run['repeat']) for run in previous['runs']}
                expected={(method,repeat) for method in original_names for repeat in range(case['repeats'])}
                if actual!=expected or len(previous['runs'])!=len(expected):
                    raise RuntimeError('Legacy group incomplete or duplicate')
                imported=dict(previous)
                imported['source_hashes']=hashes
                imported['measurement_source_hashes']=previous.get('measurement_source_hashes',previous['source_hashes'])
                imported['reuse_provenance']={'original_path':str(previous_path),
                    'original_file_sha256':hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                    'resumption_contract_source_hashes':hashes,
                    'parent_reuse_provenance':previous.get('reuse_provenance'),
                    'original_record_source_hashes':previous['source_hashes'],
                    'timings_outputs_unchanged':True,
                    'excluded_methods':[m for m in original_names if m not in methods_for(case)]}
                imported['runs']=[run for run in previous['runs'] if run['method'] in methods_for(case)]
                imported['orders']=[[m for m in order if m in methods_for(case)] for order in previous.get('orders',[])]
                for run in imported['runs']:
                    if run['summary']['output_tokens']!=sum(map(len,run['outputs'])):
                        raise RuntimeError('Legacy output-token count mismatch')
                for method in methods_for(case):
                    values=[run['outputs'] for run in imported['runs'] if run['method']==method]
                    if not values or not all(v==values[0] for v in values):
                        raise RuntimeError('Legacy within-method repeatability failed')
                write(path,imported)
                counts['reused_groups']+=1;counts['skipped_groups']+=1
                write(args.output/'progress.json',counts)
                continue
            if shutil.disk_usage(args.output).free<3*1024**3:raise RuntimeError('disk free space safety gate')
            prompts=[]
            for row in selected:
                if case['prompt_tokens'] is None:
                    ids=encode_messages(tokenizer,[{'role':'user','content':row['prompt']}],cfg,str(engine.device))
                else:
                    # Controlled synthetic context throughput only, never task-quality data.
                    text=('Background context about algorithms and numerical methods. '*max(10,case['prompt_tokens']))
                    ids=encode_messages(tokenizer,[{'role':'user','content':text}],cfg,str(engine.device))
                    ids=ids[:,-case['prompt_tokens']:].contiguous()
                    assert ids.shape[1]==case['prompt_tokens']
                prompts.append(ids)
            seeds=[case['seed']*1000003+int(hashlib.sha256(str(row['source_id']).encode()).hexdigest()[:12],16)%1000000007 for row in selected]
            record={'identity':identity,'source_hashes':hashes,'request_ids':[r['source_id'] for r in selected],
                    'prompt_sha256':[r['prompt_sha256'] for r in selected],
                    'actual_prompt_tokens':[p.shape[1] for p in prompts],'synthetic_context':case['prompt_tokens'] is not None,
                    'request_seeds':seeds,'runs':[],'orders':[]}
            try:
                for repeat in range(case['repeats']):
                    names=methods_for(case);rng=random.Random(int(key[:12],16)+repeat);rng.shuffle(names)
                    record['orders'].append(names)
                    for name in names:
                        trace=[];policy=name;decode=None
                        cleanup_before=reclaim_completed_decodes(engine)
                        torch.cuda.reset_peak_memory_stats()
                        outer=time.perf_counter()
                        if name=='ar':
                            summary,outputs=target_ar(engine,tokenizer,prompts,seeds,stops,case['max_new_tokens'],case['temperature'])
                        else:
                            original=spec[DD if name=='ddtree' else DF if name=='dflash' else OURS]
                            decode=dict(original['decode']);variant=original['variant']
                            if name=='no_weight':decode['global_tree_criticality_weight']=0.0
                            if name=='constant_cost':decode['global_tree_row_cost_curve']=tuple((r,1.0) for r,_ in curve)
                            if name=='linear_cost':
                                slope=(curve[-1][1]-curve[0][1])/(curve[-1][0]-curve[0][0])
                                decode['global_tree_row_cost_curve']=None
                                decode['global_tree_fixed_row_equivalent']=max(1.0,curve[0][1]/max(slope,1e-9)-curve[0][0])
                            if name=='no_tier_fit':decode['global_tree_require_tier_fit']=False
                            if name=='no_active_gate':decode['global_tree_expansion_active_limit']=None
                            ctx=control(policy,trace) if name not in {'dflash','ddtree'} else nullcontext()
                            with ctx:
                                result=decoder.generate_continuous_tree_blocks(engine,prompts,variant,
                                    max_new_tokens=case['max_new_tokens'],stop_ids=stops,seeds=seeds,
                                    slot_capacity=case['budget']//decode['row_cap'],row_budget=case['budget'],
                                    layout='packed_sequence',persistent_request_target_cache=True,**decode)
                            summary=summarize(result);outputs=result.outputs;del result
                        engine.sync()
                        texts=tokenizer.batch_decode(outputs,skip_special_tokens=True)
                        run={'method':name,'repeat':repeat,'summary':summary,'outputs':outputs,'texts':texts,
                            'allocator_trace':trace,'decode_settings':decode,'outer_seconds':time.perf_counter()-outer,
                            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                        assert summary['output_tokens']==sum(map(len,outputs))
                        assert all(len(o)<=case['max_new_tokens'] for o in outputs)
                        assert all(t['rows']<=case['budget'] for t in trace)
                        cleanup_after=reclaim_completed_decodes(engine)
                        run['memory_lifecycle_before']=cleanup_before
                        run['memory_lifecycle_after']=cleanup_after
                        run['maintenance_inclusive_wall_ms']=summary['wall_ms']+cleanup_before['maintenance_ms']+cleanup_after['maintenance_ms']
                        record['runs'].append(run)
                        if case['prompt_tokens'] is not None and case['prompt_tokens']>=8192:
                            print(json.dumps({'event':'memory_safe_method_complete','method':name,'repeat':repeat,
                                'case':case,'group':group,'cuda_peak_allocated_bytes':run['cuda_peak_allocated_bytes'],
                                'cleanup_after':cleanup_after}),flush=True)
                for name in methods_for(case):
                    values=[r['outputs'] for r in record['runs'] if r['method']==name]
                    assert all(v==values[0] for v in values),'within-method repeatability failed'
                write(path,record);counts['completed_groups']+=1
                print(json.dumps({'model':args.model,'phase':args.phase,'case':case,'group':group,
                    'completed':counts['completed_groups'],'run_seconds':sum(r['outer_seconds'] for r in record['runs'])}),flush=True)
                write(args.output/'progress.json',counts)
            except Exception as exc:
                counts['failed_groups']+=1;write(args.output/'failure.json',{'identity':identity,'error':repr(exc),'counts':counts})
                raise
            ordinal+=1
    write(args.output/'complete.json',counts)


if __name__=='__main__':main()
