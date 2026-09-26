#!/usr/bin/env python3
"""Main matrix with complete, unmodified natural prompts and variable lengths."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
from transformers import AutoTokenizer

from benchmark_draft_cache_reuse import write, digest, audit
from benchmark_continuous_tree_block_e2e import summarize, prepared_prompts
from gbv_experiments.config import load_config
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.continuous_tree_block_decode import generate_continuous_tree_blocks
from paper_memory_lifecycle import reclaim_completed_decodes
from rerun_audited_global_tree_matrix_h20 import DD, DF, OURS, matched_specs
from run_dp_paper_single_wave import target_ar


def eligible_methods(concurrency, specs, budget=193):
    methods = ['ar']
    for family, name in [('dflash', DF), ('ddtree', DD), ('dp', OURS)]:
        if concurrency <= budget // specs[name]['decode']['row_cap']:
            methods.extend([family + '_uncached', family + '_cached'])
    return methods


def aggregate(records):
    cells = []
    for dataset, concurrency in sorted({(r['dataset'], r['concurrency']) for r in records}):
        groups = [r for r in records if r['dataset'] == dataset and r['concurrency'] == concurrency]
        methods = list(dict.fromkeys(run['method'] for r in groups for run in r['runs']))
        stats = {}
        for method in methods:
            runs = [run for r in groups for run in r['runs'] if run['method'] == method]
            tokens = sum(run['summary']['output_tokens'] for run in runs)
            wall = sum(run['summary']['wall_ms'] for run in runs)
            stats[method] = {'tokens': tokens, 'wall_ms': wall, 'executions': len(runs),
                'tokens_per_second': 1000 * tokens / wall}
        ar = stats['ar']['tokens_per_second']
        for values in stats.values():
            values['speedup_vs_AR'] = values['tokens_per_second'] / ar
        cells.append({'dataset': dataset, 'concurrency': concurrency, 'completed_groups': len(groups),
            'seeds_seen': sorted({r['seed'] for r in groups}), 'methods': stats})
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen3_4b', 'qwen3_8b'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--datasets', default='gsm8k,math500,humaneval,mbpp_sanitized')
    parser.add_argument('--concurrencies', default='1,4,8')
    parser.add_argument('--seeds', default='17,29,43')
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    cs = list(map(int, args.concurrencies.split(',')))
    seeds = list(map(int, args.seeds.split(',')))
    datasets = args.datasets.split(',')
    if len(set(cs)) != len(cs) or len(set(seeds)) != len(seeds) or len(set(datasets)) != len(datasets):
        raise ValueError('Duplicate experiment conditions')
    if any(c not in (1, 4, 8) or args.requests % c for c in cs) or args.repeats < 1 or args.requests < 8:
        raise ValueError('Invalid condition controls')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = load_config(ROOT / f'configs/adaptive_block_{args.model}.json')['model']
    tokenizer = AutoTokenizer.from_pretrained(cfg['target'], revision=cfg.get('target_revision'))
    cpu_prompts, selections = {}, {}
    for dataset in datasets:
        cpu_prompts[dataset], identities = prepared_prompts(
            tokenizer, cfg, 'cpu', args.data_dir, dataset, 0, args.requests)
        if len({str(i['source_id']) for i in identities}) != args.requests:
            raise ValueError('Duplicate natural task identity')
        for ids, identity in zip(cpu_prompts[dataset], identities):
            identity.update(original_tokens=ids.shape[1], input_tokens=ids.shape[1],
                filler_tokens=0, prompt_preserved=True,
                input_token_sha256=hashlib.sha256(json.dumps(ids.tolist()).encode()).hexdigest())
        selections[dataset] = {'dataset': dataset,
            'data_sha256': hashlib.sha256((args.data_dir / f'{dataset}.jsonl').read_bytes()).hexdigest(),
            'selection': 'First32 natural tasks in original file order, no length filter or output-based selection',
            'filler': 'none', 'identities': identities,
            'quality_protocol': 'Original complete prompt plus original chat template, no truncation or added context'}
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src/gbv_experiments', 'scripts') for p in (ROOT / folder).glob('*.py')}
    curve = tuple(map(tuple, json.loads(args.calibration.read_text())['curve']))
    specs = matched_specs(1, curve)
    plan = [{'dataset': dataset, 'concurrency': c, 'seed': seed,
        'group': first // c, 'first': first, 'methods': eligible_methods(c, specs)}
        for dataset in datasets for c in cs for seed in seeds for first in range(0, args.requests, c)]
    contract = {'model': cfg, 'source_hashes': hashes, 'temperature': 1, 'input_tokens': 'natural_variable',
        'max_new_tokens': args.max_new_tokens, 'row_budget': 193, 'requests_per_dataset': args.requests,
        'repeats': args.repeats, 'concurrencies': cs, 'seeds': seeds,
        'datasets': datasets, 'selections': selections, 'cases': plan,
        'kind': 'complete_natural_prompt_main_cache_ablation', 'protocol_version': 1,
        'native_official_dflash': False, 'strict_BF16_AR_certified': False,
        'input_control': 'Original complete questions at actual natural lengths; no filler, truncation or length filtering',
        'precision': 'BF16 SDPA/eager, FP32 proposals/posteriors; no compile/CUDA graphs',
        'timing': 'Includes prefill and decode stages; excludes model load, text decode and method-boundary GC',
        'calibration': json.loads(args.calibration.read_text()),
        'policy_exclusions': [{'concurrency': c, 'method': family,
            'reason': 'configured concurrency exceeds single Target wave capacity',
            'capacity': 193 // specs[name]['decode']['row_cap']}
            for c in cs for family, name in [('ddtree', DD), ('dflash', DF), ('dp', OURS)]
            if c > 193 // specs[name]['decode']['row_cap']],
        'quality': 'Generation only; complete original prompts; no scores or strict exactness certification'}
    contract_path = args.output / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Resume contract drift; use a new output directory')
    write(contract_path, contract)
    print(json.dumps({'prepared': args.model, 'groups': len(plan),
        'executions': sum(len(case['methods']) * args.repeats for case in plan),
        'selected_per_dataset': {d: len(s['identities']) for d, s in selections.items()},
        'original_length_ranges': {d: [min(i['original_tokens'] for i in s['identities']),
            max(i['original_tokens'] for i in s['identities'])] for d, s in selections.items()}}), flush=True)
    if args.prepare_only:
        return
    records, pending = [], []
    for case in plan:
        path = args.output / 'groups' / (digest(case)[:24] + '.json')
        if path.exists():
            record = json.loads(path.read_text())
            expected = {(m, repeat) for m in case['methods'] for repeat in range(args.repeats)}
            actual = {(r['method'], r['repeat']) for r in record['runs']}
            if record['case'] != case or record['source_hashes'] != hashes or actual != expected or len(record['runs']) != len(expected):
                raise RuntimeError('Completed group contract mismatch')
            records.append(record)
        else:
            pending.append(case)
    if not pending:
        write(args.output / 'SUMMARY.json', {'cells': aggregate(records), 'partial': False})
        return
    torch.manual_seed(20260916)
    engine, actual_tokenizer = load_models(cfg, 'cuda:0')
    if actual_tokenizer.encode('\n', add_special_tokens=False) != tokenizer.encode('\n', add_special_tokens=False):
        raise RuntimeError('Tokenizer changed after preflight')
    engine.set_target_verification_backend('eager')
    stops = stop_token_ids(engine, actual_tokenizer)
    write(args.output / 'environment.json', {'torch': torch.__version__, 'python': sys.version,
        'gpu': torch.cuda.get_device_name(0), 'model': cfg})
    write(args.output / 'draft_correctness_audit.json', {'rows': audit(engine, actual_tokenizer, cfg, specs),
        'scope': 'bounded request-local Draft cache checks, not full-sequence exactness certification'})
    warmed = set()

    @torch.inference_mode()
    def execute(method, prompts, request_seeds, cap):
        if method == 'ar':
            return target_ar(engine, actual_tokenizer, prompts, request_seeds, stops, cap, 1)
        name = DF if method.startswith('dflash') else DD if method.startswith('ddtree') else OURS
        settings = dict(specs[name]['decode'])
        settings['reuse_request_draft_cache'] = method.endswith('_cached')
        result = generate_continuous_tree_blocks(engine, prompts, specs[name]['variant'],
            max_new_tokens=cap, stop_ids=stops, seeds=request_seeds,
            slot_capacity=193 // settings['row_cap'], row_budget=193,
            layout='packed_sequence', persistent_request_target_cache=True, **settings)
        return summarize(result), result.outputs

    for case in pending:
        dataset, c, first = case['dataset'], case['concurrency'], case['first']
        prompts = [p.to(engine.device) for p in cpu_prompts[dataset][first:first + c]]
        request_seeds = [case['seed'] * 1000003 +
            int(hashlib.sha256(str(i['source_id']).encode()).hexdigest()[:12], 16) % 1000000007
            for i in selections[dataset]['identities'][first:first + c]]
        if c not in warmed:
            for method in case['methods']:
                reclaim_completed_decodes(engine)
                _, outputs = execute(method, prompts, request_seeds, 4)
                del outputs
                reclaim_completed_decodes(engine)
            warmed.add(c)
        record = {'case': case, 'dataset': dataset, 'concurrency': c, 'seed': case['seed'],
            'group': case['group'], 'source_hashes': hashes, 'request_seeds': request_seeds,
            'identities': selections[dataset]['identities'][first:first + c],
            'prompt_lengths': [p.shape[1] for p in prompts], 'runs': [], 'orders': []}
        began = time.perf_counter()
        for repeat in range(args.repeats):
            order = case['methods'].copy()
            random.Random(digest([case, repeat])).shuffle(order)
            record['orders'].append(order)
            for method in order:
                before = reclaim_completed_decodes(engine)
                torch.cuda.reset_peak_memory_stats()
                summary, outputs = execute(method, prompts, request_seeds, args.max_new_tokens)
                lists = [list(o) for o in outputs]
                del outputs
                if summary['output_tokens'] != sum(map(len, lists)) or any(len(o) > args.max_new_tokens for o in lists):
                    raise RuntimeError('Output accounting violation')
                peak = torch.cuda.max_memory_allocated()
                after = reclaim_completed_decodes(engine)
                record['runs'].append({'method': method, 'repeat': repeat, 'summary': summary,
                    'outputs': lists, 'output_sha256': digest(lists),
                    'texts': actual_tokenizer.batch_decode(lists, skip_special_tokens=True),
                    'cuda_peak_allocated_bytes': peak, 'maintenance_before': before,
                    'maintenance_after': after})
        for method in case['methods']:
            if len({r['output_sha256'] for r in record['runs'] if r['method'] == method}) != 1:
                raise RuntimeError('Within-method repeatability violation')
        record['cached_vs_uncached_output_equal'] = {
            family: next(r['outputs'] for r in record['runs'] if r['method'] == family + '_cached') ==
                next(r['outputs'] for r in record['runs'] if r['method'] == family + '_uncached')
            for family in ('dflash', 'ddtree', 'dp') if family + '_cached' in case['methods']}
        write(args.output / 'groups' / (digest(case)[:24] + '.json'), record)
        records.append(record)
        write(args.output / 'SUMMARY.json', {'cells': aggregate(records), 'partial': len(records) != len(plan)})
        write(args.output / 'progress.json', {'completed_groups': len(records), 'planned_groups': len(plan),
            'completed_executions': sum(len(r['runs']) for r in records), 'last_case': case})
        print(json.dumps({'completed_groups': len(records), 'planned_groups': len(plan),
            'case': case, 'group_seconds': time.perf_counter() - began}), flush=True)
    write(args.output / 'complete.json', {'groups': len(records), 'planned_groups': len(plan),
        'executions': sum(len(r['runs']) for r in records), 'source_hashes': hashes})


if __name__ == '__main__':
    main()
