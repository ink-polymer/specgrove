"""Resume-safe natural-task paper phases. Audit timings are never performance."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import torch
from transformers import AutoTokenizer
from paper_natural_plan import PHASES, DATASETS, cases_for
from benchmark_continuous_tree_block_e2e import prepared_prompts, summarize
from benchmark_draft_cache_reuse import digest, write
from gbv_experiments.config import load_config
from gbv_experiments.common import digest as prompt_digest
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.continuous_tree_block_decode import generate_continuous_tree_blocks
from paper_dp_allocators import control
from paper_memory_lifecycle import reclaim_completed_decodes
from rerun_audited_global_tree_matrix_h20 import DD, DF, OURS, matched_specs
from run_dp_paper_single_wave import target_ar


def settings_for(method, temperature, curve):
    family = DF if method == 'dflash' else DD if method in ('ddtree', 'fixed_large') else OURS
    spec = copy.deepcopy(matched_specs(temperature, curve)[family])
    settings = spec['decode']
    settings['reuse_request_draft_cache'] = True
    if method == 'fixed_small':
        settings['global_tree_budget_options'] = (11,)
    elif method == 'no_weight':
        settings['global_tree_criticality_weight'] = 0.0
    elif method == 'constant_cost':
        settings['global_tree_row_cost_curve'] = tuple((r, 1.0) for r, _ in curve)
    elif method == 'linear_cost':
        slope = (curve[-1][1]-curve[0][1]) / max(1, curve[-1][0]-curve[0][0])
        settings['global_tree_row_cost_curve'] = None
        settings['global_tree_fixed_row_equivalent'] = max(1.0, curve[0][1]/max(slope, 1e-9)-curve[0][0])
    elif method == 'no_active_gate':
        settings['global_tree_expansion_active_limit'] = None
    elif method == 'no_tier_fit':
        settings['global_tree_require_tier_fit'] = False
    return spec['variant'], settings


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src/gbv_experiments', 'scripts', 'third_party/ddtree_pinned')
            for p in sorted((ROOT/folder).rglob('*.py')) if not p.name.startswith('._')}


def load_natural(tokenizer, cfg, data_dir, dataset, count):
    if dataset == 'mixed_natural':
        each = count//len(DATASETS)
        if each*len(DATASETS) != count: raise ValueError('Unbalanced mixed tasks')
        batches = [load_natural(tokenizer, cfg, data_dir, d, each) for d in DATASETS]
        prompts, identities = [], []
        for i in range(each):
            for ps, ids in batches:
                prompts.append(ps[i]); identities.append(ids[i])
        return prompts, identities
    path = data_dir/f'{dataset}.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    count = len(rows) if count == 'all' else count
    if len({str(r['source_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate task identity')
    for row in rows[:count]:
        if row['prompt_sha256'] != prompt_digest(row['prompt']):
            raise ValueError('Prepared prompt hash mismatch')
    prompts, identities = prepared_prompts(tokenizer, cfg, 'cpu', data_dir, dataset, 0, count)
    for p, identity in zip(prompts, identities):
        identity.update(input_tokens=p.shape[1], filler_tokens=0, prompt_preserved=True,
                        input_token_sha256=digest(p.tolist()))
    return prompts, identities


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--model', required=True, choices=('qwen3_4b', 'qwen3_8b'))
    ap.add_argument('--phase', required=True, choices=PHASES)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--calibration', required=True, type=Path)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='Separate non-formal first group, two output tokens')
    ap.add_argument('--reuse-directory', type=Path, help='Verified same-host original protocol; retain prespecified repeat0 only')
    args = ap.parse_args()
    torch.set_num_threads(1)
    cfg = load_config(ROOT/f'configs/adaptive_block_{args.model}.json')['model']
    tokenizer = AutoTokenizer.from_pretrained(cfg['target'], revision=cfg.get('target_revision'))
    calibration = json.loads(args.calibration.read_text())
    curve = tuple(map(tuple, calibration['curve']))
    cases = cases_for(args.phase)
    samples = {(c['dataset'], c['requests']) for c in cases}
    cpu = {key: load_natural(tokenizer, cfg, args.data_dir, *key) for key in samples}
    groups = []
    for case in cases:
        total = len(cpu[(case['dataset'], case['requests'])][0])
        for first in range(0, total, case['concurrency']):
            groups.append(dict(case, first=first, group=first//case['concurrency']))
    if args.smoke:
        groups = [dict(groups[0], max_new_tokens=2, repeats=1)]
    hashes = source_hashes()
    contract = {'model': cfg, 'phase': args.phase, 'cases': groups, 'source_hashes': hashes,
                'calibration': calibration, 'data_hashes': {d: hashlib.sha256((args.data_dir/f'{d}.jsonl').read_bytes()).hexdigest() for d in DATASETS},
                'natural_full_prompts': True, 'smoke_nonformal': args.smoke,
                'timing': 'Full prefill+decode, excludes model load/text decode/boundary reclamation; correctness instrumented timings nonperformance',
                'precision': 'BF16 eager SDPA, FP32 posterior; no compilation/CUDA graphs',
                'strict_BF16_sequence_law_certified': False,
                'selection_counts': {d: len(ps) for (d, _), (ps, _) in cpu.items()},
                'prompt_lengths': {d: [min(p.shape[1] for p in ps), max(p.shape[1] for p in ps)] for (d, _), (ps, _) in cpu.items()}}
    path = args.output/'contract.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(contract)):
        raise RuntimeError('Resume contract drift; use a new directory')
    write(path, contract)
    if args.reuse_directory is not None:
        from paper_once_migration import import_repetition_zero
        imported = import_repetition_zero(args.reuse_directory, args.output,
            json.loads(json.dumps(contract)), cpu, digest, write)
        print(json.dumps({'reused_repeat0_groups': imported, 'reuse_directory': str(args.reuse_directory)}), flush=True)
    pending, done = [], []
    for case in groups:
        path = args.output/'groups'/(digest(case)[:24]+'.json')
        if path.exists():
            record = json.loads(path.read_text())
            expected = {(m, repeat) for m in case['methods'] for repeat in range(case['repeats'])}
            if record['case'] != case or record['source_hashes'] != hashes or {(r['method'], r['repeat']) for r in record['runs']} != expected or len(record['runs']) != len(expected):
                raise RuntimeError('Completed group mismatch')
            done.append(record)
        else:
            pending.append(case)
    write(args.output/'progress.json', {'completed_groups': len(done), 'planned_groups': len(groups), 'planned_executions': sum(len(c['methods'])*c['repeats'] for c in groups), 'prepare_only': args.prepare_only})
    print(json.dumps({'phase': args.phase, 'groups': len(groups), 'pending': len(pending), 'smoke': args.smoke}), flush=True)
    if args.prepare_only:
        return
    if not pending:
        write(args.output/'complete.json', {'groups': len(done), 'source_hashes': hashes}); return
    torch.manual_seed(20260917)
    engine, tokenizer = load_models(cfg, 'cuda:0')
    engine.set_target_verification_backend('eager')
    stops = stop_token_ids(engine, tokenizer)
    write(args.output/'environment.json', {'torch': torch.__version__, 'python': sys.version, 'gpu': torch.cuda.get_device_name(0)})

    @torch.inference_mode()
    def execute(method, prompts, seeds, case, cap, audit=None):
        trace = []
        if method == 'ar':
            summary, outputs = target_ar(engine, tokenizer, prompts, seeds, stops, cap, case['temperature'])
        elif method.startswith('native_'):
            if len(prompts) != 1: raise ValueError('Native baseline only supports C1 here')
            sys.path.insert(0, str(ROOT/'third_party/ddtree_pinned'))
            from dflash import dflash_generate
            from ddtree import ddtree_generate
            torch.manual_seed(seeds[0]); torch.cuda.manual_seed_all(seeds[0])
            kwargs = dict(model=engine.draft, target=engine.target, input_ids=prompts[0],
                          mask_token_id=engine.draft.mask_token_id, max_new_tokens=cap,
                          block_size=16, stop_token_ids=stops, temperature=case['temperature'])
            engine.sync(); start = time.perf_counter()
            result = ddtree_generate(**kwargs, tree_budget=45) if method == 'native_ddtree' else dflash_generate(**kwargs)
            engine.sync(); wall = (time.perf_counter()-start)*1000
            outputs = [result.output_ids[0, result.num_input_tokens:].tolist()]
            summary = {'wall_ms': wall, 'output_tokens': len(outputs[0]), 'tokens_per_second': len(outputs[0])*1000/wall, 'decode_rounds': result.decode_rounds, 'native_official': True}
        else:
            variant, settings = settings_for(method, case['temperature'], curve)
            with control(method, trace), (audit.instrument() if audit else nullcontext()):
                result = generate_continuous_tree_blocks(engine, prompts, variant, max_new_tokens=cap,
                    stop_ids=stops, seeds=seeds, row_budget=case['budget'],
                    slot_capacity=case['budget']//settings['row_cap'], layout='packed_sequence',
                    persistent_request_target_cache=True, **settings)
            summary, outputs = summarize(result), result.outputs
            if any(t['rows'] > case['budget'] for t in trace):
                raise RuntimeError('Global budget violation')
        return summary, [list(o) for o in outputs], trace

    warmed = set()
    for case in pending:
        ps, identities = cpu[(case['dataset'], case['requests'])]
        first, c = case['first'], case['concurrency']
        selected = identities[first:first+c]
        prompts = [p.to(engine.device) for p in ps[first:first+c]]
        seeds = [case['seed']*1000003+int(hashlib.sha256((i['dataset']+':'+str(i['source_id'])).encode()).hexdigest()[:12],16)%1000000007 for i in selected]
        shape = (case['budget'], c, case['temperature'], tuple(case['methods']))
        if shape not in warmed:
            for method in case['methods']:
                reclaim_completed_decodes(engine)
                execute(method, prompts, seeds, case, 2)
                reclaim_completed_decodes(engine)
            warmed.add(shape)
        record = {'case': case, 'source_hashes': hashes, 'identities': selected, 'request_seeds': seeds, 'runs': [], 'orders': []}
        began = time.perf_counter()
        for repeat in range(case['repeats']):
            order = case['methods'].copy(); random.Random(digest([case, repeat])).shuffle(order)
            record['orders'].append(order)
            for method in order:
                before = reclaim_completed_decodes(engine)
                torch.cuda.reset_peak_memory_stats()
                summary, outputs, trace = execute(method, prompts, seeds, case, case['max_new_tokens'])
                if summary['output_tokens'] != sum(map(len, outputs)) or any(len(o)>case['max_new_tokens'] for o in outputs):
                    raise RuntimeError('Output accounting violation')
                record['runs'].append({'method': method, 'repeat': repeat, 'summary': summary, 'outputs': outputs,
                    'texts': tokenizer.batch_decode(outputs, skip_special_tokens=True), 'output_sha256': digest(outputs),
                    'allocation_trace': trace, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'maintenance_before': before, 'maintenance_after': reclaim_completed_decodes(engine)})
        for method in case['methods']:
            if len({r['output_sha256'] for r in record['runs'] if r['method']==method}) != 1:
                raise RuntimeError('Within-method repeatability violation')
        if args.phase == 'correctness':
            record['performance_eligible'] = False
            if case['temperature'] == 0:
                ar = next(r['outputs'] for r in record['runs'] if r['method']=='ar')
                record['T0_vs_matched_AR_equal_requests'] = {r['method']: sum(a==b for a,b in zip(ar,r['outputs'])) for r in record['runs']}
            # Instrument the first natural group of every dataset/budget/C/T.
            if first == 0 and 'dp' in case['methods']:
                from gbv_experiments import continuous_tree_block_decode as decoder
                from correctness_trace_shared_budget_h20 import ActualHistoryAudit, canonical_ar
                reclaim_completed_decodes(engine)
                references = [canonical_ar(engine, p, stops, case['max_new_tokens']) for p in prompts]
                audit = ActualHistoryAudit(engine, decoder, prompts, references, case['temperature'], case['budget'])
                _, instrumented, _ = execute('dp', prompts, seeds, case, case['max_new_tokens'], audit)
                expected = next(r['outputs'] for r in record['runs'] if r['method']=='dp')
                if instrumented != expected: raise RuntimeError('Audit instrumentation interfered with output')
                record['actual_history_audit'] = {'stats': audit.stats, 'prefills': audit.prefills, 'probes': audit.probes,
                    'structural_checks_passed': True, 'instrumentation_noninterference': True,
                    'strict_sequence_law_certified': False, 'scope': 'First natural group per condition; all groups get accounting/repeatability/T0 comparison'}
                del audit, references
                reclaim_completed_decodes(engine)
        write(args.output/'groups'/(digest(case)[:24]+'.json'), record)
        done.append(record)
        write(args.output/'progress.json', {'completed_groups': len(done), 'planned_groups': len(groups),
            'completed_executions': sum(len(r['runs']) for r in done), 'last_case': case})
        print(json.dumps({'completed_groups': len(done), 'planned_groups': len(groups), 'case': case, 'group_seconds': time.perf_counter()-began}), flush=True)
    write(args.output/'complete.json', {'groups': len(done), 'executions': sum(len(r['runs']) for r in done), 'source_hashes': hashes, 'strict_sequence_law_certified': False})


if __name__ == '__main__':
    main()
