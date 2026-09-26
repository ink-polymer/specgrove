#!/usr/bin/env python3
"""Matched cache ablation. Frozen historical results are never reused or overwritten."""
import argparse
from dataclasses import replace
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

from benchmark_continuous_tree_block_e2e import summarize, prepared_prompts
from gbv_experiments.config import load_config
from gbv_experiments.conversation import encode_messages
from gbv_experiments.engine import load_models
from gbv_experiments.runner import stop_token_ids
from gbv_experiments.continuous_tree_block_decode import (
    ContinuousDecodeRequest, _propose_many, generate_continuous_tree_blocks,
)
from paper_memory_lifecycle import reclaim_completed_decodes
from rerun_audited_global_tree_matrix_h20 import DF, OURS, matched_specs
from run_dp_paper_single_wave import target_ar


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.inference_mode()
def audit(engine, tokenizer, cfg, specs):
    """Same inputs, real Draft, ragged/reordered requests and incremental updates.

    Reports numerical drift, not a certificate of full-sequence BF16 AR equality.
    The identical request states are queried cached and uncached every round.
    """
    rows = []
    for name in (DF, OURS):
        states = []
        for i in range(3):
            ids = encode_messages(tokenizer, [{'role': 'user', 'content':
                'Explain a binary search invariant. ' * 30}], cfg, str(engine.device))[:, -(64 + 7 * i):]
            state = ContinuousDecodeRequest(i, ids, 17 + i)
            state.target_cache = engine.cache_factory()
            output = engine.target_forward(ids, state.target_cache, hidden=True, last_only=True)
            state.full_features = engine.features(output.hidden_states)
            state.draft_update = state.full_features
            state.draft_cache = engine.cache_factory()
            state.generated = [int(output.logits[0, -1].argmax())]
            states.append(state)
            del output
        for round_index in range(3):
            if round_index:
                states.reverse()
                for state in states:
                    count = state.request_id + 1
                    ids = torch.tensor([[state.generated[-1]] * count], device=engine.device)
                    output = engine.target_forward(ids, state.target_cache, hidden=True, last_only=True)
                    state.draft_update = engine.features(output.hidden_states)
                    state.full_features = torch.cat((state.full_features, state.draft_update), dim=1)
                    state.generated = [int(output.logits[0, -1].argmax())]
                    del output
            captured = []
            def capture(_module, _args, output):
                captured.append(output.detach().clone())
            handle = engine.draft.register_forward_hook(capture)
            try:
                _propose_many(engine, states, specs[name]['variant'], reuse_request_draft_cache=False)
                _propose_many(engine, states, specs[name]['variant'], reuse_request_draft_cache=True)
            finally:
                handle.remove()
            head = engine.target.get_output_embeddings()
            reference = head(captured[0][:, 1:16]).float()
            cached = head(captured[1][:, 1:16]).float()
            finite = bool(torch.isfinite(reference).all() and torch.isfinite(cached).all())
            if not finite:
                raise RuntimeError('Nonfinite Draft logits')
            lengths = [state.target_cache.get_seq_length() for state in states]
            cache_lengths = [state.draft_cache.get_seq_length() for state in states]
            if lengths != cache_lengths:
                raise RuntimeError('Incremental Draft cache prefix mismatch')
            rows.append({'method': name, 'round': round_index, 'prefix_lengths': lengths,
                'draft_cache_lengths': cache_lengths, 'finite': finite,
                'max_abs_logit_difference': float((reference - cached).abs().max()),
                'max_local_total_variation': float((reference.softmax(-1) - cached.softmax(-1)).abs().sum(-1).mul(.5).max()),
                'greedy_path_equal': bool(torch.equal(reference.argmax(-1), cached.argmax(-1)))})
            del captured, reference, cached
        del states, state
        reclaim_completed_decodes(engine)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen3_4b', 'qwen3_8b'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--prefixes', default='128,8192')
    parser.add_argument('--concurrencies', default='1,4,8')
    parser.add_argument('--datasets', default='gsm8k,math500')
    parser.add_argument('--seeds', default='17,29')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--batched-prefill', action='store_true',
        help='Separate protocol: applies to BOTH cached and uncached speculative methods.')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory; do not mix protocols')
    cs = list(map(int, args.concurrencies.split(',')))
    if any(c not in (1, 4, 8) for c in cs) or args.repeats < 1:
        raise ValueError('Invalid single-wave concurrency or repeat controls')
    args.output.mkdir(parents=True)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('src/gbv_experiments', 'scripts') for p in (ROOT / folder).glob('*.py')}
    cfg = load_config(ROOT / f'configs/adaptive_block_{args.model}.json')['model']
    curve = tuple(map(tuple, json.loads(args.calibration.read_text())['curve']))
    specs = matched_specs(1, curve)
    contract = {'args': vars(args), 'source_hashes': hashes, 'model': cfg,
        'calibration': json.loads(args.calibration.read_text()), 'temperature': 1,
        'row_budget': 193, 'kind': 'matched_Draft_KV_reuse_ablation',
        'native_official_dflash': False, 'strict_BF16_AR_certified': False,
        'timing': 'decode wall includes prefill, proposal, packing and verification; excludes model load and between-trial GC',
        'attention': 'BF16 SDPA; eager execution; no compile/CUDA graphs',
        'synthetic_context_is_not_dataset_quality': True}
    write(args.output / 'contract.json', contract)
    torch.set_num_threads(1)
    torch.manual_seed(20260916)
    engine, tokenizer = load_models(cfg, 'cuda:0')
    engine.set_target_verification_backend('eager')
    stops = stop_token_ids(engine, tokenizer)
    write(args.output / 'environment.json', {'torch': torch.__version__, 'python': sys.version,
        'gpu': torch.cuda.get_device_name(0), 'allocated_bytes': torch.cuda.memory_allocated()})
    write(args.output / 'draft_correctness_audit.json', {'rows': audit(engine, tokenizer, cfg, specs),
        'scope': 'same-state Draft comparison; finite logits and request-local cache lengths; BF16 numerical drift reported, no strict sequence-law certification'})
    workloads = [(f'synthetic_{p}', int(p)) for p in args.prefixes.split(',') if p]
    workloads += [(d, None) for d in args.datasets.split(',') if d]
    synthetic_ids = {}
    methods = ['ar', 'dflash_uncached', 'dflash_cached', 'dp_uncached', 'dp_cached']
    cells = []

    @torch.inference_mode()
    def execute(method, prompts, seeds, cap):
        if method == 'ar':
            return target_ar(engine, tokenizer, prompts, seeds, stops, cap, 1)
        name = DF if method.startswith('dflash') else OURS
        settings = dict(specs[name]['decode'])
        settings['reuse_request_draft_cache'] = method.endswith('_cached')
        result = generate_continuous_tree_blocks(engine, prompts, specs[name]['variant'],
            max_new_tokens=cap, stop_ids=stops, seeds=seeds,
            slot_capacity=193 // settings['row_cap'], row_budget=193,
            layout='packed_sequence', persistent_request_target_cache=True,
            batched_prefill=args.batched_prefill, **settings)
        return summarize(result), result.outputs

    for workload, prefix in workloads:
        for c in cs:
            if prefix is not None:
                if prefix not in synthetic_ids:
                    text = 'Background context about algorithms and numerical methods. ' * max(10, prefix)
                    ids = encode_messages(tokenizer, [{'role': 'user', 'content': text}], cfg, str(engine.device))
                    synthetic_ids[prefix] = ids[:, -prefix:].contiguous()
                prompts = [synthetic_ids[prefix].clone() for _ in range(c)]
                identities = [{'synthetic': True, 'prompt_tokens': prefix} for _ in prompts]
            else:
                prompts, identities = prepared_prompts(tokenizer, cfg, engine.device, args.data_dir, workload, 0, c)
            # Each method receives an unmeasured warmup at this exact shape.
            for method in methods:
                reclaim_completed_decodes(engine)
                _, warm_outputs = execute(method, prompts, list(range(17, 17 + c)), 4)
                del warm_outputs
                reclaim_completed_decodes(engine)
            for seed in map(int, args.seeds.split(',')):
                record = {'workload': workload, 'synthetic': prefix is not None, 'concurrency': c,
                    'seed': seed, 'request_seeds': [seed + 104729 * i for i in range(c)],
                    'identities': identities, 'prompt_sha256': [digest(p.cpu().tolist()) for p in prompts],
                    'prompt_lengths': [p.shape[1] for p in prompts], 'source_hashes': hashes,
                    'runs': [], 'orders': []}
                for repeat in range(args.repeats):
                    order = methods.copy()
                    random.Random(digest([workload, c, seed, repeat])).shuffle(order)
                    record['orders'].append(order)
                    for method in order:
                        before = reclaim_completed_decodes(engine)
                        torch.cuda.reset_peak_memory_stats()
                        summary, outputs = execute(method, prompts, record['request_seeds'], args.max_new_tokens)
                        output_lists = [list(o) for o in outputs]
                        del outputs
                        if summary['output_tokens'] != sum(map(len, output_lists)):
                            raise RuntimeError('Output accounting mismatch')
                        peak = torch.cuda.max_memory_allocated()
                        after = reclaim_completed_decodes(engine)
                        record['runs'].append({'method': method, 'repeat': repeat, 'summary': summary,
                            'outputs': output_lists, 'output_sha256': digest(output_lists),
                            'cuda_peak_allocated_bytes': peak, 'maintenance_before': before, 'maintenance_after': after})
                for method in methods:
                    values = [r['output_sha256'] for r in record['runs'] if r['method'] == method]
                    if len(set(values)) != 1:
                        raise RuntimeError(f'Within-method repeatability failure: {method}')
                record['cached_vs_uncached_output_equal'] = {
                    family: next(r['outputs'] for r in record['runs'] if r['method'] == family + '_cached') ==
                        next(r['outputs'] for r in record['runs'] if r['method'] == family + '_uncached')
                    for family in ('dflash', 'dp')}
                write(args.output / 'groups' / f'{workload}_c{c}_s{seed}.json', record)
                print(json.dumps({'completed': workload, 'concurrency': c, 'seed': seed}), flush=True)
                cells.append(record)
                summary_rows = []
                for work in workloads:
                    for count in cs:
                        records = [r for r in cells if r['workload'] == work[0] and r['concurrency'] == count]
                        if not records:
                            continue
                        tps = {}
                        for method in methods:
                            runs = [run for r in records for run in r['runs'] if run['method'] == method]
                            tps[method] = 1000 * sum(r['summary']['output_tokens'] for r in runs) / sum(r['summary']['wall_ms'] for r in runs)
                        summary_rows.append({'workload': work[0], 'concurrency': count, 'completed_seeds': len(records),
                            'tokens_per_second': tps, 'speedup_vs_AR': {m: t / tps['ar'] for m, t in tps.items()}})
                write(args.output / 'SUMMARY.json', {'cells': summary_rows, 'partial': True})
    write(args.output / 'complete.json', {'groups': len(cells), 'source_hashes': hashes})
    write(args.output / 'SUMMARY.json', {'cells': summary_rows, 'partial': False})


if __name__ == '__main__':
    main()
