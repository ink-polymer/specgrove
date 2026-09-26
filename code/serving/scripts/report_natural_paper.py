"""Generate only observed natural-suite cells, never borrow another candidate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from paper_natural_plan import manifest, PHASES
from benchmark_draft_cache_reuse import write


def speedup_interval(records, method):
    import numpy as np
    groups = defaultdict(lambda: [0., 0., 0., 0.])
    for record in records:
        # Repeat and seed are kept inside the same independent prompt batch.
        group = groups[record['case']['first']]
        for run in record['runs']:
            if run['method'] in (method, 'ar'):
                k = 0 if run['method'] == method else 2
                group[k] += run['summary']['output_tokens']
                group[k+1] += run['summary']['wall_ms']
    values = np.asarray(list(groups.values()))
    rng = np.random.default_rng(9091)
    samples = values[rng.integers(0, len(values), size=(2000, len(values)))].sum(axis=1)
    ratios = (samples[:,0]/samples[:,1])/(samples[:,2]/samples[:,3])
    return {'paired_prompt_batch_bootstrap95': np.quantile(ratios, [.025, .975]).tolist(),
            'independent_prompt_batches': len(values), 'resamples': 2000,
            'warning': 'Few independent batches; interpret cautiously' if len(values)<10 else None}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    write(args.results/'EXPERIMENT_PLAN.json', manifest())
    if args.plan_only: return
    cells = defaultdict(list)
    for model in ('qwen3_4b', 'qwen3_8b'):
        for phase in PHASES:
            for path in sorted((args.results/model/phase/'groups').glob('*.json')):
                record = json.loads(path.read_text())
                c = record['case']
                if phase == 'correctness': continue
                key = (model, phase, c['dataset'], c['temperature'], c['budget'], c['concurrency'], c['max_new_tokens'])
                cells[key].append(record)
    rows = []
    for key, records in sorted(cells.items()):
        model, phase, dataset, temperature, budget, concurrency, cap = key
        methods = sorted({r['method'] for g in records for r in g['runs']})
        stats = {}
        for m in methods:
            runs = [r for g in records for r in g['runs'] if r['method']==m]
            tokens = sum(r['summary']['output_tokens'] for r in runs)
            wall = sum(r['summary']['wall_ms'] for r in runs)
            stats[m] = {'tokens_per_second': 1000*tokens/wall, 'output_tokens': tokens, 'wall_ms': wall, 'executions': len(runs), 'maximum_gpu_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in runs)}
        ar = stats['ar']['tokens_per_second']
        for m, v in stats.items():
            v['speedup_vs_AR'] = v['tokens_per_second']/ar
            if m != 'ar': v['uncertainty'] = speedup_interval(records, m)
            runs = [r for g in records for r in g['runs'] if r['method']==m]
            stages = defaultdict(float)
            for r in runs:
                for name, elapsed in r['summary'].get('stage_ms', {}).items(): stages[name] += elapsed
            v['stage_ms_totals'] = dict(stages)
            v['allocation_events'] = sum(len(r.get('allocation_trace', [])) for r in runs)
        contract = json.loads((args.results/model/phase/'contract.json').read_text())
        expected = [c for c in contract['cases'] if (c['dataset'], c['temperature'], c['budget'], c['concurrency'], c['max_new_tokens'])==(dataset, temperature, budget, concurrency, cap)]
        rows.append({'model': model, 'phase': phase, 'dataset': dataset, 'temperature': temperature, 'budget': budget,
            'concurrency': concurrency, 'max_new_tokens': cap, 'completed_groups': len(records), 'planned_groups': len(expected), 'partial': len(records)!=len(expected), 'methods': stats})
    write(args.results/'OBSERVED_CELLS.json', rows)
    documents = defaultdict(list)
    for row in rows: documents[(row['temperature'], row['budget'])].append(row)
    for (temperature, budget), selected in documents.items():
        lines = [f'# Natural128: T={temperature}, global row budget={budget}', '',
            'Throughput = total generated tokens / total measured seconds; speedup = throughput / matched AR throughput. Each row is per model/dataset/concurrency/output-cap/phase. Partial rows are explicitly marked. Model loading and boundary GC are excluded; prefill is included. No pooled cross-dataset mean.', '',
            '|Model|Phase|Dataset|C|Output cap|Method|tok/s|Speedup vs AR|95% CI|Groups|Status|',
            '|---|---|---|---:|---:|---|---:|---:|---|---|---|']
        for r in selected:
            for m, s in r['methods'].items():
                interval = s.get('uncertainty', {}).get('paired_prompt_batch_bootstrap95')
                ci = f'{interval[0]:.3f}–{interval[1]:.3f}' if interval else 'reference'
                lines.append(f"|{r['model']}|{r['phase']}|{r['dataset']}|{r['concurrency']}|{r['max_new_tokens']}|{m}|{s['tokens_per_second']:.2f}|{s['speedup_vs_AR']:.3f}|{ci}|{r['completed_groups']}/{r['planned_groups']}|{'partial' if r['partial'] else 'complete'}|")
        (args.results/f'T{temperature}_B{budget}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'observed_cells': len(rows), 'tables': len(documents)}))


if __name__ == '__main__': main()
