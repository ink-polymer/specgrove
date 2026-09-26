"""Natural quality scoring, isolated code tests, prompt-cluster paired CIs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from benchmark_draft_cache_reuse import write, digest
from paper_natural_plan import DATASETS, QUALITY_SEEDS
from paper_safe_code import prepare_stdlib, selftest, run_program


def clean_code(text):
    text = text.rsplit('</think>', 1)[-1]
    blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', text)
    return (blocks[0] if blocks else text).strip('\n').rstrip()


def code_program(evaluation, prediction, reference=False):
    code = evaluation['reference_code'] if reference else clean_code(prediction)
    if evaluation['kind'] == 'humaneval':
        entry = evaluation['entry_point']
        if reference or not re.search(r'^\s*def\s+'+re.escape(entry)+r'\s*\(', code, re.M):
            code = evaluation['prompt']+code
        return code+'\n\n'+evaluation['test']+'\ncheck('+entry+')\n'
    return evaluation.get('setup', '')+'\n'+code+'\n'+'\n'.join(evaluation['tests']+evaluation.get('challenge_tests', []))+'\n'


def math_score(evaluation, prediction):
    from math_verify import parse, verify, LatexExtractionConfig
    gold = parse('$'+evaluation['answer']+'$', extraction_config=[LatexExtractionConfig()])
    if not gold or not verify(gold, gold): raise RuntimeError('Gold math parsing failed')
    try:
        pred = parse(prediction.rsplit('</think>', 1)[-1])
        return {'passed': bool(verify(gold, pred)) if pred else False,
                'status': 'graded' if pred else 'prediction_unparsable'}
    except Exception as error:
        return {'passed': False, 'status': 'prediction_parse_error', 'error_type': type(error).__name__}


def paired_interval(deltas, seed=7331):
    import numpy as np
    values = np.asarray(deltas, dtype=float)
    if not len(values): raise ValueError('No paired prompt clusters')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return {'mean_difference': float(values.mean()), 'two_sided95': np.quantile(means, [.025, .975]).tolist(),
            'one_sided95_lower': float(np.quantile(means, .05)), 'prompt_clusters': len(values)}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    contract = json.loads((args.input/'contract.json').read_text())
    rows = {}
    for d in DATASETS:
        file = args.data_dir/f'{d}.jsonl'
        if hashlib.sha256(file.read_bytes()).hexdigest() != contract['data_hashes'][d]: raise RuntimeError('Quality data drift')
        selected = list(map(json.loads, file.read_text().splitlines()))[:128]
        if len(selected) != 128: raise RuntimeError('Quality requires128 natural tasks')
        for row in selected: rows[(d, str(row['source_id']))] = row
    runtime = prepare_stdlib()
    isolation = selftest(runtime)
    write(args.input/'scoring'/'isolation.json', isolation)
    if not isolation['available']: raise RuntimeError('Code sandbox unavailable; no generated code executed')
    grader_hashes = {name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ('grade_natural_paper_quality.py', 'paper_safe_code.py')}
    scoring_contract = {'grader_hashes': grader_hashes, 'data_hashes': contract['data_hashes'], 'extraction': 'first fenced Python block, or raw continuation; no semantic repair', 'noninferiority_margin': .02, 'cluster': 'prompt identity, mean over five seeds', 'code_sandbox': isolation['policy']}
    path = args.input/'scoring'/'contract.json'
    if path.exists() and json.loads(path.read_text()) != scoring_contract: raise RuntimeError('Scorer resume drift')
    write(path, scoring_contract)
    reference_path = args.input/'scoring'/'reference_validation.json'
    if not reference_path.exists():
        validated = []
        for (d, source_id), row in rows.items():
            evaluation = row['evaluation']
            if evaluation['kind'] == 'math':
                score = math_score(evaluation, '\\boxed{'+evaluation['answer']+'}')
            else:
                score = run_program(code_program(evaluation, '', reference=True), runtime)
            validated.append({'dataset': d, 'source_id': source_id, 'score': score})
            if score['passed'] is not True:
                write(args.input/'scoring'/'reference_validation_failed.json', validated)
                raise RuntimeError('Reference scorer failed; refusing to grade generated outputs')
        write(reference_path, {'rows': validated, 'all_reference_tests_passed': True})
    if args.preflight_only: return
    observations = []
    for path in sorted((args.input/'groups').glob('*.json')):
        record = json.loads(path.read_text())
        case = record['case']
        if case['phase'] != 'quality' or case['requests'] != 128 or case['repeats'] != 1: raise RuntimeError('Nonformal quality group')
        for run in record['runs']:
            for identity, text, tokens in zip(record['identities'], run['texts'], run['outputs']):
                row = rows[(identity['dataset'], str(identity['source_id']))]
                if identity['prompt_sha256'] != row['prompt_sha256']: raise RuntimeError('Prompt identity mismatch')
                key = [identity['dataset'], str(identity['source_id']), case['seed'], run['method']]
                cache = args.input/'scoring'/'items'/(digest(key)[:24]+'.json')
                pred_hash = hashlib.sha256(text.encode()).hexdigest()
                if cache.exists():
                    observation = json.loads(cache.read_text())
                    if observation['prediction_sha256'] != pred_hash or observation['key'] != key: raise RuntimeError('Prediction drift')
                else:
                    ev = row['evaluation']
                    score = math_score(ev, text) if ev['kind'] == 'math' else run_program(code_program(ev, text), runtime)
                    if score['passed'] is None: raise RuntimeError('Sandbox failure is not a model failure')
                    observation = {'key': key, 'prediction_sha256': pred_hash, 'score': score,
                        'output_tokens': len(tokens), 'output_cap_reached': len(tokens) == case['max_new_tokens']}
                    write(cache, observation)
                observations.append(observation)
    bucket = defaultdict(dict)
    for obs in observations:
        d, identity, seed, method = obs['key']
        if (seed, method) in bucket[(d, identity)]: raise RuntimeError('Duplicate quality observation')
        bucket[(d, identity)][(seed, method)] = obs
    cells = []
    for d in DATASETS:
        clusters = [v for (dataset, _), v in bucket.items() if dataset == d and all((s,m) in v for s in QUALITY_SEEDS for m in ('ar','dflash','ddtree','dp'))]
        if not clusters: continue
        rates = {m: sum(v[(s,m)]['score']['passed'] for v in clusters for s in QUALITY_SEEDS)/(len(clusters)*len(QUALITY_SEEDS)) for m in ('ar','dflash','ddtree','dp')}
        comparisons = {}
        for comparator in ('ar','dflash','ddtree'):
            deltas = [sum(float(v[(s,'dp')]['score']['passed'])-float(v[(s,comparator)]['score']['passed']) for s in QUALITY_SEEDS)/len(QUALITY_SEEDS) for v in clusters]
            ci = paired_interval(deltas)
            ci['noninferiority_2pp_supported'] = len(clusters)==128 and ci['one_sided95_lower']>=-.02
            comparisons[comparator] = ci
        cells.append({'dataset': d, 'complete_prompt_clusters': len(clusters), 'rates': rates, 'dp_vs': comparisons,
            'cap_reached_fraction': {m: sum(v[(s,m)]['output_cap_reached'] for v in clusters for s in QUALITY_SEEDS)/(len(clusters)*len(QUALITY_SEEDS)) for m in rates}})
    complete = len(cells)==4 and all(c['complete_prompt_clusters']==128 for c in cells) and (args.input/'complete.json').exists()
    write(args.input/'QUALITY.json', {'partial': not complete, 'observations': len(observations), 'cells': cells,
        'strict_sequence_law_certified': False, 'statistical_caveat': 'Prespecified per-dataset/comparator one-sided intervals, not a simultaneous familywise guarantee'})
    print(json.dumps({'scored': len(observations), 'complete': complete}), flush=True)


if __name__ == '__main__': main()
