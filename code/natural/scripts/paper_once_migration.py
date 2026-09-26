"""Explicit same-host protocol migration, never select the fastest repeat."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ALLOWED_CHANGED = {'scripts/paper_natural_plan.py', 'scripts/run_natural_paper_phase.py'}
ALLOWED_ADDED = {'scripts/paper_once_migration.py', 'scripts/switch_natural_paper_once.py'}


def normalized_main(source):
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    retained = []
    for statement in main.body:
        if (isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call)
                and statement.value.args and isinstance(statement.value.args[0],ast.Constant)
                and statement.value.args[0].value=='--reuse-directory'):
            continue
        if isinstance(statement,ast.If) and ast.unparse(statement.test)=='args.reuse_directory is not None':
            continue
        retained.append(statement)
    main.body = retained
    return ast.dump(main,include_attributes=False)


def clean_hashes(hashes):
    return {p: h for p, h in hashes.items() if not Path(p).name.startswith('._')}


def validate_compatibility(old, new):
    ignored = {'source_hashes', 'cases'}
    if {k:v for k,v in old.items() if k not in ignored} != {k:v for k,v in new.items() if k not in ignored}:
        raise RuntimeError('Model/data/calibration/precision/selection protocol mismatch')
    a, b = clean_hashes(old['source_hashes']), clean_hashes(new['source_hashes'])
    if set(a)-set(b) or set(b)-set(a)-ALLOWED_ADDED:
        raise RuntimeError('Unexpected source bundle membership drift')
    changed = {p for p in a if a[p] != b[p]}
    if changed-ALLOWED_CHANGED:
        raise RuntimeError('Decoder/baseline/scorer code changed; reuse prohibited')
    old_cases = {json.dumps({k:v for k,v in c.items() if k!='repeats'}, sort_keys=True): c for c in old['cases']}
    if len(old_cases) != len(old['cases']): raise RuntimeError('Duplicate old case')
    for case in new['cases']:
        key = json.dumps({k:v for k,v in case.items() if k!='repeats'}, sort_keys=True)
        if key not in old_cases or case['repeats']!=1 or old_cases[key]['repeats'] not in (1,3):
            raise RuntimeError('Only3-to1 timing repetition migration is allowed')
    if len(new['cases']) != len(old_cases): raise RuntimeError('Experiment conditions were dropped/added')
    return old_cases


def import_repetition_zero(old_directory, output, contract, cpu, digest, write):
    old_directory, output = Path(old_directory), Path(output)
    if not (old_directory/'contract.json').exists(): return 0
    # Reuse is intentionally only from the known same-host original suite.
    if not str(old_directory.resolve()).startswith('/root/dp-paper-natural-results-20260917/'):
        raise RuntimeError('Unexpected old result root; same-host reuse required')
    old = json.loads((old_directory/'contract.json').read_text())
    old_cases = validate_compatibility(old, contract)
    old_runner = Path('/root/dp-paper-natural-suite-20260917/scripts/run_natural_paper_phase.py')
    new_runner = Path(__file__).with_name('run_natural_paper_phase.py')
    if hashlib.sha256(old_runner.read_bytes()).hexdigest()!=old['source_hashes']['scripts/run_natural_paper_phase.py']:
        raise RuntimeError('Original driver no longer matches measured source')
    if normalized_main(old_runner.read_text()) != normalized_main(new_runner.read_text()):
        raise RuntimeError('Execution/tokenization/seed/timing code changed beyond the import hook')
    old_ast,new_ast = ast.parse(old_runner.read_text()),ast.parse(new_runner.read_text())
    for name in ('settings_for','load_natural'):
        before = next(n for n in old_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
        after = next(n for n in new_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if ast.dump(before,include_attributes=False)!=ast.dump(after,include_attributes=False):
            raise RuntimeError('Algorithm settings or natural selection code changed')
    imported = 0
    for case in contract['cases']:
        key = json.dumps({k:v for k,v in case.items() if k!='repeats'}, sort_keys=True)
        previous = old_cases[key]
        path = old_directory/'groups'/(digest(previous)[:24]+'.json')
        target = output/'groups'/(digest(case)[:24]+'.json')
        if not path.exists() or target.exists(): continue
        original = json.loads(path.read_text())
        expected = {(m,r) for m in previous['methods'] for r in range(previous['repeats'])}
        actual = {(r['method'],r['repeat']) for r in original['runs']}
        if original['case']!=previous or original['source_hashes']!=old['source_hashes'] or actual!=expected or len(original['runs'])!=len(expected):
            raise RuntimeError('Original completed group is incomplete or inconsistent')
        _, identities = cpu[(case['dataset'],case['requests'])]
        identities = identities[case['first']:case['first']+case['concurrency']]
        seeds = [case['seed']*1000003+int(hashlib.sha256((i['dataset']+':'+str(i['source_id'])).encode()).hexdigest()[:12],16)%1000000007 for i in identities]
        if original['identities']!=identities or original['request_seeds']!=seeds:
            raise RuntimeError('Original natural prompt identities or Target seeds differ')
        runs = [r for r in original['runs'] if r['repeat']==0]
        if {r['method'] for r in runs} != set(case['methods']): raise RuntimeError('Missing repeat0 method')
        for method in case['methods']:
            if len({r['output_sha256'] for r in original['runs'] if r['method']==method})!=1:
                raise RuntimeError('Original repeats had divergent outputs')
        for run in runs:
            if run['summary']['output_tokens']!=sum(map(len,run['outputs'])) or run['output_sha256']!=digest(run['outputs']):
                raise RuntimeError('Original token accounting/hash mismatch')
        record = copy.deepcopy(original)
        record.update(case=case, source_hashes=contract['source_hashes'], runs=runs, orders=original['orders'][:1],
            measurement_source_hashes=original['source_hashes'], execution_origin='compatible_old_repeat0',
            provenance={'original_file':str(path),'original_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'original_case':previous,'selection_rule':'Always repeat0; never fastest or favorable output',
                        'algorithm_sources_verified_identical':True,'same_host_original_results_preserved':True})
        write(target,record); imported+=1
    if (old_directory/'environment.json').exists():
        write(output/'imported_environment.json',json.loads((old_directory/'environment.json').read_text()))
    write(output/'migration.json', {'imported_now':imported,'original_directory':str(old_directory),
        'old_contract_sha256':hashlib.sha256((old_directory/'contract.json').read_bytes()).hexdigest(),
        'unchanged_algorithm_verified':True,'retained_repeat':0,'original_three_repeat_results_preserved':True})
    return imported
