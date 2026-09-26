"""Frozen, auditable natural-task paper suite. No synthetic context workloads."""
DATASETS = ('gsm8k', 'math500', 'humaneval', 'mbpp_sanitized')
SEEDS = (17, 29, 43)
QUALITY_SEEDS = (17, 29, 43, 67, 101)
PRIMARY = '/root/dp-main-natural-cache-20260917'


def row_cap(method):
    if method == 'ar':
        return 0
    if method in ('ddtree', 'fixed_large') or method.startswith('native_ddtree'):
        return 46
    if method == 'dflash' or method.startswith('native_dflash'):
        return 16
    return 12


def methods_for(names, concurrency, budget):
    supported = [m for m in names if m != 'ar' and concurrency * row_cap(m) <= budget]
    return ['ar'] + supported if supported else []


def cases_for(phase):
    cases = []
    def add(datasets, budgets, cs, caps, names, requests=128, seeds=SEEDS, repeats=1, temperatures=(1,)):
        for dataset in datasets:
            for budget in budgets:
                for c in cs:
                    eligible = methods_for(names, c, budget)
                    if not eligible:
                        continue
                    for cap in caps:
                        for temperature in temperatures:
                            for seed in seeds:
                                cases.append({'phase': phase, 'dataset': dataset, 'budget': budget,
                                    'concurrency': c, 'max_new_tokens': cap, 'temperature': temperature,
                                    'seed': seed, 'requests': requests if requests != 'all' else 'all',
                                    'repeats': repeats, 'methods': eligible,
                                    'excluded_methods': [m for m in names if m not in eligible]})
    cached = ('ar', 'dflash', 'ddtree', 'dp')
    if phase == 'main128':
        add(DATASETS, (193,), (1, 4, 8, 16), (256,), cached)
    elif phase == 'main_extension':
        add(DATASETS, (193,), (16,), (256,), cached)
    elif phase == 'budget':
        add(DATASETS, (96, 384), (1, 4, 8, 16, 32), (256,), cached)
    elif phase == 'long_output':
        add(DATASETS, (193,), (1, 4, 8), (512, 1024), cached, seeds=(17, 29))
    elif phase == 'ablation':
        add(('gsm8k', 'math500'), (193,), (1, 4, 8, 16), (256,),
            ('ar', 'dp', 'equal', 'greedy', 'random', 'fixed_small', 'fixed_large',
             'no_weight', 'constant_cost', 'linear_cost', 'no_active_gate', 'no_tier_fit', 'age_admission'))
    elif phase == 'quality':
        add(DATASETS, (193,), (4,), (1024,), cached, requests=128, seeds=QUALITY_SEEDS, repeats=1)
    elif phase == 'correctness':
        add(DATASETS, (96, 193, 384), (1, 4, 8, 16, 32), (64,), cached,
            requests=32, seeds=(911,), repeats=1, temperatures=(0, 1))
    elif phase == 'native_c1':
        add(DATASETS, (193,), (1,), (256,),
            ('ar', 'native_dflash', 'native_ddtree', 'dflash', 'ddtree', 'dp'),
            requests=128, seeds=(17, 29))
    elif phase == 'heterogeneous':
        add(('mixed_natural',), (193, 384), (4, 8, 16, 32), (256,), cached, requests=128)
    else:
        raise ValueError(phase)
    return cases


PHASES = ('main128', 'quality', 'correctness', 'native_c1', 'budget',
          'ablation', 'long_output', 'heterogeneous')


def manifest():
    return {'version': 2, 'datasets': DATASETS, 'model_targets': ['qwen3_4b', 'qwen3_8b'],
        'repeat_policy': 'One timing execution per method/seed/group; seed sets, workloads, and quality five-seed protocol unchanged',
        'reuse_policy': 'Compatible completed old groups reuse only prespecified repeat0; original results and source hashes retained with provenance',
        'data_policy': 'First128 prepared natural tasks per dataset for formal timing/quality; no filler/truncation/length filtering. Correctness diagnostic uses32.',
        'previous32_pilot_in_progress': PRIMARY, 'phases': {p: cases_for(p) for p in PHASES},
        'temperature_policy': 'T=1 performance and quality; T=0 only diagnostic correctness checks',
        'native_policy': 'Pinned official code C=1; not an assertion of native high-concurrency serving results',
        'quality_policy': 'math_verify and safely isolated code tests; paired prompt-cluster confidence intervals, prespecified2pp margin',
        'sampling_policy': 'Existing prepared subsets, previously used in this project; not certified unseen data',
        'hardware_scope': 'Two H20s, one per target model; no cross-hardware generalization',
        'measurement_policy': 'Matched synchronized full decode wall including prefill; separate boundary maintenance metric; speedups versus matched AR',
        'statistical_policy': 'Resample independent request-identity groups; timing repeats/seeds are not independent prompts; no guarantee of positive outcomes',
        'serving_scope': 'Closed request batches only, logical token-ready latency; no network/open-loop SLO claim',
        'not_yet_certified': ['Strict BF16 AR sequence-law equality', 'Native optimized high-concurrency service comparison',
            'Task noninferiority before scoring', 'Generalization beyond the two available models/H20 hardware']}
