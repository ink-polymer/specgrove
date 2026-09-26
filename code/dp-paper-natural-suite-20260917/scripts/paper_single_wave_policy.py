"""Single Target verification-wave eligibility; no decoder/model changes."""
BASE_METHODS = ('ar', 'dflash', 'ddtree', 'dp')
ABLATIONS = ('dp', 'equal', 'greedy', 'random', 'no_weight', 'constant_cost',
             'linear_cost', 'no_tier_fit', 'no_active_gate', 'age_admission')
POLICY_VERSION = 'single-target-wave-20260917-v1'

def row_cap(method):
    if method == 'ar':
        return None
    if method == 'ddtree':
        return 46
    if method == 'dflash':
        return 16
    if method in ABLATIONS:
        return 12
    raise ValueError('Unknown method: ' + method)

def method_capacity(method, budget):
    if budget < 1:
        raise ValueError('Positive row budget required')
    cap = row_cap(method)
    return None if cap is None else budget // cap

def original_methods(case):
    return list(ABLATIONS if case['phase'] == 'ablation' else BASE_METHODS)

def methods_for_case(case):
    concurrency = case['concurrency']
    if concurrency < 1:
        raise ValueError('Positive concurrency required')
    methods = original_methods(case)
    supported = [m for m in methods if m != 'ar'
                 and concurrency <= method_capacity(m, case['budget'])]
    if not supported:
        return []
    return [m for m in methods if m == 'ar' or m in supported]

def exclusion_records(case):
    eligible = methods_for_case(case)
    out = []
    for method in original_methods(case):
        if method in eligible:
            continue
        capacity = method_capacity(method, case['budget'])
        reason = ('no_eligible_speculative_method; AR-only cell omitted'
                  if method == 'ar' else 'configured_concurrency_exceeds_single_target_wave_capacity')
        out.append({'method': method, 'configured_concurrency': case['concurrency'],
                    'row_cap': row_cap(method), 'single_wave_capacity': capacity,
                    'reason': reason})
    return out

def partition_cases(cases):
    included = [case for case in cases if methods_for_case(case)]
    excluded = [case for case in cases if not methods_for_case(case)]
    return included, excluded

def policy_contract(cases):
    return {'version': POLICY_VERSION,
            'amendment': 'User requested skipping unsupported nominal parallelism on 2026-09-17',
            'definition': 'All initially active requests must fit one Target verification wave at the method minimum row cap and existing slot capacity',
            'baseline': 'matched autoregressive; no AR-only cells',
            'changes_decoder_or_model': False,
            'changes_draft_cache_or_verifier': False,
            'row_caps': {'ddtree': 46, 'dflash': 16, 'dp_and_ablation': 12},
            'cases': [{'case': case, 'eligible_methods': methods_for_case(case),
                       'excluded_methods': exclusion_records(case)} for case in cases]}
