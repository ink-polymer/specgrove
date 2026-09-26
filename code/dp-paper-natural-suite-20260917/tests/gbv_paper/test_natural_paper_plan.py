from paper_natural_plan import PHASES, cases_for, methods_for, row_cap


def test_no_overcapacity_method_or_ar_only_cell_is_scheduled():
    for phase in PHASES:
        for case in cases_for(phase):
            assert case['methods'][0] == 'ar'
            assert len(case['methods']) > 1
            for method in case['methods'][1:]:
                assert case['concurrency'] * row_cap(method) <= case['budget']


def test_high_concurrency_is_only_used_where_budget_can_fit():
    assert methods_for(('ar', 'dflash', 'ddtree', 'dp'), 32, 193) == []
    assert methods_for(('ar', 'dflash', 'ddtree', 'dp'), 32, 384) == ['ar', 'dp']
    assert methods_for(('ar', 'dflash', 'ddtree', 'dp'), 8, 193) == ['ar', 'dflash', 'dp']


def test_quality_uses128_natural_questions_and_five_seeds_without_timing_duplicates():
    cases = cases_for('quality')
    assert len(cases) == 20
    assert all(c['requests'] == 128 and c['repeats'] == 1 and c['temperature'] == 1 for c in cases)
    assert all(c['methods'] == ['ar', 'dflash', 'ddtree', 'dp'] for c in cases)


def test_nonzero_temperature_performance_and_no_synthetic_workload():
    for phase in PHASES:
        for case in cases_for(phase):
            assert 'synthetic' not in case['dataset']
            if phase != 'correctness':
                assert case['temperature'] == 1


def test_formal_request_counts_are128_and_correctness_is_diagnostic32():
    for phase in PHASES:
        assert all(c['requests'] == (32 if phase == 'correctness' else 128) for c in cases_for(phase))
