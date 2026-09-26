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


def test_fixed_tier_tuning_uses_only_feasible_singleton_tiers():
    assert row_cap('fixed_tier_11') == 12
    assert row_cap('fixed_tier_23') == 24
    assert row_cap('fixed_tier_45') == 46
    by_shape = {(c['budget'], c['concurrency']): c['methods']
                for c in cases_for('fixed_tier_tuning')}
    assert by_shape[(193, 4)] == [
        'ar', 'dp', 'fixed_tier_11', 'fixed_tier_23', 'fixed_tier_45']
    assert by_shape[(193, 8)] == [
        'ar', 'dp', 'fixed_tier_11', 'fixed_tier_23']
    assert by_shape[(384, 8)] == [
        'ar', 'dp', 'fixed_tier_11', 'fixed_tier_23', 'fixed_tier_45']


def test_complex_tier_matrix_uses_mixed_requests_and_three_allocators():
    cases = cases_for('complex_tiers')
    assert {(c['budget'], c['concurrency']) for c in cases} == {
        (193, 4), (193, 8), (384, 8)}
    assert all(c['dataset'] == 'mixed_natural' and c['requests'] == 128
               for c in cases)
    assert all(c['methods'] == [
        'ar', 'dp_dense', 'greedy_dense', 'equal_dense'] for c in cases)


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
