from grade_natural_paper_quality import clean_code, code_program, paired_interval
from run_natural_paper_phase import settings_for


def test_only_text_extraction_no_semantic_repair():
    assert clean_code('<think>bad</think>```python\ndef f():\n    return 3\n```') == 'def f():\n    return 3'
    assert clean_code('def f():\n    return 4') == 'def f():\n    return 4'


def test_reference_humaneval_uses_original_prefix_and_test():
    e = {'kind':'humaneval', 'prompt':'def f(x):\n', 'reference_code':'    return x\n', 'entry_point':'f', 'test':'def check(f):\n    assert f(3)==3'}
    program = code_program(e, '', reference=True)
    exec(program, {})
    assert 'check(f)' in program
    assert 'return 9' in code_program(e, '```python\ndef f(x):\n    return 9\n```')


def test_mbpp_keeps_all_original_tests():
    e = {'kind':'mbpp','setup':'','reference_code':'def f(x): return x','tests':['assert f(1)==1','assert f(2)==2'],'challenge_tests':['assert f(3)==3']}
    p = code_program(e, '', reference=True)
    exec(p, {})
    assert p.count('assert') == 3


def test_paired_cluster_intervals_are_in_percentage_point_units():
    ci = paired_interval([0.0]*128)
    assert ci['one_sided95_lower']==0 and ci['prompt_clusters']==128


def test_ablation_settings_change_only_named_controls():
    curve = ((12,1.0),(96,2.0),(384,5.0))
    _, base = settings_for('dp', 1, curve)
    assert base['reuse_request_draft_cache']
    _, small = settings_for('fixed_small', 1, curve)
    assert small['global_tree_budget_options'] == (11,)
    _, no_weight = settings_for('no_weight', 1, curve)
    assert no_weight['global_tree_criticality_weight'] == 0
    _, no_gate = settings_for('no_active_gate', 1, curve)
    assert no_gate['global_tree_expansion_active_limit'] is None
    _, no_fit = settings_for('no_tier_fit', 1, curve)
    assert no_fit['global_tree_require_tier_fit'] is False
