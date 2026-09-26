"""Eligibility boundary checks; preserve original contracts and decoder."""
import sys
import hashlib
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from paper_single_wave_policy import methods_for_case, method_capacity, partition_cases, exclusion_records, policy_contract
from run_dp_paper_study import cases_for, source_hashes
from run_dp_paper_single_wave import cases_for as amended_cases, source_hashes as amended_hashes
from rerun_audited_global_tree_matrix_h20 import matched_specs, DD, DF, OURS

def case(c,budget=193,phase='main'):
    return {'phase':phase,'concurrency':c,'budget':budget}

@pytest.mark.parametrize('budget,expected',[(96,(2,6,8)),(193,(4,12,16)),(384,(8,24,32))])
def test_capacities(budget,expected):
    assert tuple(method_capacity(m,budget) for m in ['ddtree','dflash','dp'])==expected

@pytest.mark.parametrize('c,expected',[(1,['ar','dflash','ddtree','dp']),
    (4,['ar','dflash','ddtree','dp']),(8,['ar','dflash','dp']),
    (12,['ar','dflash','dp']),(13,['ar','dp']),(16,['ar','dp']),(17,[]),(32,[])])
def test_budget193_method_filter(c,expected):
    assert methods_for_case(case(c))==expected

@pytest.mark.parametrize('budget',[96,193,384])
def test_supported_requests_fit_and_above_boundary_is_excluded(budget):
    for method in ['ddtree','dflash','dp']:
        cap=method_capacity(method,budget)
        assert method in methods_for_case(case(cap,budget))
        assert method not in methods_for_case(case(cap+1,budget))
    for c in range(1,40):
        selected=methods_for_case(case(c,budget))
        assert not selected or 'ar' in selected
        assert not selected or any(m!='ar' for m in selected)
        assert all(c<=method_capacity(m,budget) for m in selected if m!='ar')

def test_ablation_never_adds_ar_and_drops_all_at_c32():
    assert 'ar' not in methods_for_case(case(8,phase='ablation'))
    assert len(methods_for_case(case(16,phase='ablation')))==10
    assert methods_for_case(case(32,phase='ablation'))==[]

def test_actual_decode_row_caps_match_policy():
    curve=((12,1.0),(46,1.1),(96,1.2),(193,1.3),(384,1.4))
    for temperature in [0,1]:
        specs=matched_specs(temperature,curve)
        for method,key in [('ddtree',DD),('dflash',DF),('dp',OURS)]:
            for budget in [96,193,384]:
                assert budget//specs[key]['decode']['row_cap']==method_capacity(method,budget)

def test_partition_retains_original_case_identities_and_all_exclusions():
    for phase in ['pilot','main','budget','ablation','temperature0','long_output','context','quality_math']:
        original=cases_for(phase)
        assert amended_cases(phase)==original
        included,excluded=partition_cases(original)
        assert len(included)+len(excluded)==len(original)
        assert all(methods_for_case(c) for c in included)
        assert all(not methods_for_case(c) for c in excluded)
        assert policy_contract(original)['changes_decoder_or_model'] is False
        for c in original:
            selected=methods_for_case(c)
            records=exclusion_records(c)
            assert not set(selected)&{r['method'] for r in records}

def test_original_driver_and_core_unchanged_and_new_resume_manifest_distinct():
    assert hashlib.sha256((ROOT/'scripts/run_dp_paper_study.py').read_bytes()).hexdigest()=='02f9f422293f09e7a1ff0cbc38a61b52365cd9cea6c74ac8f6da7b83683325a3'
    assert hashlib.sha256((ROOT/'src/gbv_experiments/continuous_tree_block_decode.py').read_bytes()).hexdigest()=='3f132a41b659d1306cbfc3fbb348c0f89284d097d69851f0f73d8a6e4518587f'
    assert source_hashes()!=amended_hashes()
    assert 'scripts/paper_single_wave_policy.py' in amended_hashes()

def test_amended_summary_marks_skips_and_preserves_ar_normalization(tmp_path,monkeypatch):
    import summarize_dp_paper_single_wave as summary
    measured=dict(case(16),dataset='gsm8k',temperature=1,max_new_tokens=256,
                  prompt_tokens=None,requests=16,repeats=1,seed=17)
    excluded=dict(measured,concurrency=32,requests=32)
    path=tmp_path/'main/qwen3_4b/groups/one.json'
    path.parent.mkdir(parents=True)
    record={'identity':{'case':measured},'request_ids':['one'],
            'request_seeds':[17],'prompt_sha256':['hash'],
            'runs':[{'method':m,'repeat':0,'summary':{'output_tokens':tokens,'wall_ms':100}}
                    for m,tokens in [('ar',20),('dp',40)]]}
    path.write_text(json.dumps(record))
    (path.parent.parent/'contract.json').write_text(json.dumps(
        {'cases':[measured],'single_wave_policy':policy_contract([measured,excluded])}))
    monkeypatch.setattr(sys,'argv',['summary','--directory',str(tmp_path)])
    summary.main()
    result=json.loads((tmp_path/'RESULTS.json').read_text())
    assert {c['method'] for c in result['cells']}=={'ar','dp'}
    assert next(c for c in result['cells'] if c['method']=='dp')['ar_comparison']['speedup_vs_ar']==2
    assert all(c['cell_complete'] for c in result['cells'])
    excluded_rows=result['policy_exclusions']
    assert any(x['case']['concurrency']==32 and len(x['excluded_methods'])==4 for x in excluded_rows)
    assert 'Unsupported configurations: skipped by policy' in (tmp_path/'RESULTS_T1_R193.md').read_text()
