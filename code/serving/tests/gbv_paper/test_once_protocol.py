import copy
import importlib.util
from pathlib import Path
import pytest
from paper_natural_plan import PHASES,cases_for
from paper_once_migration import validate_compatibility,normalized_main


def old_plan():
    old=Path(__file__).resolve().parents[3]/'dp-paper-natural-suite-20260917/scripts/paper_natural_plan.py'
    spec=importlib.util.spec_from_file_location('old_three_repeat_plan',old)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_all_conditions_and_seed_sets_are_preserved():
    previous=old_plan()
    assert PHASES==previous.PHASES
    for phase in PHASES:
        before=previous.cases_for(phase)
        after=cases_for(phase)
        assert [{k:v for k,v in c.items() if k!='repeats'} for c in before]==[{k:v for k,v in c.items() if k!='repeats'} for c in after]
        assert all(c['repeats']==1 for c in after)
    assert len(cases_for('quality'))==20


def contracts():
    case=cases_for('main128')[0]
    new={'model':{'target':'pinned'},'cases':[case], 'source_hashes':{'src/gbv_experiments/continuous_tree_block_decode.py':'core','scripts/run_natural_paper_phase.py':'new','scripts/paper_natural_plan.py':'new','scripts/paper_once_migration.py':'added'},'data_hashes':{'gsm8k':'data'}}
    old=copy.deepcopy(new);old['cases'][0]['repeats']=3
    old['source_hashes'].pop('scripts/paper_once_migration.py')
    old['source_hashes']['scripts/run_natural_paper_phase.py']='old'
    old['source_hashes']['scripts/paper_natural_plan.py']='old'
    return old,new


def test_compatible_migration_only_changes_repetition_count():
    old,new=contracts()
    assert len(validate_compatibility(old,new))==1


@pytest.mark.parametrize('drift',['decoder','data','model','seed','concurrency','methods'])
def test_real_protocol_drift_is_rejected(drift):
    old,new=contracts()
    if drift=='decoder':new['source_hashes']['src/gbv_experiments/continuous_tree_block_decode.py']='changed'
    elif drift=='data':new['data_hashes']['gsm8k']='changed'
    elif drift=='model':new['model']['target']='changed'
    elif drift=='methods':new['cases'][0]['methods']=['ar','dp']
    else:new['cases'][0][drift]+=1
    with pytest.raises(RuntimeError):validate_compatibility(old,new)


def test_ast_proves_execution_and_seed_code_unchanged():
    old=Path(__file__).resolve().parents[3]/'dp-paper-natural-suite-20260917/scripts/run_natural_paper_phase.py'
    new=Path(__file__).resolve().parents[2]/'scripts/run_natural_paper_phase.py'
    assert normalized_main(old.read_text())==normalized_main(new.read_text())
