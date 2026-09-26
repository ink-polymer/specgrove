import pytest

from run_natural_paper_phase import precision_config, precision_contract


def test_default_precision_contract_is_unchanged():
    cfg = {'dtype': 'bfloat16', 'allow_tf32': False}
    result = precision_config(cfg, 'quality', None)
    assert result == cfg and result is not cfg
    assert precision_contract(None) == {}


@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_experimental_precision_keeps_original_config_and_labels_reference(dtype):
    cfg = {'dtype': 'bfloat16', 'allow_tf32': True}
    result = precision_config(cfg, 'quality', dtype)
    assert cfg == {'dtype': 'bfloat16', 'allow_tf32': True}
    assert result == {'dtype': dtype, 'allow_tf32': False}
    contract = precision_contract(dtype)
    assert contract['performance_eligible'] is False
    assert contract['formal_BF16_benchmark_eligible'] is False
    assert contract['strict_sequence_law_certified'] is False
    assert 'original BF16 AR' in contract['ar_reference']


def test_experimental_precision_rejects_performance_phases_and_unknown_dtype():
    with pytest.raises(ValueError, match='quality/correctness'):
        precision_config({}, 'main128', 'float32')
    with pytest.raises(ValueError, match='Unsupported'):
        precision_config({}, 'quality', 'float64')
