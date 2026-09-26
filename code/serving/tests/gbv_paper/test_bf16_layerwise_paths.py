from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_bf16_layerwise_paths_h20 import hidden_gap, parse_cases


def test_parse_cases_contract():
    assert parse_cases("103:19,108:2") == ((103, 19), (108, 2))
    with pytest.raises(Exception):
        parse_cases("103")
    with pytest.raises(Exception):
        parse_cases("103:19,103:19")


def test_hidden_gap_detects_first_order_difference():
    result = hidden_gap(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]))
    assert result["byte_equal"] is False
    assert result["max_error"] == 0.5
    assert result["relative_l2_error"] > 0


def test_hidden_gap_is_zero_for_identical_vectors():
    result = hidden_gap(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert result["byte_equal"] is True
    assert result["max_error"] == 0
