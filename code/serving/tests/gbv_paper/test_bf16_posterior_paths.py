from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_bf16_posterior_paths_h20 import (
    distribution_gap,
    detach_target_output_head,
    ancestor_path,
    check_packed_structure,
    parse_positive_csv,
    target_verification_precision_gate,
    target_head_precision_gate,
    timing_summary,
)


def test_distribution_gap_is_zero_for_identical_logits():
    logits = torch.tensor([2.0, 1.0, -3.0])
    result = distribution_gap(logits, logits.clone())
    assert result["max_logit_error"] == 0
    assert result["total_variation"] == 0
    assert result["top1_agreement"] is True


def test_distribution_gap_detects_top1_flip():
    result = distribution_gap(
        torch.tensor([2.0, 1.0, -3.0]),
        torch.tensor([1.0, 2.0, -3.0]),
    )
    assert result["max_logit_error"] == 1
    assert result["total_variation"] > 0
    assert result["top1_agreement"] is False
    assert result["reference_top1"] == 0
    assert result["candidate_top1"] == 1


def test_parse_positive_csv_contract():
    assert parse_positive_csv("1,12,46") == (1, 12, 46)
    assert parse_positive_csv("0,16", allow_zero=True) == (0, 16)
    with pytest.raises(Exception):
        parse_positive_csv("0,16")
    with pytest.raises(Exception):
        parse_positive_csv("1,1")


def test_target_verification_precision_gate_allows_mixed_target_and_draft():
    class Engine:
        target = torch.nn.Linear(2, 2).float()
        draft = torch.nn.Linear(2, 2).bfloat16()

    result = target_verification_precision_gate(Engine(), "float32")
    assert result["passed"] is True
    assert result["formal_benchmark_eligible"] is False
    assert result["observed_parameter_dtypes"] == {
        "target": ["torch.float32"],
        "draft": ["torch.bfloat16"],
    }


def test_target_verification_precision_gate_rejects_wrong_target_dtype():
    class Engine:
        target = torch.nn.Linear(2, 2).bfloat16()
        draft = torch.nn.Linear(2, 2).bfloat16()

    with pytest.raises(RuntimeError, match="Target verification dtype mismatch"):
        target_verification_precision_gate(Engine(), "float32")


def test_timing_summary_uses_median_and_rejects_nonpositive_samples():
    result = timing_summary([4.0, 1.0, 2.0])
    assert result["median_ms"] == 2.0
    assert result["minimum_ms"] == 1.0
    assert result["maximum_ms"] == 4.0
    with pytest.raises(ValueError, match="positive"):
        timing_summary([0.0])


def test_target_head_precision_gate_accepts_isolated_fp32_head():
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(2, 2).bfloat16()
            self.head = torch.nn.Linear(2, 3).float()

        def get_output_embeddings(self):
            return self.head

    class Engine:
        target = Target()

    result = target_head_precision_gate(Engine(), "bfloat16", "float32")
    assert result["passed"] is True
    assert result["observed_parameter_dtypes"] == {
        "body": ["torch.bfloat16"],
        "head": ["torch.float32"],
    }


def test_detach_target_output_head_preserves_tied_input_embedding_dtype():
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(3, 2).bfloat16()
            self.lm_head = torch.nn.Linear(2, 3, bias=False).bfloat16()
            self.lm_head.weight = self.model.embed_tokens.weight

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, head):
            self.lm_head = head

    class Engine:
        target = Target()

    original = Engine.target.get_input_embeddings().weight
    detach_target_output_head(Engine(), "float32")
    assert Engine.target.get_input_embeddings().weight is original
    assert Engine.target.get_input_embeddings().weight.dtype == torch.bfloat16
    assert Engine.target.get_output_embeddings().weight.dtype == torch.float32


def test_ancestor_path_rejects_cycles_and_preserves_branch_history():
    tree = SimpleNamespace(parents=[-1, 0, 0, 1])
    assert ancestor_path(tree, 3) == [0, 1, 3]
    assert ancestor_path(tree, 2) == [0, 2]
    with pytest.raises(ValueError, match="precede"):
        ancestor_path(SimpleNamespace(parents=[-1, 1]), 1)


@pytest.mark.parametrize("corruption", [None, "mask", "positions", "kv"])
def test_packed_structure_checks_unequal_request_prefixes(corruption):
    def cache(values):
        tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 1)
        return SimpleNamespace(layers=[SimpleNamespace(keys=tensor, values=tensor.clone())])

    tree = SimpleNamespace(tokens=[8], parents=[-1, 0], depths=[0, 1])
    items = [
        {"tree": tree, "root": 7, "prefix_length": 2, "cache": cache([10, 11])},
        {"tree": tree, "root": 9, "prefix_length": 1, "cache": cache([12])},
    ]
    packed = cache([10, 11, 12])
    mask = torch.full((1, 1, 4, 7), -torch.inf)
    for row, visible in enumerate(([0, 1, 3], [0, 1, 3, 4], [2, 5], [2, 5, 6])):
        mask[0, 0, row, list(visible)] = 0
    positions = torch.tensor([[2, 3, 1, 2]])
    if corruption == "mask":
        mask[0, 0, 0, 2] = 0
    elif corruption == "positions":
        positions[0, 3] += 1
    elif corruption == "kv":
        packed.layers[0].keys[0, 0, 2, 0] = 99

    def check():
        check_packed_structure(items, packed, [2, 1], [0, 2], 3,
                               torch.tensor([[7, 8, 9, 8]]), positions, mask, [0, 2])
    if corruption is None:
        check()
    else:
        with pytest.raises(RuntimeError):
            check()
