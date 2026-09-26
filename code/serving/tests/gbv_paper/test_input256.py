import pytest
import torch

from paper_input256 import pad_natural_ids


def test_padding_preserves_header_and_entire_task_and_generation_suffix():
    ids = torch.tensor([[10, 11, 12, 20, 21, 99]])
    header = ids[:, :3].clone()
    result = pad_natural_ids(ids, header, 7, target=10)
    assert result.tolist() == [[10, 11, 12, 7, 7, 7, 7, 20, 21, 99]]


def test_already_target_length_is_not_modified():
    ids = torch.arange(256).reshape(1, 256)
    assert torch.equal(pad_natural_ids(ids, ids[:, :3], 7), ids)


def test_long_tasks_are_not_silently_truncated():
    ids = torch.arange(257).reshape(1, 257)
    with pytest.raises(ValueError, match='exceeds target'):
        pad_natural_ids(ids, ids[:, :3], 7)


def test_wrong_chat_header_is_rejected():
    with pytest.raises(ValueError, match='boundary'):
        pad_natural_ids(torch.tensor([[1, 2, 3]]), torch.tensor([[8]]), 7)
