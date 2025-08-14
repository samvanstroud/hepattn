import matplotlib.pyplot as plt
import pytest
import torch

from hepattn.utils.tensor_utils import pad_to_size

plt.rcParams["figure.dpi"] = 300

torch.manual_seed(42)


def test_all_empty():
    x = torch.empty((0, 0, 0))
    d = (0, 0, 0)
    padded = pad_to_size(x, d, pad_value=0)
    assert torch.equal(padded, x)
    assert padded.shape == torch.Size(d)


def test_empty_after_nopad():
    x = torch.empty((1, 39, 0))
    d = (1, 39, 0)
    padded = pad_to_size(x, d, pad_value=0)
    assert padded.shape == torch.Size(d)


def test_empty_after_pad():
    x = torch.empty((1, 39, 0))
    d = (1, 96, 0)
    padded = pad_to_size(x, d, pad_value=0)
    assert padded.shape == torch.Size(d)


def test_no_padding_needed():
    x = torch.tensor([[1, 2], [3, 4]])
    d = (2, 2)
    padded = pad_to_size(x, d, pad_value=0)
    assert torch.equal(padded, x)
    assert padded.shape == torch.Size(d)


def test_padding_2d_tensor():
    x = torch.tensor([[1, 2], [3, 4]])
    d = (3, 4)
    padded = pad_to_size(x, d, pad_value=0)
    expected = torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0]])
    assert torch.equal(padded, expected)
    assert padded.shape == torch.Size(d)


def test_padding_1d_tensor():
    x = torch.tensor([1, 2, 3])
    d = (5,)
    padded = pad_to_size(x, d, pad_value=-1)
    expected = torch.tensor([1, 2, 3, -1, -1])
    assert torch.equal(padded, expected)
    assert padded.shape == torch.Size(d)


def test_padding_3d_tensor():
    x = torch.ones((2, 3, 1))
    d = (3, 4, 2)
    padded = pad_to_size(x, d, pad_value=0)
    assert padded.shape == torch.Size(d)
    assert torch.all(padded[:2, :3, :1] == 1)
    assert torch.all(padded[2:, :, :] == 0)
    assert torch.all(padded[:, 3:, :] == 0)
    assert torch.all(padded[:, :, 1:] == 0)


def test_error_on_dimension_mismatch():
    x = torch.zeros((2, 2))
    d = (2, 2, 2)
    with pytest.raises(ValueError):
        pad_to_size(x, d, pad_value=0)


def test_error_on_negative_padding():
    x = torch.ones((4,))
    d = (3,)
    with pytest.raises(ValueError):
        pad_to_size(x, d, pad_value=0)


def test_padding_from_zero_last_dim():
    x = torch.empty((2, 0))
    d = (2, 3)
    padded = pad_to_size(x, d, pad_value=7)
    expected = torch.full((2, 3), 7)
    assert torch.equal(padded, expected)
    assert padded.shape == torch.Size(d)


def test_padding_no_pad_opt():
    x = torch.ones((2, 3, 5))
    d = (3, 4, -1)
    padded = pad_to_size(x, d, pad_value=0)
    assert padded.shape == (3, 4, 5)
