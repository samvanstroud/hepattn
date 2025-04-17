import pytest
import torch

from hepattn.utils.masks import indices_from_mask, mask_from_indices


@pytest.fixture
def indices_1d():
    return torch.tensor([0, 1, 1])


@pytest.fixture
def indices_2d():
    return torch.tensor([[0, 1, 1], [0, 0, 1]])


@pytest.fixture
def mask_2d():
    return torch.tensor([[True, False, False], [False, True, True]])


@pytest.fixture
def mask_3d():
    return torch.tensor([
        [[True, False, False], [False, True, True]],
        [[True, True, False], [False, False, True]],
    ])


def test_mask_from_indices_1d(indices_1d, mask_2d):
    sparse_from_1d = mask_from_indices(indices_1d)
    assert torch.all(sparse_from_1d == mask_2d)


def test_mask_from_indices_2d(indices_2d, mask_3d):
    sparse_from_2d = mask_from_indices(indices_2d)
    assert torch.all(sparse_from_2d == mask_3d)


def test_mask_from_negative_indices():
    indices = torch.tensor([-1, 0, 0, 1, 2])
    out = mask_from_indices(indices, num_masks=3)
    expected = torch.tensor([
        [False, True, True, False, False],
        [False, False, False, True, False],
        [False, False, False, False, True],
    ])
    assert torch.all(out == expected)

    indices = torch.tensor([[-1, -1, -1], [0, 0, 1]])
    out = mask_from_indices(indices, num_masks=3)
    expected = torch.tensor([
        [[False, False, False], [False, False, False], [False, False, False]],
        [[True, True, False], [False, False, True], [False, False, False]],
    ])
    assert torch.all(out == expected)


def test_indices_from_mask_2d(mask_2d, indices_1d):
    indices = indices_from_mask(mask_2d)
    assert torch.all(indices == indices_1d)


def test_indices_from_mask_3d(mask_3d, indices_2d):
    indices = indices_from_mask(mask_3d)
    assert torch.all(indices == indices_2d)


def test_indices_from_mask_empty():
    mask = torch.tensor([
        [False, False],
        [False, False],
        [False, False],
    ])
    indices = indices_from_mask(mask)
    print(mask)
    print(indices)
    assert torch.all(indices == torch.tensor([-1, -1]))


def test_indices_from_mask_with_non_mask_item():
    mask = torch.tensor([
        [True, True, True, False, False, False],
        [False, False, False, True, True, False],
    ])
    indices = indices_from_mask(mask)
    assert torch.all(indices == torch.tensor([0, 0, 0, 1, 1, -1]))
