import pytest
import torch

from hepattn.utils.kmeans_ca import KMeansCrossAttention


def test_requires_values():
    module = KMeansCrossAttention(dim=2)
    q = torch.zeros(1, 1, 2)
    k = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="requires v"):
        module(q, k, v=None)


def test_requires_k():
    module = KMeansCrossAttention(dim=2)
    q = torch.zeros(1, 1, 2)
    v = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="requires k"):
        module(q, v=v)


def test_basic_sum_update():
    module = KMeansCrossAttention(dim=2, update="sum")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    out = module(q, k, v)
    expected = torch.tensor([[[6.0, 8.0], [3.0, 4.0]]])
    assert torch.allclose(out, expected)


def test_mean_update():
    module = KMeansCrossAttention(dim=2, update="mean")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    out = module(q, k, v)
    summed = torch.tensor([[[6.0, 8.0], [3.0, 4.0]]])
    denom = torch.tensor([[[2.0], [1.0]]]) + module.eps
    expected = summed / denom
    assert torch.allclose(out, expected)


def test_q_mask_forces_single_query():
    module = KMeansCrossAttention(dim=2, update="sum")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[0.0, 1.0], [0.0, 2.0]]])
    v = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    q_mask = torch.tensor([[True, False]])
    out = module(q, k=k, v=v, q_mask=q_mask)
    expected = torch.tensor([[[3.0, 3.0], [0.0, 0.0]]])
    assert torch.allclose(out, expected)


def test_kv_mask_excludes_key():
    module = KMeansCrossAttention(dim=2, update="sum")
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[2.0, 0.0], [2.0, 0.0]]])
    v = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    kv_mask = torch.tensor([[True, False]])
    out = module(q, k=k, v=v, kv_mask=kv_mask)
    expected = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    assert torch.allclose(out, expected)


def test_attn_mask_respected():
    module = KMeansCrossAttention(dim=2, update="sum", respect_attn_mask=True)
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[0.0, 2.0]]])
    v = torch.tensor([[[2.0, 3.0]]])
    attn_mask = torch.tensor([[[True], [False]]])
    out = module(q, k=k, v=v, attn_mask=attn_mask)
    expected = torch.tensor([[[2.0, 3.0], [0.0, 0.0]]])
    assert torch.allclose(out, expected)
