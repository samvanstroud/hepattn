import matplotlib.pyplot as plt
import torch

from hepattn.utils.local_ca import auto_local_ca_mask, get_local_ca_mask


def test_basic_functionality():
    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=0)
    expected = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result == expected)


def test_window_size():
    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=2)
    expected = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)


def test_window_stride():
    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=0, stride=2)
    expected = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)

    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=2, stride=3)
    expected = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)


def test_window_size_larger_than_input():
    result = get_local_ca_mask(n_objects=2, n_inputs=5, window_size=10)
    expected = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)


def test_single_object():
    result = get_local_ca_mask(n_objects=1, n_inputs=5, window_size=2)
    expected = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)


def test_auto_mask_no_wrap():
    q = torch.randn(1, 5, 5)
    kv = torch.randn(1, 15, 5)
    result = auto_local_ca_mask(q, kv, window_size=4, wrap=False)
    expected = torch.tensor([
        [True, True, True, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, True, True, True, True, True, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, True, True, True, True, True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, True, True, True, True, True, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True],
    ])
    assert torch.all(result.squeeze(0) == expected)


def test_auto_mask():
    q = torch.randn(1, 5, 5)
    kv = torch.randn(1, 15, 5)
    result = auto_local_ca_mask(q, kv, window_size=4)
    expected = torch.tensor([
        [True, True, True, False, False, False, False, False, False, False, False, False, False, True, True],
        [False, True, True, True, True, True, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, True, True, True, True, True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, True, True, True, True, True, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, True, True, True, True, True],
    ])
    assert torch.all(result.squeeze(0) == expected)


def test_realistic():
    q = torch.randn(1, 1000, 5)
    kv = torch.randn(1, 5462, 5)
    mask = auto_local_ca_mask(q, kv, window_size=2048).squeeze(0)
    # save the mask as a maplotlib image using imshow
    plt.imshow(mask.numpy(), aspect="auto")
    plt.savefig("local_ca_mask.png")

    mask = auto_local_ca_mask(q, kv, window_size=2048, wrap=True).squeeze(0)
    # save the mask as a maplotlib image using imshow
    plt.imshow(mask.numpy(), aspect="auto")
    plt.savefig("local_ca_mask_wrap.png")


def test_wrap():
    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=2, wrap=True)
    expected = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)

    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=4, wrap=True)
    expected = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)

    result = get_local_ca_mask(n_objects=3, n_inputs=10, window_size=2, stride=2, wrap=True)
    expected = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)

    result = get_local_ca_mask(n_objects=3, n_inputs=6, window_size=4, stride=2, wrap=True)
    expected = torch.tensor([[1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.all(result.squeeze(0) == expected)


if __name__ == "__main__":
    q = torch.randn(1, 5, 5)
    kv = torch.randn(1, 15, 5)
    result = auto_local_ca_mask(q, kv, window_size=4)
    print("Result shape:", result.shape)
    print("Result:")
    print(result.squeeze(0))
