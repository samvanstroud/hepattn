import pytest
import torch

from hepattn.flex.fast_local_ca import build_strided_sliding_window_blockmask
from hepattn.flex.local_ca import sliding_window_mask_strided, sliding_window_mask_strided_wrapped


@pytest.fixture
def test_config():
    """Common test configuration."""
    return {"window_size": 32, "stride": 2.0, "q_len": 100, "kv_len": 1000, "device": "cpu", "block_size": 128}


def test_non_wrapped_equivalence(test_config):
    """Test that fast_local_ca and local_ca produce equivalent masks for non-wrapped case."""
    # Create masks using both approaches
    fast_mask = build_strided_sliding_window_blockmask(wrap=False, **test_config)

    local_mask = sliding_window_mask_strided(
        window_size=test_config["window_size"],
        stride=test_config["stride"],
        q_len=test_config["q_len"],
        kv_len=test_config["kv_len"],
        device=test_config["device"],
    )

    # They should be identical even though they're different types
    assert torch.equal(fast_mask.to_dense(), local_mask.to_dense()), "Fast and local CA masks should be identical for non-wrapped case"


def test_wrapped_equivalence(test_config):
    """Test that fast_local_ca and local_ca produce equivalent masks for non-wrapped case."""
    # Create masks using both approaches
    fast_mask = build_strided_sliding_window_blockmask(wrap=True, **test_config)

    local_mask = sliding_window_mask_strided_wrapped(
        window_size=test_config["window_size"],
        stride=test_config["stride"],
        q_len=test_config["q_len"],
        kv_len=test_config["kv_len"],
        device=test_config["device"],
    )

    # They should be identical even though they're different types
    assert torch.equal(fast_mask.to_dense(), local_mask.to_dense()), "Fast and local CA masks should be identical for non-wrapped case"
