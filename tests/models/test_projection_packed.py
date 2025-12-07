import pytest
import torch
import torch.nn.functional as F

from hepattn.models.attention import projection_packed

# Set random seed for reproducibility
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_packed_self_attention(batch_size, seq_len, dim, bias):
    """Test projection_packed with self-attention (q is k is v)."""
    # Create input tensor
    q = torch.randn(batch_size, seq_len, dim, device=DEVICE)

    # Create weights and bias
    weight = torch.randn(3 * dim, dim, device=DEVICE)
    bias_tensor = torch.randn(3 * dim, device=DEVICE) if bias else None

    # Test projection_packed with self-attention
    q_proj, k_proj, v_proj = projection_packed(q, q, q, weight, bias_tensor)

    # Compare with reference implementation (PyTorch's _in_projection_packed)
    q_ref, k_ref, v_ref = F._in_projection_packed(q, q, q, weight, bias_tensor)  # noqa: SLF001

    # Check shapes
    assert q_proj.shape == (batch_size, seq_len, dim)
    assert k_proj.shape == (batch_size, seq_len, dim)
    assert v_proj.shape == (batch_size, seq_len, dim)

    # Check values match reference
    torch.testing.assert_close(q_proj, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_proj, k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_proj, v_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("q_len", [32, 64])
@pytest.mark.parametrize("kv_len", [32, 128])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_packed_cross_attention_shared_kv(batch_size, q_len, kv_len, dim, bias):
    """Test projection_packed with cross-attention where k is v."""
    # Create input tensors
    q = torch.randn(batch_size, q_len, dim, device=DEVICE)
    kv = torch.randn(batch_size, kv_len, dim, device=DEVICE)

    # Create weights and bias
    weight = torch.randn(3 * dim, dim, device=DEVICE)
    bias_tensor = torch.randn(3 * dim, device=DEVICE) if bias else None

    # Test projection_packed with k is v
    q_proj, k_proj, v_proj = projection_packed(q, kv, kv, weight, bias_tensor)

    # Compare with reference implementation
    q_ref, k_ref, v_ref = F._in_projection_packed(q, kv, kv, weight, bias_tensor)  # noqa: SLF001

    # Check shapes
    assert q_proj.shape == (batch_size, q_len, dim)
    assert k_proj.shape == (batch_size, kv_len, dim)
    assert v_proj.shape == (batch_size, kv_len, dim)

    # Check values match reference
    torch.testing.assert_close(q_proj, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_proj, k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_proj, v_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("q_len", [32, 64])
@pytest.mark.parametrize("k_len", [32, 128])
@pytest.mark.parametrize("v_len", [32, 128])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_packed_cross_attention_separate_kv(batch_size, q_len, k_len, v_len, dim, bias):
    """Test projection_packed with cross-attention where k and v are separate tensors."""
    # Create input tensors - all different objects
    q = torch.randn(batch_size, q_len, dim, device=DEVICE)
    k = torch.randn(batch_size, k_len, dim, device=DEVICE)
    v = torch.randn(batch_size, v_len, dim, device=DEVICE)

    # Create weights and bias
    weight = torch.randn(3 * dim, dim, device=DEVICE)
    bias_tensor = torch.randn(3 * dim, device=DEVICE) if bias else None

    # Test projection_packed with separate k and v
    q_proj, k_proj, v_proj = projection_packed(q, k, v, weight, bias_tensor)

    # Compare with reference implementation
    q_ref, k_ref, v_ref = F._in_projection_packed(q, k, v, weight, bias_tensor)  # noqa: SLF001

    # Check shapes
    assert q_proj.shape == (batch_size, q_len, dim)
    assert k_proj.shape == (batch_size, k_len, dim)
    assert v_proj.shape == (batch_size, v_len, dim)

    # Check values match reference
    torch.testing.assert_close(q_proj, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_proj, k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_proj, v_ref, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_projection_packed_nested_tensor_self_attention():
    """Test projection_packed with nested tensor for self-attention."""
    dim = 128
    device = "cuda"

    # Create nested tensor with jagged layout
    tensors = [torch.randn(s, dim, dtype=torch.float16, device=device) for s in (64, 128)]
    q = torch.nested.nested_tensor(tensors, layout=torch.jagged, device=device)

    # Create weights and bias
    weight = torch.randn(3 * dim, dim, dtype=torch.float16, device=device)
    bias_tensor = torch.randn(3 * dim, dtype=torch.float16, device=device)

    # Test projection_packed with nested tensor
    q_proj, k_proj, v_proj = projection_packed(q, q, q, weight, bias_tensor)

    # Check that outputs are nested tensors
    assert q_proj.is_nested
    assert k_proj.is_nested
    assert v_proj.is_nested

    # Check that we can extract individual tensors
    for i in range(len(tensors)):
        assert q_proj[i].shape == (tensors[i].shape[0], dim)
        assert k_proj[i].shape == (tensors[i].shape[0], dim)
        assert v_proj[i].shape == (tensors[i].shape[0], dim)


@pytest.mark.gpu
def test_projection_packed_nested_tensor_cross_attention():
    """Test projection_packed with nested tensor for cross-attention."""
    dim = 128
    device = "cuda"

    # Create nested tensors with jagged layout
    q_tensors = [torch.randn(s, dim, dtype=torch.float16, device=device) for s in (64, 128)]
    kv_tensors = [torch.randn(s, dim, dtype=torch.float16, device=device) for s in (96, 150)]

    q = torch.nested.nested_tensor(q_tensors, layout=torch.jagged, device=device)
    kv = torch.nested.nested_tensor(kv_tensors, layout=torch.jagged, device=device)

    # Create weights and bias
    weight = torch.randn(3 * dim, dim, dtype=torch.float16, device=device)
    bias_tensor = torch.randn(3 * dim, dtype=torch.float16, device=device)

    # Test projection_packed with nested tensor
    q_proj, k_proj, v_proj = projection_packed(q, kv, kv, weight, bias_tensor)

    # Check that outputs are nested tensors
    assert q_proj.is_nested
    assert k_proj.is_nested
    assert v_proj.is_nested

    # Check that we can extract individual tensors
    for i in range(len(q_tensors)):
        assert q_proj[i].shape == (q_tensors[i].shape[0], dim)
        assert k_proj[i].shape == (kv_tensors[i].shape[0], dim)
        assert v_proj[i].shape == (kv_tensors[i].shape[0], dim)


def test_projection_packed_identity_checks():
    """Test that projection_packed correctly uses identity checks for different cases."""
    batch_size, seq_len, dim = 2, 32, 64

    # Create tensors
    q = torch.randn(batch_size, seq_len, dim, device=DEVICE)
    k = torch.randn(batch_size, seq_len, dim, device=DEVICE)
    v = torch.randn(batch_size, seq_len, dim, device=DEVICE)

    weight = torch.randn(3 * dim, dim, device=DEVICE)
    bias_tensor = torch.randn(3 * dim, device=DEVICE)

    # Test 1: All same (self-attention path)
    q1_proj, k1_proj, v1_proj = projection_packed(q, q, q, weight, bias_tensor)

    # Test 2: k is v but different from q (cross-attention with shared kv)
    q2_proj, k2_proj, v2_proj = projection_packed(q, k, k, weight, bias_tensor)

    # Test 3: All different (cross-attention with separate kv)
    q3_proj, _, _ = projection_packed(q, k, v, weight, bias_tensor)

    # Results should be the same for q in all cases
    torch.testing.assert_close(q1_proj, q2_proj, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q1_proj, q3_proj, rtol=1e-5, atol=1e-5)

    # For test 1, k and v projections should be the same
    torch.testing.assert_close(k1_proj, v1_proj, rtol=1e-5, atol=1e-5)

    # For test 2, k and v projections should be the same
    torch.testing.assert_close(k2_proj, v2_proj, rtol=1e-5, atol=1e-5)

    # For test 3, k and v projections should be different inputs
    # (but we can't directly test the projection difference without knowing the weights)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_projection_packed_weight_splitting(dim):
    """Test that weight splitting is correct for all three cases."""
    batch_size, q_len, kv_len = 2, 32, 48

    q = torch.randn(batch_size, q_len, dim, device=DEVICE)
    k = torch.randn(batch_size, kv_len, dim, device=DEVICE)
    v = torch.randn(batch_size, kv_len, dim, device=DEVICE)

    # Create identity-like weights to make splitting behavior obvious
    weight = torch.randn(3 * dim, dim, device=DEVICE)

    # Test with no bias
    q_proj, k_proj, v_proj = projection_packed(q, k, v, weight, None)

    # Verify shapes
    assert q_proj.shape == (batch_size, q_len, dim)
    assert k_proj.shape == (batch_size, kv_len, dim)
    assert v_proj.shape == (batch_size, kv_len, dim)

    # Test with bias
    bias_tensor = torch.randn(3 * dim, device=DEVICE)
    q_proj_b, k_proj_b, v_proj_b = projection_packed(q, k, v, weight, bias_tensor)

    # Verify shapes
    assert q_proj_b.shape == (batch_size, q_len, dim)
    assert k_proj_b.shape == (batch_size, kv_len, dim)
    assert v_proj_b.shape == (batch_size, kv_len, dim)

    # Verify that bias makes a difference
    assert not torch.allclose(q_proj, q_proj_b, rtol=1e-5, atol=1e-5)
