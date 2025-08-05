import torch
import torch.nn.functional as F


def test_weighted_binary_cross_entropy_equivalence():
    """Test that weighted BCE with logits gives same result as manual weighting."""
    # Test case 1: [batch_size, num_targets] - classification task
    batch_size, num_targets = 32, 10

    # Create sample data
    pred_logits_2d = torch.randn(batch_size, num_targets, requires_grad=True)
    true_2d = torch.randint(0, 2, (batch_size, num_targets)).float()
    weight_2d = torch.rand(batch_size, num_targets)

    # Method 1: Using built-in weight parameter
    loss_builtin_2d = F.binary_cross_entropy_with_logits(pred_logits_2d, true_2d, weight=weight_2d, reduction="none")
    weighted_mean_builtin_2d = loss_builtin_2d.mean()

    # Method 2: Manual weighting
    loss_unweighted_2d = F.binary_cross_entropy_with_logits(pred_logits_2d, true_2d, reduction="none")
    loss_manual_2d = loss_unweighted_2d * weight_2d
    weighted_mean_manual_2d = loss_manual_2d.mean()

    # Test that per-element losses are identical
    torch.testing.assert_close(loss_builtin_2d, loss_manual_2d, rtol=1e-6, atol=1e-6)

    # Test that mean losses are identical
    torch.testing.assert_close(weighted_mean_builtin_2d, weighted_mean_manual_2d, rtol=1e-6, atol=1e-6)

    print(f"2D case - Built-in weighted mean: {weighted_mean_builtin_2d:.6f}")
    print(f"2D case - Manual weighted mean: {weighted_mean_manual_2d:.6f}")
    print("✓ 2D case passed")


def test_weighted_binary_cross_entropy_3d():
    """Test weighted BCE for 3D tensors - object detection scenario."""
    # Test case 2: [batch_size, num_targets, num_inputs] - object detection task
    batch_size, num_targets, num_inputs = 16, 5, 8

    # Create sample data
    pred_logits_3d = torch.randn(batch_size, num_targets, num_inputs, requires_grad=True)
    true_3d = torch.randint(0, 2, (batch_size, num_targets, num_inputs)).float()
    weight_3d = torch.rand(batch_size, num_targets, num_inputs)

    # Method 1: Using built-in weight parameter
    loss_builtin_3d = F.binary_cross_entropy_with_logits(pred_logits_3d, true_3d, weight=weight_3d, reduction="none")
    weighted_mean_builtin_3d = loss_builtin_3d.mean()

    # Method 2: Manual weighting
    loss_unweighted_3d = F.binary_cross_entropy_with_logits(pred_logits_3d, true_3d, reduction="none")
    loss_manual_3d = loss_unweighted_3d * weight_3d
    weighted_mean_manual_3d = loss_manual_3d.mean()

    # Test that per-element losses are identical
    torch.testing.assert_close(loss_builtin_3d, loss_manual_3d, rtol=1e-6, atol=1e-6)

    # Test that mean losses are identical
    torch.testing.assert_close(weighted_mean_builtin_3d, weighted_mean_manual_3d, rtol=1e-6, atol=1e-6)

    print(f"3D case - Built-in weighted mean: {weighted_mean_builtin_3d:.6f}")
    print(f"3D case - Manual weighted mean: {weighted_mean_manual_3d:.6f}")
    print("✓ 3D case passed")


def test_edge_cases():
    """Test edge cases with zero weights and extreme values."""
    # Test with some zero weights
    pred_logits = torch.randn(4, 3)
    true = torch.randint(0, 2, (4, 3)).float()
    weight = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.0, 1.5], [2.0, 1.0, 0.0], [0.0, 0.0, 3.0]])

    loss_builtin = F.binary_cross_entropy_with_logits(pred_logits, true, weight=weight, reduction="none")
    loss_manual = F.binary_cross_entropy_with_logits(pred_logits, true, reduction="none") * weight

    torch.testing.assert_close(loss_builtin, loss_manual, rtol=1e-6, atol=1e-6)

    # Test that zero weights produce zero loss
    zero_weight_mask = weight == 0.0
    assert torch.all(loss_builtin[zero_weight_mask] == 0.0)
    assert torch.all(loss_manual[zero_weight_mask] == 0.0)

    print("✓ Edge cases passed")


def test_gradient_equivalence():
    """Test that gradients are the same for both methods."""
    # Create tensors with gradients
    pred_logits_builtin = torch.randn(8, 4, requires_grad=True)
    pred_logits_manual = pred_logits_builtin.clone().detach().requires_grad_(True)
    true = torch.randint(0, 2, (8, 4)).float()
    weight = torch.rand(8, 4)

    # Built-in method
    loss_builtin = F.binary_cross_entropy_with_logits(pred_logits_builtin, true, weight=weight, reduction="mean")
    loss_builtin.backward()

    # Manual method
    loss_manual = (F.binary_cross_entropy_with_logits(pred_logits_manual, true, reduction="none") * weight).mean()
    loss_manual.backward()

    # Compare gradients
    torch.testing.assert_close(pred_logits_builtin.grad, pred_logits_manual.grad, rtol=1e-6, atol=1e-6)

    print("✓ Gradient equivalence passed")


def test_focal_loss_sample_weight():
    gamma = 2.0
    shape = (100, 50)
    # generate dummy inputs
    pred_logits = torch.randn(*shape, requires_grad=True)
    targets = torch.randint(0, 2, shape).float()
    sample_weight = torch.rand(*shape)
    pred = pred_logits.sigmoid()

    # sample weight inside bce loss
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.type_as(pred_logits), weight=sample_weight, reduction="none")
    p_t = pred * targets + (1 - pred) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    loss1 = loss.mean()

    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets.type_as(pred_logits), reduction="none")
    p_t = pred * targets + (1 - pred) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    loss *= sample_weight
    loss2 = loss.mean()

    torch.testing.assert_close(loss1, loss2, rtol=1e-6, atol=1e-6)
