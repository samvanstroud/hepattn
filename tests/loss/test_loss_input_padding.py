import pytest
import torch

from hepattn.models.loss import mask_bce_cost, mask_bce_loss, mask_dice_cost, mask_dice_loss, mask_focal_cost, mask_focal_loss, mask_iou_cost

torch.manual_seed(42)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_objects", [2, 3, 5])
@pytest.mark.parametrize("num_inputs", [8, 10, 20])
def test_mask_bce_loss_input_padding(batch_size, num_objects, num_inputs):
    """Test that mask_bce_loss gives same results with and without input padding."""
    # Create original data
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()
    object_valid_mask = torch.ones(batch_size, num_objects, dtype=torch.bool)

    # Compute loss without padding
    loss_no_pad = mask_bce_loss(pred_logits, targets, object_valid_mask, None)

    # Add padding
    pad_length = 5
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    # Compute loss with padding
    loss_with_pad = mask_bce_loss(pred_logits_padded, targets_padded, object_valid_mask, input_pad_mask)

    # Should be equal
    assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_objects", [2, 3, 5])
@pytest.mark.parametrize("num_inputs", [8, 10, 20])
def test_mask_dice_loss_input_padding(batch_size, num_objects, num_inputs):
    """Test that mask_dice_loss gives same results with and without input padding."""
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()
    object_valid_mask = torch.ones(batch_size, num_objects, dtype=torch.bool)

    loss_no_pad = mask_dice_loss(pred_logits, targets, object_valid_mask, None)

    # Add padding
    pad_length = 3
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    loss_with_pad = mask_dice_loss(pred_logits_padded, targets_padded, object_valid_mask, input_pad_mask)

    assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_objects", [3, 4])
@pytest.mark.parametrize("num_inputs", [10, 15])
@pytest.mark.parametrize("gamma", [1.5, 2.0])
def test_mask_focal_loss_input_padding(batch_size, num_objects, num_inputs, gamma):
    """Test that mask_focal_loss gives same results with and without input padding."""
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()
    object_valid_mask = torch.ones(batch_size, num_objects, dtype=torch.bool)

    loss_no_pad = mask_focal_loss(pred_logits, targets, gamma, object_valid_mask, None)

    # Add padding
    pad_length = 4
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    loss_with_pad = mask_focal_loss(pred_logits_padded, targets_padded, gamma, object_valid_mask, input_pad_mask)

    assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-6)


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("num_objects", [3, 4])
@pytest.mark.parametrize("num_inputs", [10, 12])
def test_mask_bce_cost_input_padding(batch_size, num_objects, num_inputs):
    """Test that mask_bce_cost gives same results with and without input padding."""
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()

    cost_no_pad = mask_bce_cost(pred_logits, targets, None)

    # Add padding
    pad_length = 6
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    cost_with_pad = mask_bce_cost(pred_logits_padded, targets_padded, input_pad_mask)

    assert torch.allclose(cost_no_pad, cost_with_pad, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_objects", [3, 5])
@pytest.mark.parametrize("num_inputs", [8, 10])
def test_mask_dice_cost_input_padding(batch_size, num_objects, num_inputs):
    """Test that mask_dice_cost gives same results with and without input padding."""
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()

    cost_no_pad = mask_dice_cost(pred_logits, targets, None)

    # Add padding
    pad_length = 5
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    cost_with_pad = mask_dice_cost(pred_logits_padded, targets_padded, input_pad_mask)

    assert torch.allclose(cost_no_pad, cost_with_pad, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_objects", [3, 4])
@pytest.mark.parametrize("num_inputs", [10, 12])
@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_mask_focal_cost_input_padding(batch_size, num_objects, num_inputs, gamma):
    """Test that mask_focal_cost gives same results with and without input padding."""
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()

    cost_no_pad = mask_focal_cost(pred_logits, targets, gamma, None)

    # Add padding
    pad_length = 7
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    cost_with_pad = mask_focal_cost(pred_logits_padded, targets_padded, gamma, input_pad_mask)

    assert torch.allclose(cost_no_pad, cost_with_pad, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_objects", [3, 4])
@pytest.mark.parametrize("num_inputs", [10, 15])
def test_mask_iou_cost_input_padding(batch_size, num_objects, num_inputs):
    """Test that mask_iou_cost gives same results with and without input padding."""
    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()

    cost_no_pad = mask_iou_cost(pred_logits, targets, None)

    # Add padding
    pad_length = 4
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    cost_with_pad = mask_iou_cost(pred_logits_padded, targets_padded, input_pad_mask)

    assert torch.allclose(cost_no_pad, cost_with_pad, atol=1e-6)


def test_edge_cases():
    """Test edge cases like all zeros, all ones, etc."""
    batch_size, num_objects, num_inputs = 2, 3, 8

    # Test with all zeros
    pred_logits = torch.zeros(batch_size, num_objects, num_inputs)
    targets = torch.zeros(batch_size, num_objects, num_inputs)

    # Add padding
    pad_length = 3
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    # Test BCE loss
    loss_no_pad = mask_bce_loss(pred_logits, targets, None, None)
    loss_with_pad = mask_bce_loss(pred_logits_padded, targets_padded, None, input_pad_mask)
    assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-6)

    # Test BCE cost
    cost_no_pad = mask_bce_cost(pred_logits, targets, None)
    cost_with_pad = mask_bce_cost(pred_logits_padded, targets_padded, input_pad_mask)
    assert torch.allclose(cost_no_pad, cost_with_pad, atol=1e-6)


def test_sample_weight_consistency():
    """Test that sample weights work consistently with padding."""
    batch_size, num_objects, num_inputs = 2, 3, 10

    pred_logits = torch.randn(batch_size, num_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_objects, num_inputs)).float()
    sample_weight = torch.rand(batch_size, num_objects, num_inputs)

    # Test focal loss with sample weights
    loss_no_pad = mask_focal_loss(pred_logits, targets, 2.0, None, None, sample_weight)

    # Add padding
    pad_length = 4
    pred_logits_padded = torch.cat([pred_logits, torch.randn(batch_size, num_objects, pad_length)], dim=-1)
    targets_padded = torch.cat([targets, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    sample_weight_padded = torch.cat([sample_weight, torch.zeros(batch_size, num_objects, pad_length)], dim=-1)
    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    loss_with_pad = mask_focal_loss(pred_logits_padded, targets_padded, 2.0, None, input_pad_mask, sample_weight_padded)

    assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-6)
