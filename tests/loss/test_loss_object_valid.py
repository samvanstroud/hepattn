import pytest
import torch

from hepattn.models.loss import mask_bce_loss, mask_dice_loss, mask_focal_loss

torch.manual_seed(42)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_valid_objects", [2, 3, 5])
@pytest.mark.parametrize("num_inputs", [8, 10, 20])
def test_mask_bce_loss_object_valid(batch_size, num_valid_objects, num_inputs):
    """Test that mask_bce_loss gives same results with and without invalid objects."""
    # Create original data with valid objects only
    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()

    # Compute loss without invalid objects
    loss_no_invalid = mask_bce_loss(pred_logits, targets, None, None)

    # Add invalid objects
    num_invalid_objects = 3
    invalid_pred_logits = torch.randn(batch_size, num_invalid_objects, num_inputs)
    invalid_targets = torch.randint(0, 2, (batch_size, num_invalid_objects, num_inputs)).float()

    pred_logits_with_invalid = torch.cat([pred_logits, invalid_pred_logits], dim=1)
    targets_with_invalid = torch.cat([targets, invalid_targets], dim=1)

    # Create object validity mask
    object_valid_mask = torch.cat(
        [torch.ones(batch_size, num_valid_objects, dtype=torch.bool), torch.zeros(batch_size, num_invalid_objects, dtype=torch.bool)], dim=1
    )

    # Compute loss with invalid objects masked out
    loss_with_invalid = mask_bce_loss(pred_logits_with_invalid, targets_with_invalid, object_valid_mask, None)

    # Should be equal
    assert torch.allclose(loss_no_invalid, loss_with_invalid, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_valid_objects", [2, 3, 5])
@pytest.mark.parametrize("num_inputs", [8, 10, 20])
def test_mask_dice_loss_object_valid(batch_size, num_valid_objects, num_inputs):
    """Test that mask_dice_loss gives same results with and without invalid objects."""
    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()

    loss_no_invalid = mask_dice_loss(pred_logits, targets, None, None)

    # Add invalid objects
    num_invalid_objects = 4
    invalid_pred_logits = torch.randn(batch_size, num_invalid_objects, num_inputs)
    invalid_targets = torch.randint(0, 2, (batch_size, num_invalid_objects, num_inputs)).float()

    pred_logits_with_invalid = torch.cat([pred_logits, invalid_pred_logits], dim=1)
    targets_with_invalid = torch.cat([targets, invalid_targets], dim=1)

    object_valid_mask = torch.cat(
        [torch.ones(batch_size, num_valid_objects, dtype=torch.bool), torch.zeros(batch_size, num_invalid_objects, dtype=torch.bool)], dim=1
    )

    loss_with_invalid = mask_dice_loss(pred_logits_with_invalid, targets_with_invalid, object_valid_mask, None)

    assert torch.allclose(loss_no_invalid, loss_with_invalid, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_valid_objects", [3, 4])
@pytest.mark.parametrize("num_inputs", [10, 15])
@pytest.mark.parametrize("gamma", [1.5, 2.0])
def test_mask_focal_loss_object_valid(batch_size, num_valid_objects, num_inputs, gamma):
    """Test that mask_focal_loss gives same results with and without invalid objects."""
    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()

    loss_no_invalid = mask_focal_loss(pred_logits, targets, gamma, None, None)

    # Add invalid objects
    num_invalid_objects = 2
    invalid_pred_logits = torch.randn(batch_size, num_invalid_objects, num_inputs)
    invalid_targets = torch.randint(0, 2, (batch_size, num_invalid_objects, num_inputs)).float()

    pred_logits_with_invalid = torch.cat([pred_logits, invalid_pred_logits], dim=1)
    targets_with_invalid = torch.cat([targets, invalid_targets], dim=1)

    object_valid_mask = torch.cat(
        [torch.ones(batch_size, num_valid_objects, dtype=torch.bool), torch.zeros(batch_size, num_invalid_objects, dtype=torch.bool)], dim=1
    )

    loss_with_invalid = mask_focal_loss(pred_logits_with_invalid, targets_with_invalid, gamma, object_valid_mask, None)

    assert torch.allclose(loss_no_invalid, loss_with_invalid, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_valid_objects", [2, 3])
@pytest.mark.parametrize("num_inputs", [8, 12])
def test_combined_object_valid_and_input_padding(batch_size, num_valid_objects, num_inputs):
    """Test that loss functions work correctly with both object validity mask and input padding."""
    # Create original data
    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()

    # Compute loss with no masks
    loss_no_masks = mask_bce_loss(pred_logits, targets, None, None)

    # Add invalid objects
    num_invalid_objects = 2
    invalid_pred_logits = torch.randn(batch_size, num_invalid_objects, num_inputs)
    invalid_targets = torch.randint(0, 2, (batch_size, num_invalid_objects, num_inputs)).float()

    # Add input padding
    pad_length = 4
    pad_pred = torch.randn(batch_size, num_valid_objects + num_invalid_objects, pad_length)
    pad_targets = torch.zeros(batch_size, num_valid_objects + num_invalid_objects, pad_length)

    pred_logits_full = torch.cat([torch.cat([pred_logits, invalid_pred_logits], dim=1), pad_pred], dim=-1)

    targets_full = torch.cat([torch.cat([targets, invalid_targets], dim=1), pad_targets], dim=-1)

    # Create masks
    object_valid_mask = torch.cat(
        [torch.ones(batch_size, num_valid_objects, dtype=torch.bool), torch.zeros(batch_size, num_invalid_objects, dtype=torch.bool)], dim=1
    )

    input_pad_mask = torch.cat([torch.ones(batch_size, num_inputs), torch.zeros(batch_size, pad_length)], dim=-1)

    # Compute loss with both masks
    loss_with_masks = mask_bce_loss(pred_logits_full, targets_full, object_valid_mask, input_pad_mask)

    assert torch.allclose(loss_no_masks, loss_with_masks, atol=1e-6)


def test_edge_cases_object_valid():
    """Test edge cases with object validity masks."""
    batch_size, num_valid_objects, num_inputs = 2, 3, 8

    # Test with all valid objects (mask all True)
    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()
    object_valid_mask = torch.ones(batch_size, num_valid_objects, dtype=torch.bool)

    loss_no_mask = mask_bce_loss(pred_logits, targets, None, None)
    loss_with_mask = mask_bce_loss(pred_logits, targets, object_valid_mask, None)

    assert torch.allclose(loss_no_mask, loss_with_mask, atol=1e-6)

    # Test with all invalid objects (should handle gracefully)
    object_valid_mask_all_false = torch.zeros(batch_size, num_valid_objects, dtype=torch.bool)

    # This should not crash and should return some reasonable value
    loss_all_invalid = mask_bce_loss(pred_logits, targets, object_valid_mask_all_false, None)
    assert torch.is_tensor(loss_all_invalid)


def test_sample_weight_consistency_object_valid():
    """Test that sample weights work consistently with object validity masks."""
    batch_size, num_valid_objects, num_inputs = 2, 3, 10

    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()
    sample_weight = torch.rand(batch_size, num_valid_objects, num_inputs)

    # Test focal loss with sample weights
    loss_no_invalid = mask_focal_loss(pred_logits, targets, 2.0, None, None, sample_weight)

    # Add invalid objects
    num_invalid_objects = 2
    invalid_pred_logits = torch.randn(batch_size, num_invalid_objects, num_inputs)
    invalid_targets = torch.randint(0, 2, (batch_size, num_invalid_objects, num_inputs)).float()
    invalid_sample_weight = torch.rand(batch_size, num_invalid_objects, num_inputs)

    pred_logits_with_invalid = torch.cat([pred_logits, invalid_pred_logits], dim=1)
    targets_with_invalid = torch.cat([targets, invalid_targets], dim=1)
    sample_weight_with_invalid = torch.cat([sample_weight, invalid_sample_weight], dim=1)

    object_valid_mask = torch.cat(
        [torch.ones(batch_size, num_valid_objects, dtype=torch.bool), torch.zeros(batch_size, num_invalid_objects, dtype=torch.bool)], dim=1
    )

    loss_with_invalid = mask_focal_loss(pred_logits_with_invalid, targets_with_invalid, 2.0, object_valid_mask, None, sample_weight_with_invalid)

    assert torch.allclose(loss_no_invalid, loss_with_invalid, atol=1e-6)


def test_different_invalid_object_counts():
    """Test with different numbers of invalid objects across batches."""
    batch_size, num_valid_objects, num_inputs = 2, 3, 8

    pred_logits = torch.randn(batch_size, num_valid_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, num_valid_objects, num_inputs)).float()

    loss_no_invalid = mask_dice_loss(pred_logits, targets, None, None)

    # Add different numbers of invalid objects
    num_invalid_objects = 5
    invalid_pred_logits = torch.randn(batch_size, num_invalid_objects, num_inputs)
    invalid_targets = torch.randint(0, 2, (batch_size, num_invalid_objects, num_inputs)).float()

    pred_logits_with_invalid = torch.cat([pred_logits, invalid_pred_logits], dim=1)
    targets_with_invalid = torch.cat([targets, invalid_targets], dim=1)

    # Create object validity mask with different patterns per batch
    object_valid_mask = torch.zeros(batch_size, num_valid_objects + num_invalid_objects, dtype=torch.bool)
    object_valid_mask[:, :num_valid_objects] = True  # First objects are valid for all batches

    loss_with_invalid = mask_dice_loss(pred_logits_with_invalid, targets_with_invalid, object_valid_mask, None)

    assert torch.allclose(loss_no_invalid, loss_with_invalid, atol=1e-6)


def test_partial_validity_patterns():
    """Test with various partial validity patterns."""
    batch_size, total_objects, num_inputs = 2, 6, 10

    pred_logits = torch.randn(batch_size, total_objects, num_inputs)
    targets = torch.randint(0, 2, (batch_size, total_objects, num_inputs)).float()

    # Test different validity patterns
    patterns = [
        torch.tensor([True, True, False, False, True, False]),  # Mixed pattern
        torch.tensor([True, False, True, False, True, False]),  # Alternating pattern
        torch.tensor([False, False, True, True, True, True]),  # Valid objects at end
    ]

    for pattern in patterns:
        object_valid_mask = pattern.unsqueeze(0).expand(batch_size, -1)

        # Extract only valid objects for comparison
        valid_indices = pattern.nonzero(as_tuple=True)[0]
        pred_logits_valid = pred_logits[:, valid_indices]
        targets_valid = targets[:, valid_indices]

        loss_valid_only = mask_bce_loss(pred_logits_valid, targets_valid, None, None)
        loss_with_mask = mask_bce_loss(pred_logits, targets, object_valid_mask, None)

        assert torch.allclose(loss_valid_only, loss_with_mask, atol=1e-6)
