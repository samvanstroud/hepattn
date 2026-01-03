import torch

from hepattn.models.task import IoUPredictionTask, ObjectHitMaskTask


def test_object_hit_mask_task_iou():
    """Test IoUPredictionTask with ObjectHitMaskTask."""
    dim = 16

    # Create mask task
    mask_task = ObjectHitMaskTask(
        name="test_mask_task",
        input_constituent="hit",
        input_object="track",
        output_object="track",
        target_object="track",
        losses={"mask_bce": 1.0},
        costs={"mask_bce": 1.0},
        dim=dim,
    )

    # Create IoU prediction task
    iou_task = IoUPredictionTask(
        name="test_iou_task",
        input_object="track",
        mask_logit_key="track_hit_logit",
        target_mask_key="track_hit",
        dim=dim,
        loss_weight=0.5,
    )

    batch_size = 2
    num_objects = 3
    num_hits = 4

    # Create dummy inputs
    x = {
        "track_embed": torch.randn(batch_size, num_objects, dim),
        "hit_embed": torch.randn(batch_size, num_hits, dim),
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    }

    # Forward pass for mask task
    mask_outputs = mask_task(x)

    assert "track_hit_logit" in mask_outputs
    assert "track_iou_logit" not in mask_outputs  # IoU logit should not be in mask task outputs

    # Forward pass for IoU task
    iou_outputs = iou_task(x)
    assert "track_iou_logit" in iou_outputs
    assert iou_outputs["track_iou_logit"].shape == (batch_size, num_objects)

    # Predict IoU
    preds = iou_task.predict(iou_outputs)
    assert "track_iou" in preds
    assert preds["track_iou"].shape == (batch_size, num_objects)

    # Loss - inject mask logits into iou_outputs (simulating what maskformer does)
    iou_outputs["track_hit_logit"] = mask_outputs["track_hit_logit"]

    targets = {
        "track_hit_valid": torch.randint(0, 2, (batch_size, num_objects, num_hits)).float(),
        "track_valid": torch.ones(batch_size, num_objects, dtype=torch.bool),
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    }

    losses = iou_task.loss(iou_outputs, targets)
    assert "iou_mse" in losses
    assert losses["iou_mse"] > 0
