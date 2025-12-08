import torch

from hepattn.models.task import ObjectHitMaskTask


def test_object_hit_mask_task_iou():
    dim = 16
    task = ObjectHitMaskTask(
        name="test_task",
        input_constituent="hit",
        input_object="track",
        output_object="track",
        target_object="track",
        losses={"mask_bce": 1.0},
        costs={"mask_bce": 1.0},
        dim=dim,
        predict_iou=True,
        iou_loss_weight=0.5,
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

    # Forward pass
    outputs = task(x)

    assert "track_hit_logit" in outputs
    assert "track_iou_logit" in outputs
    assert outputs["track_iou_logit"].shape == (batch_size, num_objects)

    # Predict
    preds = task.predict(outputs)
    assert "track_iou" in preds
    assert preds["track_iou"].shape == (batch_size, num_objects)

    # Loss
    targets = {
        "track_hit_valid": torch.randint(0, 2, (batch_size, num_objects, num_hits)).float(),
        "track_valid": torch.ones(batch_size, num_objects, dtype=torch.bool),
        "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
    }

    losses = task.loss(outputs, targets)
    assert "iou_mse" in losses
    assert losses["iou_mse"] > 0
