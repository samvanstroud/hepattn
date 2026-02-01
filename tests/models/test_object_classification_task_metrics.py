import torch

from hepattn.models.task import ObjectClassificationTask


def test_object_classification_task_metrics_basic():
    # 2 batches, 4 queries
    # true_valid has 4 positives total and 4 negatives total
    true_valid = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
        ],
        dtype=torch.bool,
    )

    # predicted valid: 4 predicted positives total
    # TP = 3 (positions: (0,0), (0,1), (1,2))
    # FP = 1 (position: (1,1))
    pred_valid = torch.tensor(
        [
            [True, True, False, False],
            [False, True, True, False],
        ],
        dtype=torch.bool,
    )

    task = ObjectClassificationTask(
        name="track_valid",
        input_object="query",
        output_object="track",
        target_object="particle",
        dim=8,
        losses={"object_bce": 1.0},
        costs={"object_bce": 1.0},
        num_classes=1,
    )

    preds = {"track_valid": pred_valid}
    targets = {"particle_valid": true_valid}

    metrics = task.metrics(preds, targets)

    assert metrics["num_queries"] == 4.0
    assert torch.isclose(metrics["query_frac_pred_valid"], torch.tensor(0.5))  # 4/8
    assert torch.isclose(metrics["query_eff"], torch.tensor(0.75))  # 3/4 (formerly query_tpr)
    assert torch.isclose(metrics["query_fr"], torch.tensor(0.25))  # 1/4 (formerly query_fpr)
