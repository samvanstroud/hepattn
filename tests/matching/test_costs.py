import pytest
import torch

from hepattn.models.loss import object_bce_cost, object_ce_cost

torch.manual_seed(42)


def compute_class_cost_unbatched(pred_logits, target_labels):
    """Compute classification cost for object matching."""
    if pred_logits.dim() == 1:
        # Binary classification - 2D input [num_queries]
        out_prob = pred_logits.sigmoid()
        cost_class = -out_prob.unsqueeze(1) * target_labels.unsqueeze(0) - (1 - out_prob.unsqueeze(1)) * (1 - target_labels.unsqueeze(0))
    elif pred_logits.shape[-1] == 1:
        # Binary classification - 3D input [num_queries, 1]
        out_prob = pred_logits.sigmoid().squeeze(-1)
        cost_class = -out_prob.unsqueeze(1) * target_labels.unsqueeze(0) - (1 - out_prob.unsqueeze(1)) * (1 - target_labels.unsqueeze(0))
    else:
        # Multi-class
        out_prob = pred_logits.softmax(-1)
        cost_class = -out_prob[:, target_labels]
    return cost_class


def test_binary_classification_equivalence():
    """Test that binary classification with num_classes=1 and num_classes=2 produce equivalent results."""
    num_queries = 5
    num_targets = 3

    pred_logits_1class = torch.randn(num_queries, 1)

    pred_logits_2class = torch.zeros(num_queries, 2)
    pred_logits_2class[:, 1] = pred_logits_1class.squeeze(-1)

    target_labels = torch.randint(0, 2, (num_targets,))

    cost_1class = compute_class_cost_unbatched(pred_logits_1class, target_labels)
    cost_2class = compute_class_cost_unbatched(pred_logits_2class, target_labels)

    torch.testing.assert_close(cost_1class, cost_2class, rtol=1e-5, atol=1e-6)
    assert cost_1class.shape == (num_queries, num_targets)
    assert cost_2class.shape == (num_queries, num_targets)


@pytest.mark.repeat(10)
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_object_bce_cost(batch_size):
    """Test that batched and unbatched implementations produce equivalent results."""
    num_queries = 100
    num_targets = 100

    pred_logits_batched = torch.randn(batch_size, num_queries)
    target_labels_batched = torch.randint(0, 2, (batch_size, num_targets))

    cost_batched = object_bce_cost(pred_logits_batched, target_labels_batched)

    costs_unbatched = []
    for b in range(batch_size):
        cost_single = compute_class_cost_unbatched(pred_logits_batched[b], target_labels_batched[b])
        costs_unbatched.append(cost_single)

    cost_unbatched_stacked = torch.stack(costs_unbatched, dim=0)

    torch.testing.assert_close(cost_batched, cost_unbatched_stacked, rtol=1e-5, atol=1e-6)
    assert cost_batched.shape == (batch_size, num_queries, num_targets)
    assert cost_unbatched_stacked.shape == (batch_size, num_queries, num_targets)


@pytest.mark.repeat(10)
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_object_ce_cost(batch_size):
    """Test that batched and unbatched implementations produce equivalent results."""
    num_queries = 100
    num_targets = 100
    num_classes = 3

    pred_logits_batched = torch.randn(batch_size, num_queries, num_classes)
    target_labels_batched = torch.randint(0, num_classes, (batch_size, num_targets))

    cost_batched = object_ce_cost(pred_logits_batched, target_labels_batched)

    costs_unbatched = []
    for b in range(batch_size):
        cost_single = compute_class_cost_unbatched(pred_logits_batched[b], target_labels_batched[b])
        costs_unbatched.append(cost_single)

    cost_unbatched_stacked = torch.stack(costs_unbatched, dim=0)

    torch.testing.assert_close(cost_batched, cost_unbatched_stacked, rtol=1e-5, atol=1e-6)
    assert cost_batched.shape == (batch_size, num_queries, num_targets)
    assert cost_unbatched_stacked.shape == (batch_size, num_queries, num_targets)
