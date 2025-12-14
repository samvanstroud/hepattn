"""Test cases for ClassificationTask covering binary, multilabel, and multi-class modes."""

import pytest
import torch

from hepattn.models.dense import Dense
from hepattn.models.task import ClassificationTask


class TestClassificationTask:  # noqa: PLR0904
    """Test ClassificationTask in different modes."""

    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def num_hits(self):
        return 100

    @pytest.fixture
    def dim(self):
        return 64

    @pytest.fixture
    def binary_single_output_task(self, dim):
        """Binary classification with single output (e.g., hit is_first prediction)."""
        return ClassificationTask(
            name="binary_classifier",
            input_object="hit",
            output_object="hit",
            target_object="hit",
            classes=["is_first"],
            net=Dense(input_size=dim, output_size=1),
            multilabel=False,
            loss_weight=1.0,
        )

    @pytest.fixture
    def multilabel_task(self, dim):
        """Multilabel classification (e.g., hit has multiple properties)."""
        return ClassificationTask(
            name="multilabel_classifier",
            input_object="hit",
            output_object="hit",
            target_object="hit",
            classes=["is_first", "is_last", "is_vertex"],
            net=Dense(input_size=dim, output_size=3),
            multilabel=True,
            loss_weight=1.0,
        )

    @pytest.fixture
    def multiclass_task(self, dim):
        """Multi-class classification (e.g., particle type: electron, muon, pion)."""
        return ClassificationTask(
            name="multiclass_classifier",
            input_object="particle",
            output_object="particle",
            target_object="particle",
            classes=["electron", "muon", "pion"],
            net=Dense(input_size=dim, output_size=3),
            multilabel=False,
            loss_weight=1.0,
        )

    @pytest.fixture
    def weighted_binary_task(self, dim):
        """Binary classification with class weighting."""
        return ClassificationTask(
            name="weighted_binary",
            input_object="hit",
            output_object="hit",
            target_object="hit",
            classes=["is_rare_event"],
            net=Dense(input_size=dim, output_size=1),
            class_weights={"is_rare_event": 5.0},
            multilabel=False,
            loss_weight=1.0,
        )

    def test_binary_single_output_forward(self, binary_single_output_task, batch_size, num_hits, dim):
        """Test forward pass for binary classification with single output."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = binary_single_output_task.forward(x)

        assert "hit_logits" in outputs
        # Single output should be squeezed or have shape (..., 1)
        assert outputs["hit_logits"].shape[0] == batch_size
        assert outputs["hit_logits"].shape[1] == num_hits
        # Should be either (B, N) or (B, N, 1)
        assert outputs["hit_logits"].dim() in {2, 3}

    def test_binary_single_output_predict(self, binary_single_output_task, batch_size, num_hits, dim):
        """Test predict method for binary classification."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = binary_single_output_task.forward(x)
        predictions = binary_single_output_task.predict(outputs)

        assert "hit_is_first" in predictions
        assert predictions["hit_is_first"].shape == (batch_size, num_hits)
        assert predictions["hit_is_first"].dtype == torch.bool

    def test_binary_single_output_loss(self, binary_single_output_task, batch_size, num_hits, dim):
        """Test loss computation for binary classification."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = binary_single_output_task.forward(x)

        targets = {
            "hit_is_first": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        }

        losses = binary_single_output_task.loss(outputs, targets)
        assert "bce" in losses
        assert losses["bce"].shape == ()
        assert losses["bce"] >= 0

    def test_binary_with_invalid_hits(self, binary_single_output_task, batch_size, num_hits, dim):
        """Test that invalid hits are properly masked in loss."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = binary_single_output_task.forward(x)

        # Mark half of hits as invalid
        valid_mask = torch.zeros(batch_size, num_hits, dtype=torch.bool)
        valid_mask[:, : num_hits // 2] = True

        targets = {
            "hit_is_first": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_valid": valid_mask,
        }

        losses = binary_single_output_task.loss(outputs, targets)
        assert "bce" in losses
        assert not torch.isnan(losses["bce"])
        assert losses["bce"] >= 0

    def test_multilabel_forward(self, multilabel_task, batch_size, num_hits, dim):
        """Test forward pass for multilabel classification."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = multilabel_task.forward(x)

        assert "hit_logits" in outputs
        assert outputs["hit_logits"].shape == (batch_size, num_hits, 3)

    def test_multilabel_predict(self, multilabel_task, batch_size, num_hits, dim):
        """Test predict method for multilabel classification."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = multilabel_task.forward(x)
        predictions = multilabel_task.predict(outputs)

        assert "hit_is_first" in predictions
        assert "hit_is_last" in predictions
        assert "hit_is_vertex" in predictions

        for class_name in ["is_first", "is_last", "is_vertex"]:
            assert predictions[f"hit_{class_name}"].shape == (batch_size, num_hits)
            assert predictions[f"hit_{class_name}"].dtype == torch.bool

    def test_multilabel_loss(self, multilabel_task, batch_size, num_hits, dim):
        """Test loss computation for multilabel classification."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = multilabel_task.forward(x)

        targets = {
            "hit_is_first": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_is_last": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_is_vertex": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        }

        losses = multilabel_task.loss(outputs, targets)
        assert "bce" in losses
        assert losses["bce"].shape == ()
        assert losses["bce"] >= 0

    def test_multilabel_allows_multiple_labels(self, multilabel_task, batch_size, num_hits, dim):
        """Test that multilabel allows multiple True values per hit."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = multilabel_task.forward(x)

        # Create targets where some hits have multiple True labels
        targets = {
            "hit_is_first": torch.ones(batch_size, num_hits, dtype=torch.bool),
            "hit_is_last": torch.ones(batch_size, num_hits, dtype=torch.bool),
            "hit_is_vertex": torch.zeros(batch_size, num_hits, dtype=torch.bool),
            "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        }

        losses = multilabel_task.loss(outputs, targets)
        assert "bce" in losses
        assert not torch.isnan(losses["bce"])

    def test_multiclass_forward(self, multiclass_task, batch_size, dim):
        """Test forward pass for multi-class classification."""
        num_particles = 50
        x = {"particle_embed": torch.randn(batch_size, num_particles, dim)}
        outputs = multiclass_task.forward(x)

        assert "particle_logits" in outputs
        assert outputs["particle_logits"].shape == (batch_size, num_particles, 3)

    def test_multiclass_predict(self, multiclass_task, batch_size, dim):
        """Test predict method for multi-class classification."""
        num_particles = 50
        x = {"particle_embed": torch.randn(batch_size, num_particles, dim)}
        outputs = multiclass_task.forward(x)
        predictions = multiclass_task.predict(outputs)

        assert "particle_electron" in predictions
        assert "particle_muon" in predictions
        assert "particle_pion" in predictions

        for class_name in ["electron", "muon", "pion"]:
            assert predictions[f"particle_{class_name}"].shape == (batch_size, num_particles)
            assert predictions[f"particle_{class_name}"].dtype == torch.bool

        # Multi-class should have exactly one True per particle
        total_predictions = predictions["particle_electron"].int() + predictions["particle_muon"].int() + predictions["particle_pion"].int()
        assert torch.all(total_predictions == 1)

    def test_multiclass_loss(self, multiclass_task, batch_size, dim):
        """Test loss computation for multi-class classification."""
        num_particles = 50
        x = {"particle_embed": torch.randn(batch_size, num_particles, dim)}
        outputs = multiclass_task.forward(x)

        # Create one-hot encoded targets
        targets = {
            "particle_electron": torch.zeros(batch_size, num_particles, dtype=torch.bool),
            "particle_muon": torch.zeros(batch_size, num_particles, dtype=torch.bool),
            "particle_pion": torch.zeros(batch_size, num_particles, dtype=torch.bool),
            "particle_valid": torch.ones(batch_size, num_particles, dtype=torch.bool),
        }
        # Randomly assign each particle to one class
        for b in range(batch_size):
            for p in range(num_particles):
                class_idx = torch.randint(0, 3, (1,)).item()
                if class_idx == 0:
                    targets["particle_electron"][b, p] = True
                elif class_idx == 1:
                    targets["particle_muon"][b, p] = True
                else:
                    targets["particle_pion"][b, p] = True

        losses = multiclass_task.loss(outputs, targets)
        assert "bce" in losses
        assert losses["bce"].shape == ()
        assert losses["bce"] >= 0

    def test_weighted_binary_loss(self, weighted_binary_task, batch_size, num_hits, dim):
        """Test that class weights are applied in binary classification."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = weighted_binary_task.forward(x)

        # Create imbalanced targets (mostly False)
        targets = {
            "hit_is_rare_event": torch.zeros(batch_size, num_hits, dtype=torch.bool),
            "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        }
        # Only 10% are True
        targets["hit_is_rare_event"][:, : num_hits // 10] = True

        losses = weighted_binary_task.loss(outputs, targets)
        assert "bce" in losses
        assert not torch.isnan(losses["bce"])
        assert losses["bce"] >= 0

    def test_metrics(self, binary_single_output_task, batch_size, num_hits, dim):
        """Test metrics computation."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}
        outputs = binary_single_output_task.forward(x)
        predictions = binary_single_output_task.predict(outputs)

        targets = {
            "hit_is_first": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        }

        metrics = binary_single_output_task.metrics(predictions, targets)
        assert "is_first_eff" in metrics
        assert "is_first_pur" in metrics
        assert 0 <= metrics["is_first_eff"] <= 1
        assert 0 <= metrics["is_first_pur"] <= 1

    def test_threshold_effect_on_predictions(self, batch_size, num_hits, dim):
        """Test that threshold affects predictions."""
        # Create two tasks with different thresholds
        task_low_threshold = ClassificationTask(
            name="binary_low",
            input_object="hit",
            output_object="hit",
            target_object="hit",
            classes=["is_first"],
            net=Dense(input_size=dim, output_size=1),
            multilabel=False,
            loss_weight=1.0,
            threshold=0.1,
        )
        task_high_threshold = ClassificationTask(
            name="binary_high",
            input_object="hit",
            output_object="hit",
            target_object="hit",
            classes=["is_first"],
            net=Dense(input_size=dim, output_size=1),
            multilabel=False,
            loss_weight=1.0,
            threshold=0.9,
        )

        # Use the same network weights for fair comparison
        task_high_threshold.net.load_state_dict(task_low_threshold.net.state_dict())

        x = {"hit_embed": torch.randn(batch_size, num_hits, dim)}

        # Get outputs from both tasks (should be identical since same weights)
        outputs_low = task_low_threshold.forward(x)
        outputs_high = task_high_threshold.forward(x)

        pred_low = task_low_threshold.predict(outputs_low)
        pred_high = task_high_threshold.predict(outputs_high)

        # Lower threshold should predict more True values
        assert pred_low["hit_is_first"].sum() >= pred_high["hit_is_first"].sum()

    def test_gradient_flow(self, binary_single_output_task, batch_size, num_hits, dim):
        """Test that gradients flow properly through the task."""
        x = {"hit_embed": torch.randn(batch_size, num_hits, dim, requires_grad=True)}
        outputs = binary_single_output_task.forward(x)

        targets = {
            "hit_is_first": torch.randint(0, 2, (batch_size, num_hits), dtype=torch.bool),
            "hit_valid": torch.ones(batch_size, num_hits, dtype=torch.bool),
        }

        losses = binary_single_output_task.loss(outputs, targets)
        losses["bce"].backward()

        assert x["hit_embed"].grad is not None
        assert not torch.isnan(x["hit_embed"].grad).any()
