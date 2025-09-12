import pytest
import torch

from hepattn.models.task import ObjectHitMaskTask

BATCH_SIZE = 2
NUM_QUERIES = 5
NUM_CONSTITUENTS = 10
DIM = 64


class TestObjectHitMaskTaskBasics:
    """Test class for ObjectHitMaskTask in both traditional and unified decoding modes."""

    @pytest.fixture
    def task_config(self):
        return {
            "name": "track_hit_valid",
            "input_constituent": "hit",
            "input_object": "query",
            "output_object": "track",
            "target_object": "particle",
            "losses": {"mask_bce": 1.0, "mask_focal": 0.5},
            "costs": {"mask_bce": 1.0, "mask_focal": 0.5},
            "dim": DIM,
            "null_weight": 1.0,
            "mask_attn": True,
            "target_field": "valid",
            "logit_scale": 1.0,
            "pred_threshold": 0.5,
            "has_intermediate_loss": True,
        }

    @pytest.fixture
    def traditional_task(self, task_config):
        """ObjectHitMaskTask with unified_decoding=False (traditional mode)."""
        return ObjectHitMaskTask(**task_config, unified_decoding=False)

    @pytest.fixture
    def unified_task(self, task_config):
        """ObjectHitMaskTask with unified_decoding=True (unified mode)."""
        return ObjectHitMaskTask(**task_config, unified_decoding=True)

    @pytest.fixture
    def traditional_inputs(self):
        """Input data for traditional mode with separate embeddings."""
        return {
            "query_embed": torch.randn(BATCH_SIZE, NUM_QUERIES, DIM),
            "hit_embed": torch.randn(BATCH_SIZE, NUM_CONSTITUENTS, DIM),
            "hit_valid": torch.ones(BATCH_SIZE, NUM_CONSTITUENTS, dtype=torch.bool),
        }

    @pytest.fixture
    def unified_inputs(self):
        """Input data for unified mode with merged embeddings."""
        return {
            "query_embed": torch.randn(BATCH_SIZE, NUM_QUERIES, DIM),
            "key_embed": torch.randn(BATCH_SIZE, NUM_CONSTITUENTS, DIM),
            "key_valid": torch.ones(BATCH_SIZE, NUM_CONSTITUENTS, dtype=torch.bool),
        }

    @pytest.fixture
    def sample_targets(self):
        """Sample target data for testing loss and cost functions."""
        return {
            "particle_hit_valid": torch.randint(0, 2, (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS), dtype=torch.float),
            "hit_valid": torch.ones(BATCH_SIZE, NUM_CONSTITUENTS, dtype=torch.bool),
            "particle_valid": torch.ones(BATCH_SIZE, NUM_QUERIES, dtype=torch.bool),
        }

    def test_traditional_initialization(self, traditional_task):
        """Test traditional task initialization."""
        assert traditional_task.name == "track_hit_valid"
        assert traditional_task.input_constituent == "hit"
        assert traditional_task.input_object == "query"
        assert traditional_task.unified_decoding is False
        assert traditional_task.inputs == ["query_embed", "hit_embed"]
        assert traditional_task.outputs == ["track_hit_logit"]

    def test_unified_initialization(self, unified_task):
        """Test unified task initialization."""
        assert unified_task.name == "track_hit_valid"
        assert unified_task.input_constituent == "hit"
        assert unified_task.input_object == "query"
        assert unified_task.unified_decoding is True
        assert unified_task.inputs == ["query_embed", "key_embed"]
        assert unified_task.outputs == ["track_hit_logit"]

    def test_traditional_forward(self, traditional_task, traditional_inputs):
        """Test forward pass in traditional mode."""
        outputs = traditional_task(traditional_inputs)

        # Check output structure
        assert "track_hit_logit" in outputs
        track_hit_logit = outputs["track_hit_logit"]

        # Check output shape: (batch_size, num_queries, num_constituents)
        assert track_hit_logit.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS)
        assert track_hit_logit.dtype == torch.float32

    def test_unified_forward(self, unified_task, unified_inputs):
        """Test forward pass in unified mode."""
        outputs = unified_task(unified_inputs)

        # Check output structure
        assert "track_hit_logit" in outputs
        track_hit_logit = outputs["track_hit_logit"]

        # Check output shape: (batch_size, num_queries, num_constituents)
        assert track_hit_logit.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS)
        assert track_hit_logit.dtype == torch.float32

    def test_traditional_attn_mask(self, traditional_task, traditional_inputs):
        """Test attention mask generation in traditional mode."""
        outputs = traditional_task(traditional_inputs)
        attn_masks = traditional_task.attn_mask(outputs, threshold=0.1)

        # Should return mask for the specific input constituent
        assert "hit" in attn_masks
        assert "key" not in attn_masks

        attn_mask = attn_masks["hit"]
        assert attn_mask.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS)
        assert attn_mask.dtype == torch.bool

    def test_unified_attn_mask(self, unified_task, unified_inputs):
        """Test attention mask generation in unified mode."""
        outputs = unified_task(unified_inputs)
        attn_masks = unified_task.attn_mask(outputs, threshold=0.1)

        # Should return mask for the full merged tensor
        assert "key" in attn_masks
        assert "hit" not in attn_masks

        attn_mask = attn_masks["key"]
        assert attn_mask.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS)
        assert attn_mask.dtype == torch.bool

    def test_traditional_predict(self, traditional_task, traditional_inputs):
        """Test prediction in traditional mode."""
        outputs = traditional_task(traditional_inputs)
        predictions = traditional_task.predict(outputs)

        assert "track_hit_valid" in predictions
        pred = predictions["track_hit_valid"]
        assert pred.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS)
        assert pred.dtype == torch.bool

    def test_unified_predict(self, unified_task, unified_inputs):
        """Test prediction in unified mode."""
        outputs = unified_task(unified_inputs)
        predictions = unified_task.predict(outputs)

        assert "track_hit_valid" in predictions
        pred = predictions["track_hit_valid"]
        assert pred.shape == (BATCH_SIZE, NUM_QUERIES, NUM_CONSTITUENTS)
        assert pred.dtype == torch.bool

    def test_mask_consistency_between_modes(self):
        """Test that attention masks are consistent between traditional and unified modes."""
        # Create tasks with identical configurations except for unified_decoding
        config = {
            "name": "test_task",
            "input_constituent": "hit",
            "input_object": "query",
            "output_object": "track",
            "target_object": "particle",
            "losses": {"mask_bce": 1.0},
            "costs": {"mask_bce": 1.0},
            "dim": DIM,
            "mask_attn": True,
        }

        traditional_task = ObjectHitMaskTask(**config, unified_decoding=False)
        unified_task = ObjectHitMaskTask(**config, unified_decoding=True)

        # Create inputs with the same underlying data
        torch.manual_seed(42)  # Ensure reproducible results
        query_embed = torch.randn(BATCH_SIZE, NUM_QUERIES, DIM)
        constituent_embed = torch.randn(BATCH_SIZE, NUM_CONSTITUENTS, DIM)
        valid_mask = torch.ones(BATCH_SIZE, NUM_CONSTITUENTS, dtype=torch.bool)

        traditional_inputs = {
            "query_embed": query_embed,
            "hit_embed": constituent_embed,
            "hit_valid": valid_mask,
        }

        unified_inputs = {
            "query_embed": query_embed,
            "key_embed": constituent_embed,  # Same data, different key
            "key_valid": valid_mask,  # Same data, different key
        }

        # Get outputs from both modes
        traditional_outputs = traditional_task(traditional_inputs)
        unified_outputs = unified_task(unified_inputs)

        # Get attention masks with the same threshold
        threshold = 0.1
        traditional_masks = traditional_task.attn_mask(traditional_outputs, threshold=threshold)
        unified_masks = unified_task.attn_mask(unified_outputs, threshold=threshold)

        # Traditional mode should return mask for "hit", unified mode for "key"
        assert "hit" in traditional_masks
        assert "key" in unified_masks
        assert "key" not in traditional_masks
        assert "hit" not in unified_masks

        # The actual mask values should be functionally equivalent
        # since they're computed from the same underlying logits
        traditional_mask = traditional_masks["hit"]
        unified_mask = unified_masks["key"]

        # Both masks should have the same shape
        assert traditional_mask.shape == unified_mask.shape
        assert traditional_mask.dtype == unified_mask.dtype

        # The masks should be computed from logits that have the same shape
        # and similar distributions (though exact values may differ due to
        # potentially different embedding processing)
        assert traditional_outputs["track_hit_logit"].shape == unified_outputs["track_hit_logit"].shape

        # Test with a mix of valid and invalid constituents
        mixed_valid_mask = torch.tensor(
            [[True, True, False, True, False, True, False, True, False, True], [False, True, True, False, True, True, False, False, True, True]],
            dtype=torch.bool,
        )

        traditional_inputs["hit_valid"] = mixed_valid_mask
        unified_inputs["key_valid"] = mixed_valid_mask

        traditional_outputs_mixed = traditional_task(traditional_inputs)
        unified_outputs_mixed = unified_task(unified_inputs)

        traditional_masks_mixed = traditional_task.attn_mask(traditional_outputs_mixed, threshold=threshold)
        unified_masks_mixed = unified_task.attn_mask(unified_outputs_mixed, threshold=threshold)

        # Both should respect the validity mask in the same way
        traditional_mask_mixed = traditional_masks_mixed["hit"]
        unified_mask_mixed = unified_masks_mixed["key"]

        # Check that invalid positions are consistently masked to False in both modes
        # In both modes, attention should be masked (False) for invalid constituents
        # Check across all query positions for invalid constituents
        for batch_idx in range(BATCH_SIZE):
            for const_idx in range(NUM_CONSTITUENTS):
                if not mixed_valid_mask[batch_idx, const_idx]:
                    # All queries should have False attention to invalid constituents
                    assert torch.all(~traditional_mask_mixed[batch_idx, :, const_idx])
                    assert torch.all(~unified_mask_mixed[batch_idx, :, const_idx])
