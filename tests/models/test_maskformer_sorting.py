import pytest
import torch
from hepattn.models.transformer import Encoder
from torch import nn

from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.maskformer import MaskFormer
from hepattn.utils.sorting import Sorter


class MockInputNet(nn.Module):
    def __init__(self, input_name: str, dim: int):
        super().__init__()
        self.input_name = input_name
        self.embedding = nn.Linear(1, dim)

    def forward(self, inputs):
        return self.embedding(inputs[f"{self.input_name}_data"].unsqueeze(-1))


class MockTask(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.outputs = ["output"]
        self.has_intermediate_loss = False
        self.permute_loss = True

    def forward(self, x):
        return {"output": torch.randn(1, 5, 10)}

    def predict(self, outputs):
        return {"prediction": outputs["output"] > 0}

    def cost(self, outputs, targets):
        return {"cost": torch.randn(1, 5, 5)}

    def loss(self, outputs, targets):
        return {"loss": torch.tensor(0.1)}


class TestMaskFormerSorting:
    @pytest.fixture
    def input_nets(self):
        return nn.ModuleList([MockInputNet("input1", dim=64), MockInputNet("input2", dim=64)])

    @pytest.fixture
    def encoder(self):
        return Encoder(num_layers=2, dim=64, window_size=4)

    @pytest.fixture
    def decoder(self):
        decoder_layer_config = {
            "dim": 64,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {},
            "bidirectional_ca": True,
            "hybrid_norm": False,
        }
        return MaskFormerDecoder(
            num_queries=5,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=2,
            mask_attention=False,  # Disable for simpler testing
        )

    @pytest.fixture
    def tasks(self):
        return nn.ModuleList([MockTask("test_task")])

    @pytest.fixture
    def sample_inputs(self):
        return {
            "input1_data": torch.randn(1, 10),
            "input1_valid": torch.ones(1, 10, dtype=torch.bool),
            "input1_phi": torch.randn(1, 10),  # Sort field
            "input2_data": torch.randn(1, 15),
            "input2_valid": torch.ones(1, 15, dtype=torch.bool),
            "input2_phi": torch.randn(1, 15),  # Sort field
        }

    def test_sort_before_encoder_creates_sorter(self, input_nets, encoder, decoder, tasks):
        """Test that sort_before_encoder=True creates a Sorter instance."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            input_sort_field="phi",
            sort_before_encoder=True,
        )

        assert isinstance(model.sorting, Sorter)

    def test_sorting_functionality(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that the sorting functionality works correctly."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            input_sort_field="phi",
            sort_before_encoder=True,
        )

        # Run forward pass
        outputs = model(sample_inputs)

        # Check that outputs are produced
        assert "final" in outputs
        assert "test_task" in outputs["final"]
        assert "output" in outputs["final"]["test_task"]
        assert "sort_indices" in outputs["final"]

    def test_sorting_without_encoder(self, input_nets, decoder, tasks, sample_inputs):
        """Test that sorting works even without an encoder."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=None,  # No encoder
            decoder=decoder,
            tasks=tasks,
            dim=64,
            input_sort_field="phi",
            sort_before_encoder=True,
        )

        # Run forward pass
        outputs = model(sample_inputs)

        # Check that outputs are produced
        assert "final" in outputs
        assert "test_task" in outputs["final"]
        assert model.sorting is not None
        assert "sort_indices" in outputs["final"]

    def test_sorting_without_sort_field_raises_error(self, input_nets, encoder, decoder, tasks):
        """Test that model raises error when sort_before_encoder=True but no sort field is provided."""
        with pytest.raises(AssertionError, match="input_sort_field must be provided if sort_before_encoder is True"):
            MaskFormer(
                input_nets=input_nets,
                encoder=encoder,
                decoder=decoder,
                tasks=tasks,
                dim=64,
                input_sort_field=None,
                sort_before_encoder=True,
            )

    def test_sorting_without_sort_field_works_when_disabled(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that model works when sort_before_encoder=False and no sort field is provided."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            input_sort_field=None,
            sort_before_encoder=False,
        )

        outputs = model(sample_inputs)

        # Check that outputs are produced
        assert "final" in outputs
        assert "test_task" in outputs["final"]

    def test_matcher_none_handling(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that the model handles None matcher correctly."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            input_sort_field="phi",
            sort_before_encoder=True,
            matcher=None,  # No matcher
        )

        # Create mock targets
        targets = {
            "particle_valid": torch.ones(1, 5, dtype=torch.bool),
            "test_task_target": torch.randn(1, 5, 10),
        }

        # Run forward pass
        outputs = model(sample_inputs)

        # Run loss computation (should not raise error with None matcher)
        losses = model.loss(outputs, targets)

        # Check that losses are computed
        assert "final" in losses
        assert "test_task" in losses["final"]

    def test_sorter_sort_indices_persistence(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that sort indices are properly stored and accessible."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            input_sort_field="phi",
            sort_before_encoder=True,
        )

        outputs = model(sample_inputs)

        assert hasattr(model.sorting, "sort_indices")
        assert "key" in model.sorting.sort_indices
        assert outputs["final"]["sort_indices"] == model.sorting.sort_indices
