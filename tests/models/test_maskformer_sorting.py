import pytest
import torch
from torch import nn

from hepattn.models import Encoder
from hepattn.models.decoder import MaskFormerDecoder
from hepattn.models.maskformer import MaskFormer
from hepattn.utils.sorting import Sorter


class MockInputNet(nn.Module):
    def __init__(self, input_name: str, dim: int):
        super().__init__()
        self.input_name = input_name
        self.net = nn.Linear(2, dim)

    def forward(self, inputs):
        input_data = torch.cat(
            [v.unsqueeze(-1) for k, v in inputs.items() if k.startswith(self.input_name) and "valid" not in k and "phi" not in k], dim=-1
        )
        return self.net(input_data)


class MockTask(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.outputs = ["output"]
        self.has_intermediate_loss = False
        self.permute_loss = True

    def forward(self, x):
        return {"output": torch.randn(2, 5, 10)}

    def predict(self, outputs):
        return {"prediction": outputs["output"] > 0}

    def cost(self, outputs, targets):
        return {"cost": torch.randn(2, 5, 5)}

    def loss(self, outputs, targets):
        return {"loss": torch.tensor(0.1)}


class MockMatcher(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, costs, object_valid_mask=None):
        # Return identity permutation for simplicity
        batch_size, num_pred, _ = costs.shape
        return torch.arange(num_pred).unsqueeze(0).expand(batch_size, -1)


class TestMaskFormerSorting:
    @pytest.fixture
    def input_nets(self):
        return nn.ModuleList([MockInputNet("input1", dim=64), MockInputNet("input2", dim=64)])

    @pytest.fixture
    def encoder(self):
        return Encoder(num_layers=2, dim=64)

    @pytest.fixture
    def decoder(self):
        decoder_layer_config = {
            "dim": 64,
            "norm": "LayerNorm",
            "dense_kwargs": {},
            "attn_kwargs": {},
            "bidirectional_ca": True,
        }
        return MaskFormerDecoder(
            num_queries=5,
            decoder_layer_config=decoder_layer_config,
            num_decoder_layers=2,
            mask_attention=False,  # Disable for simpler testing # TODO: fix
        )

    @pytest.fixture
    def tasks(self):
        return nn.ModuleList([MockTask("test_task")])

    @pytest.fixture
    def sample_inputs(self):
        return {
            "input1_x": torch.randn(2, 10),
            "input1_y": torch.randn(2, 10),
            "input1_valid": torch.ones(2, 10, dtype=torch.bool),
            "input1_phi": torch.tensor([
                [3.0, 1.0, 4.0, 2.0, 5.0, 0.0, 6.0, 7.0, 8.0, 9.0],
                [3.0, 1.0, 4.0, 2.0, 5.0, 0.0, 6.0, 7.0, 8.0, 9.0],
            ]),  # Specific unsorted values
            "input2_x": torch.randn(2, 15),
            "input2_y": torch.randn(2, 15),
            "input2_valid": torch.ones(2, 15, dtype=torch.bool),
            "input2_phi": torch.tensor([
                [15.0, 11.0, 14.0, 12.0, 13.0, 10.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [15.0, 11.0, 14.0, 12.0, 13.0, 10.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            ]),  # Specific unsorted values
        }

    def test_sorter_sort_indices_persistence(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that sort indices are properly stored and accessible."""
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            matcher=MockMatcher(),
            sorter=Sorter(
                input_sort_field="phi",
            ),
        )

        outputs = model(sample_inputs)

        # The sort indices should be accessible from the model's sorting attribute
        assert outputs["final"]["phi"]["input1_phi"] is not None
        assert outputs["final"]["phi"]["input2_phi"] is not None

    def test_no_sorting_does_not_break(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that nothing breaks when no sorting is applied."""
        # Create a model without sorting
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            sorter=None,
            matcher=MockMatcher(),
        )

        # Run forward pass - should not raise any errors
        outputs = model(sample_inputs)

        # Check that outputs are produced correctly
        assert "final" in outputs
        assert "test_task" in outputs["final"]

        # Test loss computation without sorting
        targets = {
            "particle_hit_valid": torch.ones(1, 10, 5, dtype=torch.bool),  # Match the default target_object
            "particle_valid": torch.ones(1, 10, 5, dtype=torch.bool),
        }

        # Should not raise any errors
        losses, _ = model.loss(outputs, targets)

        # Check that losses are computed
        assert "final" in losses
        assert "test_task" in losses["final"]

    def test_sorting_with_multiple_input_types_and_targets(self, input_nets, encoder, decoder, tasks, sample_inputs):
        """Test that sorting works correctly with multiple input types and that targets are sorted properly."""
        # Create a model with sorting enabled for multiple input types
        model = MaskFormer(
            input_nets=input_nets,
            encoder=encoder,
            decoder=decoder,
            tasks=tasks,
            dim=64,
            matcher=MockMatcher(),
            sorter=Sorter(
                input_sort_field="phi",
            ),
        )

        # Test target sorting with multiple target fields
        # Create targets with known unsorted values to make sorting obvious
        original_input1_phi = sample_inputs["input1_phi"].clone()
        original_input2_phi = sample_inputs["input2_phi"].clone()

        # Run forward pass
        outputs = model(sample_inputs)

        assert outputs["final"]["phi"]["input1_phi"] is not None
        assert outputs["final"]["phi"]["input2_phi"] is not None

        # Create random boolean tensors for target masks
        original_particle_input1_valid = torch.randint(0, 2, (2, 5, 10), dtype=torch.bool)
        original_particle_input2_valid = torch.randint(0, 2, (2, 5, 15), dtype=torch.bool)

        targets = {
            "particle_valid": torch.ones(2, 5, dtype=torch.bool),
            "particle_input1_valid": original_particle_input1_valid,
            "particle_input2_valid": original_particle_input2_valid,
            "input1_target": torch.randn(2, 10),
            "input2_target": torch.randn(2, 15),
        }

        # Run loss computation which should sort targets
        _, sorted_targets = model.loss(outputs, targets)

        # Get the sort indices that were used for the input phi values
        input1_sort_idx = torch.argsort(original_input1_phi)
        input2_sort_idx = torch.argsort(original_input2_phi)

        # Check that the targets are sorted by verifying they're different from the original
        # (since we're using unsorted values, sorting should change the order)
        assert not torch.allclose(sorted_targets["particle_input1_valid"], original_particle_input1_valid)
        assert not torch.allclose(sorted_targets["particle_input2_valid"], original_particle_input2_valid)

        # Verify that the valid tensors are reordered according to the phi sorting
        # The valid tensors should be reordered using the same indices that sort the phi values
        expected_input1_valid = torch.gather(
            original_particle_input1_valid, 2, input1_sort_idx.unsqueeze(1).expand_as(original_particle_input1_valid)
        )
        expected_input2_valid = torch.gather(
            original_particle_input2_valid, 2, input2_sort_idx.unsqueeze(1).expand_as(original_particle_input2_valid)
        )

        # Verify that the sorted targets are actually sorted
        assert torch.allclose(sorted_targets["particle_input1_valid"], expected_input1_valid)
        assert torch.allclose(sorted_targets["particle_input2_valid"], expected_input2_valid)
