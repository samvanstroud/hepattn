import pytest
import torch

from hepattn.experiments.atlas_muon.data import AtlasMuonDataset


class TestAtlasMuonEvent:
    @pytest.fixture
    def atlas_muon_event(self):
        input_fields = {
            "hit": [
                "spacePoint_globEdgeHighX",
                "spacePoint_globEdgeHighY",
                "spacePoint_globEdgeHighZ",
                "spacePoint_globEdgeLowX",
                "spacePoint_globEdgeLowY",
                "spacePoint_globEdgeLowZ",
                "spacePoint_driftR",
                "spacePoint_channel",
                "spacePoint_layer",
                "spacePoint_stationPhi",
                "spacePoint_stationEta",
                "spacePoint_stationIndex",
                "spacePoint_technology",
                "r",
                "s",
                "theta",
                "phi",
                "eta",
            ]
        }

        target_fields = {
            "particle": ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"],
            "hit": ["on_valid_particle"],
        }

        dataset = AtlasMuonDataset(
            dirpath="dummy",
            inputs=input_fields,
            targets=target_fields,
            num_events=4,
            event_max_num_particles=2,
            dummy_data=True,
        )

        return dataset[0]

    def test_atlas_muon_event_masks(self, atlas_muon_event):
        """Test that particle hit masks are consistent with particle validity."""
        _inputs, targets = atlas_muon_event

        particle_valid = targets["particle_valid"]
        particle_hit_mask = targets["particle_hit_valid"]

        # Invalid particle slots should have no hits
        assert torch.all(~particle_hit_mask[~particle_valid.unsqueeze(-1).expand_as(particle_hit_mask)])

    def test_atlas_muon_event_shapes(self, atlas_muon_event):
        """Test that event tensors have expected shapes."""
        inputs, targets = atlas_muon_event

        # Check that hit inputs exist and have correct batch dimension
        assert "hit_valid" in inputs
        assert inputs["hit_valid"].dim() == 2  # [batch, num_hits]
        assert inputs["hit_valid"].shape[0] == 1  # batch size 1 for single event

        # Check particle targets
        assert "particle_valid" in targets
        assert targets["particle_valid"].dim() == 2  # [batch, num_particles]
        assert targets["particle_valid"].shape[0] == 1  # batch size 1

        # Check particle-hit mask
        assert "particle_hit_valid" in targets
        assert targets["particle_hit_valid"].dim() == 3  # [batch, num_particles, num_hits]

    def test_atlas_muon_event_hit_inputs(self, atlas_muon_event):
        """Test that all expected hit input fields are present."""
        inputs, _targets = atlas_muon_event

        expected_fields = [
            "hit_spacePoint_globEdgeHighX",
            "hit_spacePoint_globEdgeHighY",
            "hit_spacePoint_globEdgeHighZ",
            "hit_r",
            "hit_theta",
            "hit_phi",
            "hit_eta",
        ]

        for field in expected_fields:
            assert field in inputs, f"Missing expected input field: {field}"

    def test_atlas_muon_event_particle_targets(self, atlas_muon_event):
        """Test that all expected particle target fields are present."""
        _inputs, targets = atlas_muon_event

        expected_fields = [
            "particle_truthMuon_pt",
            "particle_truthMuon_eta",
            "particle_truthMuon_phi",
            "particle_truthMuon_q",
        ]

        for field in expected_fields:
            assert field in targets, f"Missing expected target field: {field}"

    def test_atlas_muon_dataset_length(self):
        """Test that dataset reports correct length."""
        dataset = AtlasMuonDataset(
            dirpath="dummy",
            inputs={"hit": ["r", "phi"]},
            targets={"particle": ["truthMuon_pt"]},
            num_events=10,
            event_max_num_particles=2,
            dummy_data=True,
        )

        assert len(dataset) == 10

    def test_atlas_muon_dataset_deterministic(self):
        """Test that dummy data generation is deterministic with same seed."""
        kwargs = {
            "dirpath": "dummy",
            "inputs": {"hit": ["r", "phi", "eta"]},
            "targets": {"particle": ["truthMuon_pt"]},
            "num_events": 4,
            "event_max_num_particles": 2,
            "dummy_data": True,
        }

        dataset1 = AtlasMuonDataset(**kwargs)
        dataset2 = AtlasMuonDataset(**kwargs)

        inputs1, _targets1 = dataset1[0]
        inputs2, _targets2 = dataset2[0]

        # Same seed should produce same data
        assert torch.allclose(inputs1["hit_r"], inputs2["hit_r"])
        assert torch.allclose(inputs1["hit_phi"], inputs2["hit_phi"])
