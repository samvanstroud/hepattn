"""Tests for dynamic query target building and prediction alignment."""

import pytest
import torch

from hepattn.models.maskformer import MaskFormer


class TestDynamicTargets:
    """Tests for build_dynamic_targets and _align_predictions_to_full_targets round-trip consistency."""

    @pytest.fixture
    def synthetic_targets(self):
        """Create synthetic targets with known hit-to-particle mappings (grouped by particle).

        Scenario:
        - 10 hits total
        - 4 particles with IDs [100, 200, 300, 400]
        - Hit-to-particle mapping (grouped):
            - Hit 0, 1 -> Particle 100 (index 0)
            - Hit 2, 3, 4 -> Particle 200 (index 1)
            - Hit 5 -> Particle 300 (index 2)
            - Hit 6, 7, 8, 9 -> Particle 400 (index 3)
        """
        num_hits = 10
        num_particles = 4

        # Particle IDs
        particle_id = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)

        # Hit to particle ID mapping (grouped by particle)
        hit_particle_id = torch.tensor([[100, 100, 200, 200, 200, 300, 400, 400, 400, 400]], dtype=torch.long)

        # Ground truth particle_hit_valid: which hits belong to each particle
        # Shape: (1, N_particles, N_hits) - computed via broadcast comparison
        particle_hit_valid = (particle_id[0].unsqueeze(-1) == hit_particle_id[0].unsqueeze(-2)).unsqueeze(0)

        # All particles are valid
        particle_valid = torch.ones(1, num_particles, dtype=torch.bool)

        return {
            "particle_id": particle_id,
            "hit_particle_id": hit_particle_id,
            "particle_hit_valid": particle_hit_valid,
            "particle_valid": particle_valid,
            "num_hits": num_hits,
            "num_particles": num_particles,
        }

    @pytest.fixture
    def interleaved_targets(self):
        """Create synthetic targets with interleaved hit-to-particle mappings (realistic).

        In real HEP data, hits are ordered by detector layer (radially outward),
        so hits from the same particle are scattered throughout the sequence.

        Scenario:
        - 10 hits total (representing 10 detector layers)
        - 4 particles with IDs [100, 200, 300, 400]
        - Hit-to-particle mapping (interleaved, as if ordered by layer):
            - Hit 0 -> P100, Hit 1 -> P200, Hit 2 -> P100, Hit 3 -> P300
            - Hit 4 -> P200, Hit 5 -> P300, Hit 6 -> P400, Hit 7 -> P100
            - Hit 8 -> P400, Hit 9 -> P400

        Particle hit counts:
            - Particle 100 (index 0): hits 0, 2, 7 (3 hits)
            - Particle 200 (index 1): hits 1, 4 (2 hits)
            - Particle 300 (index 2): hits 3, 5 (2 hits)
            - Particle 400 (index 3): hits 6, 8, 9 (3 hits)
        """
        num_hits = 10
        num_particles = 4

        # Particle IDs
        particle_id = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)

        # Hit to particle ID mapping (interleaved - realistic detector ordering)
        hit_particle_id = torch.tensor([[100, 200, 100, 300, 200, 300, 400, 100, 400, 400]], dtype=torch.long)

        # Ground truth particle_hit_valid: which hits belong to each particle
        # Shape: (1, N_particles, N_hits) - computed via broadcast comparison
        particle_hit_valid = (particle_id[0].unsqueeze(-1) == hit_particle_id[0].unsqueeze(-2)).unsqueeze(0)

        # All particles are valid
        particle_valid = torch.ones(1, num_particles, dtype=torch.bool)

        return {
            "particle_id": particle_id,
            "hit_particle_id": hit_particle_id,
            "particle_hit_valid": particle_hit_valid,
            "particle_valid": particle_valid,
            "num_hits": num_hits,
            "num_particles": num_particles,
        }

    def test_build_dynamic_targets_basic(self, synthetic_targets):
        """Test that build_dynamic_targets correctly builds query-to-particle mappings."""
        # Select hits 0, 2, 5, 6 (one hit from each particle)
        selected_hit_indices = torch.tensor([0, 2, 5, 6])

        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # All queries should be first occurrence since each is from a different particle
        assert torch.all(first_occurrence), "All queries should be first occurrence when selecting one hit per particle"
        assert torch.all(query_particle_valid), "All queries should be valid"

        # Check particle index mapping
        expected_particle_idx = torch.tensor([[0, 1, 2, 3]])  # Hit 0->P0, Hit 2->P1, Hit 5->P2, Hit 6->P3
        assert torch.equal(query_particle_idx, expected_particle_idx), (
            f"Expected particle indices {expected_particle_idx.tolist()}, got {query_particle_idx.tolist()}"
        )

        # Check hit validity masks
        # Query 0 (particle 0) should have hits 0, 1 valid
        assert query_hit_valid[0, 0, 0] and query_hit_valid[0, 0, 1], "Query 0 should have hits 0, 1 valid"
        assert not query_hit_valid[0, 0, 2], "Query 0 should not have hit 2 valid"

    def test_build_dynamic_targets_duplicate_particles(self, synthetic_targets):
        """Test that duplicate hits from same particle are handled correctly.

        When multiple hits from the same particle are selected, only the first
        occurrence should have valid targets (to avoid duplicate predictions).
        """
        # Select hits 0, 1 (both from particle 100), and 6, 8 (both from particle 400)
        # Order: hit 0 (P0), hit 1 (P0), hit 6 (P3), hit 8 (P3)
        selected_hit_indices = torch.tensor([0, 1, 6, 8])

        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # First occurrence should be True for queries 0 and 2 only
        expected_first = torch.tensor([[True, False, True, False]])
        assert torch.equal(first_occurrence, expected_first), f"Expected first occurrence {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # query_particle_valid should match first_occurrence
        assert torch.equal(query_particle_valid, expected_first), "query_particle_valid should match first_occurrence"

        # Particle index mapping
        expected_particle_idx = torch.tensor([[0, 0, 3, 3]])  # Hit 0,1->P0, Hit 6,8->P3
        assert torch.equal(query_particle_idx, expected_particle_idx), (
            f"Expected particle indices {expected_particle_idx.tolist()}, got {query_particle_idx.tolist()}"
        )

        # Non-first queries should have zeroed hit masks
        assert not query_hit_valid[0, 1].any(), "Query 1 (duplicate of P0) should have all-zero hit mask"
        assert not query_hit_valid[0, 3].any(), "Query 3 (duplicate of P3) should have all-zero hit mask"

        # First queries should have correct hit masks
        assert query_hit_valid[0, 0, 0] and query_hit_valid[0, 0, 1], "Query 0 should have hits 0, 1 valid"
        assert query_hit_valid[0, 2, 6:10].all(), "Query 2 should have hits 6-9 valid"

    def test_round_trip_consistency_no_duplicates(self, synthetic_targets):
        """Test that build_dynamic_targets followed by _align_predictions_to_full_targets
        correctly recovers predictions when there are no duplicate hits per particle.
        """
        # Select one hit per particle
        selected_hit_indices = torch.tensor([0, 2, 5, 6])
        num_full_particles = synthetic_targets["num_particles"]
        num_hits = synthetic_targets["num_hits"]

        # Build dynamic targets
        query_hit_valid, _first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # Create mock predictions that match the query targets
        # track_hit_valid predictions should match query_hit_valid
        mock_preds = {
            "final": {
                "track": {
                    "track_hit_valid": query_hit_valid.float(),  # (1, N_queries, N_hits)
                    "track_valid": query_particle_valid.float(),  # (1, N_queries)
                }
            }
        }

        # Create a minimal mock ModelWrapper to use _align_predictions_to_full_targets
        aligned_preds = self._align_predictions(mock_preds, query_particle_idx, num_full_particles)

        # Verify aligned predictions match original targets
        aligned_track_hit_valid = aligned_preds["final"]["track"]["track_hit_valid"]
        original_particle_hit_valid = synthetic_targets["particle_hit_valid"].float()

        assert aligned_track_hit_valid.shape == (1, num_full_particles, num_hits), (
            f"Expected shape (1, {num_full_particles}, {num_hits}), got {aligned_track_hit_valid.shape}"
        )

        # Check each particle's hit mask was correctly recovered
        for p_idx in range(num_full_particles):
            assert torch.equal(aligned_track_hit_valid[0, p_idx], original_particle_hit_valid[0, p_idx]), (
                f"Particle {p_idx} hit mask not correctly recovered. "
                f"Expected {original_particle_hit_valid[0, p_idx].tolist()}, "
                f"got {aligned_track_hit_valid[0, p_idx].tolist()}"
            )

    def test_round_trip_consistency_with_duplicates(self, synthetic_targets):
        """Test round-trip consistency when multiple hits from the same particle are selected.

        This is the critical test that verifies the implementation handles duplicates correctly.
        The scatter operation in _align_predictions_to_full_targets must not overwrite valid
        targets with zeroed ones from duplicate queries.

        WARNING: This test is expected to FAIL with the current implementation if the scatter
        loop processes queries in order and later (zeroed) duplicates overwrite earlier valid values.
        """
        # Select hits with duplicates: hits 0, 1 (both P0), hits 6, 8 (both P3)
        # Also select hit 2 (P1) to have a non-duplicate particle
        selected_hit_indices = torch.tensor([0, 1, 2, 6, 8])
        num_full_particles = synthetic_targets["num_particles"]

        # Build dynamic targets
        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # Verify build_dynamic_targets correctly identified first occurrences
        # Queries: [0, 1, 2, 6, 8] -> Particles: [0, 0, 1, 3, 3]
        # First occurrence: [T, F, T, T, F]
        expected_first = torch.tensor([[True, False, True, True, False]])
        assert torch.equal(first_occurrence, expected_first), f"Expected first occurrence {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # Create mock predictions matching dynamic targets
        mock_preds = {
            "final": {
                "track": {
                    "track_hit_valid": query_hit_valid.float(),  # (1, N_queries, N_hits)
                    "track_valid": query_particle_valid.float(),  # (1, N_queries)
                }
            }
        }

        # Align predictions back to full particle dimension
        aligned_preds = self._align_predictions(mock_preds, query_particle_idx, num_full_particles)

        aligned_track_hit_valid = aligned_preds["final"]["track"]["track_hit_valid"]
        original_particle_hit_valid = synthetic_targets["particle_hit_valid"].float()

        # Check that particles with at least one selected hit have correct masks
        # Particle 0: should have hits 0, 1 valid (from query 0, NOT overwritten by query 1)
        assert aligned_track_hit_valid[0, 0, 0] == 1.0, "Particle 0, hit 0 should be valid. Possible overwrite by duplicate query."
        assert aligned_track_hit_valid[0, 0, 1] == 1.0, "Particle 0, hit 1 should be valid. Possible overwrite by duplicate query."

        # Particle 1: should have hits 2, 3, 4 valid (no duplicate, query 2)
        assert aligned_track_hit_valid[0, 1, 2:5].sum() == 3.0, (
            f"Particle 1 should have 3 valid hits. Got {aligned_track_hit_valid[0, 1, 2:5].tolist()}"
        )

        # Particle 2: not selected, should be all zeros
        assert aligned_track_hit_valid[0, 2].sum() == 0.0, "Particle 2 was not selected, should be all zeros"

        # Particle 3: should have hits 6, 7, 8, 9 valid (from query 3, NOT overwritten by query 4)
        assert aligned_track_hit_valid[0, 3, 6:10].sum() == 4.0, (
            f"Particle 3 should have 4 valid hits. Got {aligned_track_hit_valid[0, 3, 6:10].tolist()}. Possible overwrite by duplicate query."
        )

        # Full comparison for selected particles
        for p_idx in [0, 1, 3]:  # Particles that were selected
            assert torch.equal(aligned_track_hit_valid[0, p_idx], original_particle_hit_valid[0, p_idx]), (
                f"Particle {p_idx} hit mask not correctly recovered after round-trip. "
                f"Expected {original_particle_hit_valid[0, p_idx].tolist()}, "
                f"got {aligned_track_hit_valid[0, p_idx].tolist()}"
            )

    def test_round_trip_with_reversed_duplicate_order(self, synthetic_targets):
        """Test when the duplicate query comes BEFORE the first-occurrence query in index order.

        This tests the case where selected_hit_indices = [1, 0, ...] instead of [0, 1, ...].
        If the scatter loop processes in order, it might write the first-occurrence (query 1)
        and then... well, query 0 was marked as not-first so it would also be zeroed.

        Actually with our ordering preservation fix, indices are sorted, so this tests that
        first_occurrence detection still works correctly after sorting.
        """
        # Select hits in reverse order for some particles
        # Hit 1 before hit 0 (both P0), hit 8 before hit 6 (both P3)
        selected_hit_indices = torch.tensor([1, 0, 8, 6, 5])  # 5 is P2 (only one hit)
        num_full_particles = synthetic_targets["num_particles"]

        # Build dynamic targets
        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # With hits [1, 0, 8, 6, 5] -> Particles [0, 0, 3, 3, 2]
        # First occurrence: query 0 (hit 1, P0) is first for P0
        #                   query 1 (hit 0, P0) is NOT first for P0
        #                   query 2 (hit 8, P3) is first for P3
        #                   query 3 (hit 6, P3) is NOT first for P3
        #                   query 4 (hit 5, P2) is first for P2
        expected_first = torch.tensor([[True, False, True, False, True]])
        assert torch.equal(first_occurrence, expected_first), f"Expected first occurrence {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # Create mock predictions
        mock_preds = {
            "final": {
                "track": {
                    "track_hit_valid": query_hit_valid.float(),
                    "track_valid": query_particle_valid.float(),
                }
            }
        }

        # Align predictions
        aligned_preds = self._align_predictions(mock_preds, query_particle_idx, num_full_particles)
        aligned_track_hit_valid = aligned_preds["final"]["track"]["track_hit_valid"]

        # Particle 0: query 0 (hit 1) is first occurrence, should have hits 0, 1 valid
        assert aligned_track_hit_valid[0, 0, 0:2].sum() == 2.0, (
            f"Particle 0 should have 2 valid hits. Got {aligned_track_hit_valid[0, 0, :].tolist()}. First occurrence detection may be incorrect."
        )

        # Particle 2: single hit at index 5
        assert aligned_track_hit_valid[0, 2, 5] == 1.0, "Particle 2 should have hit 5 valid"

        # Particle 3: query 2 (hit 8) is first occurrence, should have hits 6-9 valid
        assert aligned_track_hit_valid[0, 3, 6:10].sum() == 4.0, (
            f"Particle 3 should have 4 valid hits. Got {aligned_track_hit_valid[0, 3, 6:10].tolist()}. First occurrence detection may be incorrect."
        )

    # ==================== Interleaved (realistic) data tests ====================

    def test_build_dynamic_targets_interleaved_basic(self, interleaved_targets):
        """Test build_dynamic_targets with interleaved hits (one hit per particle)."""
        # Select one hit per particle: hit 0 (P100), hit 1 (P200), hit 3 (P300), hit 6 (P400)
        selected_hit_indices = torch.tensor([0, 1, 3, 6])

        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        # All queries should be first occurrence
        assert torch.all(first_occurrence), "All queries should be first occurrence"
        assert torch.all(query_particle_valid), "All queries should be valid"

        # Check particle index mapping
        expected_particle_idx = torch.tensor([[0, 1, 2, 3]])
        assert torch.equal(query_particle_idx, expected_particle_idx), (
            f"Expected particle indices {expected_particle_idx.tolist()}, got {query_particle_idx.tolist()}"
        )

        # Check hit validity masks for interleaved data
        # Query 0 (particle 0, ID 100) should have hits 0, 2, 7 valid
        assert query_hit_valid[0, 0, [0, 2, 7]].all(), "Query 0 should have hits 0, 2, 7 valid"
        assert query_hit_valid[0, 0].sum() == 3, f"Query 0 should have exactly 3 valid hits, got {query_hit_valid[0, 0].sum()}"

        # Query 1 (particle 1, ID 200) should have hits 1, 4 valid
        assert query_hit_valid[0, 1, [1, 4]].all(), "Query 1 should have hits 1, 4 valid"
        assert query_hit_valid[0, 1].sum() == 2, "Query 1 should have exactly 2 valid hits"

    def test_build_dynamic_targets_interleaved_duplicates(self, interleaved_targets):
        """Test duplicate handling with interleaved hits.

        Select multiple hits from same particle where hits are not adjacent.
        """
        # Select hits 0 and 7 (both P100, but separated), and hit 6 (P400)
        selected_hit_indices = torch.tensor([0, 7, 6])

        query_hit_valid, first_occurrence, _query_particle_valid, _query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        # First occurrence: query 0 (hit 0, P100) is first, query 1 (hit 7, P100) is NOT first
        expected_first = torch.tensor([[True, False, True]])
        assert torch.equal(first_occurrence, expected_first), f"Expected first occurrence {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # Query 1 should have zeroed mask
        assert not query_hit_valid[0, 1].any(), "Query 1 (duplicate) should have all-zero hit mask"

        # Query 0 should still have all hits for P100
        assert query_hit_valid[0, 0, [0, 2, 7]].all(), "Query 0 should have hits 0, 2, 7 valid"

    def test_round_trip_interleaved_no_duplicates(self, interleaved_targets):
        """Test round-trip with interleaved data, no duplicates."""
        # Select one hit per particle
        selected_hit_indices = torch.tensor([0, 1, 3, 6])
        num_full_particles = interleaved_targets["num_particles"]

        query_hit_valid, _first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        mock_preds = {
            "final": {
                "track": {
                    "track_hit_valid": query_hit_valid.float(),
                    "track_valid": query_particle_valid.float(),
                }
            }
        }

        aligned_preds = self._align_predictions(mock_preds, query_particle_idx, num_full_particles)
        aligned_track_hit_valid = aligned_preds["final"]["track"]["track_hit_valid"]
        original_particle_hit_valid = interleaved_targets["particle_hit_valid"].float()

        # Check each particle's hit mask was correctly recovered
        for p_idx in range(num_full_particles):
            assert torch.equal(aligned_track_hit_valid[0, p_idx], original_particle_hit_valid[0, p_idx]), (
                f"Particle {p_idx} hit mask not correctly recovered. "
                f"Expected {original_particle_hit_valid[0, p_idx].nonzero().squeeze().tolist()}, "
                f"got {aligned_track_hit_valid[0, p_idx].nonzero().squeeze().tolist()}"
            )

    def test_round_trip_interleaved_with_duplicates(self, interleaved_targets):
        """Test round-trip with interleaved data and duplicate hits from same particle.

        This is a realistic scenario: hits 0, 2, 7 all belong to P100 but are scattered.
        Selecting hits 0 and 7 should still recover all of P100's hits (0, 2, 7).
        """
        # Select: hit 0 (P100), hit 7 (P100 - duplicate), hit 1 (P200), hit 8 (P400)
        selected_hit_indices = torch.tensor([0, 7, 1, 8])
        num_full_particles = interleaved_targets["num_particles"]

        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        # Queries: hit 0 (P0), hit 7 (P0), hit 1 (P1), hit 8 (P3)
        # First occurrence: [T, F, T, T]
        expected_first = torch.tensor([[True, False, True, True]])
        assert torch.equal(first_occurrence, expected_first)

        mock_preds = {
            "final": {
                "track": {
                    "track_hit_valid": query_hit_valid.float(),
                    "track_valid": query_particle_valid.float(),
                }
            }
        }

        aligned_preds = self._align_predictions(mock_preds, query_particle_idx, num_full_particles)
        aligned_track_hit_valid = aligned_preds["final"]["track"]["track_hit_valid"]
        original_particle_hit_valid = interleaved_targets["particle_hit_valid"].float()

        # Particle 0 (P100): should have hits 0, 2, 7 valid
        assert aligned_track_hit_valid[0, 0, [0, 2, 7]].sum() == 3.0, (
            f"Particle 0 should have 3 valid hits at indices 0, 2, 7. Got {aligned_track_hit_valid[0, 0].nonzero().squeeze().tolist()}"
        )

        # Particle 1 (P200): should have hits 1, 4 valid
        assert aligned_track_hit_valid[0, 1, [1, 4]].sum() == 2.0, (
            f"Particle 1 should have 2 valid hits. Got {aligned_track_hit_valid[0, 1].nonzero().squeeze().tolist()}"
        )

        # Particle 2 (P300): not selected, should be all zeros
        assert aligned_track_hit_valid[0, 2].sum() == 0.0, "Particle 2 was not selected"

        # Particle 3 (P400): should have hits 6, 8, 9 valid
        assert aligned_track_hit_valid[0, 3, [6, 8, 9]].sum() == 3.0, (
            f"Particle 3 should have 3 valid hits. Got {aligned_track_hit_valid[0, 3].nonzero().squeeze().tolist()}"
        )

        # Full comparison for selected particles
        for p_idx in [0, 1, 3]:
            assert torch.equal(aligned_track_hit_valid[0, p_idx], original_particle_hit_valid[0, p_idx]), f"Particle {p_idx} hit mask mismatch"

    def test_interleaved_all_hits_from_one_particle(self, interleaved_targets):
        """Test when all selected hits belong to the same particle (edge case)."""
        # Select all hits from P100: hits 0, 2, 7
        selected_hit_indices = torch.tensor([0, 2, 7])
        num_full_particles = interleaved_targets["num_particles"]

        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        # Only first query should be valid
        expected_first = torch.tensor([[True, False, False]])
        assert torch.equal(first_occurrence, expected_first), f"Expected {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # All map to particle 0
        expected_idx = torch.tensor([[0, 0, 0]])
        assert torch.equal(query_particle_idx, expected_idx)

        mock_preds = {
            "final": {
                "track": {
                    "track_hit_valid": query_hit_valid.float(),
                    "track_valid": query_particle_valid.float(),
                }
            }
        }

        aligned_preds = self._align_predictions(mock_preds, query_particle_idx, num_full_particles)
        aligned_track_hit_valid = aligned_preds["final"]["track"]["track_hit_valid"]

        # Only particle 0 should have valid hits
        assert aligned_track_hit_valid[0, 0, [0, 2, 7]].sum() == 3.0
        assert aligned_track_hit_valid[0, 1:].sum() == 0.0, "Other particles should have no valid hits"

    def test_align_outputs_nested_structure(self, synthetic_targets):
        """Test that _align_outputs_to_original_targets handles nested layers and tasks."""
        num_queries = 3
        num_original_targets = 5
        query_target_idx = torch.tensor([[0, 2, 4]])  # Queries map to targets 0, 2, 4

        mock_outputs = {
            "layer_0": {
                "task_a": {
                    "field_1": torch.randn(1, num_queries, 8),
                },
            },
            "layer_1": {
                "task_a": {
                    "field_1": torch.randn(1, num_queries, 8),
                },
                "task_b": {
                    "field_2": torch.randn(1, num_queries),
                },
            },
            "final": {
                "task_c": {
                    "field_3": torch.randn(1, num_queries, 10, 10),
                }
            },
            "encoder": {"some_metadata": True},  # Should be preserved as is
        }

        aligned = MaskFormer._align_outputs_to_original_targets(mock_outputs, query_target_idx, num_original_targets)

        # Check structure preservation
        assert set(aligned.keys()) == {"layer_0", "layer_1", "final", "encoder"}
        assert aligned["encoder"]["some_metadata"] is True

        # Check alignment and shapes
        assert aligned["layer_0"]["task_a"]["field_1"].shape == (1, num_original_targets, 8)
        assert aligned["layer_1"]["task_b"]["field_2"].shape == (1, num_original_targets)
        assert aligned["final"]["task_c"]["field_3"].shape == (1, num_original_targets, 10, 10)

        # Check specific values (e.g., target 2 should have data from query 1)
        torch.testing.assert_close(aligned["layer_1"]["task_b"]["field_2"][0, 2], mock_outputs["layer_1"]["task_b"]["field_2"][0, 1])
        # Target 1 was not mapped to, should be zero
        assert aligned["layer_1"]["task_b"]["field_2"][0, 1] == 0.0

    def test_build_dynamic_targets_missing_particles(self, synthetic_targets):
        """Test build_dynamic_targets when some particles are not selected."""
        # Select hits from particles 0 and 2 only. Particle 1 and 3 are missing.
        selected_hit_indices = torch.tensor([0, 5])  # Hit 0 (P0), Hit 5 (P2)

        query_hit_valid, first_occurrence, query_particle_valid, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        assert query_particle_idx.tolist() == [[0, 2]]
        assert query_particle_valid.all()

        # Original particles 1 and 3 are not in query_particle_idx
        assert 1 not in query_particle_idx
        assert 3 not in query_particle_idx

    def test_build_dynamic_targets_unmatched_hits(self, synthetic_targets):
        """Test build_dynamic_targets behavior with hits that don't match any target particle ID."""
        # Add a noise hit that doesn't match any particle_id
        targets = synthetic_targets.copy()
        targets["hit_particle_id"] = torch.cat([targets["hit_particle_id"], torch.tensor([[999]])], dim=1)

        # Select hits [0, 10] where 10 is the noise hit
        selected_hit_indices = torch.tensor([0, 10])

        _, first_occurrence, _, query_particle_idx = MaskFormer.build_dynamic_targets(
            selected_hit_indices, targets, source_name="hit", target_name="particle"
        )

        # The noise hit (999) doesn't match any ID in [100, 200, 300, 400]
        # Matches mask will be all False for that row.
        # Long().argmax(dim=1) returns 0 if all are False.
        # So query 1 (the noise) will map to target index 0 if not handled.
        assert query_particle_idx[0, 1] == 0

        # However, it SHOULD still be marked as a first occurrence if it's the first time 999 is seen
        assert first_occurrence[0, 1]

    def test_align_outputs_different_shapes(self):
        """Test alignment of outputs that don't have a query dimension."""
        num_queries = 2
        num_original_targets = 4
        query_target_idx = torch.tensor([[1, 3]])

        mock_outputs = {
            "final": {
                "global_task": {"scalar": torch.tensor([1.23]), "batch_only": torch.randn(1, 10), "query_field": torch.randn(1, num_queries, 5)}
            }
        }

        aligned = MaskFormer._align_outputs_to_original_targets(mock_outputs, query_target_idx, num_original_targets)

        # Scalars and batch-only tensors should be preserved exactly
        torch.testing.assert_close(aligned["final"]["global_task"]["scalar"], mock_outputs["final"]["global_task"]["scalar"])
        torch.testing.assert_close(aligned["final"]["global_task"]["batch_only"], mock_outputs["final"]["global_task"]["batch_only"])

        # Query field should be expanded
        assert aligned["final"]["global_task"]["query_field"].shape == (1, num_original_targets, 5)

    # Use the production code directly - it's a static method
    _align_predictions = staticmethod(MaskFormer._align_outputs_to_original_targets)
