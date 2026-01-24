"""Tests for dynamic query target building."""

import pytest
import torch

from hepattn.models.maskformer import MaskFormer


class TestDynamicTargets:
    """Tests for build_dynamic_targets."""

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

        query_hit_valid, first_occurrence, query_particle_valid = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # All queries should be first occurrence since each is from a different particle
        assert torch.all(first_occurrence), "All queries should be first occurrence when selecting one hit per particle"
        assert torch.all(query_particle_valid), "All queries should be valid"

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

        query_hit_valid, first_occurrence, query_particle_valid = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        # First occurrence should be True for queries 0 and 2 only
        expected_first = torch.tensor([[True, False, True, False]])
        assert torch.equal(first_occurrence, expected_first), f"Expected first occurrence {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # query_particle_valid should match first_occurrence
        assert torch.equal(query_particle_valid, expected_first), "query_particle_valid should match first_occurrence"

        # Non-first queries should have zeroed hit masks
        assert not query_hit_valid[0, 1].any(), "Query 1 (duplicate of P0) should have all-zero hit mask"
        assert not query_hit_valid[0, 3].any(), "Query 3 (duplicate of P3) should have all-zero hit mask"

        # First queries should have correct hit masks
        assert query_hit_valid[0, 0, 0] and query_hit_valid[0, 0, 1], "Query 0 should have hits 0, 1 valid"
        assert query_hit_valid[0, 2, 6:10].all(), "Query 2 should have hits 6-9 valid"

    def test_build_dynamic_targets_reversed_duplicate_order(self, synthetic_targets):
        """Test when the duplicate query comes BEFORE the first-occurrence query in index order.

        This tests the case where selected_hit_indices = [1, 0, ...] instead of [0, 1, ...].
        """
        # Select hits in reverse order for some particles
        # Hit 1 before hit 0 (both P0), hit 8 before hit 6 (both P3)
        selected_hit_indices = torch.tensor([1, 0, 8, 6, 5])  # 5 is P2 (only one hit)

        # Build dynamic targets
        query_hit_valid, first_occurrence, query_particle_valid = MaskFormer.build_dynamic_targets(
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

    # ==================== Interleaved (realistic) data tests ====================

    def test_build_dynamic_targets_interleaved_basic(self, interleaved_targets):
        """Test build_dynamic_targets with interleaved hits (one hit per particle)."""
        # Select one hit per particle: hit 0 (P100), hit 1 (P200), hit 3 (P300), hit 6 (P400)
        selected_hit_indices = torch.tensor([0, 1, 3, 6])

        query_hit_valid, first_occurrence, query_particle_valid = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        # All queries should be first occurrence
        assert torch.all(first_occurrence), "All queries should be first occurrence"
        assert torch.all(query_particle_valid), "All queries should be valid"

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

        query_hit_valid, first_occurrence, _query_particle_valid = MaskFormer.build_dynamic_targets(
            selected_hit_indices, interleaved_targets, source_name="hit", target_name="particle"
        )

        # First occurrence: query 0 (hit 0, P100) is first, query 1 (hit 7, P100) is NOT first
        expected_first = torch.tensor([[True, False, True]])
        assert torch.equal(first_occurrence, expected_first), f"Expected first occurrence {expected_first.tolist()}, got {first_occurrence.tolist()}"

        # Query 1 should have zeroed mask
        assert not query_hit_valid[0, 1].any(), "Query 1 (duplicate) should have all-zero hit mask"

        # Query 0 should still have all hits for P100
        assert query_hit_valid[0, 0, [0, 2, 7]].all(), "Query 0 should have hits 0, 2, 7 valid"

    def test_build_dynamic_targets_missing_particles(self, synthetic_targets):
        """Test build_dynamic_targets when some particles are not selected."""
        # Select hits from particles 0 and 2 only. Particle 1 and 3 are missing.
        selected_hit_indices = torch.tensor([0, 5])  # Hit 0 (P0), Hit 5 (P2)

        query_hit_valid, first_occurrence, query_particle_valid = MaskFormer.build_dynamic_targets(
            selected_hit_indices, synthetic_targets, source_name="hit", target_name="particle"
        )

        assert query_particle_valid.all()
        # Both queries are first occurrence (different particles)
        assert first_occurrence.all()

    def test_build_dynamic_targets_unmatched_hits(self, synthetic_targets):
        """Test build_dynamic_targets behavior with hits that don't match any target particle ID."""
        # Add a noise hit that doesn't match any particle_id
        targets = synthetic_targets.copy()
        targets["hit_particle_id"] = torch.cat([targets["hit_particle_id"], torch.tensor([[999]])], dim=1)

        # Select hits [0, 10] where 10 is the noise hit
        selected_hit_indices = torch.tensor([0, 10])

        _, first_occurrence, _ = MaskFormer.build_dynamic_targets(
            selected_hit_indices, targets, source_name="hit", target_name="particle"
        )

        # The noise hit (999) doesn't match any ID in [100, 200, 300, 400]
        # However, it SHOULD still be marked as a first occurrence if it's the first time 999 is seen
        assert first_occurrence[0, 1]
