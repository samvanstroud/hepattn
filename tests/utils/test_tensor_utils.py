import numpy as np
import pytest

from hepattn.utils.array_utils import (
    join_structured_arrays,
    masked_angle_diff_last_axis,
    masked_diff_last_axis,
    maybe_pad,
)


class TestMaskedDiffLastAxis:
    """Tests for masked_diff_last_axis function."""

    def test_simple_case(self):
        """Test basic functionality with a simple array."""
        data = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
        mask = np.array([[False, False, False, False], [False, False, False, False]])
        m = np.ma.MaskedArray(data, mask=mask)

        result = masked_diff_last_axis(m)

        # First column should be masked (no previous valid)
        assert result.mask[0, 0]
        assert result.mask[1, 0]

        # Differences should be correct
        np.testing.assert_array_equal(result.data[0, 1:], [1, 1, 1])
        np.testing.assert_array_equal(result.data[1, 1:], [10, 10, 10])

        # Only first column should be masked
        assert not np.any(result.mask[:, 1:])

    def test_with_masked_values(self):
        """Test with some masked values in the input."""
        data = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
        mask = np.array([[False, True, False, False], [False, False, True, False]])
        m = np.ma.MaskedArray(data, mask=mask)

        result = masked_diff_last_axis(m)

        # First column should be masked
        assert result.mask[0, 0]
        assert result.mask[1, 0]

        # Second column of first row should be masked (input was masked)
        assert result.mask[0, 1]

        # Third column of first row should use value from index 0 (since index 1 was masked)
        assert result.data[0, 2] == 2  # 3 - 1
        assert not result.mask[0, 2]

    def test_all_masked_row(self):
        """Test with a completely masked row."""
        data = np.array([[1, 2, 3, 4]])
        mask = np.array([[True, True, True, True]])
        m = np.ma.MaskedArray(data, mask=mask)

        result = masked_diff_last_axis(m)

        # All should be masked
        assert np.all(result.mask)

    def test_single_column(self):
        """Test with single column array."""
        data = np.array([[5], [10]])
        mask = np.array([[False], [False]])
        m = np.ma.MaskedArray(data, mask=mask)

        result = masked_diff_last_axis(m)

        # Single column should be masked (no previous)
        assert np.all(result.mask)

    def test_empty_array(self):
        """Test with empty array."""
        data = np.empty((0, 0))
        mask = np.empty((0, 0), dtype=bool)
        m = np.ma.MaskedArray(data, mask=mask)

        result = masked_diff_last_axis(m)

        assert result.shape == (0, 0)


class TestMaskedAngleDiffLastAxis:
    """Tests for masked_angle_diff_last_axis function."""

    def test_simple_vectors(self):
        """Test with simple unit vectors."""
        # Two 2D vectors: (1,0,0) and (0,1,0) - should have 90 degree angle
        ax = np.array([[1.0, 0.0]])
        ay = np.array([[0.0, 1.0]])
        az = np.array([[0.0, 0.0]])
        mask = np.array([[False, False]])

        result = masked_angle_diff_last_axis(ax, ay, az, mask)

        # First should be masked (no previous)
        assert result.mask[0, 0]

        # Second should be 90 degrees (π/2)
        np.testing.assert_allclose(result.data[0, 1], np.pi / 2, rtol=1e-10)
        assert not result.mask[0, 1]

    def test_parallel_vectors(self):
        """Test with parallel vectors (angle should be 0)."""
        ax = np.array([[1.0, 1.0]])
        ay = np.array([[0.0, 0.0]])
        az = np.array([[0.0, 0.0]])
        mask = np.array([[False, False]])

        result = masked_angle_diff_last_axis(ax, ay, az, mask)

        # Second vector is parallel to first, angle should be 0
        np.testing.assert_allclose(result.data[0, 1], 0.0, atol=1e-10)

    def test_antiparallel_vectors(self):
        """Test with antiparallel vectors (angle should be π)."""
        ax = np.array([[1.0, -1.0]])
        ay = np.array([[0.0, 0.0]])
        az = np.array([[0.0, 0.0]])
        mask = np.array([[False, False]])

        result = masked_angle_diff_last_axis(ax, ay, az, mask)

        # Second vector is antiparallel to first, angle should be π
        np.testing.assert_allclose(result.data[0, 1], np.pi, rtol=1e-10)

    def test_with_masked_values(self):
        """Test with masked values in input."""
        ax = np.array([[1.0, 0.0, 1.0]])
        ay = np.array([[0.0, 1.0, 0.0]])
        az = np.array([[0.0, 0.0, 0.0]])
        mask = np.array([[False, True, False]])

        result = masked_angle_diff_last_axis(ax, ay, az, mask)

        # First should be masked (no previous)
        assert result.mask[0, 0]

        # Second should be masked (input was masked)
        assert result.mask[0, 1]

        # Third should use first vector as reference (since second was masked)
        np.testing.assert_allclose(result.data[0, 2], 0.0, atol=1e-10)
        assert not result.mask[0, 2]

    def test_3d_vectors(self):
        """Test with 3D vectors."""
        # (1,0,0) and (0,0,1) should have 90 degree angle
        ax = np.array([[1.0, 0.0]])
        ay = np.array([[0.0, 0.0]])
        az = np.array([[0.0, 1.0]])
        mask = np.array([[False, False]])

        result = masked_angle_diff_last_axis(ax, ay, az, mask)

        np.testing.assert_allclose(result.data[0, 1], np.pi / 2, rtol=1e-10)


class TestJoinStructuredArrays:
    """Tests for join_structured_arrays function."""

    def test_simple_join(self):
        """Test joining two simple structured arrays."""
        dt1 = np.dtype([("x", "f8"), ("y", "f8")])
        dt2 = np.dtype([("z", "f8")])

        arr1 = np.array([(1.0, 2.0), (3.0, 4.0)], dtype=dt1)
        arr2 = np.array([(5.0,), (6.0,)], dtype=dt2)

        result = join_structured_arrays([arr1, arr2])

        assert result.shape == (2,)
        assert "x" in result.dtype.names
        assert "y" in result.dtype.names
        assert "z" in result.dtype.names

        np.testing.assert_array_equal(result["x"], [1.0, 3.0])
        np.testing.assert_array_equal(result["y"], [2.0, 4.0])
        np.testing.assert_array_equal(result["z"], [5.0, 6.0])

    def test_multiple_arrays(self):
        """Test joining multiple structured arrays."""
        dt1 = np.dtype([("a", "i4")])
        dt2 = np.dtype([("b", "i4")])
        dt3 = np.dtype([("c", "i4")])

        arr1 = np.array([(1,), (2,)], dtype=dt1)
        arr2 = np.array([(3,), (4,)], dtype=dt2)
        arr3 = np.array([(5,), (6,)], dtype=dt3)

        result = join_structured_arrays([arr1, arr2, arr3])

        assert set(result.dtype.names) == {"a", "b", "c"}
        np.testing.assert_array_equal(result["a"], [1, 2])
        np.testing.assert_array_equal(result["b"], [3, 4])
        np.testing.assert_array_equal(result["c"], [5, 6])

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Input list of arrays cannot be empty"):
            join_structured_arrays([])

    def test_mismatched_shapes_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        dt = np.dtype([("x", "f8")])
        arr1 = np.array([(1.0,), (2.0,)], dtype=dt)
        arr2 = np.array([(3.0,)], dtype=dt)

        with pytest.raises(ValueError, match="All arrays in the list must have the same shape"):
            join_structured_arrays([arr1, arr2])

    def test_overlapping_field_names(self):
        """Test joining arrays with overlapping field names."""
        # Note: NumPy doesn't actually allow duplicate field names in practice
        # This test verifies that join_structured_arrays will fail gracefully
        # if someone tries to create overlapping field names
        dt1 = np.dtype([("x", "f8"), ("y", "f8")])
        dt2 = np.dtype([("x", "f8")])  # Overlapping field name

        arr1 = np.array([(1.0, 2.0)], dtype=dt1)
        arr2 = np.array([(3.0,)], dtype=dt2)

        # This should raise a ValueError due to duplicate field names
        with pytest.raises(ValueError, match=r"field.*occurs more than once"):
            join_structured_arrays([arr1, arr2])


class TestMaybePad:
    """Tests for maybe_pad function."""

    def test_no_padding_needed(self):
        """Test when no padding is needed."""
        x = np.array([[1, 2], [3, 4]])
        target_shape = (2, 2)

        result = maybe_pad(x, target_shape)

        np.testing.assert_array_equal(result, x)
        assert result.shape == target_shape

    def test_padding_2d_array(self):
        """Test padding a 2D array."""
        x = np.array([[1, 2], [3, 4]])
        target_shape = (3, 4)

        result = maybe_pad(x, target_shape, pad_value=0)

        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_equal(result, expected)
        assert result.shape == target_shape

    def test_custom_pad_value(self):
        """Test with custom pad value."""
        x = np.array([1, 2, 3])
        target_shape = (5,)

        result = maybe_pad(x, target_shape, pad_value=-1)

        expected = np.array([1, 2, 3, -1, -1])
        np.testing.assert_array_equal(result, expected)

    def test_3d_padding(self):
        """Test padding a 3D array."""
        x = np.ones((2, 3, 1))
        target_shape = (3, 4, 2)

        result = maybe_pad(x, target_shape, pad_value=0)

        assert result.shape == target_shape
        # Original data should be preserved
        assert np.all(result[:2, :3, :1] == 1)
        # Padded areas should be zero
        assert np.all(result[2:, :, :] == 0)
        assert np.all(result[:, 3:, :] == 0)
        assert np.all(result[:, :, 1:] == 0)

    def test_match_input_dimension(self):
        """Test using -1 to match input dimension."""
        x = np.ones((2, 3, 5))
        target_shape = (3, 4, -1)

        result = maybe_pad(x, target_shape, pad_value=0)

        assert result.shape == (3, 4, 5)

    def test_multiple_match_dimensions(self):
        """Test using multiple -1 values."""
        x = np.ones((2, 3, 5))
        target_shape = (-1, 4, -1)

        result = maybe_pad(x, target_shape, pad_value=0)

        assert result.shape == (2, 4, 5)

    def test_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises ValueError."""
        x = np.zeros((2, 2))
        target_shape = (2, 2, 2)

        with pytest.raises(ValueError, match="Target shape must have the same number of dimensions"):
            maybe_pad(x, target_shape)

    def test_negative_padding_raises_error(self):
        """Test that negative padding raises ValueError."""
        x = np.ones((4, 3))
        target_shape = (3, 2)

        with pytest.raises(ValueError, match=r"Cannot pad: dimension .* is .*, which is larger than target"):
            maybe_pad(x, target_shape)

    def test_empty_array(self):
        """Test with empty array."""
        x = np.empty((0, 0))
        target_shape = (2, 3)

        result = maybe_pad(x, target_shape, pad_value=7)

        expected = np.full((2, 3), 7)
        np.testing.assert_array_equal(result, expected)

    def test_different_dtypes(self):
        """Test that padding preserves dtype."""
        x = np.array([1, 2], dtype=np.int32)
        target_shape = (4,)

        result = maybe_pad(x, target_shape, pad_value=0)

        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, [1, 2, 0, 0])

    def test_float_array(self):
        """Test with float array."""
        x = np.array([[1.5, 2.5]], dtype=np.float32)
        target_shape = (2, 3)

        result = maybe_pad(x, target_shape, pad_value=0.0)

        assert result.dtype == np.float32
        expected = np.array([[1.5, 2.5, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_padding_from_zero_dimension(self):
        """Test padding when one dimension is zero."""
        x = np.empty((2, 0))
        target_shape = (2, 3)

        result = maybe_pad(x, target_shape, pad_value=7)

        expected = np.full((2, 3), 7)
        np.testing.assert_array_equal(result, expected)
