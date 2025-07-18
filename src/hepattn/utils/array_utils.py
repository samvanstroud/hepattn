import operator
from functools import reduce

import numpy as np


def masked_diff_last_axis(m: np.ma.MaskedArray) -> np.ma.MaskedArray:
    data = m.data
    mask = m.mask
    valid = ~mask

    m, n = data.shape
    # Build an index array [0,1,2,…,N-1] and broadcast it to shape (M, N)
    idxs = np.arange(n).reshape(1, n)

    # For each row, find “last valid index up to and including j”:
    last_valid = np.maximum.accumulate(np.where(valid, idxs, -1), axis=1)

    # Shift that right by one to get “previous valid before j”:
    prev_idx = np.concatenate([np.full((m, 1), -1, dtype=int), last_valid[:, :-1]], axis=1)

    # Clip negatives just for safe indexing (we will mask them out anyway)
    prev_idx_clipped = np.where(prev_idx < 0, 0, prev_idx)

    # Gather the “previous” values and subtract
    prev_vals = np.take_along_axis(data, prev_idx_clipped, axis=1)
    diffs = data - prev_vals

    # Mask out any slot where either the current was masked or there was no previous valid
    out_mask = mask | (prev_idx < 0)
    return np.ma.masked_array(diffs, mask=out_mask)


def masked_angle_diff_last_axis(ax, ay, az, mask) -> np.ma.MaskedArray:
    valid = ~mask

    m, n = mask.shape
    idxs = np.arange(n).reshape(1, n)

    last_valid = np.maximum.accumulate(np.where(valid, idxs, -1), axis=1)
    prev_idx = np.concatenate([np.full((m, 1), -1, dtype=int), last_valid[:, :-1]], axis=1)
    prev_idx_clipped = np.where(prev_idx < 0, 0, prev_idx)

    bx = np.take_along_axis(ax, prev_idx_clipped, axis=1)
    by = np.take_along_axis(ay, prev_idx_clipped, axis=1)
    bz = np.take_along_axis(az, prev_idx_clipped, axis=1)

    out_mask = mask | (prev_idx < 0)

    bx = np.ma.masked_array(bx, mask=out_mask)
    by = np.ma.masked_array(by, mask=out_mask)
    bz = np.ma.masked_array(bz, mask=out_mask)

    a_mag = np.ma.sqrt(ax**2 + ay**2 + az**2)
    b_mag = np.ma.sqrt(bx**2 + by**2 + bz**2)

    costheta = (ax * bx + ay * by + az * bz) / (a_mag * b_mag)
    theta = np.ma.arccos(costheta)

    return theta


def join_structured_arrays(arrays: list):
    """Join a list of structured numpy arrays. Taken from hepformer repo.

    See https://github.com/numpy/numpy/issues/7811

    Args:
        arrays (list): List of structured numpy arrays to join

    Returns:
        np.array: Merged structured array
        A merged structured array

    Raises:
        ValueError: If the input list is empty or if the arrays do not have the same
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    first_shape = arrays[0].shape
    if any(a.shape != first_shape for a in arrays):
        raise ValueError("All arrays in the list must have the same shape.")

    dtype: list = reduce(operator.add, (a.dtype.descr for a in arrays))
    newrecarray = np.empty(arrays[0].shape, dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]

    return newrecarray


def maybe_pad(x: np.ndarray, target_shape: tuple, pad_value: float = 0.0) -> np.ndarray:
    """Pads a numpy array `x` to match `target_shape`, using `pad_value`.
    numpy version of pad_to_size from hepattn.utils.tensor_utils.

    Args:
        x (np.ndarray): The input array to pad.
        target_shape (tuple of int): The desired shape of the output array. Use -1 to match input dim.
        pad_value (float): The constant value to use for padding.
        The constant value to use for padding.

    Returns:
        np.ndarray: Padded array of shape `target_shape`.

    Raises:
        ValueError: If `target_shape` does not have the same number of dimensions as `x` or if any dimension of `x` is larger than the corresponding
        dimension in `target_shape
    """
    current_shape = x.shape
    if len(target_shape) != x.ndim:
        raise ValueError(f"Target shape must have the same number of dimensions as x: {current_shape} vs {target_shape}")

    target_shape_ = []

    for i, (current, target) in enumerate(zip(current_shape, target_shape, strict=False)):
        if target == -1:
            target = current

        if current > target:
            raise ValueError(f"Cannot pad: dimension {i} of x is {current}, which is larger than target {target}.")

        target_shape_.append(target)

    target_shape = tuple(target_shape_)

    if current_shape == target_shape:
        return x

    new_array = np.full(target_shape, pad_value, dtype=x.dtype)

    index_slices = tuple(slice(0, cur) for cur in current_shape)

    new_array[index_slices] = x

    return new_array
