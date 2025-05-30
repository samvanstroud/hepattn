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
