
"""
Unified, vectorized, and fast evaluation (PID or Incidence matrix)
Based on metrics in https://arxiv.org/pdf/1904.06778

Truth modes:
  - 'incidence': use provided truth incidence matrix [P, H]
  - 'pid':       build incidence from parts/hits pid arrays (pid==0 -> noise)
  - 'auto':      prefer 'incidence' if available, else 'pid'
"""

import h5py
import numpy as np
import pandas as pd

# Optional sparse acceleration for very sparse truth matrices
try:
    from scipy.sparse import csr_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# -------------------------------
# Fast utilities
# -------------------------------

def get_masks(logits, threshold=0.5):
    """
    Convert mask logits to boolean masks without calling sigmoid.
    Note: sigmoid(x) > 0.5  <=>  x > 0
          For a different threshold t: x > log(t/(1-t))
    """
    if threshold == 0.5:
        return logits > 0
    tau = np.log(threshold / (1.0 - threshold))
    return logits > tau


def is_valid_track_old(n_assigned, class_pred):
    """Track is valid if it has >=3 hits and class_pred == 0."""
    return (n_assigned >= 3) & (class_pred == 0)

def is_valid_track_new(n_assigned, class_pred):
    """Track is valid if it has >=3 hits and class_pred == 0."""
    return (n_assigned >= 3) & (class_pred)

def _has(g, key):
    try:
        _ = g[key]
        return True
    except KeyError:
        return False


def _safe_get(g, key, required=True):
    if not _has(g, key):
        if required:
            raise KeyError(f"Missing dataset '{key}'")
        return None
    return g[key][:]


def _build_incidence_from_pids(parts_pid, hits_pid):
    """
    Build boolean incidence [P, H] from arrays of particle pids and hit pids.
    pid==0 (or pid missing in parts_pid) => unclaimed hit.
    Returns (incidence_bool, pid_to_pidx)
    """
    parts_pid = np.asarray(parts_pid, dtype=int)
    hits_pid = np.asarray(hits_pid, dtype=int)

    P = parts_pid.shape[0]
    H = hits_pid.shape[0]
    # Map pid -> pidx using parts order
    pid_to_pidx = {int(pid): i for i, pid in enumerate(parts_pid)}

    incidence = np.zeros((P, H), dtype=bool)
    # hits with valid (non-zero) pid that exists in parts
    valid = (hits_pid != 0) & np.vectorize(pid_to_pidx.__contains__)(hits_pid)
    if valid.any():
        hidx = np.nonzero(valid)[0]
        pidx = np.fromiter((pid_to_pidx[int(pid)] for pid in hits_pid[valid]), count=hidx.size, dtype=int)
        incidence[pidx, hidx] = True

    return incidence, pid_to_pidx


def _maybe_sparse(A_bool, auto=True, force_dense=False):
    """
    Optionally convert a boolean matrix to CSR if it's quite sparse and SciPy is available.
    Returns (matrix, is_sparse)
    """
    if force_dense or (not auto) or (not _HAS_SCIPY):
        return A_bool.astype(np.int8), False
    density = A_bool.mean() if A_bool.size else 1.0
    if density < 0.10:  # heuristic threshold
        return csr_matrix(A_bool.astype(np.int8)), True
    return A_bool.astype(np.int8), False


def _rowwise_unique_mask(masks_bool):
    """
    Detect duplicate rows in a boolean matrix.
    Returns a boolean array 'is_duplicate' marking every row that is a duplicate
    of a previous row (first occurrence is NOT marked).
    """
    N, H = masks_bool.shape
    if N == 0:
        return np.zeros((0,), dtype=bool)

    # Pack bits along the hit axis to reduce width; ensure contiguous.
    packed = np.packbits(masks_bool, axis=1)
    packed = np.ascontiguousarray(packed)

    # View each row as a single fixed-size blob so np.unique works row-wise.
    row_bytes = packed.dtype.itemsize * packed.shape[1]
    flat_rows = packed.view(np.dtype((np.void, row_bytes))).ravel()

    _, first_idx = np.unique(flat_rows, return_index=True)
    is_dup = np.ones(N, dtype=bool)
    is_dup[first_idx] = False
    return is_dup


# -------------------------------
# Loading & truth incidence
# -------------------------------

def load_event(fname, idx, key_mode="old"):
    """
    Returns:
      hits_df : DataFrame indexed by hidx
      tracks  : DataFrame (class_pred, n_assigned, kinematics if present)
      masks_b : bool array [N, H]
      parts   : DataFrame indexed by pidx (pt, eta, phi, vz, ...), may include 'original_pid'
      A_bool  : truth incidence boolean [P, H]
    """
    with h5py.File(fname, "r") as f:
        if key_mode == "old":
            g = f[f"event_{idx}"]
        else:
            g = f[str(idx)]

        # predictions
        if key_mode == "old":
            masks_logits = _safe_get(g, "preds/masks")
            masks_b = get_masks(masks_logits)  # [N, H]
        elif key_mode == "new":
            masks_b = g["preds/final/track_hit_valid/track_hit_valid"][0]

        if key_mode == "old":
            class_preds = _safe_get(g, "preds/class_preds")
            class_pred = class_preds.argmax(-1)
        elif key_mode == "new":
            class_pred = g["preds/final/track_valid/track_valid"][0]

        tracks = pd.DataFrame({"class_pred": class_pred})
        tracks["n_assigned"] = masks_b.sum(axis=1)

        if _has(g, "preds/regression"):
            reg = g["preds/regression"]
            tracks["px"] = reg["px"][:]
            tracks["py"] = reg["py"][:]
            tracks["pz"] = reg["pz"][:]
            tracks["pt"] = np.sqrt(tracks["px"].to_numpy()**2 + tracks["py"].to_numpy()**2)
            tracks["phi"] = np.arctan2(tracks["py"].to_numpy(), tracks["px"].to_numpy())
            theta = np.arctan2(tracks["pt"].to_numpy(), tracks["pz"].to_numpy())
            tracks["eta"] = -np.log(np.tan(0.5 * theta))
        else:
            tracks["pt"] = np.nan
            tracks["phi"] = np.nan
            tracks["eta"] = np.nan

        # parts kinematics
        if key_mode == "old":
            parts = pd.DataFrame({
                "pt":  _safe_get(g, "parts/pts"),
                "eta": _safe_get(g, "parts/etas"),
                "phi": _safe_get(g, "parts/phis"),
                # "vz":  _safe_get(g, "parts/vzs"),
            })
        elif key_mode == "new":
            parts = pd.DataFrame({
                "pt": g["targets/particle_pt"][0],
                "eta": g["targets/particle_eta"][0],
                "phi": g["targets/particle_phi"][0],
            })
        parts.index.name = "pidx"

        # truth incidence (choose mode)
        chosen = "incidence" if _has(g, "targets/particle_hit_valid") else "pid"

        if chosen == "incidence":
            A_bool = _safe_get(g, "targets/particle_hit_valid").astype(bool)
            A_bool = np.squeeze(A_bool, axis=0)
            print(A_bool.shape)
            P, H = A_bool.shape
            if P != len(parts):
                raise ValueError(f"Incidence rows (P={P}) must equal len(parts) ({len(parts)}).")
            hits_df = pd.DataFrame(index=pd.RangeIndex(H, name="hidx"))

        elif chosen == "pid":
            parts_pid = _safe_get(g, "parts/pids")
            hits_pid = _safe_get(g, "hits/pids")
            A_bool, pid_to_pidx = _build_incidence_from_pids(parts_pid, hits_pid)
            P, H = A_bool.shape
            if P != len(parts):
                raise ValueError(f"Parts length ({len(parts)}) must match P from parts/pids ({P}).")
            parts["original_pid"] = parts_pid
            hits_df = pd.DataFrame(index=pd.RangeIndex(H, name="hidx"))
        else:
            raise ValueError("truth_mode must be one of {'auto','incidence','pid'}")

        if masks_b.shape[1] != hits_df.index.size:
            raise ValueError(f"Mask width H={masks_b.shape[1]} != number of hits H={hits_df.index.size}")

    return hits_df, tracks, masks_b, parts, A_bool


# -------------------------------
# Metrics (all vectorized)
# -------------------------------

def process_particles(parts, A_bool, eta_cut=2.5, pt_cut=0.6):
    parts = parts.copy()
    parts["n_hits"] = A_bool.sum(axis=1)
    parts["reconstructable"] = (parts["n_hits"] >= 3) & (np.abs(parts["eta"].to_numpy()) < eta_cut) & ((np.abs(parts["pt"].to_numpy()) > pt_cut))
    return parts


def process_tracks(parts, hits_df, tracks_df, masks_bool, A_bool, use_sparse_auto=True, key_mode="old"):
    """
    Vectorized matching & metrics.
    - masks_bool: [N, H] bool
    - A_bool:     [P, H] bool (truth incidence)
    """
    P, H = A_bool.shape
    N = masks_bool.shape[0]
    if H != masks_bool.shape[1]:
        raise ValueError("masks/hits size mismatch")

    # Validity
    n_assigned = masks_bool.sum(axis=1).astype(int)
    class_pred = tracks_df["class_pred"].to_numpy()
    if key_mode == "old":
        valid = is_valid_track_old(n_assigned, class_pred)
    elif key_mode == "new":
        valid = is_valid_track_new(n_assigned, class_pred)
    # If nothing valid, return minimal frame
    out = tracks_df.copy()
    out["n_assigned"] = n_assigned
    out["valid"] = valid
    if not valid.any():
        out = out[[]]
        return out

    # Int8 representations; maybe sparse truth
    A_int, is_sparse = _maybe_sparse(A_bool, auto=use_sparse_auto, force_dense=False)
    M_int = masks_bool.astype(np.int8)  # [N, H]

    # Overlaps for ALL tracks at once: [P, H] @ [H, N] -> [P, N]
    overlaps = A_int @ M_int.T

    # Track-wise best particle and overlap
    best_pidx = overlaps.argmax(axis=0)          # [N]
    best_ovlp = overlaps.max(axis=0).astype(int) # [N]
    matched = best_ovlp > 0

    # Greedy single-assignment: keep first available particle by descending overlap
    order = np.argsort(-best_ovlp, kind="mergesort")  # stable
    keep = np.zeros(N, dtype=bool)
    taken = np.zeros(P, dtype=bool)
    for t in order:
        if not (valid[t] and matched[t]):
            continue
        p = best_pidx[t]
        if not taken[p]:
            keep[t] = True
            taken[p] = True

    # Metrics
    n_truth_hits = A_bool.sum(axis=1).astype(int)       # [P]
    denom_track = np.maximum(n_assigned, 1)
    denom_truth = np.maximum(n_truth_hits[best_pidx], 1)

    recall = best_ovlp / denom_truth
    precision = best_ovlp / denom_track

    eff_dm      = (recall > 0.5) & (precision > 0.5)
    eff_lhc     = (precision > 0.75)
    eff_perfect = (recall == 1.0) & (precision == 1.0)

    # Duplicate masks (identical assigned hits). Vectorized.
    dup_mask = _rowwise_unique_mask(masks_bool) & valid

    # Build output once
    out = tracks_df.copy()
    out["n_assigned"] = n_assigned
    out["valid"] = valid
    matched_idx = valid & matched & keep
    out["matched_pidx"] = np.where(matched_idx, best_pidx, -1)
    out["recall"]    = np.where(matched_idx, recall, 0.0)
    out["precision"] = np.where(matched_idx, precision, 0.0)
    out["duplicate"] = dup_mask
    out["eff_dm"] = matched_idx & eff_dm & (~dup_mask)
    out["eff_lhc"] = matched_idx & eff_lhc & (~dup_mask)
    out["eff_perfect"] = matched_idx & eff_perfect & (~dup_mask)

    # Matched kinematics
    N = out.shape[0]
    mp = out["matched_pidx"].to_numpy()
    good = mp != -1
    for col in ["pt", "eta", "phi"]:
        arr = np.full(N, np.nan, dtype=float)
        if good.any():
            arr[good] = parts[col].to_numpy()[mp[good]]
        out[f"matched_{col}"] = arr
    rec_arr = np.zeros(N, dtype=bool)
    if "reconstructable" in parts.columns and good.any():
        rec_arr[good] = parts["reconstructable"].to_numpy()[mp[good]]
    out["matched_reconstructable"] = rec_arr

    # Keep only valid rows (mimic original pipeline)
    out = out[out["valid"]]

    return out


# -------------------------------
# Top-level evaluation
# -------------------------------

def eval_event(fname, idx, eta_cut=2.5, key_mode="old"):
    hits_df, tracks_df, masks_b, parts_df, A_bool = load_event(
        fname, idx, key_mode=key_mode
    )
    parts_df = process_particles(parts_df, A_bool, eta_cut=eta_cut)
    tracks_df = process_tracks(parts_df, hits_df, tracks_df, masks_b, A_bool, use_sparse_auto=True, key_mode=key_mode)

    # particle-level efficiency flags from matched reco tracks
    dm_pidx = tracks_df.loc[tracks_df["eff_dm"], "matched_pidx"].astype(int).to_numpy()
    perfect_pidx = tracks_df.loc[tracks_df["eff_perfect"], "matched_pidx"].astype(int).to_numpy()
    lhc_pidx = tracks_df.loc[tracks_df["eff_lhc"], "matched_pidx"].astype(int).to_numpy()

    parts_df = parts_df.copy()
    idx_arr = parts_df.index.to_numpy()
    parts_df["eff_dm"] = np.isin(idx_arr, dm_pidx)
    parts_df["eff_perfect"] = np.isin(idx_arr, perfect_pidx)
    parts_df["eff_lhc"] = np.isin(idx_arr, lhc_pidx)

    tracks_df["n_trk"] = int(tracks_df["valid"].sum())
    # parts_df["n_vtx"] = parts_df["vz"].nunique()
    tracks_df["event_idx"] = idx
    parts_df["event_idx"] = idx

    return tracks_df, parts_df


def eval_events(fname, num_events, eta_cut=2.5, key_mode="old", event_idx_start=0):
    tracks_list, parts_list = [], []
    for i in range(event_idx_start, event_idx_start+num_events):
        print(f"Processing event {i}", end="\r")
        trk_i, prt_i = eval_event(fname, i, eta_cut=eta_cut, key_mode=key_mode)
        tracks_list.append(trk_i.reset_index(drop=True))
        parts_list.append(prt_i.reset_index(drop=True))
    print()
    tracks_all = pd.concat(tracks_list, ignore_index=True) if tracks_list else pd.DataFrame()
    parts_all = pd.concat(parts_list, ignore_index=True) if parts_list else pd.DataFrame()
    return parts_all, tracks_all


# -------------------------------
# Script entry
# -------------------------------

if __name__ == "__main__":
    fnames = {
        "Paper": "/lus/lfs1aip2/home/u5ar/pduckett.u5ar/hepattn-scale-up/epoch=028-val_loss=1.29786__test_test.h5"
    }

    EVAL_SET = "test"
    REGRESSION = False
    if EVAL_SET == "test":
        event_idx_start = 0

    # Choose among: 'auto', 'incidence', 'pid'
    TRUTH_MODE = "auto"
    INCIDENCE_KEY = "targets/particle_hit_valid"  # change if your dataset path differs

    for name, fname in fnames.items():
        print(f"\nRunning {name} (truth_mode={TRUTH_MODE})")
        parts, tracks = eval_events(
            fname,
            num_events=5,
            eta_cut=2.5,
        )
    print("\nEff DM sum:", int(parts["eff_dm"].sum()))
