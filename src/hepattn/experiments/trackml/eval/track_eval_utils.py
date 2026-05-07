import hashlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def matched_kinematics(tracks, parts, particle_targets):
    """Add kinematic information of matched particles to track DataFrame.

    Arguments:
    ----------
    tracks: DataFrame
        The DataFrame for tracks, matched to a particle
    parts: DataFrame
        The DataFrame for particles
    particle_targets: list[str]
        List of kinematic quantities to be added

    """
    for x in particle_targets:
        # From matched particle id, pair pt prediction with its true value
        tracks["matched_" + x] = np.array(parts["particle_" + x])[tracks["matched_pid"]]
        # unmatched tracks are assigned pid of -1, label with np.nan
        tracks.loc[tracks["matched_pid"] == -1, "matched_" + x] = np.nan


def load_regression(f, idx, tracks, parts, valid, key_mode=None):
    """Load regression information for tracks.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: str
        Identifier for events in the collection
    tracks: DataFrame
        The DataFrame for tracks, into which new columns of the regression quantities are added
    parts:
        The DataFrame for particle physical quantities
    valid: ndarray
        1D array containg data with `bool` type, labels valid tracks
    key_mode: str
        specify if output file structure type

    """
    if key_mode == "old":
        if "regression" not in list(f[idx]["preds"]):
            warnings.warn("No regression values found in evaluation file")
        else:
            regr_qty = list(f[idx]["preds"]["regression"])
            for x in regr_qty:
                tracks["track_" + x] = np.array(f[idx]["preds"]["regression"][x])[valid]
            if ("px" in regr_qty) and ("py" in regr_qty):
                if "pt" not in regr_qty:
                    tracks["track_pt"] = np.sqrt(tracks["track_px"] ** 2 + tracks["track_py"] ** 2)
                if "phi" not in regr_qty:
                    tracks["track_phi"] = np.arctan2(tracks["track_py"], tracks["track_px"])
                    # From matched particle id, pair phi prediction with its true value
                if ("pz" in regr_qty) and ("eta" not in regr_qty):
                    p = np.sqrt(tracks["track_px"] ** 2 + tracks["track_py"] ** 2 + tracks["track_pz"] ** 2, dtype=np.float64)
                    tracks["track_eta"] = 0.5 * np.log((p + tracks["track_pz"]) / (p - tracks["track_pz"]), dtype=np.float64)
                    # From matched particle id, pair eta prediction with its true value
                    tracks["matched_eta"] = np.array(parts["particle_eta"])[tracks["matched_pid"]]
                    tracks.loc[tracks["matched_pid"] == -1, "matched_eta"] = np.nan  # unmatched tracks are assigned pid of -1, label with np.nan
    elif "track_regr" not in list(f[idx]["preds"]["final"]):
        warnings.warn("No regression values found in evaluation file")
    else:
        regr_qty = list(f[idx]["preds"]["final"]["track_regr"])
        for x in regr_qty:
            tracks[x] = np.array(f[idx]["preds"]["final"]["track_regr"][x][:][0])[valid]
        if ("track_px" in regr_qty) and ("track_py" in regr_qty):
            if "track_pt" not in regr_qty:
                tracks["track_pt"] = np.sqrt(tracks["track_px"] ** 2 + tracks["track_py"] ** 2)
            if "track_phi" not in regr_qty:
                tracks["track_phi"] = np.arctan2(tracks["track_py"], tracks["track_px"])
            if ("track_pz" in regr_qty) and ("track_eta" not in regr_qty):
                p = np.sqrt(tracks["track_px"] ** 2 + tracks["track_py"] ** 2 + tracks["track_pz"] ** 2, dtype=np.float64)
                tracks["track_eta"] = 0.5 * np.log((p + tracks["track_pz"]) / (p - tracks["track_pz"]), dtype=np.float64)


def check_valid(f, idx, parts, tracks, key_mode=None, iou_threshold=0.0, track_valid_threshold=0.5):
    """Label valid tracks and particles.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: int or str
        Identifier for events in the collection
    tracks: DataFrame
        The DataFrame for tracks, into which the "valid" and "reconstructable" columns are added
    parts: DataFrame
        The DataFrame for particles, into which the "valid" and "reconstructable" columns are added
    key_mode: str
        specify if output file structure type
    iou_threshold: float
        IoU threshold for valid tracks (default: 0.0)
    track_valid_threshold: float
        Track valid probability threshold (default: 0.5)

    """
    if key_mode == "old":
        # tag particles that are valid
        parts["class_pred"] = parts["n_true_hits"] >= 3
        # tag tracks that are valid
        tracks["class_pred"] = np.array(f[idx]["preds"]["class_preds"]).argmax(-1) == 0
    else:
        # tag particles that are valid
        parts["class_pred"] = np.array(f[idx]["targets"]["particle_valid"][:][0])
        # tag tracks that are valid
        tracks["class_pred"] = np.array(f[idx]["preds"]["final"]["track_valid"]["track_valid_prob"][:][0] >= track_valid_threshold)
    parts["valid"] = (parts["class_pred"]) & (parts["n_true_hits"] >= 3)
    tracks["valid"] = (tracks["class_pred"]) & (tracks["n_pred_hits"] >= 3)

    # Load and apply IoU threshold
    if iou_threshold > 0:
        if key_mode != "old":
            # Try new location in encoder_tasks first, then fall back to old location in track_hit_valid
            try:
                tracks["track_iou"] = np.array(f[idx]["preds"]["final"]["track_iou"]["query_iou"][:][0])
            except (KeyError, AttributeError):
                tracks["track_iou"] = np.array(f[idx]["preds"]["final"]["track_hit_valid"]["track_iou"][:][0])
            tracks["valid"] = tracks["valid"] & (tracks["track_iou"] >= iou_threshold)


def check_reconstructable(tracks, parts, eta_cut=2.5, pt_cut=1):
    """Label reconstructable tracks and particles.

    Arguments:
    ----------
    tracks: DataFrame
        The DataFrame for tracks, into which the "valid" and "reconstructable" columns are added
    parts: DataFrame
        The DataFrame for particles, into which the "valid" and "reconstructable" columns are added
    eta_cut: float
        Pseudorapidity acceptance window, must be in (-eta_cut , eta_cut) for tracks/particles to be considered reconstructable
    pt_cut: float
        Transverse momentum threshold, lower limit for tracks/particles to be considered reconstructable

    """
    parts["reconstructable"] = parts["valid"]
    if "particle_eta" in parts.columns:
        parts["reconstructable"] = (parts["reconstructable"]) & (parts["particle_eta"].abs() < eta_cut)
    if "particle_pt" in parts.columns:
        parts["reconstructable"] = (parts["reconstructable"]) & (parts["particle_pt"] > pt_cut)

    tracks["reconstructable_parts"] = np.array(parts["reconstructable"])[tracks["matched_pid"]]
    tracks.loc[tracks["matched_pid"] == -1, "reconstructable_parts"] = False

    tracks["reconstructable"] = tracks["valid"]
    if "track_eta" in tracks.columns:
        tracks["reconstructable"] = tracks["reconstructable"] & (tracks["track_eta"].abs() < eta_cut)
    if "track_pt" in tracks.columns:
        tracks["reconstructable"] = tracks["reconstructable"] & (tracks["track_pt"] > pt_cut)
    if "out_of_acceptance_match" in tracks.columns:
        tracks["reconstructable"] = tracks["reconstructable"] & (~tracks["out_of_acceptance_match"])


def recover_out_of_acceptance_matches(f, idx, tracks, masks, key_mode=None, hit_order_mode="auto"):
    """Relabel fake tracks that actually reconstruct particles excluded from target slots."""
    tracks["out_of_acceptance_match"] = np.zeros(len(tracks), dtype=bool)
    tracks["out_of_acceptance_particle_id"] = np.full(len(tracks), -1, dtype=np.int64)
    if key_mode == "old" or tracks.empty:
        return

    hit_particle_ids = _load_hit_particle_ids(f, idx, hit_order_mode=hit_order_mode)
    if hit_particle_ids is None:
        return

    hit_particle_ids = np.asarray(hit_particle_ids, dtype=np.int64)
    if hit_particle_ids.shape[0] != masks.shape[1]:
        warnings.warn(
            f"Cannot recover out-of-acceptance matches for event {idx!r}: hit particle id length {hit_particle_ids.shape[0]} does not match mask hit axis {masks.shape[1]}.",
            stacklevel=2,
        )
        return

    target_particle_ids = np.array(f[idx]["targets"]["particle_id"][:][0], dtype=np.int64)
    target_particle_valid = np.array(f[idx]["targets"]["particle_valid"][:][0], dtype=bool)
    target_particle_ids = target_particle_ids[target_particle_valid]

    valid_hit_particle_ids = hit_particle_ids[hit_particle_ids > 0]
    out_of_acceptance_particle_ids = np.unique(valid_hit_particle_ids[~np.isin(valid_hit_particle_ids, target_particle_ids)])
    if out_of_acceptance_particle_ids.size == 0:
        return

    candidate_mask = (~tracks["duplicate"].to_numpy()) & (~tracks["eff_dm"].to_numpy())
    if not np.any(candidate_mask):
        return

    candidate_track_rows = tracks.index.to_numpy(dtype=np.int64)[candidate_mask]
    candidate_masks = masks[candidate_track_rows]
    if candidate_masks.size == 0:
        return

    truth_masks = out_of_acceptance_particle_ids[:, None] == hit_particle_ids[None, :]
    overlap = truth_masks.astype(np.int8) @ candidate_masks.T.astype(np.int8)
    best_match_n = np.asarray(np.max(overlap, axis=0)).reshape(-1)
    matched = best_match_n > 0
    if not np.any(matched):
        return

    best_match_idx = np.asarray(np.argmax(overlap, axis=0)).reshape(-1)
    truth_hits = np.sum(truth_masks, axis=-1)
    matched_truth_hits = truth_hits[best_match_idx]
    pred_hits = tracks.loc[tracks.index[candidate_mask], "n_pred_hits"].to_numpy()

    precision = np.where(best_match_n > 0, best_match_n / np.maximum(pred_hits, 1), -1.0)
    recall = np.where(best_match_n > 0, best_match_n / np.maximum(matched_truth_hits, 1), -1.0)
    recovered = matched & (precision > 0.5) & (recall > 0.5)
    if not np.any(recovered):
        return

    recovered_rows = tracks.index.to_numpy()[candidate_mask][recovered]
    recovered_particle_ids = out_of_acceptance_particle_ids[best_match_idx[recovered]]
    recovered_precision = precision[recovered]
    recovered_recall = recall[recovered]
    recovered_true_hits = matched_truth_hits[recovered]
    recovered_matched_hits = best_match_n[recovered]

    tracks.loc[recovered_rows, "out_of_acceptance_match"] = True
    tracks.loc[recovered_rows, "out_of_acceptance_particle_id"] = recovered_particle_ids
    tracks.loc[recovered_rows, "n_true_hits"] = recovered_true_hits
    tracks.loc[recovered_rows, "n_matched_hits"] = recovered_matched_hits
    tracks.loc[recovered_rows, "precision"] = recovered_precision
    tracks.loc[recovered_rows, "recall"] = recovered_recall
    tracks.loc[recovered_rows, "eff_dm"] = True
    tracks.loc[recovered_rows, "eff_perfect"] = (recovered_precision == 1.0) & (recovered_recall == 1.0)
    tracks.loc[recovered_rows, "eff_lhc"] = recovered_precision > 0.75


def build_incidence(f, idx):  # for key_mode = "old"
    """Construct incidence matrix for particles.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: int or str
        Identifier for events in the collection

    """
    parts_pid = np.array(f[idx]["parts"]["pids"])
    hits_pid = np.array(f[idx]["hits"]["pids"])
    p, h = parts_pid.shape[0], hits_pid.shape[0]
    incidence = np.zeros((p, h), dtype=bool)
    pid_to_pidx = {int(pid): i for i, pid in enumerate(parts_pid)}
    valid = (hits_pid != 0) & np.vectorize(pid_to_pidx.__contains__)(hits_pid)
    if valid.any():
        hidx = np.nonzero(valid)[0]
        pidx = np.fromiter((pid_to_pidx[int(pid)] for pid in hits_pid[valid]), count=hidx.size, dtype=int)
        incidence[pidx, hidx] = True
    return incidence


def _has_aligned_targets_attr(f):
    value = f.attrs.get("targets_model_aligned")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return False


def _mask_target_match_score(masks, targets):
    """Heuristic quality score for a given hit-axis alignment.

    Higher means track masks and particle targets are more mutually consistent.
    """
    if masks.shape[1] != targets.shape[1]:
        return -np.inf
    if masks.size == 0:
        return -np.inf

    density = masks.mean()
    if density < 0.10:  # noqa: SIM108
        int_masks = csr_matrix(masks.T.astype(np.int8))
    else:
        int_masks = masks.T.astype(np.int8)

    overlap = targets.astype(np.int8) @ int_masks  # (P, H) @ (H, N) -> (P, N)
    best_match_n = np.max(overlap, axis=0)
    best_match_n = np.asarray(best_match_n).reshape(-1)

    n_pred_hits = np.maximum(np.sum(masks, axis=-1), 1)
    return float(np.mean(best_match_n / n_pred_hits))


def _infer_auto_hit_order_mode(f, masks, targets, unsort_idx):
    """Infer whether predictions should be unsorted to match targets."""
    if "targets_model_aligned" in f.attrs:
        return "as_saved" if _has_aligned_targets_attr(f) else "unsort_preds"

    score_saved = _mask_target_match_score(masks, targets)
    score_unsorted = _mask_target_match_score(masks[:, unsort_idx], targets)
    return "unsort_preds" if score_unsorted > score_saved else "as_saved"


def _get_hit_sort_idx(f, idx):
    sort_field = f.attrs.get("input_sort_field", "phi")
    if isinstance(sort_field, bytes):
        sort_field = sort_field.decode()
    sort_field = str(sort_field)

    try:
        sort_values = np.array(f[idx][f"outputs/final/{sort_field}/hit_{sort_field}"][:][0])
    except KeyError:
        return None, sort_field

    return np.argsort(sort_values, kind="stable"), sort_field


def _align_hit_order(f, idx, masks, targets, mode="auto"):
    valid_modes = {"auto", "as_saved", "unsort_preds", "sort_targets"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown hit_order_mode={mode!r}. Expected one of {sorted(valid_modes)}.")

    # Keep the file content untouched.
    if mode == "as_saved":
        return masks, targets

    sort_idx, sort_field = _get_hit_sort_idx(f, idx)
    if sort_idx is None:
        if mode != "auto":
            warnings.warn(
                f"Cannot apply hit_order_mode={mode!r}: missing sort values for field {sort_field!r}. Using saved ordering.",
                stacklevel=2,
            )
        return masks, targets

    if masks.shape[-1] != sort_idx.shape[0] or targets.shape[-1] != sort_idx.shape[0]:
        warnings.warn(
            f"Cannot apply hit_order_mode={mode!r}: sort length {sort_idx.shape[0]} does not match masks/targets hit axis.",
            stacklevel=2,
        )
        return masks, targets

    unsort_idx = np.argsort(sort_idx, kind="stable")

    if mode == "auto":
        mode = _infer_auto_hit_order_mode(f, masks, targets, unsort_idx)

    if mode == "unsort_preds":
        masks = masks[:, unsort_idx]
    elif mode == "sort_targets":
        targets = targets[:, sort_idx]

    return masks, targets


def _load_hit_particle_ids(f, idx, hit_order_mode="auto"):
    targets_group = f[idx].get("targets")
    if targets_group is None:
        return None

    hit_particle_ids = None
    for dataset_name in ("hit_particle_id", "key_particle_id"):
        if dataset_name in targets_group:
            hit_particle_ids = np.array(targets_group[dataset_name][:][0])
            break

    if hit_particle_ids is None:
        return None

    if hit_order_mode == "sort_targets":
        sort_idx, sort_field = _get_hit_sort_idx(f, idx)
        if sort_idx is None:
            warnings.warn(
                f"Cannot align hit particle ids for hit_order_mode={hit_order_mode!r}: missing sort values for field {sort_field!r}. Using saved ordering.",
                stacklevel=2,
            )
            return hit_particle_ids
        if hit_particle_ids.shape[0] != sort_idx.shape[0]:
            warnings.warn(
                f"Cannot align hit particle ids for hit_order_mode={hit_order_mode!r}: sort length {sort_idx.shape[0]} does not match hit axis.",
                stacklevel=2,
            )
            return hit_particle_ids
        return hit_particle_ids[sort_idx]

    return hit_particle_ids


def get_masks(f, idx, tracks, parts, key_mode=None, hit_order_mode="auto"):
    """Retrieve predicted hit masks.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: int or str
        Identifier for events in the collection
    tracks: DataFrame
        The DataFrame for tracks, count assigned hits for each track
    parts: DataFrame
        The DataFrame for particles, count true hits for each particle
    key_mode: str
        specify if output file structure type
    hit_order_mode: str
        Hit-axis alignment mode for predicted masks vs target masks.
    """
    # extract hit mask and its target
    if key_mode == "old":
        masks = np.array(f[idx]["preds"]["masks"]) > 0  # if sigmoid threshold is 0.5
        targets = build_incidence(f, idx)
    else:
        # predicted tracks and associated hits, shape = (n_max_particles, n_hits)
        masks = np.array(f[idx]["preds"]["final"]["track_hit_valid"]["track_hit_valid"][:][0])
        # truth tracks and associated hits, shape = (n_max_particles, n_hits)
        targets = np.array(f[idx]["targets"]["particle_hit_valid"][:][0])
        masks, targets = _align_hit_order(f, idx, masks, targets, mode=hit_order_mode)

    # number of predicted hits for each track (retained hits), shape = (n_max_particles, )
    tracks["n_pred_hits"] = np.sum(masks, axis=-1)
    # number of truth hits for each track (true hits), shape = (n_max_particles, )
    parts["n_true_hits"] = np.sum(targets, axis=-1)
    return masks, targets


def process_particles(f, idx, parts, particle_targets=None, key_mode=None):
    """Load truth level particle physical quantities.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: int or str
        Identifier for events in the collection
    parts: DataFrame
        The DataFrame for particle physical quantities
    particle_targets: list[str]
        List of physical quantities that are regression targets
    key_mode: str
        specify if output file structure type

    """
    if particle_targets is None:
        particle_targets = ["pt", "eta", "phi"]
    if key_mode == "old":
        for x in particle_targets:
            parts["particle_" + x] = np.array(f[idx]["parts"][x + "s"])
    else:
        # physical quantity regression values/targets (and derived quantities)
        for x in particle_targets:
            parts["particle_" + x] = np.array(f[idx]["targets"]["particle_" + x][:][0])


def process_tracks(f, idx, tracks, parts, masks, targets, key_mode=None, iou_threshold=0.0, track_valid_threshold=0.5, match_min_hits=0, match_min_pt=None):
    """Track matching, fits track masks to target masks.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: int or str
        Identifier for events in the collection
    tracks: DataFrame
        The DataFrame for track physical quantities
    parts: DataFrame
        The DataFrame for particle physical quantities
    masks: ndarray
        Track masks, 2 dimensional boolean array with shape (N, H) where N is the number of tracks, and H is the number of total hits.
        Along the last dimension is equivalent to a n-hot encoded array of length H, with n as the number of predicted hits.
    targets: ndarray
        Target masks, 2 dimensional boolean array with shape (P, H) where P is the number of particles, and H is the number of total hits.
        Along the last dimension is equivalent to a n-hot encoded array of length H, with n as the number of true hits.
    key_mode: str
        specify if output file structure type
    iou_threshold: float
        IoU threshold for valid tracks (default: 0.1)
    track_valid_threshold: float
        Track valid probability threshold (default: 0.5)
    match_min_hits: int
        Minimum number of true hits for a truth particle to be eligible for matching (default: 0, no cut).
        Predicted tracks whose best-matching particle fails this cut are treated as unmatched.
    match_min_pt: float, optional
        Minimum pT [GeV] for a truth particle to be eligible for matching (default: None, no cut).
        Predicted tracks whose best-matching particle fails this cut are treated as unmatched.

    Returns:
    ----------
    tracks_out: DataFrame
        The DataFrame holding track physical quantities and track matching results
    valid: ndarray
        1D array containg data with `bool` type, labels valid tracks

    Raises:
    --------
    ValueError:
        If the hit dimension of the predicted and true incidence matrix don't match

    """
    # check valid
    check_valid(f, idx, parts, tracks, key_mode, iou_threshold, track_valid_threshold)
    valid = tracks["valid"]  # valid tracks (N,)
    tracks_out = tracks.copy()
    tracks_out["n_true_hits"] = -1
    tracks_out["n_matched_hits"] = -1
    tracks_out["duplicate"] = False
    tracks_out["matched_pid"] = np.full(len(tracks_out), -1, dtype=np.int64)
    if not np.any(valid):  # none valid tracks
        return tracks_out[valid], valid

    # if valid tracks exist
    if targets.shape[1] != masks.shape[1]:
        raise ValueError("masks/hits size mismatch")

    density = masks.mean() if masks.size else 1.0
    if density < 0.10:  # heuristic threshold  # noqa: SIM108
        int_masks = csr_matrix(masks.T.astype(np.int8))
    else:
        int_masks = masks.T.astype(np.int8)

    # Build an eligibility mask for truth particles. Only eligible particles can be
    # assigned to a predicted track; predicted tracks whose best match is an ineligible
    # particle are treated as unmatched (and counted as fakes if they are valid).
    part_eligible = np.ones(targets.shape[0], dtype=bool)
    if match_min_hits > 0:
        part_eligible &= parts["n_true_hits"].to_numpy() >= match_min_hits
    if match_min_pt is not None and "particle_pt" in parts.columns:
        part_eligible &= parts["particle_pt"].to_numpy() >= match_min_pt
    eligible_idx = np.where(part_eligible)[0]

    if eligible_idx.size > 0:
        # (P_eligible, H) @ (H, N) -> (P_eligible, N): hit overlap between eligible particles and all tracks
        match_targets = targets[eligible_idx]
        overlap = match_targets.astype(np.int8) @ int_masks
        # best_match_* are indexed over all N tracks; best_match_pid holds full particle indices
        best_match_n = np.asarray(np.max(overlap, axis=0)).flatten()
        best_match_pid = eligible_idx[np.asarray(np.argmax(overlap, axis=0)).flatten()]
    else:
        best_match_n = np.zeros(masks.shape[0], dtype=np.int32)
        best_match_pid = np.zeros(masks.shape[0], dtype=np.int64)

    matched = best_match_n > 0  # label tracks that match some eligible particle
    order = np.argsort(-1 * best_match_n, kind="mergesort")  # descending matched-hit count
    keep = np.zeros(masks.shape[0], dtype=bool)
    taken = np.zeros(targets.shape[0], dtype=bool)
    seen_masks = {}

    # Use numpy arrays for per-track accumulation; assign to DataFrame in bulk after the loop
    # to avoid the overhead of pandas .loc scalar writes on every iteration.
    N = masks.shape[0]
    duplicate_arr = np.zeros(N, dtype=bool)
    n_true_hits_arr = np.full(N, -1, dtype=np.int64)
    n_matched_hits_arr = np.full(N, -1, dtype=np.int64)
    parts_n_true_hits = parts["n_true_hits"].to_numpy()

    for tid in order:
        mask_bytes = masks[tid, :].tobytes()
        if (mask_bytes in seen_masks) and (valid[tid]):
            duplicate_arr[tid] = True
        else:
            seen_masks[mask_bytes] = tid
        if not (valid[tid] and matched[tid]):
            continue
        pid = best_match_pid[tid]
        if not taken[pid]:
            keep[tid] = True
            taken[pid] = True
            n_true_hits_arr[tid] = parts_n_true_hits[pid]
            n_matched_hits_arr[tid] = best_match_n[tid]

    tracks_out["duplicate"] = duplicate_arr
    tracks_out["n_true_hits"] = n_true_hits_arr
    tracks_out["n_matched_hits"] = n_matched_hits_arr
    matched_pid = valid & matched & keep
    tracks_out["matched_pid"] = np.where(matched_pid, best_match_pid, -1)
    tracks_out = tracks_out[valid]  # Keep only valid tracks

    return tracks_out, valid


def eval_tracks(tracks, parts):
    """Evaluate track match metrics.

    Arguments:
    ----------
    tracks: DataFrame
        The DataFrame holding track physical quantities and track matching results
    parts: DataFrame
        The DataFrame holding particle physical quantities

    """
    # calculate track matching metrics
    # true positives / predicted positives
    precision = np.where(tracks["n_matched_hits"] > 0, tracks["n_matched_hits"] / tracks["n_pred_hits"], -1)
    # true positives / positives
    recall = np.where(tracks["n_matched_hits"] > 0, tracks["n_matched_hits"] / tracks["n_true_hits"], -1)
    tracks["precision"] = precision
    tracks["recall"] = recall
    # perfect match scheme: all hits are assigned to true track (recall and precision == 1)
    tracks["eff_perfect"] = (precision == 1) & (recall == 1) & (~tracks["duplicate"])
    # Double Majority scheme: 50% of hits are assigned to true track (recall and precision > 0.5)
    tracks["eff_dm"] = (precision > 0.5) & (recall > 0.5) & (~tracks["duplicate"])
    # LHC eff: precision > 0.75
    tracks["eff_lhc"] = (precision > 0.75) & (~tracks["duplicate"])
    # assign tracks to particles
    dm_pid = tracks.loc[tracks["eff_dm"], "matched_pid"].astype(int).to_numpy()
    perfect_pid = tracks.loc[tracks["eff_perfect"], "matched_pid"].astype(int).to_numpy()
    lhc_pid = tracks.loc[tracks["eff_lhc"], "matched_pid"].astype(int).to_numpy()
    pid = parts.index.to_numpy()
    parts["eff_dm"] = np.isin(pid, dm_pid)
    parts["eff_perfect"] = np.isin(pid, perfect_pid)
    parts["eff_lhc"] = np.isin(pid, lhc_pid)


def load_event(
    f,
    idx,
    eta_cut=2.5,
    pt_cut=1,
    particle_targets=None,
    regression=False,
    key_mode=None,
    iou_threshold=0.0,
    track_valid_threshold=0.5,
    hit_order_mode="auto",
    match_min_hits=0,
    match_min_pt=None,
):
    """Load an event from an evaluation file and create a DataFrame.

    Arguments:
    ----------
    f: File
        h5 file in buffer
    idx: str or int
        the event identifier (e.g. "29800" to "29899")
    eta_cut: float
        Accepted pseudorapidity range
    pt_cut: float
        Transverse momentum [GeV] threshold for tracks
    particle_targets: list[str]
        List of physical quantities that are regression targets
    regression: bool
        specify whether regression quantities are included in the evaluation file
    key_mode: str
        specify if output file structure type
    iou_threshold: float
        IoU threshold for valid tracks (default: 0.0)
    track_valid_threshold: float
        Track valid probability threshold (default: 0.5)
    hit_order_mode: str
        One of {"auto", "as_saved", "unsort_preds", "sort_targets"}.
        "auto" uses file metadata when available and otherwise infers the best alignment.
    match_min_hits: int
        Minimum true hits for a truth particle to be eligible for matching.
    match_min_pt: float, optional
        Minimum pT [GeV] for a truth particle to be eligible for matching.

    Returns:
    --------
    tracks: DataFrame
        Predicted information of each track in an event.
    parts: DataFrame
        Truth information of each track in an event.

    """
    # Declare DataFrame for tracks, particles
    tracks = pd.DataFrame()
    parts = pd.DataFrame()

    # Load truth level particle physical quantities
    process_particles(f, idx, parts, particle_targets, key_mode)

    # Extract masks from incidence boolean matrix
    masks, targets = get_masks(f, idx, tracks, parts, key_mode, hit_order_mode=hit_order_mode)

    # Perform track matching
    tracks, valid = process_tracks(f, idx, tracks, parts, masks, targets, key_mode, iou_threshold, track_valid_threshold, match_min_hits=match_min_hits, match_min_pt=match_min_pt)

    # Evaluate track match metrics
    eval_tracks(tracks, parts)
    recover_out_of_acceptance_matches(f, idx, tracks, masks, key_mode=key_mode, hit_order_mode=hit_order_mode)
    matched_kinematics(tracks, parts, particle_targets)
    # Load regression values
    if regression:
        load_regression(f, idx, tracks, parts, valid, key_mode)

    # Identify reconstructable tracks
    check_reconstructable(tracks, parts, eta_cut, pt_cut)

    # Tag rows with the event identifier
    tracks["event_id"] = idx
    parts["event_id"] = idx

    return tracks, parts


def _resolve_event_id_list(f, index_list=None, randomize=None, random_seed=None):
    if index_list is not None:
        return [str(idx) for idx in index_list]

    file_keys = list(f.keys())
    if randomize is None:
        return file_keys

    if (randomize <= 0) or (not isinstance(randomize, int)):
        raise ValueError("Only positive integer amounts allowed.")

    if randomize >= len(file_keys):
        if randomize > len(file_keys):
            warnings.warn(f"Requested amount of events exceeds record. Using all {len(file_keys)} events.")
        return file_keys

    rng = np.random.default_rng(random_seed)
    return [str(idx) for idx in rng.choice(file_keys, size=randomize, replace=False)]


def _cache_signature(
    *,
    fname,
    event_ids,
    eta_cut,
    pt_cut,
    particle_targets,
    regression,
    key_mode,
    iou_threshold,
    track_valid_threshold,
    hit_order_mode,
    extra_options=None,
):
    extra_options = {} if extra_options is None else dict(extra_options)
    return {
        "fname": str(Path(fname).resolve()),
        "event_ids": [str(idx) for idx in event_ids],
        "eta_cut": float(eta_cut),
        "pt_cut": float(pt_cut),
        "particle_targets": list(particle_targets),
        "regression": bool(regression),
        "key_mode": key_mode,
        "iou_threshold": float(iou_threshold),
        "track_valid_threshold": float(track_valid_threshold),
        "hit_order_mode": str(hit_order_mode),
        "extra_options": extra_options,
    }


def _cache_base_path(cache_dir, cache_key, signature):
    signature_blob = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    signature_hash = hashlib.sha1(signature_blob.encode("utf-8")).hexdigest()[:12]
    stem = cache_key or f"{Path(signature['fname']).stem}_{signature_hash}"
    return Path(cache_dir) / stem


def _load_saved_events(cache_base, expected_signature):
    meta_path = cache_base.with_name(cache_base.name + "_meta.json")
    tracks_path = cache_base.with_name(cache_base.name + "_tracks.pkl")
    parts_path = cache_base.with_name(cache_base.name + "_parts.pkl")
    if not (meta_path.exists() and tracks_path.exists() and parts_path.exists()):
        return None

    with meta_path.open() as f:
        saved_meta = json.load(f)
    if saved_meta.get("signature") != expected_signature:
        return None

    tracks = pd.read_pickle(tracks_path)
    parts = pd.read_pickle(parts_path)
    return tracks, parts


def _write_saved_events(cache_base, signature, tracks, parts):
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_base.with_name(cache_base.name + "_meta.json")
    tracks_path = cache_base.with_name(cache_base.name + "_tracks.pkl")
    parts_path = cache_base.with_name(cache_base.name + "_parts.pkl")

    tracks.to_pickle(tracks_path)
    parts.to_pickle(parts_path)
    with meta_path.open("w") as f:
        json.dump({"signature": signature}, f, indent=2, sort_keys=True)


def _load_event_worker(fname, idx, eta_cut, pt_cut, particle_targets, regression, key_mode, iou_threshold, track_valid_threshold, hit_order_mode, match_min_hits, match_min_pt):
    """Open a fresh HDF5 handle and load a single event. Used by ThreadPoolExecutor for parallel loading."""
    with h5py.File(fname, "r") as f:
        return load_event(
            f, idx, eta_cut, pt_cut, particle_targets, regression, key_mode,
            iou_threshold, track_valid_threshold, hit_order_mode,
            match_min_hits=match_min_hits, match_min_pt=match_min_pt,
        )


def load_events(
    fname,
    index_list=None,
    randomize=None,
    random_seed=None,
    eta_cut=2.5,
    pt_cut=1,
    particle_targets=None,
    regression=False,
    key_mode=None,
    iou_threshold=0.0,
    track_valid_threshold=0.5,
    hit_order_mode="auto",
    match_min_hits=0,
    match_min_pt=None,
    cache_dir=None,
    cache_key=None,
    reuse_saved=False,
    write_cache=False,
    n_workers=1,
):
    """Load events from an evaluation file and aggregate into a single DataFrame.

    Arguments:
    ----------
    fname: str
        filepath of the evaluation file
    index_list: list[int or str]
        specify a list of indexes to load
    randomize: int
        specify the size for a random set of events from evaluation file
    random_seed: int, optional
        Seed used when drawing a randomized event subset. When set together with
        caching, the same subset can be reused across runs.
    eta_cut: float
        Accepted pseudorapidity range
    pt_cut: float
        Transverse momentum [GeV] threshold for tracks
    particle_targets: list[str]
        list of physical quantities that are regression targets
    regression: bool
        specify whether regression quantities are included in the evaluation file
    key_mode: str
        specify if output file structure type
    iou_threshold: float
        IoU threshold for valid tracks (default: 0.1)
    track_valid_threshold: float
        Track valid probability threshold (default: 0.5)
    hit_order_mode: str
        One of {"auto", "as_saved", "unsort_preds", "sort_targets"}.
        "auto" uses file metadata when available and otherwise infers the best alignment.
    match_min_hits: int
        Minimum number of true hits for a truth particle to be eligible for matching.
        Predicted tracks whose best match fails this cut are treated as unmatched.
    match_min_pt: float, optional
        Minimum pT [GeV] for a truth particle to be eligible for matching.
        Predicted tracks whose best match fails this cut are treated as unmatched.
    cache_dir: str or Path, optional
        Directory holding saved `tracks`/`parts` DataFrames from previous runs.
    cache_key: str, optional
        Human-readable cache stem. When omitted, one is derived from the input file
        name and evaluation settings.
    reuse_saved: bool
        If True, load saved `tracks`/`parts` from `cache_dir` when the cached
        signature matches the current request.
    write_cache: bool
        If True, persist the computed `tracks`/`parts` so future runs can reuse them.
    n_workers: int
        Number of threads for parallel event loading (default: 1 = sequential).
        Each worker opens its own HDF5 handle; requires a thread-safe HDF5 build
        or a read-only file on a POSIX filesystem.

    Returns:
    --------
    tracks: DataFrame
        Predicted information of each track in the collection of events.
    parts: DataFrame
        Truth information of each track in the collection of events.

    Raises:
    --------
    ValueError:
        If specified size for a random sample is non-positive

    """
    if particle_targets is None:
        particle_targets = ["pt", "eta", "phi"]

    # Resolve event IDs — requires a file handle only for key listing.
    with h5py.File(fname, "r") as f:
        id_list = _resolve_event_id_list(f, index_list=index_list, randomize=randomize, random_seed=random_seed)

    signature = _cache_signature(
        fname=fname,
        event_ids=id_list,
        eta_cut=eta_cut,
        pt_cut=pt_cut,
        particle_targets=particle_targets,
        regression=regression,
        key_mode=key_mode,
        iou_threshold=iou_threshold,
        track_valid_threshold=track_valid_threshold,
        hit_order_mode=hit_order_mode,
        extra_options={
            "match_min_hits": int(match_min_hits),
            "match_min_pt": float(match_min_pt) if match_min_pt is not None else None,
        },
    )
    cache_base = _cache_base_path(cache_dir, cache_key, signature) if cache_dir is not None else None
    if reuse_saved and cache_base is not None:
        cached = _load_saved_events(cache_base, signature)
        if cached is not None:
            return cached

    tracks_list = []
    parts_list = []

    if n_workers > 1:
        # Submit all events concurrently; each worker opens its own HDF5 handle.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futs = [
                executor.submit(
                    _load_event_worker,
                    fname, idx, eta_cut, pt_cut, particle_targets, regression,
                    key_mode, iou_threshold, track_valid_threshold, hit_order_mode,
                    match_min_hits, match_min_pt,
                )
                for idx in id_list
            ]
        # Collect in submission order to preserve event ordering.
        for fut in futs:
            t, p = fut.result()
            tracks_list.append(t)
            parts_list.append(p)
    else:
        with h5py.File(fname, "r") as f:
            for idx in id_list:
                t, p = load_event(
                    f, idx, eta_cut, pt_cut, particle_targets, regression,
                    key_mode, iou_threshold, track_valid_threshold, hit_order_mode,
                    match_min_hits=match_min_hits, match_min_pt=match_min_pt,
                )
                tracks_list.append(t)
                parts_list.append(p)

    tracks = pd.concat(tracks_list, ignore_index=True)
    parts = pd.concat(parts_list, ignore_index=True)

    if write_cache and cache_base is not None:
        _write_saved_events(cache_base, signature, tracks, parts)

    return (tracks, parts)
