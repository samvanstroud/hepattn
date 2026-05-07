import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _read_first_available(group, dataset_names):
    for dataset_name in dataset_names:
        if dataset_name in group:
            return np.asarray(group[dataset_name][:][0])
    raise KeyError(f"None of {dataset_names} found under {group.name}")


def _read_if_present(group, dataset_name):
    if dataset_name not in group:
        return None
    return np.asarray(group[dataset_name][:][0])


def _populate_track_unique_layers(f, idx, tracks, masks, key_mode=None, require_layer_ids=False, hit_perm=None):
    if key_mode == "old":
        if require_layer_ids:
            warnings.warn(
                f"min_num_layers was requested for event {idx}, but old-format evaluation files do not store "
                "per-hit layer IDs. Continuing without applying the layer cut.",
                stacklevel=2,
            )
        return False

    target_group = f[idx]["targets"]
    layer_lengths = {}
    for dataset_name in ("key_layer_id", "hit_layer_id"):
        layer_ids = _read_if_present(target_group, dataset_name)
        if layer_ids is None:
            continue
        layer_lengths[dataset_name] = int(layer_ids.shape[0])
        if layer_ids.shape[0] != masks.shape[1]:
            continue
        # Apply the same hit-axis permutation that was applied to masks so indices align
        if hit_perm is not None:
            layer_ids = layer_ids[hit_perm]
        tracks["n_unique_layers"] = np.array(
            [int(np.unique(layer_ids[m]).shape[0]) if m.any() else 0 for m in masks],
            dtype=np.int32,
        )
        return True

    if require_layer_ids:
        if layer_lengths:
            available = ", ".join(f"{name} length={length}" for name, length in layer_lengths.items())
            raise RuntimeError(
                f"min_num_layers was requested for event {idx}, but the available layer-ID targets do not match "
                f"the track mask hit dimension ({masks.shape[1]}). Found {available}. "
                "Regenerate the evaluation file with aligned layer IDs or disable the layer cut."
            )
        warnings.warn(
            f"min_num_layers was requested for event {idx}, but neither key_layer_id nor hit_layer_id exists under "
            f"{target_group.name}. Continuing without applying the layer cut.",
            stacklevel=2,
        )
    return False


def _select_track_mask_pair(f, idx):
    pred_group = f[idx]["preds"]["final"]["track_hit_valid"]
    target_group = f[idx]["targets"]

    candidate_pairs = []
    for pred_name, target_name in (
        ("track_key_valid", "particle_key_valid"),
        ("track_hit_valid", "particle_hit_valid"),
    ):
        pred = _read_if_present(pred_group, pred_name)
        target = _read_if_present(target_group, target_name)
        if pred is None or target is None:
            continue
        candidate_pairs.append((pred_name, target_name, pred, target))

    if not candidate_pairs:
        raise KeyError(
            f"Could not find a matching track/particle mask pair in {pred_group.name} and {target_group.name}."
        )

    exact_shape_pairs = [pair for pair in candidate_pairs if pair[2].shape == pair[3].shape]
    if exact_shape_pairs:
        candidate_pairs = exact_shape_pairs

    compatible_pairs = [pair for pair in candidate_pairs if pair[2].shape[-1] == pair[3].shape[-1]]
    if compatible_pairs:
        candidate_pairs = compatible_pairs

    return candidate_pairs[0]


def _populate_track_pixel_hit_counts(f, idx, tracks, masks, key_mode=None):
    """Count predicted pixel hits per track when the eval file exposes key-level feature tags."""
    if key_mode == "old":
        return False

    key_is_pixel = _read_if_present(f[idx]["targets"], "key_is_pixel")
    if key_is_pixel is None:
        return False
    if key_is_pixel.shape[0] != masks.shape[1]:
        return False

    pixel_mask = key_is_pixel.astype(bool)
    tracks["n_pred_pixel_hits"] = np.sum(masks & pixel_mask[np.newaxis, :], axis=-1).astype(np.int32)
    return True


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


def _load_track_quality_score(f, idx, key_mode=None):
    if key_mode == "old":
        raise KeyError("Track quality score is not available for old-format evaluation files.")

    try:
        return np.array(f[idx]["preds"]["final"]["track_iou"]["query_iou"][:][0])
    except (KeyError, AttributeError):
        return np.array(f[idx]["preds"]["final"]["track_hit_valid"]["track_iou"][:][0])


def _load_saved_track_valid(f, idx, key_mode=None):
    if key_mode == "old":
        raise KeyError("Saved track-valid predictions are not available for old-format evaluation files.")
    return np.array(f[idx]["preds"]["final"]["track_valid"]["track_valid"][:][0], dtype=bool)


def _load_particle_num_pixel_hits(f, idx, key_mode=None):
    if key_mode == "old":
        return None
    return _read_if_present(f[idx]["targets"], "particle_num_pixel_hits")


def _resolve_track_combination_threshold(
    track_valid_threshold,
    track_quality_threshold,
    track_validity_mode,
    track_combination_threshold=None,
):
    if track_combination_threshold is not None:
        return track_combination_threshold
    if track_validity_mode == "track_valid_prob_plus_track_quality_score":
        return track_valid_threshold + track_quality_threshold
    if track_validity_mode == "track_valid_prob_times_track_quality_score":
        return track_valid_threshold * track_quality_threshold
    return None


def check_valid(
    f,
    idx,
    parts,
    tracks,
    key_mode=None,
    iou_threshold=0.0,
    track_valid_threshold=0.5,
    track_quality_threshold=0.0,
    track_combination_threshold=None,
    track_validity_mode="track_valid_prob",
    require_individual_cuts_for_combination=False,
    min_num_hits=3,
    min_num_pixel_hits=0,
    min_num_layers=0,
):
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
    track_quality_threshold: float
        Track quality threshold used by combined validity modes (default: 0.0)
    track_combination_threshold: float | None
        Explicit threshold for sum/product combination modes. If None, the threshold is
        derived from the component thresholds.
    track_validity_mode: str
        Validity scoring mode. "track_valid_prob" uses the probability threshold only.
        "saved_track_valid" uses the saved boolean `track_valid` prediction exactly as
        written in the evaluation file (including any query-mask handling done by the model).
        "track_valid_prob_and_track_quality_score" applies separate cuts on valid probability
        and predicted quality score.
        "track_valid_prob_plus_track_quality_score" thresholds the sum of valid probability
        and the predicted quality score against the sum of their thresholds.
        "track_valid_prob_times_track_quality_score" thresholds the product of valid probability
        and predicted quality score.
    require_individual_cuts_for_combination: bool
        If True, combination modes also require the individual per-score thresholds.
    min_num_hits: int
        Minimum number of true or predicted hits required for validity/reconstructability.
    min_num_pixel_hits: int
        Minimum number of true or predicted pixel hits required for validity/reconstructability.
    min_num_layers: int
        Minimum number of unique detector layers a predicted track must span to be valid (default: 0).

    """
    if key_mode == "old":
        # tag particles that are valid
        parts["class_pred"] = parts["n_true_hits"] >= min_num_hits
        # tag tracks that are valid
        tracks["class_pred"] = np.array(f[idx]["preds"]["class_preds"]).argmax(-1) == 0
    else:
        # tag particles that are valid
        parts["class_pred"] = np.array(f[idx]["targets"]["particle_valid"][:][0])
        particle_num_pixel_hits = _load_particle_num_pixel_hits(f, idx, key_mode)
        if particle_num_pixel_hits is not None:
            parts["n_true_pixel_hits"] = particle_num_pixel_hits
        if track_validity_mode == "saved_track_valid":
            tracks["class_pred"] = _load_saved_track_valid(f, idx, key_mode)
        else:
            tracks["track_valid_prob"] = np.array(f[idx]["preds"]["final"]["track_valid"]["track_valid_prob"][:][0])
            individual_cuts = tracks["track_valid_prob"] >= track_valid_threshold
            if track_validity_mode == "track_valid_prob":
                tracks["class_pred"] = tracks["track_valid_prob"] >= track_valid_threshold
            elif track_validity_mode == "track_valid_prob_and_track_quality_score":
                tracks["track_quality_score"] = _load_track_quality_score(f, idx, key_mode)
                tracks["track_iou"] = tracks["track_quality_score"]
                individual_cuts = individual_cuts & (tracks["track_quality_score"] >= track_quality_threshold)
                tracks["class_pred"] = individual_cuts
            elif track_validity_mode == "track_valid_prob_plus_track_quality_score":
                tracks["track_quality_score"] = _load_track_quality_score(f, idx, key_mode)
                tracks["track_iou"] = tracks["track_quality_score"]
                individual_cuts = individual_cuts & (tracks["track_quality_score"] >= track_quality_threshold)
                tracks["track_valid_score"] = tracks["track_valid_prob"] + tracks["track_quality_score"]
                combination_threshold = _resolve_track_combination_threshold(
                    track_valid_threshold,
                    track_quality_threshold,
                    track_validity_mode,
                    track_combination_threshold,
                )
                tracks["class_pred"] = tracks["track_valid_score"] >= combination_threshold
                if require_individual_cuts_for_combination:
                    tracks["class_pred"] = tracks["class_pred"] & individual_cuts
            elif track_validity_mode == "track_valid_prob_times_track_quality_score":
                tracks["track_quality_score"] = _load_track_quality_score(f, idx, key_mode)
                tracks["track_iou"] = tracks["track_quality_score"]
                individual_cuts = individual_cuts & (tracks["track_quality_score"] >= track_quality_threshold)
                tracks["track_valid_product"] = tracks["track_valid_prob"] * tracks["track_quality_score"]
                combination_threshold = _resolve_track_combination_threshold(
                    track_valid_threshold,
                    track_quality_threshold,
                    track_validity_mode,
                    track_combination_threshold,
                )
                tracks["class_pred"] = tracks["track_valid_product"] >= combination_threshold
                if require_individual_cuts_for_combination:
                    tracks["class_pred"] = tracks["class_pred"] & individual_cuts
            else:
                raise ValueError(
                    f"Unknown track_validity_mode={track_validity_mode!r}. "
                    "Expected 'saved_track_valid', 'track_valid_prob', "
                    "'track_valid_prob_and_track_quality_score', "
                    "'track_valid_prob_plus_track_quality_score', or "
                    "'track_valid_prob_times_track_quality_score'."
                )

    parts["valid"] = parts["class_pred"]
    parts["valid"] = parts["valid"] & (parts["n_true_hits"] >= min_num_hits)
    tracks["valid"] = tracks["class_pred"] & (tracks["n_pred_hits"] >= min_num_hits)

    if min_num_pixel_hits > 0:
        if "n_true_pixel_hits" in parts.columns:
            parts["valid"] = parts["valid"] & (parts["n_true_pixel_hits"] >= min_num_pixel_hits)
        elif key_mode != "old":
            warnings.warn(
                "min_num_pixel_hits was requested, but particle_num_pixel_hits is missing in the evaluation file. "
                "Falling back to particle_valid without an explicit pixel-hit cross-check.",
                stacklevel=2,
            )
        if "n_pred_pixel_hits" in tracks.columns:
            tracks["valid"] = tracks["valid"] & (tracks["n_pred_pixel_hits"] >= min_num_pixel_hits)
        else:
            raise RuntimeError(
                "min_num_pixel_hits was requested, but predicted track pixel-hit counts could not be derived "
                f"from the evaluation file for event {idx}. Regenerate the evaluation file with key_is_pixel "
                "targets, or disable the pixel-hit cut."
            )

    if min_num_layers > 0:
        if "n_unique_layers" in tracks.columns:
            tracks["valid"] = tracks["valid"] & (tracks["n_unique_layers"] >= min_num_layers)

    # Load and apply IoU threshold
    if iou_threshold > 0:
        if key_mode != "old":
            if "track_iou" not in tracks.columns:
                tracks["track_iou"] = _load_track_quality_score(f, idx, key_mode)
            if "track_quality_score" not in tracks.columns:
                tracks["track_quality_score"] = tracks["track_iou"]
            tracks["valid"] = tracks["valid"] & (tracks["track_iou"] >= iou_threshold)


def _resolve_matching_mode(f, key_mode, masks, targets, parts, mode):
    valid_modes = {"auto", "best_overlap", "slot_aligned"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown matching_mode={mode!r}. Expected one of {sorted(valid_modes)}.")

    if mode != "auto":
        return mode

    if key_mode == "old":
        return "best_overlap"

    if not _has_aligned_targets_attr(f):
        return "best_overlap"

    if masks.shape[0] != targets.shape[0]:
        return "best_overlap"

    if targets.shape[0] != len(parts):
        return "best_overlap"

    return "slot_aligned"


def _process_tracks_slot_aligned(tracks, parts, masks, targets, valid, truth_valid, match_valid_truth_only=False):
    if masks.shape != targets.shape:
        raise ValueError("slot_aligned matching requires masks and targets to have identical shapes")

    n_slots = masks.shape[0]
    overlap_diag = np.sum(masks & targets, axis=-1)

    tracks_out = tracks.copy()
    tracks_out["n_true_hits"] = parts["n_true_hits"].to_numpy(copy=True)
    tracks_out["n_matched_hits"] = overlap_diag
    tracks_out["duplicate"] = False
    tracks_out["matched_pid"] = -1

    slot_truth_valid = truth_valid.copy()
    if match_valid_truth_only:
        slot_truth_valid = slot_truth_valid & truth_valid

    matched_pid = np.where(valid & slot_truth_valid, np.arange(n_slots, dtype=np.int64), -1)
    tracks_out["matched_pid"] = matched_pid
    return tracks_out.loc[valid].copy(), valid


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


def _align_hit_order(f, idx, masks, targets, mode="auto"):
    valid_modes = {"auto", "as_saved", "unsort_preds", "sort_targets"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown hit_order_mode={mode!r}. Expected one of {sorted(valid_modes)}.")

    # Keep the file content untouched.
    if mode == "as_saved":
        return masks, targets, None

    sort_field = f.attrs.get("input_sort_field", "phi")
    if isinstance(sort_field, bytes):
        sort_field = sort_field.decode()
    sort_field = str(sort_field)

    # Sort values are stored under outputs/final/<sort_field>/hit_<sort_field> when write_outputs=True.
    try:
        sort_values = np.array(f[idx][f"outputs/final/{sort_field}/hit_{sort_field}"][:][0])
    except KeyError:
        if mode != "auto":
            warnings.warn(
                f"Cannot apply hit_order_mode={mode!r}: missing sort values for field {sort_field!r}. Using saved ordering.",
                stacklevel=2,
            )
        return masks, targets, None

    sort_idx = np.argsort(sort_values, kind="stable")

    if masks.shape[-1] != sort_idx.shape[0] or targets.shape[-1] != sort_idx.shape[0]:
        warnings.warn(
            f"Cannot apply hit_order_mode={mode!r}: sort length {sort_idx.shape[0]} does not match masks/targets hit axis.",
            stacklevel=2,
        )
        return masks, targets, None

    unsort_idx = np.argsort(sort_idx, kind="stable")

    if mode == "auto":
        mode = _infer_auto_hit_order_mode(f, masks, targets, unsort_idx)

    if mode == "unsort_preds":
        masks = masks[:, unsort_idx]
        # masks are now in original (storage) order; raw HDF5 arrays are also in storage order → no perm needed
        hit_perm = None
    elif mode == "sort_targets":
        targets = targets[:, sort_idx]
        # masks stay in model (sorted) order; raw HDF5 arrays are in storage order → apply unsort to align
        hit_perm = unsort_idx
    else:
        hit_perm = None

    return masks, targets, hit_perm


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
    hit_perm = None
    if key_mode == "old":
        masks = np.array(f[idx]["preds"]["masks"]) > 0  # if sigmoid threshold is 0.5
        targets = build_incidence(f, idx)
    else:
        pred_name, target_name, masks, targets = _select_track_mask_pair(f, idx)
        if pred_name == "track_hit_valid" and target_name == "particle_hit_valid":
            masks, targets, hit_perm = _align_hit_order(f, idx, masks, targets, mode=hit_order_mode)

    # number of predicted hits for each track (retained hits), shape = (n_max_particles, )
    tracks["n_pred_hits"] = np.sum(masks, axis=-1)
    _populate_track_pixel_hit_counts(f, idx, tracks, masks, key_mode=key_mode)
    # number of truth hits for each track (true hits), shape = (n_max_particles, )
    parts["n_true_hits"] = np.sum(targets, axis=-1)

    # number of unique detector layers spanned by each predicted track
    _populate_track_unique_layers(f, idx, tracks, masks, key_mode=key_mode, require_layer_ids=False, hit_perm=hit_perm)

    return masks, targets, hit_perm


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


def process_tracks(
    f,
    idx,
    tracks,
    parts,
    masks,
    targets,
    key_mode=None,
    iou_threshold=0.0,
    track_valid_threshold=0.5,
    track_quality_threshold=0.0,
    track_combination_threshold=None,
    track_validity_mode="track_valid_prob",
    require_individual_cuts_for_combination=False,
    match_valid_truth_only=False,
    matching_mode="auto",
    min_num_hits=3,
    min_num_pixel_hits=0,
    min_num_layers=0,
):
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
    track_quality_threshold: float
        Track quality threshold used by combined validity modes.
    track_combination_threshold: float | None
        Explicit threshold for sum/product combination modes.
    track_validity_mode: str
        Validity scoring mode passed to `check_valid`.
    require_individual_cuts_for_combination: bool
        If True, combination modes also require the individual per-score thresholds.
    match_valid_truth_only: bool
        If True, only truth particles passing `parts["valid"]` are eligible for matching.
    min_num_hits: int
        Minimum number of hits required for reconstructable particles/tracks.
    min_num_pixel_hits: int
        Minimum number of true or predicted pixel hits required for reconstructable particles/tracks.
    min_num_layers: int
        Minimum number of unique detector layers a predicted track must span to be valid.

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
    check_valid(
        f,
        idx,
        parts,
        tracks,
        key_mode,
        iou_threshold,
        track_valid_threshold,
        track_quality_threshold,
        track_combination_threshold,
        track_validity_mode,
        require_individual_cuts_for_combination,
        min_num_hits,
        min_num_pixel_hits,
        min_num_layers,
    )
    valid = tracks["valid"].to_numpy(dtype=bool)  # valid tracks (N,)
    truth_valid = parts["valid"].to_numpy(dtype=bool)
    tracks_out = tracks.copy()
    tracks_out["n_true_hits"] = -1
    tracks_out["n_matched_hits"] = -1
    tracks_out["duplicate"] = False
    tracks_out["matched_pid"] = -1
    if not np.any(valid):  # none valid tracks
        return tracks_out.loc[valid].copy(), valid

    resolved_matching_mode = _resolve_matching_mode(f, key_mode, masks, targets, parts, matching_mode)
    if resolved_matching_mode == "slot_aligned":
        return _process_tracks_slot_aligned(
            tracks,
            parts,
            masks,
            targets,
            valid,
            truth_valid,
            match_valid_truth_only=match_valid_truth_only,
        )

    # if valid tracks exist
    if targets.shape[1] != masks.shape[1]:
        raise ValueError("masks/hits size mismatch")

    density = masks.mean() if masks.size else 1.0
    if density < 0.10:  # heuristic threshold  # noqa: SIM108
        int_masks = csr_matrix(masks.T.astype(np.int8))
    else:
        int_masks = masks.T.astype(np.int8)

    overlap = np.asarray(targets.astype(np.int8) @ int_masks)  # matrix multiplication (P,H) * (H,N) -> (P,N)
    if match_valid_truth_only:
        if truth_valid.shape[0] != overlap.shape[0]:
            raise ValueError("targets/parts size mismatch")
        overlap = overlap.copy()
        overlap[~truth_valid, :] = -1
    # Hint to self: index ij in this overlap matrix is the number of matched hits between particle i and track j
    best_match_n = np.asarray(np.max(overlap, axis=0)).reshape(-1)  # find the maximum amount of matched hits to a particle for each track (N,)
    best_match_pid = np.asarray(np.argmax(overlap, axis=0)).reshape(-1)  # identify the particle index that best matches with each track (N,)
    matched = best_match_n > 0  # label tracks that matches some particle
    order = np.argsort(-1 * best_match_n, kind="mergesort")  # track index ordered in descending amount of matched hits (N,)
    keep = np.zeros(masks.shape[0], dtype=bool)  # Tracks to keep, shape = (N, )
    taken = np.zeros(targets.shape[0], dtype=bool)  # Particles with matching track, shape = (P,)
    seen_masks = {}  # dict to store unique masks
    for tid in order:  # loop over array of track indices, in descending order of matching hits
        if not valid[tid]:
            continue
        # identify duplicated tracks
        mask_bytes = masks[tid, :].tobytes()
        if mask_bytes in seen_masks:
            tracks_out.loc[tid, "duplicate"] = True
        else:
            seen_masks[mask_bytes] = tid
        if not matched[tid]:  # check if a valid track is matched
            continue
        # for valid tracks that are not yet matched
        pid = best_match_pid[tid]  # locate the particle that best matches with this track
        if not taken[pid]:  # check whether this particle paired with a track
            keep[tid] = True  # label this track to matched
            taken[pid] = True  # label this particle as taken
            tracks_out.loc[tid, "n_true_hits"] = parts.loc[pid, "n_true_hits"]
            # number of predicted hits that match true hits (true positive hit), shape = (n_max_particles, )
            tracks_out.loc[tid, "n_matched_hits"] = best_match_n[tid]

    matched_pid = valid & matched & keep
    tracks_out["matched_pid"] = np.where(matched_pid, best_match_pid, -1)
    tracks_out = tracks_out[valid]  # Keep only valid tracks

    return tracks_out, valid


def _reclassify_oop_fakes(f, idx, tracks, masks, valid, key_mode=None, hit_perm=None):
    """Reclassify fake tracks that genuinely match out-of-acceptance (oop) truth particles.

    Particles failing acceptance cuts (pt, eta, num_hits) are excluded from target slots.
    Tracks correctly matching their surviving hits would otherwise be counted as fakes.
    This corrects eff_dm=True for such tracks so the fake rate is not over-inflated.

    Requires key_particle_id (or hit_particle_id) and particle_id to be present in targets.
    Skips silently if these are missing or if the hit dimension cannot be aligned with masks.
    """
    if key_mode == "old":
        raise NotImplementedError("OOP fake reclassification is not supported for old-format evaluation files.")

    # In-acceptance particle IDs stored in target slots (-999 marks padding, skip those)
    in_acc_particle_ids = _read_if_present(f[idx]["targets"], "particle_id")
    if in_acc_particle_ids is None:
        raise KeyError(f"particle_id not found in targets for event {idx}. Required for OOP fake reclassification.")
    in_acc_set = set(in_acc_particle_ids[in_acc_particle_ids >= 0].tolist())

    # Per-hit particle IDs aligned with the model's input hit ordering.
    # Try key_particle_id first (unified pixel+strip), fall back to hit_particle_id.
    hit_particle_ids = _read_if_present(f[idx]["targets"], "key_particle_id")
    if hit_particle_ids is None:
        hit_particle_ids = _read_if_present(f[idx]["targets"], "hit_particle_id")
    if hit_particle_ids is None:
        raise KeyError(f"Neither key_particle_id nor hit_particle_id found in targets for event {idx}. Required for OOP fake reclassification.")

    # Apply the same hit-axis permutation that was applied to masks so indices align
    if hit_perm is not None:
        hit_particle_ids = hit_particle_ids[hit_perm]

    # Verify hit dimension matches masks so we catch hit-ordering mismatches early
    if hit_particle_ids.shape[0] != masks.shape[1]:
        raise ValueError(
            f"OOP fake reclassification hit dimension mismatch for event {idx}: "
            f"hit_particle_id length ({hit_particle_ids.shape[0]}) != mask hit dimension ({masks.shape[1]})."
        )

    # Particle IDs present in hits but absent from target slots are out-of-acceptance
    unique_pids = np.unique(hit_particle_ids)
    oop_pids = unique_pids[(unique_pids > 0) & ~np.isin(unique_pids, list(in_acc_set))]
    if len(oop_pids) == 0:
        return

    # Non-duplicate fake tracks are what the fake rate counts — only reclassify those
    is_fake = ~tracks["eff_dm"].to_numpy(dtype=bool) & ~tracks["duplicate"].to_numpy(dtype=bool)
    if not is_fake.any():
        return

    valid_masks = masks[valid]         # (N_valid_tracks, N_hits)
    fake_masks = valid_masks[is_fake]  # (N_fakes, N_hits)

    reclassify = np.zeros(is_fake.sum(), dtype=bool)
    for pid in oop_pids:
        oop_truth = hit_particle_ids == pid  # (N_hits,)
        n_true = int(oop_truth.sum())
        if n_true == 0:
            continue
        n_matched = (fake_masks & oop_truth).sum(axis=-1)  # (N_fakes,)
        n_pred = fake_masks.sum(axis=-1)                    # (N_fakes,)
        precision = n_matched / np.maximum(n_pred, 1)
        recall = n_matched / n_true
        reclassify |= (precision > 0.5) & (recall > 0.5)

    if reclassify.any():
        fake_indices = tracks.index[is_fake][reclassify]
        tracks.loc[fake_indices, "eff_dm"] = True


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
    track_quality_threshold=0.0,
    track_combination_threshold=None,
    track_validity_mode="track_valid_prob",
    require_individual_cuts_for_combination=False,
    match_valid_truth_only=False,
    hit_order_mode="auto",
    matching_mode="auto",
    min_num_hits=3,
    min_num_pixel_hits=0,
    min_num_layers=0,
    reclassify_oop_fakes=False,
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
    track_quality_threshold: float
        Track quality threshold used by combined validity modes.
    track_combination_threshold: float | None
        Explicit threshold for sum/product combination modes.
    track_validity_mode: str
        Validity scoring mode passed to `check_valid`.
    require_individual_cuts_for_combination: bool
        If True, combination modes also require the individual per-score thresholds.
    match_valid_truth_only: bool
        If True, only valid truth particles are eligible for matching.
    hit_order_mode: str
        One of {"auto", "as_saved", "unsort_preds", "sort_targets"}.
        "auto" uses file metadata when available and otherwise infers the best alignment.
    matching_mode: str
        One of {"auto", "best_overlap", "slot_aligned"}.
        "auto" uses slot-aligned matching for model-aligned files and falls back to
        best-overlap rematching otherwise.
    min_num_hits: int
        Minimum number of hits required for reconstructable particles/tracks.
    min_num_pixel_hits: int
        Minimum number of true or predicted pixel hits required for reconstructable particles/tracks.

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
    masks, targets, hit_perm = get_masks(f, idx, tracks, parts, key_mode, hit_order_mode=hit_order_mode)
    if min_num_layers > 0 and "n_unique_layers" not in tracks.columns:
        _populate_track_unique_layers(f, idx, tracks, masks, key_mode=key_mode, require_layer_ids=True, hit_perm=hit_perm)

    # Perform track matching
    tracks, valid = process_tracks(
        f,
        idx,
        tracks,
        parts,
        masks,
        targets,
        key_mode,
        iou_threshold,
        track_valid_threshold,
        track_quality_threshold,
        track_combination_threshold,
        track_validity_mode,
        require_individual_cuts_for_combination,
        match_valid_truth_only,
        matching_mode,
        min_num_hits,
        min_num_pixel_hits,
        min_num_layers,
    )

    # Evaluate track match metrics
    eval_tracks(tracks, parts)
    if reclassify_oop_fakes:
        _reclassify_oop_fakes(f, idx, tracks, masks, valid, key_mode, hit_perm=hit_perm)
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


def load_events(
    fname,
    index_list=None,
    randomize=None,
    eta_cut=2.5,
    pt_cut=1,
    particle_targets=None,
    regression=False,
    key_mode=None,
    iou_threshold=0.0,
    track_valid_threshold=0.5,
    track_quality_threshold=0.0,
    track_combination_threshold=None,
    track_validity_mode="track_valid_prob",
    require_individual_cuts_for_combination=False,
    match_valid_truth_only=False,
    hit_order_mode="auto",
    matching_mode="auto",
    min_num_hits=3,
    min_num_pixel_hits=0,
    min_num_layers=0,
    reclassify_oop_fakes=False,
):
    """Sequentially load events from an evaluation file and aggregate into a single DataFrame.

    Arguments:
    ----------
    fname: str
        filepath of the evaluation file
    index_list: list[int or str]
        specify a list of indexes to load
    randomize: int
        specify the size for a random set of events from evaluation file
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
    track_quality_threshold: float
        Track quality threshold used by combined validity modes.
    track_combination_threshold: float | None
        Explicit threshold for sum/product combination modes.
    track_validity_mode: str
        Validity scoring mode passed to `check_valid`.
    require_individual_cuts_for_combination: bool
        If True, combination modes also require the individual per-score thresholds.
    match_valid_truth_only: bool
        If True, only valid truth particles are eligible for matching.
    hit_order_mode: str
        One of {"auto", "as_saved", "unsort_preds", "sort_targets"}.
        "auto" uses file metadata when available and otherwise infers the best alignment.
    matching_mode: str
        One of {"auto", "best_overlap", "slot_aligned"}.
    min_num_hits: int
        Minimum number of hits required for reconstructable particles/tracks.
    min_num_pixel_hits: int
        Minimum number of true or predicted pixel hits required for reconstructable particles/tracks.

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

    with h5py.File(fname, "r") as f:
        event_keys = list(f.keys())
        if not event_keys:
            raise ValueError(
                f"Evaluation file {fname} contains no events. "
                "Check that the eval job wrote event groups before running plotting."
            )
        if index_list is not None:
            # index list takes priority over randomized sample
            id_list = index_list
        elif randomize is not None:
            if (randomize <= 0) or (not isinstance(randomize, int)):
                raise ValueError("Only positive integer amounts allowed.")

            if randomize >= len(event_keys):
                id_list = event_keys
                # if requested amount exceeds record, use all events sequentially
                if randomize > len(event_keys):
                    warnings.warn(f"Requested amount of events exceeds record. Using all {len(event_keys)} events.")
            else:
                # generate a random list of indices
                id_list = np.random.default_rng().choice(event_keys, size=randomize, replace=False)
        else:
            id_list = event_keys

        if len(id_list) == 0:
            raise ValueError(
                f"No events were selected from evaluation file {fname}. "
                "Check index_list/randomize and ensure the file contains event groups."
            )

        tracks_list = []
        parts_list = []

        for idx in id_list:
            tracks, parts = load_event(
                f,
                idx,
                eta_cut,
                pt_cut,
                particle_targets,
                regression,
                key_mode,
                iou_threshold,
                track_valid_threshold,
                track_quality_threshold,
                track_combination_threshold,
                track_validity_mode,
                require_individual_cuts_for_combination,
                match_valid_truth_only,
                hit_order_mode,
                matching_mode,
                min_num_hits,
                min_num_pixel_hits,
                min_num_layers,
                reclassify_oop_fakes,
            )
            tracks_list.append(tracks)
            parts_list.append(parts)

        tracks = pd.concat(tracks_list, ignore_index=True)
        parts = pd.concat(parts_list, ignore_index=True)

    return (tracks, parts)
