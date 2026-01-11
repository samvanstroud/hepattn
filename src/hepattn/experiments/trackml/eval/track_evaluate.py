import warnings

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


def get_masks(f, idx, tracks, parts, key_mode=None):
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

    Raises:
    -------
    ValueError:
        If the eval file contains inconsistent particle-axis shapes (e.g. particle-level scalars
        and particle_hit_valid have different lengths).

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

        # Sanity check: particle-level scalars must match the particle axis of particle_hit_valid.
        # If this fails, the eval file is internally inconsistent (often caused by dynamic-query
        # code mutating only a subset of particle-shaped targets before writing).
        n_targets = int(targets.shape[0])
        n_parts = len(parts)
        if n_parts != n_targets:
            raise ValueError(
                f"Inconsistent eval file for event {idx}: particle scalars have {n_parts} entries "
                f"but targets/particle_hit_valid has {n_targets} rows. "
                "Regenerate the eval file after ensuring dynamic-query target construction keeps "
                "all particle_* targets (particle_pt/eta/phi, particle_valid, particle_id, particle_hit_valid, ...) "
                "on the same particle/query axis."
            )

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


def process_tracks(f, idx, tracks, parts, masks, targets, key_mode=None, iou_threshold=0.0, track_valid_threshold=0.5):
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
    if not np.any(valid):  # none valid tracks
        return tracks_out  # return minimal DataFrame

    # if valid tracks exist
    if targets.shape[1] != masks.shape[1]:
        raise ValueError("masks/hits size mismatch")

    density = masks.mean() if masks.size else 1.0
    if density < 0.10:  # heuristic threshold  # noqa: SIM108
        int_masks = csr_matrix(masks.T.astype(np.int8))
    else:
        int_masks = masks.T.astype(np.int8)

    overlap = targets.astype(np.int8) @ int_masks  # matrix multiplication (P,H) * (H,N) -> (P,N)
    # Hint to self: index ij in this overlap matrix is the number of matched hits between particle i and track j
    best_match_n = np.max(overlap, axis=0)  # find the maximum amount of matched hits to a particle for each track (N,)
    best_match_pid = np.argmax(overlap, axis=0)  # identify the particle index that best matches with each track (N,)
    matched = best_match_n > 0  # label tracks that matches some particle
    order = np.argsort(-1 * best_match_n, kind="mergesort")  # track index ordered in descending amount of matched hits (N,)
    keep = np.zeros(masks.shape[0], dtype=bool)  # Tracks to keep, shape = (N, )
    taken = np.zeros(targets.shape[0], dtype=bool)  # Particles with matching track, shape = (P,)
    seen_masks = {}  # dict to store unique masks
    for tid in order:  # loop over array of track indices, in descending order of matching hits
        # identify duplicated tracks
        mask_bytes = masks[tid, :].tobytes()
        if (mask_bytes in seen_masks) and (valid[tid]):
            tracks_out.loc[tid, "duplicate"] = True
        else:
            seen_masks[mask_bytes] = tid
        if not (valid[tid] and matched[tid]):  # check if a valid track is matched
            continue  # continue if a valid track is matched
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


def load_event(f, idx, eta_cut=2.5, pt_cut=1, particle_targets=None, regression=False, key_mode=None, iou_threshold=0.0, track_valid_threshold=0.5):
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
    masks, targets = get_masks(f, idx, tracks, parts, key_mode)

    # Perform track matching
    tracks, valid = process_tracks(f, idx, tracks, parts, masks, targets, key_mode, iou_threshold, track_valid_threshold)

    # Evaluate track match metrics
    eval_tracks(tracks, parts)
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
        if index_list is not None:
            # index list takes priority over randomized sample
            id_list = index_list
        elif randomize is not None:
            if (randomize <= 0) or (not isinstance(randomize, int)):
                raise ValueError("Only positive integer amounts allowed.")

            if randomize >= len(f.keys()):
                id_list = list(f.keys())
                # if requested amount exceeds record, use all events sequentially
                if randomize > len(f.keys()):
                    warnings.warn(f"Requested amount of events exceeds record. Using all {len(f.keys())} events.")
            else:
                # generate a random list of indices
                id_list = np.random.default_rng().choice(list(f.keys()), size=randomize, replace=False)
        else:
            id_list = list(f.keys())

        tracks_list = []
        parts_list = []

        for idx in id_list:
            tracks, parts = load_event(f, idx, eta_cut, pt_cut, particle_targets, regression, key_mode, iou_threshold, track_valid_threshold)
            tracks_list.append(tracks)
            parts_list.append(parts)

        tracks = pd.concat(tracks_list, ignore_index=True)
        parts = pd.concat(parts_list, ignore_index=True)

    return (tracks, parts)
