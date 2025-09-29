import warnings

import h5py
import numpy as np
import pandas as pd


def load_event(f, idx, eta_cut=2.5, pt_cut=1, particle_targets=None):

    """Load an event from an evaluation file and create a DataFrame

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

    Returns:
    --------
    tracks: DataFrame
        Predicted information of each track in an event. shape = (n_max_particles, )
    parts: DataFrame
        Truth information of each track in this event. shape = (n_max_particles, )

    """

    if particle_targets is None:
        particle_targets = ["particle_pt", "particle_eta", "particle_phi"]

    # load hit masks, targets,  and declare DataFrame for tracks, particles
    tracks = pd.DataFrame()
    parts = pd.DataFrame()

    tracks["event"] = int(idx)
    parts["event"] = int(idx)

    # physical quantity regression values/targets (and derived quantities)
    if "track_regr" not in list(f[idx]["preds"]["final"]):
        warnings.warn("No regression values found in evaluation file")
    else:
        regr_qty = list(f[idx]["preds"]["final"]["track_regr"])
        for x in regr_qty:
            tracks[x] = np.array(f[idx]["preds"]["final"]["track_regr"][x][:][0])
        if ("track_px" in regr_qty) and ("track_py" in regr_qty):
            if "track_pt" not in regr_qty:
                tracks["track_pt"] = np.sqrt(tracks["track_px"] ** 2 + tracks["track_py"] ** 2)
            if "track_phi" not in regr_qty:
                tracks["track_phi"] = np.arctan2(tracks["track_py"], tracks["track_px"])
            if "track_pz" in regr_qty:
                if "track_eta" not in regr_qty:
                    tracks["track_eta"] = np.arctanh(
                        tracks["track_pz"] / np.sqrt(tracks["track_px"] ** 2 + tracks["track_py"] ** 2 + tracks["track_pz"] ** 2)
                    )
    for x in particle_targets:
        parts[x] = np.array(np.array(f[idx]["targets"][x][:][0]))

    # extract hit mask and its target
    # predicted tracks and associated hits, shape = (n_max_particles, n_hits)
    masks = np.array(f[idx]["preds"]["final"]["track_hit_valid"]["track_hit_valid"][:][0])
    # truth tracks and associated hits, shape = (n_max_particles, n_hits)
    targets = np.array(f[idx]["targets"]["particle_hit_valid"][:][0])

    # match hit mask to target
    # number of predicted hits that match true hits (true positive hit), shape = (n_max_particles, )
    tracks["n_matched_hits"] = np.sum(masks & targets, axis=-1)
    # number of predicted hits for each track (retained hits), shape = (n_max_particles, )
    tracks["n_pred_hits"] = np.sum(masks, axis=-1)
    # number of truth hits for each track (true hits), shape = (n_max_particles, )
    parts["n_true_hits"] = np.sum(targets, axis=-1)
    
    # calculate track matching metrics
    # true positives / predicted positives
    precision = np.where(tracks["n_pred_hits"] > 0, tracks["n_matched_hits"] / tracks["n_pred_hits"], 0)
    # true positives / positives
    recall = np.where(parts["n_true_hits"] > 0, tracks["n_matched_hits"] / parts["n_true_hits"], 0)
    tracks["precision"] = precision
    tracks["recall"] = recall
    # perfect match scheme: all hits are assigned to true track (recall and precision == 1)
    tracks["eff_perfect"] = (precision == 1) & (recall == 1)
    parts["eff_perfect"] = (precision == 1) & (recall == 1)
    # Double Majority scheme: 50% of hits are assigned to true track (recall and precision > 0.5)
    tracks["eff_dm"] = (precision > 0.5) & (recall > 0.5)
    parts["eff_dm"] = (precision > 0.5) & (recall > 0.5)
    # LHC eff: precision > 0.75
    tracks["eff_lhc"] = precision > 0.75
    parts["eff_lhc"] = precision > 0.75
    # tag particles that are valid
    parts["valid"] = np.array(f[idx]["targets"]["particle_valid"][:][0])
    parts["reconstructable"] = (
        (parts["valid"]) & (parts["n_true_hits"] >= 3) & (parts["particle_eta"].abs() < eta_cut) & (parts["particle_pt"] > pt_cut)
    )
    # tag tracks that are valid
    tracks["valid"] = np.array(f[idx]["preds"]["final"]["track_valid"]["track_valid"][:][0])
    tracks["reconstructable"] = (tracks["valid"]) & (tracks["n_pred_hits"] >= 3)
    if "track_eta" in tracks.columns:
        tracks["reconstructable"] = tracks["reconstructable"] & (tracks["track_eta"].abs() < eta_cut)
    if "track_pt" in tracks.columns:
        tracks["reconstructable"] = tracks["reconstructable"] & (tracks["track_pt"] > pt_cut)

    # find duplicate track predictions
    seen_masks = {}
    tracks["duplicate"] = False
    for i, mask in enumerate(masks):
        if not tracks.loc[i, "reconstructable"]:
            continue
        mask_bytes = mask.tobytes()
        if mask_bytes in seen_masks:
            tracks.loc[i, "duplicate"] = True
            tracks.loc[i, "eff_dm"] = False
            tracks.loc[i, "eff_perfect"] = False
            tracks.loc[i, "eff_lhc"] = False
        else:
            seen_masks[mask_bytes] = i

    return tracks, parts


def load_events(fname, num_events=None, randomize=False, index_list=None, eta_cut=2.5, pt_cut=1, particle_targets=None):

    """Sequentially load events from an evaluation file and aggregate into a single DataFrame

    Arguments:
    ----------
    fname: str
        filepath of the evaluation file
    num_events: int
        number of events to load
    randomize: bool
        create a random set of events from evaluation file
    index_list: list[int or str]
        specify a list of indexes to load
    eta_cut: float
        Accepted pseudorapidity range
    pt_cut: float
        Transverse momentum [GeV] threshold for tracks
    particle_targets: list[str]
        list of physical quantities that are regression targets

    Returns:
    --------
    tracks: DataFrame
        Predicted information of each track in an event. shape = (n_events * n_max_particles, )
    parts: DataFrame
        Truth information of each track in this event. shape = (n_events * n_max_particles, )

    """

    if particle_targets is None:
        particle_targets=["particle_pt", "particle_eta", "particle_phi"]
        
    f = h5py.File(fname)
    if num_events is None:
        num_events = len(f.keys())

    if (num_events <= 0) or (not isinstance(num_events, int)):
        raise ValueError("Only positive integer amounts allowed.")

    if num_events > len(f.keys()):
        warnings.warn("Requested amount of events exceeds record. Using all %d events." % (len(f.keys())))

    if index_list is not None:
        id_list = index_list
    elif randomize and (num_events < len(f.keys())):
        id_list = np.random.choice(list(f.keys()), size=num_events, replace=False)
    else:
        id_list = list(f.keys())

    for i, idx in enumerate(id_list):
        if i == 0:
            tracks, parts = load_event(f, idx, eta_cut, pt_cut, particle_targets)
        else:
            tmp_tracks, tmp_parts = load_event(f, idx, eta_cut, pt_cut, particle_targets)
            tracks = pd.concat([tracks, tmp_tracks])
            parts = pd.concat([parts, tmp_parts])
        print("loaded event #" + idx)

    return (tracks, parts)
