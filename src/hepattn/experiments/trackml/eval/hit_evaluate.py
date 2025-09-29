import warnings

import h5py
import numpy as np
import pandas as pd
import sklearn
import torch


def load_event(f, idx, write_inputs=None, write_parts=True, threshold=0.1):

    """Load an event from an evaluation file and create a DataFrame

    Arguments:
    ----------
    f: File
        h5 file read into buffer
    idx: str or int
        the event identifier (e.g. "29800" to "29899")
    write_inputs: list[str]
        specify a list of input features to load
    write_parts: bool
        specify whether to load particle level information
    threshold: float
        threshold value between [0,1] set on discriminant to identify valid hits

    Returns:
    --------
    hits: DataFrame
        Prediction of whether hits in an event are valid. shape = (hits, )
    target: DataFrame
        Truth information of each hit in an event. shape = (hits, )
    parts: DataFrame
        Truth information of particles in this event. shape = (max_particles, )

    """

    hits = pd.DataFrame()
    targets = pd.DataFrame()

    hits["event_id"] = int(idx)
    targets["event_id"] = int(idx)

    hits["score"] = np.array(f[idx]["outputs"]["final"]["hit_filter"]["hit_logit"][:][0])
    hits["score_sigmoid"] = torch.tensor(hits["score"]).float().sigmoid().numpy()
    hits["score_bool"] = np.where(hits["score_sigmoid"] < threshold, False, True)
    hits["pred"] = np.array(f[idx]["preds"]["final"]["hit_filter"]["hit_on_valid_particle"][:][0])

    targets["hit_on_valid_particle"] = np.array(f[idx]["targets"]["hit_on_valid_particle"][:][0])
    targets["hit_valid"] = np.array(f[idx]["targets"]["hit_valid"][:][0])

    if (write_inputs is not None) or (write_parts):
        hit_particle = np.array(f[idx]["targets"]["particle_hit_valid"][:][0])
        targets["particle_id"] = np.argmax(hit_particle, axis=0)
        if write_inputs is not None:
            for x in write_inputs:
                hits[x] = np.array(f[idx]["inputs"][x][:][0])
        if write_parts:
            parts = pd.DataFrame()
            parts["particle_pt"] = np.array(f[idx]["targets"]["particle_pt"][:][0])
            parts["pred_hits"] = np.sum(hit_particle[:, hits["score_bool"]], axis=-1)
            parts["valid"] = np.array(f[idx]["targets"]["particle_valid"][:][0])

    return hits, targets, parts


def eval_event(hits, targets, threshold=0.1):

    """Calculate matrics for binary classification task

    Arguments:
    ----------
    hits: DataFrame
        The `hits` DataFrame with prediction values (discriminant score)
    targets: DataFrame
        The `targets` DataFrame with true value
    threshold: float
        Threshold value on discriminant score to predict (in)valid hits

    Returns:
    ----------
    metrics: dict
        dict containing confusion matrix elements and precision-recall curve

    """

    y_pred = np.where(hits["score_sigmoid"] >= threshold, True, False)
    y_true = targets["hit_on_valid_particle"]
    metrics = dict()
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize="all").ravel().tolist()
    metrics["true_negative_rate"] = tn
    metrics["false_positive_rate"] = fp
    metrics["false_negative_rate"] = fn
    metrics["true_positive_rate"] = tp
    metrics["precision_score"] = sklearn.metrics.precision_score(y_pred, y_true)
    metrics["recall_score"] = sklearn.metrics.recall_score(y_pred, y_true)
    pre, rec, thr = sklearn.metrics.precision_recall_curve(y_true, hits["score_sigmoid"])
    metrics["roc_eff"] = rec
    metrics["roc_pur"] = pre
    metrics["roc_eff_pur_thr"] = thr
    metrics["roc_eff_pur_auc"] = sklearn.metrics.auc(rec, pre)
    fpr, tpr, thr = sklearn.metrics.roc_curve(y_true, hits["score_sigmoid"])
    metrics["roc_tpr"] = tpr
    metrics["roc_fpr"] = fpr
    metrics["roc_fpr_tpr_thr"] = thr
    metrics["roc_fpr_tpr_auc"] = sklearn.metrics.auc(fpr, tpr)

    return metrics


def load_events(fname, num_events=None, randomize=False, index_list=None, write_inputs=None, write_parts=True, threshold=0.1):

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
    write_inputs: list[str]
        specify a list of input features to load
    write_parts: bool
        specify whether to load particle level information
    threshold: float
        threshold value between [0,1] set on discriminant to identify valid hits

    Returns:
    --------
    hits: DataFrame
        Prediction of whether hits in an event are valid. shape = (hits, )
    target: DataFrame
        Truth information of each hit in an event. shape = (hits, )
    parts: DataFrame
        Truth information of particles in this event. shape = (max_particles, )
    metrics:
        dict containing confusion matrix elements and precision-recall curve

    """

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
            hits, targets, parts = load_event(f, idx, write_inputs, write_parts, threshold)
        else:
            tmp_hits, tmp_targets, tmp_parts = load_event(f, idx, write_inputs, write_parts, threshold)
            hits = pd.concat([hits, tmp_hits])
            targets = pd.concat([targets, tmp_targets])
            parts = pd.concat([parts, tmp_parts])
        print("loaded event #" + idx)
    metrics = eval_event(hits, targets)

    return (hits, targets, parts, metrics)
