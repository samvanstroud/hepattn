from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_event(fname, i):
    f = h5py.File(fname)
    events = list(f.keys())
    event = f[events[i]]
    targets = event["targets"]
    hit_prob = torch.tensor(event["hit_pred"][:]).float().sigmoid().numpy()
    hit_tgt = targets["hit"]["hit_tgt"][:].astype(int)
    hit_pid = targets["hit"]["tgt_pid"][:]
    hits = pd.DataFrame({"pid": hit_pid, "prob": hit_prob, "tgt": hit_tgt})
    hits.index = hits.pid
    return hits


def eval_event(fname, i, hit_cut):
    hits = load_event(fname, i)
    hits["pred"] = hits.prob > hit_cut

    return hits


def load_events(fname, num_events, hit_cut=0.1):
    for i in range(num_events):
        print(f"Processing event {i}", end="\r")
        if i == 0:
            hits = eval_event(fname, i, hit_cut)
        else:
            hits_ = eval_event(fname, i, hit_cut)
            hits = pd.concat([hits, hits_])
    return hits


def make_plots(hits, parts, out_dir):
    plt.figure()
    plt.title("Check the hit count distribution")
    kwargs = {"bins": 15, "range": (0, 15), "histtype": "step", "log": False}
    hit_level_eff = parts[parts.pid != 0].hits_post.sum() / parts[parts.pid != 0].hits_pre.sum()
    plt.hist(parts[parts.pid != 0].hits_pre, **kwargs, label="Pre Filter")
    plt.hist(parts[parts.pid != 0].hits_post, **kwargs, label=f"Post Filter {hit_level_eff:.2%}")
    plt.xlabel("Number of hits left by particle")
    plt.ylabel("Number of particles")
    plt.legend()
    plt.savefig(out_dir / "hit_count.png")

    plt.figure()
    kwargs = {"bins": 10, "range": (0, 5), "histtype": "step", "log": False}
    track_level_eff = parts.reconstructable_post.sum() / parts.reconstructable_pre.sum()
    plt.hist(parts.loc[parts.reconstructable_pre].pt, **kwargs, label="inital: 100%")
    plt.hist(parts.loc[parts.reconstructable_post].pt, **kwargs, label=f"filtered {track_level_eff:.2%}")
    plt.xlabel("Particle pT [GeV]")
    plt.ylabel("Number of reconstructable particles (>=3 hits)")
    plt.legend()
    plt.savefig(out_dir / "reconstructable.png")

    plt.figure()
    kwargs = {"bins": 10, "range": (0, 5), "histtype": "step", "log": True}
    plt.hist(parts[parts.hits_pre >= 3].pt, **kwargs, label="Pre")
    plt.hist(parts[parts.hits_post >= 3].pt, **kwargs, label="Post")
    plt.xlabel("Particle pT [GeV]")
    plt.ylabel("Number of reconstructable particles (>=3 hits)")
    plt.savefig(out_dir / "reconstructable_all.png")

    plt.figure()
    kwargs = {"bins": 10, "range": (0, 1), "histtype": "step", "log": True}
    plt.hist(hits.prob, **kwargs, label="all")
    plt.hist(hits.loc[hits.tgt].prob, **kwargs, label="valid")
    plt.hist(hits.loc[~hits.tgt].prob, **kwargs, label="noise")
    plt.legend()
    plt.xlabel("model output")
    plt.savefig(out_dir / "hit_prob.png")


fname1 = "/share/rcifdata/svanstroud/hepattn/logs/HF-ftest-1024ws-10layer-rPE0.1-hybridnorm_20250411-T210657/ckpts/epoch=010-validate_loss=0.17719__test.h5"  # noqa: E501
fname2 = "/share/rcifdata/svanstroud/hepattn/logs/HF-ftest-1024ws-10layer-rPE0.1-hybridnorm_20250411-T210657/ckpts/epoch=039-validate_loss=0.30950__test.h5"  # noqa: E501


HIT_CUT = 0.01
NUM_EVENTS = 100

if __name__ == "__main__":
    fpath = Path(fname1)
    print(fpath)
    print("hit cut:", HIT_CUT)
    out_dir = fpath.parent.parent / ("plots_" + fpath.stem.split("_")[-1])
    out_dir.mkdir(exist_ok=True)

    num_events = NUM_EVENTS
    hits = load_events(fpath, num_events, hit_cut=HIT_CUT)

    # get recall and precision of
    recall = (hits.pred & hits.tgt).sum() / hits.tgt.sum()
    precision = (hits.pred & hits.tgt).sum() / hits.pred.sum()

    # get the standard error on the mean for the recall and precision
    se_recall = (recall * (1 - recall) / hits.tgt.sum()) ** 0.5
    se_precision = (precision * (1 - precision) / hits.pred.sum()) ** 0.5

    # get the average number of hits pre and post filter per event
    avg_hits_pre = len(hits) / num_events
    avg_hits_post = hits.pred.sum() / num_events

    # track level eff

    # recall before filtering
    precision_pre = hits.tgt.mean()

    # number of positive and negative samples
    print(f"Number of positive samples: {(hits.tgt == 1).sum() / NUM_EVENTS:.1f}")
    print(f"Number of negative samples: {(hits.tgt == 0).sum() / NUM_EVENTS:.1f}")

    print(f"Precision pre filter: {precision_pre:.1%}")
    print(f"Precision post: {precision:.1%}")
    print(f"Recall post: {recall:.1%}")
    print(f"Hits before filter: {avg_hits_pre:.1f}")
    print(f"Hits after filter: {avg_hits_post:.1f}")

    # plot histograms of model outputs for positive and negtaive samples

    plt.figure(constrained_layout=True)
    hits = load_events(fname1, num_events, hit_cut=HIT_CUT)

    probs = hits.prob.to_numpy()
    positive = np.clip(probs[hits.tgt == 1], 0, 1)
    negative = np.clip(probs[hits.tgt == 0], 0, 1)
    plt.hist(positive, bins=100, range=(0, 1), label="valid epoch 10", histtype="step", log=True)
    plt.hist(negative, bins=100, range=(0, 1), label="noise epoch 10", histtype="step", log=True)

    hits = load_events(fname2, num_events, hit_cut=HIT_CUT)
    probs = hits.prob.to_numpy()
    positive = np.clip(probs[hits.tgt == 1], 0, 1)
    negative = np.clip(probs[hits.tgt == 0], 0, 1)
    plt.hist(positive, bins=100, range=(0, 1), label="valid epoch 39", histtype="step", log=True, ls="dashed")
    plt.hist(negative, bins=100, range=(0, 1), label="noise epoch 39", histtype="step", log=True, ls="dashed")

    plt.xlabel("model output")
    plt.ylabel("Number of hits")
    plt.legend()

    plt.savefig(out_dir / "hit_prob.png")

    # plot roc and auc using sklearn
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thresholds = roc_curve(hits.tgt, hits.prob)
    auc = roc_auc_score(hits.tgt, hits.prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2%}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(out_dir / "roc.png")
