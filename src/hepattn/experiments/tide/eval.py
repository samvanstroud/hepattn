from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binned_statistic
from tqdm import tqdm

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    eval_path = Path("/share/rcifdata/maxhart/hepattn/logs/TIDE_100k_100_32trk_20250514-T142829/ckpts/epoch=092-train_loss=24.68639_train_eval.h5")

    pred_names = ["sudo", "sisp", "reco", "pred"]
    colors = {
        "sudo": "black",
        "sisp": "red",
        "reco": "blue",
        "pred": "purple",
    }

    trk_qtys = [
        ("sudo_pt", r"Track $p_\mathrm{T}$", "log", np.geomspace(1, 3.5e3, 24)),
        ("sudo_eta", r"Track $\eta$", "linear", np.linspace(-4, 4, 24)),
        ("sudo_phi", r"Track $\phi$", "linear", np.linspace(-np.pi, np.pi, 24)),
    ]

    trk_all_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in trk_qtys} for pred_name in pred_names}
    trk_eff_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in trk_qtys} for pred_name in pred_names}

    with h5py.File(eval_path) as file:
        for i, sample_id in tqdm(enumerate(file.keys())):
            if i == 100:
                break

            targets = file[sample_id]["targets"]
            outputs = file[sample_id]["outputs"]["final"]

            true_valid = targets["sudo_valid"][0]
            true_pix_valid = targets["sudo_pix_valid"][0]
            true_sct_valid = targets["sudo_sct_valid"][0]

            preds = {}
            for pred_name in ["sudo", "sisp", "reco"]:
                preds[f"{pred_name}_valid"] = targets[f"{pred_name}_valid"][0]
                preds[f"{pred_name}_pix_valid"] = targets[f"{pred_name}_pix_valid"][0]
                preds[f"{pred_name}_sct_valid"] = targets[f"{pred_name}_sct_valid"][0]

            preds["pred_valid"] = sigmoid(outputs["pred_valid"]["pred_logit"][0]) >= 0.5
            preds["pred_pix_valid"] = sigmoid(outputs["pred_pix_assignment"]["pred_pix_logit"][0]) >= 0.5
            preds["pred_sct_valid"] = sigmoid(outputs["pred_sct_assignment"]["pred_sct_logit"][0]) >= 0.5

            for pred_name in pred_names:
                pred_valid = preds[f"{pred_name}_valid"]
                pred_pix_valid = preds[f"{pred_name}_pix_valid"]
                pred_sct_valid = preds[f"{pred_name}_sct_valid"]

                pix_tp = np.einsum("nc,mc->nm", true_pix_valid, pred_pix_valid)
                pix_fp = np.einsum("nc,mc->nm", true_pix_valid, (~pred_pix_valid))
                pix_fn = np.einsum("nc,mc->nm", (~true_pix_valid), pred_pix_valid)

                sct_tp = np.einsum("nc,mc->nm", true_sct_valid, pred_sct_valid)
                sct_fp = np.einsum("nc,mc->nm", true_sct_valid, (~pred_sct_valid))
                sct_fn = np.einsum("nc,mc->nm", (~true_sct_valid), pred_sct_valid)

                eps = 1e-6
                metric = "iou"
                score_threshold = 0.75

                # Using the masks we calculate the desired score - the eps term prevents any division by zero
                if metric == "tmp":
                    scores = (2 * pix_tp + sct_tp) / (2 * (pix_tp + pix_fp) + sct_tp + sct_fp + eps)
                elif metric == "iou":
                    scores = (2 * pix_tp + sct_tp) / (2 * (pix_tp + pix_fp + pix_fn) + sct_tp + sct_fp + sct_fn + eps)
                elif metric == "dice":
                    scores = 2 * (pix_tp + sct_tp) / (2 * pix_tp + pix_fp + pix_fn + 2 * sct_tp + sct_fp + sct_fn + eps)

                _true_idx, pred_idx = linear_sum_assignment(scores)

                pred_valid = pred_valid[pred_idx]
                pred_pix_valid = pred_pix_valid[pred_idx]
                pred_sct_valid = pred_sct_valid[pred_idx]

                scores = scores[:, pred_idx]

                true_is_eff = (scores >= score_threshold).any(-1)[true_valid]

                for qty_name, _, _, bins in trk_qtys:
                    qty = targets[qty_name][0][true_valid]

                    num_all, _, _ = binned_statistic(qty, true_is_eff, statistic="count", bins=bins)
                    num_eff, _, _ = binned_statistic(qty, true_is_eff, statistic="sum", bins=bins)

                    trk_all_bins[pred_name][qty_name] += num_all
                    trk_eff_bins[pred_name][qty_name] += num_eff

    for qty_name, qty_label, scale, bins in trk_qtys:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)

        for pred_name in pred_names:
            freq_e = trk_eff_bins[pred_name][qty_name] / trk_all_bins[pred_name][qty_name]

            for bin_idx in range(len(bins) - 1):
                px = np.array([bins[bin_idx], bins[bin_idx + 1]])
                py = np.array([freq_e[bin_idx], freq_e[bin_idx]])
                # pe = np.array([freq_e_err[bin_idx], freq_e_err[bin_idx]])
                ax.plot(px, py, color=colors[pred_name], linewidth=1.0)
                # ax[0,qty_idx].fill_between(px, py - pe, py + pe, color=colors[pred], alpha=0.1, ec="none")

        ax.set_xscale(scale)

        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.grid(zorder=0, alpha=0.25, linestyle="--")

        ax.set_xlabel(qty_label)
        ax.set_ylabel("Track Efficiency")

        fig.tight_layout()
        fig.savefig(f"/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/{qty_name}.png")


if __name__ == "__main__":
    main()
