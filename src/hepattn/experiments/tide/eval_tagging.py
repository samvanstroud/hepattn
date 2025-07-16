from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    log_dir = Path("/share/rcifdata/maxhart/hepattn/logs/")

    eval_paths = {
        "hits": log_dir / Path("TIDE_10M_100_40trk_F32_tagging_hits_20250618-T161204/ckpts/epoch=000-train_loss=3.10198_test_eval.h5"),
        "tide": log_dir / Path("TIDE_10M_250_40trk_F32_tagging_tide_20250617-T140243/ckpts/epoch=000-train_loss=250.25084_test_eval.h5"),
    }

    eval_aliases = {
        "sudo": "Pseudo Tracks",
        "reco": "Reco Tracks",
        "reco_hits": "Reco Tracks + Hits",
        "tide": "MF Tracks",
        "hits": "Hits",
    }

    eval_colors = {
        "sudo": "gray",
        "reco": "cornflowerblue",
        "reco_hits": "mediumseagreen",
        "tide": "mediumvioletred",
        "hits": "mediumseagreen",
    }

    class_names = ["b", "c", "tau", "other"]
    classes_alises = {"b": r"$b$", "c": r"$c$", "tau": r"$\tau$", "other": "Light / Other", "notb": r"Not $b$"}
    class_linestyles = {"b": "dashed", "c": "solid", "tau": "dotted", "other": "dashdot"}

    fig_scores, ax_scores = plt.subplots()
    fig_scores.set_size_inches(8, 3)

    fig_rocs, ax_rocs = plt.subplots(nrows=4, ncols=1)
    fig_rocs.set_size_inches(8, 8)

    for eval_name, eval_path in eval_paths.items():
        labels = {class_name: [] for class_name in class_names}
        discriminants = []

        with h5py.File(Path(eval_path)) as file:
            for _i, sample_id in tqdm(enumerate(file.keys())):
                # Load the targets and predictions
                targets = file[sample_id]["targets"]
                outputs = file[sample_id]["outputs"]["final"]["roi_tagging"]

                # Record the true classes
                for class_name in class_names:
                    labels[class_name].append(targets[f"roi_is_{class_name}"][0].item())

                # Calculate the pred discriminant
                logits = outputs["jet_logits"][0]

                p_b = sigmoid(logits[0].item())
                p_c = sigmoid(logits[1].item())
                p_tau = sigmoid(logits[2].item())
                p_other = sigmoid(logits[3].item())

                f_c = 0.2
                f_tau = 0.01

                discriminant = np.log(p_b / (f_c * p_c + f_tau * p_tau + (1 - f_c - f_tau) * p_other))
                discriminants.append(discriminant)

        labels = {k: np.array(v) for k, v in labels.items()}
        discriminants = np.array(discriminants)

        for class_name in ["b", "other"]:
            ax_scores.hist(
                discriminants[labels[class_name].astype(bool)],
                bins=np.linspace(-5, 5, 32),
                density=True,
                histtype="step",
                color=eval_colors[eval_name],
                linestyle=class_linestyles[class_name],
                label=f"{eval_aliases[eval_name]} {classes_alises[class_name]}",
            )

        working_points = np.linspace(-5, 5, 1000)

        effs = np.array([np.sum((discriminants >= wp) & labels["b"].astype(bool)) / np.sum(labels["b"]) for wp in working_points])
        class_rejs = {}

        class_rejs["notb"] = class_rejs[class_name] = np.array([
            1 / (np.sum((discriminants >= wp) & (~labels["b"].astype(bool))) / np.sum(~labels["b"].astype(bool))) for wp in working_points
        ])

        for class_name in class_names:
            class_rejs[class_name] = np.array([
                1 / (np.sum((discriminants >= wp) & labels[class_name].astype(bool)) / np.sum(labels[class_name])) for wp in working_points
            ])

        for ax_idx, class_name in enumerate(["notb", "c", "tau", "other"]):
            ax_rocs[ax_idx].plot(effs, class_rejs[class_name], color=eval_colors[eval_name], label=eval_aliases[eval_name])
            ax_rocs[ax_idx].set_yscale("log")
            ax_rocs[ax_idx].set_ylabel(f"{classes_alises[class_name]} Rejection")
            ax_rocs[ax_idx].grid(alpha=0.5, linestyle="--")
            ax_rocs[ax_idx].set_xlim(0.2, 1.0)
            ax_rocs[ax_idx].set_ylim(1.0, 100.0)

    ax_rocs[-1].set_xlabel(r"$b$ Efficiency")
    ax_rocs[0].legend(fontsize=8)

    ax_scores.set_xlabel(r"$D_b$")
    ax_scores.set_ylabel("Density")
    ax_scores.set_yscale("log")
    ax_scores.grid(alpha=0.5, linestyle="--")
    ax_scores.legend(fontsize=8)

    fig_scores.tight_layout()
    fig_scores.savefig("src/hepattn/experiments/tide/plots/scores.png")

    fig_rocs.tight_layout()
    fig_rocs.savefig("/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/plots/rocs.png")


if __name__ == "__main__":
    main()
