# ruff: noqa: E501

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    eval_path = Path("/share/rcifdata/maxhart/hepattn/logs/pixel_mini_20250627-T112451/ckpts/epoch=007-train_loss=-0.68976_new_eval.h5")

    data = {
        "true_valid": [],
        "pred_valid": [],
        "true_x": [],
        "true_y": [],
        "pred_x": [],
        "pred_y": [],
    }

    with h5py.File(eval_path) as file:
        for _i, sample_id in tqdm(enumerate(file.keys())):
            sample_targets = file[sample_id]["targets"]
            sample_outputs = file[sample_id]["outputs"]["final"]

            data["true_valid"].append(sample_targets["particle_valid"][0])
            data["pred_valid"].append(sigmoid(sample_outputs["track_valid"]["track_logit"][0]) >= 0.5)
            data["true_x"].append(sample_targets["particle_x"][0])
            data["true_y"].append(sample_targets["particle_y"][0])
            data["pred_x"].append(sample_outputs["track_regr"]["track_x"][0])
            data["pred_y"].append(sample_outputs["track_regr"]["track_y"][0])

    for k, v in data.items():
        data[k] = np.stack(v, axis=0)
        print(k, data[k].shape)

    data["true_multiplicity"] = data["true_valid"].sum(-1)
    data["pred_multiplicity"] = data["pred_valid"].sum(-1)

    data["res_x"] = data["pred_x"] - data["true_x"]
    data["res_y"] = data["pred_y"] - data["true_y"]

    labels = np.arange(8) + 1

    conf_mat = 100.0 * confusion_matrix(
        data["true_multiplicity"],
        data["pred_multiplicity"],
        labels=labels,
        normalize="true")

    conf_mat = np.flipud(conf_mat)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(4, 4)

    ax.imshow(conf_mat)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            if conf_mat[i, j] > 0:
                ax.text(x=j, y=i, s=f"{np.round(conf_mat[i, j], 2)}", va="center", ha="center", fontsize=6,
                path_effects=[patheffects.withStroke(linewidth=0.0, foreground="white")])

    ax.set_xticks(labels - 1, [str(i) for i in labels])
    ax.set_yticks(labels - 1, reversed([str(i) for i in labels]))

    ax.set_xlabel("Predicted Particle Count", fontsize=8)
    ax.set_ylabel("True Particle Count", fontsize=8)

    fig.tight_layout()
    fig.savefig("src/hepattn/experiments/pixel/plots/multiplicity_cmat.png")

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(8, 2)

    bins = np.linspace(-1, 1, 24)

    for multiplicity in [1, 2, 3, 4]:
        mask = (data["true_multiplicity"] == data["pred_multiplicity"]) & (data["true_multiplicity"] == multiplicity)
        masked_res_x = data["res_x"][mask][data["true_valid"][mask]]
        masked_res_y = data["res_y"][mask][data["true_valid"][mask]]

        rmse_x = np.sqrt(np.mean(masked_res_x**2))
        rmse_y = np.sqrt(np.mean(masked_res_y**2))

        label_x = f"$N_P=${multiplicity}, RMSE={rmse_x:.2f}"
        label_y = f"$N_P=${multiplicity}, RMSE={rmse_y:.2f}"

        ax[0].hist(masked_res_x, bins=bins, histtype="step", label=label_x, density=True)
        ax[1].hist(masked_res_y, bins=bins, histtype="step", label=label_y, density=True)

    ax[0].set_xlabel(r"Residual $x$", fontsize=8)
    ax[1].set_xlabel(r"Residual $y$", fontsize=8)

    ax[0].set_ylabel("Density", fontsize=8)
    ax[1].set_ylabel("Density", fontsize=8)

    ax[0].grid(alpha=0.25, linestyle="--")
    ax[1].grid(alpha=0.25, linestyle="--")

    ax[0].legend(fontsize=6)
    ax[1].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig("src/hepattn/experiments/pixel/plots/residuals.png")

    ax[0].set_yscale("log")
    ax[1].set_yscale("log")

    fig.savefig("src/hepattn/experiments/pixel/plots/residuals_logscale.png")


if __name__ == "__main__":
    main()
