# ruff: noqa: E501

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import binned_statistic
from tqdm import tqdm

from hepattn.experiments.itk.data import ITkDataset
from hepattn.utils.eval_plots import bayesian_binomial_error, plot_hist_to_ax

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    # Arguments for the evaluation
    # First specify the config which will provide things like input info etc
    config_path = Path("/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/itk/configs/filtering_pixel.yaml")
    recon_max_eta = 4.0
    recon_min_pt = 1.0
    recon_min_num_pixel = 3.0

    # Now create the dataset
    config = yaml.safe_load(config_path.read_text())["data"]
    inputs = config["inputs"]

    # Add in extra target fields that will allow us to recompute reconstructability
    targets = config["targets"]
    targets["particle"] = ["pt", "eta", "phi"]
    targets["particle_pixel"] = []

    dataset = ITkDataset(
        dirpath=config["test_dir"],
        inputs=inputs,
        targets=targets,
        num_events=-1,
        hit_regions=config["hit_regions"],
        particle_min_pt=recon_min_pt,
        particle_max_abs_eta=recon_max_eta,
        particle_min_num_hits={"pixel": recon_min_num_pixel},
        event_max_num_particles=5000,
    )

    # Give the test eval file we are evaluating and setup the file
    hit_eval_path = "/share/rcifdata/maxhart/hepattn-test/hepattn/logs/ITk_pixel_region135_eta4_900mev_20250428-T091926/ckpts/epoch=061-val_loss=0.43928_test_eval.h5"

    # Define bins for particle retention rate under the nominal working point
    particle_bins = {"pt": np.linspace(0.5, 10.0, 32), "eta": np.linspace(-4, 4, 32), "phi": np.linspace(-np.pi, np.pi, 32)}
    particle_pre_counts = {field: np.zeros(len(particle_bins[field]) - 1) for field in particle_bins}
    particle_post_counts = {field: np.zeros(len(particle_bins[field]) - 1) for field in particle_bins}

    num_hits_pre = []
    num_recon_parts_pre = []

    working_points = [0.01, 0.05, 0.1]
    wp_num_hits_post = {wp: [] for wp in working_points}
    wp_num_recon_parts_post = {wp: [] for wp in working_points}

    hit = "pixel"

    # Working point that is used for the bulk plots
    nominal_wp = 0.05

    # Iterate over the events
    for idx in tqdm(range(100)):
        # Load the data from the event
        sample_id = dataset.sample_ids[idx]

        # Note we are using load event, so evenerying is numpy arrays that are unbatched
        inputs, targets = dataset.load_event(sample_id)

        with h5py.File(hit_eval_path, "r") as hit_eval_file:
            hit_logits = hit_eval_file[f"{sample_id}/outputs/final/{hit}_filter/{hit}_logit"][0]

        # Particles which are deemed reconstructable pre-filter
        particle_recon_pre = targets["particle_valid"]
        particle_hit_valid_pre = targets["particle_pixel_valid"]

        # Drop any invalid particles that are not reconstructable
        # The pt and eta cuts should be applied by the dataloader
        particle_hit_valid_pre = particle_hit_valid_pre[particle_recon_pre]
        particle_recon_pre = particle_recon_pre[particle_recon_pre]

        # Record number of reconstructable particles and hits before filtering
        num_hits_pre.append(particle_hit_valid_pre.shape[-1])
        num_recon_parts_pre.append(particle_recon_pre.sum())

        # Mark hits which pass the filter
        hit_filter_pred = sigmoid(hit_logits) >= nominal_wp

        # The post filter mask is just the pre filter mask, but with filtered hits removed
        particle_hit_valid_post = particle_hit_valid_pre & hit_filter_pred[None, :]
        particle_recon_post = particle_hit_valid_post.sum(-1) >= 3

        # Fill the particle histograms
        for field, bins in particle_bins.items():
            particle_field = targets[f"particle_{field}"][targets["particle_valid"]]

            post_count, _, _ = binned_statistic(particle_field, particle_recon_post, statistic="sum", bins=bins)
            pre_count, _, _ = binned_statistic(particle_field, particle_recon_pre, statistic="sum", bins=bins)

            particle_pre_counts[field] += pre_count
            particle_post_counts[field] += post_count

        # Now calculate metrics for different working points
        for working_point in working_points:
            # Mark hits which pass the filter under this new working point
            hit_filter_pred = sigmoid(hit_logits) >= working_point
            particle_hit_valid_post = particle_hit_valid_pre & hit_filter_pred[None, :]
            particle_recon_post = particle_hit_valid_post.sum(-1) >= 3

            wp_num_hits_post[working_point].append(hit_filter_pred.sum())
            wp_num_recon_parts_post[working_point].append(particle_recon_post.sum())

    plot_save_dir = Path(__file__).resolve().parent / Path("evalplots")

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4)

    for wp in working_points:
        frac_recon_parts_retained = np.array(wp_num_recon_parts_post[wp]) / np.array(num_recon_parts_pre)
        ax.scatter(np.array(wp_num_hits_post[wp]), frac_recon_parts_retained, label=wp, alpha=0.5)

    ax.grid(zorder=0, alpha=0.25, linestyle="--")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Number of Hits Retained")
    ax.set_ylabel("Fraction of Reconstructable Particles Retained")

    fig.savefig(plot_save_dir / Path("wp_scan.png"))

    pre_count = particle_pre_counts["pt"]
    post_count = particle_post_counts["pt"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["pt"], eff_errors)

    ax.set_xlabel("Truth Particle $p_T$ [GeV]")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.97, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_pt.png"))

    pre_count = particle_pre_counts["eta"]
    post_count = particle_post_counts["eta"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["eta"], eff_errors)

    ax.set_xlabel(r"Truth Particle $\eta$")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.96, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_eta.png"))

    pre_count = particle_pre_counts["phi"]
    post_count = particle_post_counts["phi"]

    eff = post_count / pre_count
    eff_errors = bayesian_binomial_error(post_count, pre_count)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 2)

    plot_hist_to_ax(ax, eff, particle_bins["phi"], eff_errors)

    ax.set_xlabel(r"Truth Particle $\phi$")
    ax.set_ylabel("Fraction of Reconstructable \n Particles Retained")
    ax.set_ylim(0.97, 1.005)
    ax.grid(zorder=0, alpha=0.25, linestyle="--")

    fig.savefig(plot_save_dir / Path("particle_recon_phi.png"))


if __name__ == "__main__":
    main()
