import math
import pathlib
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import binned, hist_plot, profile_plot
from track_evaluate import load_events

# ----------------------------------------------------
# Plotting setup
# ----------------------------------------------------

plt.rcParams["figure.dpi"] = 400
# plt.rcParams["text.usimport math
import pathlib
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import binned, hist_plot, profile_plot
from track_evaluate import load_events

# ----------------------------------------------------
# Plotting setup
# ----------------------------------------------------

plt.rcParams["figure.dpi"] = 400
# plt.rcParams["text.usetex"] = True
plt.rcParams["text.usetex"] = False
# disabled due to missing font in texlive on the Nikhef clusters
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True

# training_colours = {
#     "Paper": "tab:green",
#     "Tracking DQ": "tab:orange",  # |eta| < 4.0
#     "LCA": "tab:blue",
#     "LCA_reduce_window": "tab:green",
#     "LCA_no_wrap": "tab:orange",
# }

training_colours = {
    "Paper": "tab:green",
    "MA 600 Strip": "tab:orange",  # |eta| < 4.0
    "LCA 600 Strip": "tab:blue",
    "MA 900": "tab:red",
    "LCA 900": "tab:orange",
}

qty_bins = {
    "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
    # "eta": np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": ""}
out_dir = "/share/rcif2/pduckett/hepattn-dq-pr/src/hepattn/experiments/trackml/eval/"


def _event_id_to_event_index(event_id, key_mode=None):
    event_str = str(event_id)
    if event_str.startswith("event_"):
        event_idx = int(event_str.removeprefix("event_"))
    else:
        event_idx = int(event_str)
    if key_mode == "old":
        event_idx += 29800
    return event_idx


@lru_cache(maxsize=None)
def _event_file_map(test_dir, suffix):
    event_map = {}
    pattern = f"event*-{suffix}.parquet"
    for path in pathlib.Path(test_dir).glob(pattern):
        event_name = path.stem.removesuffix(f"-{suffix}")
        try:
            event_map[int(event_name.removeprefix("event"))] = path
        except ValueError:
            continue
    return event_map


def _read_first_available(group, dataset_names):
    for dataset_name in dataset_names:
        if dataset_name in group:
            return np.asarray(group[dataset_name][:][0])
    raise KeyError(f"None of {dataset_names} found under {group.name}")


def _input_types(data_config):
    inputs = data_config.get("inputs", {})
    if isinstance(inputs, dict):
        return list(inputs)
    return []


def _reconstructable_min_num_hits(data_config):
    if "reconstructable_min_num_hits" in data_config:
        return int(data_config["reconstructable_min_num_hits"])
    return int(data_config.get("particle_min_num_hits", 3))


def _reconstructable_min_num_pixel_hits(data_config):
    return int(data_config.get("particle_min_num_pixel_hits", 0))


def _pixel_volume_ids(data_config):
    feature_volume_ids = data_config.get("feature_volume_ids") or {}
    return feature_volume_ids.get("pixel")


def _reconstructable_definition_label(data_config):
    min_num_hits = _reconstructable_min_num_hits(data_config)
    min_num_pixel_hits = _reconstructable_min_num_pixel_hits(data_config)
    if min_num_pixel_hits > 0:
        return f">={min_num_hits} hits and >={min_num_pixel_hits} pixel hits"
    return f">={min_num_hits} hits"


def _postfilter_hits_per_event(fname, event_ids, key_mode=None):
    event_keys = [str(event_id) for event_id in event_ids]
    if not event_keys:
        return np.array([], dtype=np.int32)

    with h5py.File(fname, "r") as f:
        hit_counts = []
        for event_key in event_keys:
            if key_mode == "old":
                hit_counts.append(np.array(f[event_key]["hits"]["pids"]).shape[0])
            else:
                hit_counts.append(_read_first_available(f[event_key]["targets"], ["particle_key_valid", "particle_hit_valid"]).shape[-1])
    return np.asarray(hit_counts, dtype=np.int32)


def _prefilter_hits_per_event(data_config, event_ids, key_mode=None):
    if len(event_ids) == 0:
        return np.array([], dtype=np.int32)

    test_dir = pathlib.Path(data_config["test_dir"])
    hit_volume_ids = data_config.get("hit_volume_ids")
    event_file_map = _event_file_map(test_dir, "hits")

    hit_counts = []
    for event_id in event_ids:
        event_idx = _event_id_to_event_index(event_id, key_mode=key_mode)
        hits_path = event_file_map[event_idx]
        hits = pd.read_parquet(hits_path, columns=["volume_id"])
        if hit_volume_ids:
            hits = hits[hits["volume_id"].isin(hit_volume_ids)]
        hit_counts.append(len(hits))
    return np.asarray(hit_counts, dtype=np.int32)


def _load_prefilter_truth_particles_for_event(data_config, event_id, key_mode=None):
    test_dir = pathlib.Path(data_config["test_dir"])
    event_idx = _event_id_to_event_index(event_id, key_mode=key_mode)
    parts_path = _event_file_map(test_dir, "parts")[event_idx]
    hits_path = _event_file_map(test_dir, "hits")[event_idx]

    particles = pd.read_parquet(parts_path, columns=["particle_id", "px", "py", "pz"]).copy()
    hits = pd.read_parquet(hits_path, columns=["particle_id", "volume_id"])

    hit_volume_ids = data_config.get("hit_volume_ids")
    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)]

    particles["particle_pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
    particles["particle_phi"] = np.arctan2(particles["py"], particles["px"])
    particle_p = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
    particles["particle_eta"] = np.arctanh(particles["pz"] / particle_p)

    particles = particles[particles["particle_pt"] > data_config["particle_min_pt"]]
    particles = particles[particles["particle_eta"].abs() < data_config["particle_max_abs_eta"]]

    counts = hits["particle_id"].value_counts()
    keep_particle_ids = counts[counts >= _reconstructable_min_num_hits(data_config)].index.to_numpy()
    particles = particles[particles["particle_id"].isin(keep_particle_ids)].copy()

    min_num_pixel_hits = _reconstructable_min_num_pixel_hits(data_config)
    if min_num_pixel_hits > 0:
        pixel_volume_ids = _pixel_volume_ids(data_config)
        if not pixel_volume_ids:
            raise ValueError("particle_min_num_pixel_hits > 0 requires data.feature_volume_ids.pixel in the config.")
        pixel_hits = hits[hits["volume_id"].isin(pixel_volume_ids)]
        pixel_counts = pixel_hits["particle_id"].value_counts()
        keep_particle_ids = pixel_counts[pixel_counts >= min_num_pixel_hits].index.to_numpy()
        particles = particles[particles["particle_id"].isin(keep_particle_ids)].copy()

    event_max_num_particles = data_config["event_max_num_particles"]
    if len(particles) > event_max_num_particles:
        if data_config.get("strict_max_objects", False):
            raise ValueError(f"Event {event_id} has {len(particles)} particles, but limit is {event_max_num_particles}")
        particles = particles.iloc[:event_max_num_particles].copy()

    particles["event_id"] = event_id
    particles["valid"] = True
    particles["reconstructable"] = True
    return particles[["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable"]]


def _attach_eval_particle_ids(fname, tracks, parts, key_mode=None):
    tracks = tracks.copy()
    parts = parts.copy()
    if len(parts) == 0:
        tracks["matched_particle_id"] = np.array([], dtype=np.int64)
        parts["particle_id"] = np.array([], dtype=np.int64)
        return tracks, parts

    parts_particle_ids = np.full(len(parts), -1, dtype=np.int64)
    tracks_particle_ids = np.full(len(tracks), -1, dtype=np.int64)
    particle_ids_by_event = {}

    with h5py.File(fname, "r") as f:
        for event_id, event_parts in parts.groupby("event_id", sort=False):
            event_key = str(event_id)
            if key_mode == "old":
                particle_ids = np.array(f[event_key]["parts"]["pids"])
            else:
                particle_ids = np.array(f[event_key]["targets"]["particle_id"][:][0])
            if len(event_parts) > len(particle_ids):
                raise ValueError(f"Event {event_key} has more parts rows than particle IDs in the eval file.")
            parts_particle_ids[event_parts.index.to_numpy()] = particle_ids[: len(event_parts)]
            particle_ids_by_event[event_key] = parts_particle_ids[event_parts.index.to_numpy()]

    parts["particle_id"] = parts_particle_ids

    for event_id, event_tracks in tracks.groupby("event_id", sort=False):
        matched_pid = event_tracks["matched_pid"].to_numpy(dtype=np.int64)
        event_particle_ids = particle_ids_by_event[str(event_id)]
        valid_match = (matched_pid >= 0) & (matched_pid < len(event_particle_ids))
        matched_particle_ids = np.full(len(event_tracks), -1, dtype=np.int64)
        matched_particle_ids[valid_match] = event_particle_ids[matched_pid[valid_match]]
        tracks_particle_ids[event_tracks.index.to_numpy()] = matched_particle_ids

    tracks["matched_particle_id"] = tracks_particle_ids
    return tracks, parts


def _build_prefilter_truth_reference_parts(data_config, fname, tracks, parts, key_mode=None):
    tracks_with_ids, _parts_with_ids = _attach_eval_particle_ids(fname, tracks, parts, key_mode=key_mode)
    event_ids = np.sort(parts["event_id"].unique())
    truth_parts = []

    for event_id in event_ids:
        event_truth = _load_prefilter_truth_particles_for_event(data_config, event_id, key_mode=key_mode)
        event_tracks = tracks_with_ids[tracks_with_ids["event_id"] == event_id]
        dm_particle_ids = set(event_tracks.loc[event_tracks["eff_dm"], "matched_particle_id"].astype(np.int64).tolist())
        perfect_particle_ids = set(event_tracks.loc[event_tracks["eff_perfect"], "matched_particle_id"].astype(np.int64).tolist())
        dm_particle_ids.discard(-1)
        perfect_particle_ids.discard(-1)
        event_truth["eff_dm"] = event_truth["particle_id"].isin(dm_particle_ids)
        event_truth["eff_perfect"] = event_truth["particle_id"].isin(perfect_particle_ids)
        truth_parts.append(event_truth)

    if not truth_parts:
        return pd.DataFrame(columns=["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable", "eff_dm", "eff_perfect"])
    return pd.concat(truth_parts, ignore_index=True)


def _threshold_tag(track_valid_threshold, iou_threshold):
    track_str = f"{track_valid_threshold:.2f}".replace(".", "p")
    iou_str = f"{iou_threshold:.2f}".replace(".", "p")
    return f"tv{track_str}_iou{iou_str}"


def _as_threshold_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    return [float(value)]


def _resolve_effective_track_threshold(
    track_valid_threshold,
    track_quality_threshold,
    track_validity_mode,
    track_combination_threshold=None,
):
    if track_combination_threshold is not None:
        return float(track_combination_threshold)
    if track_validity_mode == "track_valid_prob_plus_track_quality_score":
        return float(track_valid_threshold + track_quality_threshold)
    if track_validity_mode == "track_valid_prob_times_track_quality_score":
        return float(track_valid_threshold * track_quality_threshold)
    return float(track_valid_threshold)


def _build_threshold_configs(
    track_valid_thresholds,
    track_quality_thresholds,
    iou_thresholds,
    track_validity_mode,
    track_combination_threshold=None,
    require_individual_cuts_for_combination=False,
):
    configs = []
    seen_effective = set()

    for track_valid_threshold in _as_threshold_list(track_valid_thresholds):
        for track_quality_threshold in _as_threshold_list(track_quality_thresholds):
            effective_track_threshold = _resolve_effective_track_threshold(
                track_valid_threshold,
                track_quality_threshold,
                track_validity_mode,
                track_combination_threshold,
            )
            for iou_threshold in _as_threshold_list(iou_thresholds):
                if (
                    track_validity_mode in {"track_valid_prob_plus_track_quality_score", "track_valid_prob_times_track_quality_score"}
                    and not require_individual_cuts_for_combination
                ):
                    dedupe_key = (round(effective_track_threshold, 12), round(iou_threshold, 12))
                    if dedupe_key in seen_effective:
                        continue
                    seen_effective.add(dedupe_key)

                configs.append(
                    {
                        "track_valid_threshold": track_valid_threshold,
                        "track_quality_threshold": track_quality_threshold,
                        "effective_track_threshold": effective_track_threshold,
                        "iou_threshold": iou_threshold,
                    }
                )

    return configs

# ----------------------------------------------------
# Read configuration file information
# ----------------------------------------------------

CONFIG_DIR = pathlib.Path("/share/rcif2/pduckett/hepattn-dq-pr/src/hepattn/experiments/trackml/configs")

tracking_fnames = {
    "MA 600 Strip": "/share/rcif2/pduckett/hepattn-dq-pr/logs/TRK-v8-STRIP-3100-0p3-th0p42-epochs30_20260415-T094006/ckpts/epoch=028-val_loss=0.29386_test_eval.h5",
    # "LCA 600 Strip": "/share/rcif2/pduckett/hepattn-dq-pr/logs/TRK-v8-strip-lca-2900-th0p4-0p42-epochs30_20260421-T150024/ckpts/epoch=026-val_loss=0.39419_test_eval.h5",
}

tracking_config_fname = {
    "MA 600 Strip": str(CONFIG_DIR / "tracking-strip.yaml"),
    "LCA 600 Strip": str(CONFIG_DIR / "tracking-strip-lca.yaml"),
}

tracking_params = ["particle_min_pt", "particle_max_abs_eta"]
tracking_configs = {}
for name in tracking_config_fname:
    with pathlib.Path(tracking_config_fname[name]).open() as f:
        fconfig = yaml.safe_load(f)
        print("name: " + fconfig["name"])
        for i in tracking_params:
            print("> " + i + "\t: ", fconfig["data"][i])
        print("> reconstructable\t: ", _reconstructable_definition_label(fconfig["data"]))
    tracking_configs[name] = fconfig

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

tracking_results = {}
num_events = 100
# track_valid_thresholds = [0.6, 0.7, 0.8, 0.9]
# iou_thresholds = [0.6, 0.7, 0.8, 0.9]
track_valid_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
particle_targets = ["pt", "eta", "phi"]
track_quality_thresholds = [0.0]
track_combination_threshold = None
track_validity_mode = "saved_track_valid"
require_individual_cuts_for_combination = False
match_valid_truth_only = False
matching_mode = "auto"
truth_reference_mode = "pre_filter"
# "post_filter": current behavior, denominator uses truth particles after hit filtering
# "pre_filter": denominator uses reconstructable truth particles before hit filtering
# Fake-rate metrics remain track-denominated.

# Eval files written by PredictionWriter already store the targets in the model-aligned
# order when a sorter is used, so the saved ordering is the correct default.
hit_order_by_name = {
    "Paper": "as_saved",
    "Tracking DQ": "as_saved",
    "LCA": "as_saved",
}


def _hit_order_mode_for_model(name):
    if name in hit_order_by_name:
        return hit_order_by_name[name]
    return "as_saved"

threshold_configs = _build_threshold_configs(
    track_valid_thresholds=track_valid_thresholds,
    track_quality_thresholds=track_quality_thresholds,
    iou_thresholds=iou_thresholds,
    track_validity_mode=track_validity_mode,
    track_combination_threshold=track_combination_threshold,
    require_individual_cuts_for_combination=require_individual_cuts_for_combination,
)

truth_reference_results = {}
index_list = None  # ["event_0", ..., "event_99"] or ["29800", ..., "29899"]
for threshold_config in threshold_configs:
    track_valid_threshold = threshold_config["track_valid_threshold"]
    track_quality_threshold = threshold_config["track_quality_threshold"]
    effective_track_threshold = threshold_config["effective_track_threshold"]
    iou_threshold = threshold_config["iou_threshold"]
    threshold_key = (track_valid_threshold, track_quality_threshold, iou_threshold)
    tracking_results[threshold_key] = {}
    truth_reference_results[threshold_key] = {}
    print(
        f"Evaluating threshold combination "
        f"track_valid_threshold={track_valid_threshold:.2f}, "
        f"track_quality_threshold={track_quality_threshold:.2f}, "
        f"effective_track_threshold={effective_track_threshold:.2f}, "
        f"iou_threshold={iou_threshold:.2f}, "
        f"require_individual_cuts_for_combination={require_individual_cuts_for_combination}, "
        f"matching_mode={matching_mode}"
    )
    for name, fname in tracking_fnames.items():
        eta_cut = tracking_configs[name]["data"]["particle_max_abs_eta"]
        pt_cut = tracking_configs[name]["data"]["particle_min_pt"]
        min_num_hits = _reconstructable_min_num_hits(tracking_configs[name]["data"])
        min_num_pixel_hits = _reconstructable_min_num_pixel_hits(tracking_configs[name]["data"])
        key_mode = "old" if name == "Paper" else None  # None or "old"
        print(
            f"Loading {name} model with PT > {pt_cut}, |eta| < {eta_cut}, "
            f"reconstructable={_reconstructable_definition_label(tracking_configs[name]['data'])}",
            f"",
        )
        tracking_results[threshold_key][name] = load_events(
            fname=fname,
            eta_cut=eta_cut,
            index_list=index_list,
            pt_cut=pt_cut,
            randomize=num_events,
            particle_targets=particle_targets,
            regression=False,
            key_mode=key_mode,
            track_valid_threshold=track_valid_threshold,
            track_quality_threshold=track_quality_threshold,
            track_combination_threshold=track_combination_threshold,
            iou_threshold=iou_threshold,
            track_validity_mode=track_validity_mode,
            require_individual_cuts_for_combination=require_individual_cuts_for_combination,
            match_valid_truth_only=match_valid_truth_only,
            hit_order_mode=_hit_order_mode_for_model(name),
            matching_mode=matching_mode,
            min_num_hits=min_num_hits,
            min_num_pixel_hits=min_num_pixel_hits,
        )
        tracks, parts = tracking_results[threshold_key][name]
        if truth_reference_mode == "pre_filter":
            truth_reference_results[threshold_key][name] = _build_prefilter_truth_reference_parts(
                data_config=tracking_configs[name]["data"],
                fname=fname,
                tracks=tracks,
                parts=parts,
                key_mode=key_mode,
            )
        else:
            truth_reference_results[threshold_key][name] = parts.copy()

print("loaded events")

# ----------------------------------------------------
# Efficiency and fake rate plots
# ----------------------------------------------------

single_threshold_combo = len(threshold_configs) == 1
truth_reference_suffix = "" if truth_reference_mode == "post_filter" else f"_{truth_reference_mode}_truth"

for (track_valid_threshold, track_quality_threshold, iou_threshold), threshold_results in tracking_results.items():
    effective_track_threshold = _resolve_effective_track_threshold(
        track_valid_threshold,
        track_quality_threshold,
        track_validity_mode,
        track_combination_threshold,
    )
    threshold_suffix = "" if single_threshold_combo else "_" + _threshold_tag(effective_track_threshold, iou_threshold)
    for qty in particle_targets:
        if qty not in {"pt", "eta", "phi"}:
            continue

        axlist = []
        fig, ax = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
        axlist.append(ax)

        names = []
        for name, (tracks, parts) in threshold_results.items():
            if name in tracking_fnames:
                names.append(name)
                """Efficiency plots"""
                truth_parts = truth_reference_results[(
                    track_valid_threshold,
                    track_quality_threshold,
                    iou_threshold,
                )][name]
                reconstructable = truth_parts["reconstructable"]
                # double majority
                bin_count, bin_error = binned(
                    truth_parts["eff_dm"][reconstructable],
                    truth_parts["particle_" + qty][reconstructable],
                    qty_bins[qty],
                    underflow=False,
                    overflow=False,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

                # perfect
                bin_count, bin_error = binned(
                    truth_parts["eff_perfect"][reconstructable],
                    truth_parts["particle_" + qty][reconstructable],
                    qty_bins[qty],
                    underflow=False,
                    overflow=False,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

        print("got this far")
        # axis ranges
        ax.set_ylim(0.8, 1.04)
        ax.set_ylabel("Efficiency")
        ax.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")

        for i in axlist:
            i.grid(zorder=0, alpha=0.25, linestyle="--")
            if qty == "pt":
                i.set_xlim([0, 10.5])
                i.set_xticks(np.arange(start=2, stop=11, step=2))
            if qty == "eta":
                i.set_xlim([-4.5, 4.5])
                i.set_xticks(np.arange(start=-4, stop=4.5, step=1))
            if qty == "phi":
                i.set_xlim([-3.5, 3.5])
                i.set_xticks(np.arange(start=-3, stop=3.5, step=1))

        # custom legends
        legend_elements_0 = [Line2D([0], [0], color=training_colours[training], label=training) for training in names]
        leg1_0 = ax.legend(handles=legend_elements_0, frameon=False, loc="upper left")
        ax.add_artist(leg1_0)

        legend_elements_eff = [Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")]
        leg2_0 = ax.legend(handles=legend_elements_eff, frameon=False, loc="upper right")
        ax.add_artist(leg2_0)

        fig.savefig(out_dir + f"{qty}_eff{threshold_suffix}{truth_reference_suffix}.png")
        plt.close(fig)


# ----------------------------------------------------
# Efficiency and fake rate numbers
# ----------------------------------------------------

summary_rows = []

for (track_valid_threshold, track_quality_threshold, iou_threshold), threshold_results in tracking_results.items():
    effective_track_threshold = _resolve_effective_track_threshold(
        track_valid_threshold,
        track_quality_threshold,
        track_validity_mode,
        track_combination_threshold,
    )
    print(
        f"Threshold combination: "
        f"track_valid_threshold={track_valid_threshold:.2f}, "
        f"track_quality_threshold={track_quality_threshold:.2f}, "
        f"effective_track_threshold={effective_track_threshold:.2f}, "
        f"iou_threshold={iou_threshold:.2f}, "
        f"require_individual_cuts_for_combination={require_individual_cuts_for_combination}"
    )
    for name, (tracks, parts) in threshold_results.items():
        print(name)
        event_ids = np.sort(parts["event_id"].unique())
        n_loaded_events = len(event_ids)
        truth_parts = truth_reference_results[(
            track_valid_threshold,
            track_quality_threshold,
            iou_threshold,
        )][name]
        reference_truth_parts = truth_parts[truth_parts.reconstructable]
        reference_particles_per_event = reference_truth_parts.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
        postfilter_valid_parts = parts[parts.valid]
        postfilter_valid_particles_per_event = postfilter_valid_parts.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
        tracks_per_event = tracks.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
        key_mode = "old" if name == "Paper" else None
        prefilter_hits_per_event = _prefilter_hits_per_event(tracking_configs[name]["data"], event_ids, key_mode=key_mode)
        postfilter_hits_per_event = _postfilter_hits_per_event(tracking_fnames[name], event_ids, key_mode=key_mode)
        tgts = reference_truth_parts
        trks = tracks[tracks.reconstructable]
        # compute high pt integrated metrics
        high_pt_parts = tgts[tgts.particle_pt > 1.0]
        high_pt_parts_900 = tgts[tgts.particle_pt > 0.9]
        high_pt_eff = high_pt_parts.eff_dm.mean()
        high_pt_eff_900 = high_pt_parts_900.eff_dm.mean()
        high_pt_tracks = trks[trks.matched_pt > 1.0]
        high_pt_tracks_900 = trks[trks.matched_pt > 0.9]
        high_pt_tracks_600 = trks[trks.matched_pt > 0.6]
        high_pt_fr = (~high_pt_tracks.eff_dm & ~trks.duplicate).mean()
        high_pt_fr_900 = (~high_pt_tracks_900.eff_dm & ~trks.duplicate).mean()
        high_pt_fr_600 = (~high_pt_tracks_600.eff_dm & ~trks.duplicate).mean()

        # compute the overall fake rate
        integrated_fr = (~trks.eff_dm & ~trks.duplicate).mean()
        integrated_eff = tgts.eff_dm.mean()

        # print summary
        print(f"N events: {n_loaded_events}")
        print(
            f"Reference truth particles ({truth_reference_mode}, all loaded events): total={len(reference_truth_parts)}, "
            f"mean/event={reference_particles_per_event.mean():.1f}, std/event={reference_particles_per_event.std():.1f}"
        )
        print(
            f"Post-filter valid truth particles (all loaded events): total={len(postfilter_valid_parts)}, "
            f"mean/event={postfilter_valid_particles_per_event.mean():.1f}, std/event={postfilter_valid_particles_per_event.std():.1f}"
        )
        print(
            f"Valid predicted tracks (all loaded events): total={len(tracks)}, "
            f"mean/event={tracks_per_event.mean():.1f}, std/event={tracks_per_event.std():.1f}"
        )
        print(
            f"Hits before hit filter: total={prefilter_hits_per_event.sum()}, "
            f"mean/event={prefilter_hits_per_event.mean():.1f}, std/event={prefilter_hits_per_event.std():.1f}"
        )
        print(
            f"Hits after hit filter: total={postfilter_hits_per_event.sum()}, "
            f"mean/event={postfilter_hits_per_event.mean():.1f}, std/event={postfilter_hits_per_event.std():.1f}"
        )
        print(f"DM Integrated efficiency: {integrated_eff:.1%}")
        print(f"DM Efficiency for pT > 1.0 GeV: {high_pt_eff:.1%}")
        print(f"DM Efficiency for pT > 0.9 GeV: {high_pt_eff_900:.1%}")
        print()
        print(f"DM Integrated fake rate: {integrated_fr:.1%}")
        print(f"DM Fake rate for pT > 1.0 GeV: {high_pt_fr:.1%}")
        print(f"DM Fake rate for pT > 0.9 GeV: {high_pt_fr_900:.1%}")
        print(f"DM Fake rate for pT > 0.6 GeV: {high_pt_fr_600:.1%}")
        print()
        print(f"Perfect integrated Efficiency: {tgts.eff_perfect.mean():.1%}")
        print(f"Perfect Efficiency for pT > 1.0 GeV: {high_pt_parts.eff_perfect.mean():.1%}")
        print(f"Perfect Efficiency for pT > 0.9 GeV: {high_pt_parts_900.eff_perfect.mean():.1%}")
        print()
        print(f"Duplicate rate: {tracks.duplicate.mean():.1%}")
        print("\n")

        summary_rows.append(
            {
                "model": name,
                "metric_mode": "dm",
                "truth_reference_mode": truth_reference_mode,
                "track_validity_mode": track_validity_mode,
                "match_valid_truth_only": match_valid_truth_only,
                "track_valid_threshold": track_valid_threshold,
                "track_quality_threshold": track_quality_threshold,
                "track_combination_threshold": track_combination_threshold,
                "effective_track_threshold": effective_track_threshold,
                "iou_threshold": iou_threshold,
                "require_individual_cuts_for_combination": require_individual_cuts_for_combination,
                "min_num_hits": int(_reconstructable_min_num_hits(tracking_configs[name]["data"])),
                "min_num_pixel_hits": int(_reconstructable_min_num_pixel_hits(tracking_configs[name]["data"])),
                "integrated_efficiency": float(integrated_eff),
                "integrated_fake_rate": float(integrated_fr),
                "n_events": int(n_loaded_events),
                "reference_truth_particles": int(len(reference_truth_parts)),
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_name = "track_eval_threshold_summary.csv"
if truth_reference_mode != "post_filter":
    summary_name = f"track_eval_threshold_summary_{truth_reference_mode}_truth.csv"
summary_path = pathlib.Path(out_dir) / summary_name
summary_df.to_csv(summary_path, index=False)
print(f"Wrote threshold summary to {summary_path}")
plt.rcParams["text.usetex"] = False
# disabled due to missing font in texlive on the Nikhef clusters
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True

# training_colours = {
#     "Paper": "tab:green",
#     "Tracking DQ": "tab:orange",  # |eta| < 4.0
#     "LCA": "tab:blue",
#     "LCA_reduce_window": "tab:green",
#     "LCA_no_wrap": "tab:orange",
# }

training_colours = {
    "Paper": "tab:green",
    "MA 600 Strip": "tab:orange",  # |eta| < 4.0
    "LCA 600 Strip": "tab:blue",
    "MA 900": "tab:red",
    "LCA 900": "tab:orange",
}

qty_bins = {
    "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
    # "eta": np.array([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]),
    "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
    "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
}

qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi"}
qty_units = {"pt": "[GeV]", "eta": "", "phi": ""}
out_dir = "/share/rcif2/pduckett/hepattn-dq-pr/src/hepattn/experiments/trackml/eval/"


def _event_id_to_event_index(event_id, key_mode=None):
    event_str = str(event_id)
    if event_str.startswith("event_"):
        event_idx = int(event_str.removeprefix("event_"))
    else:
        event_idx = int(event_str)
    if key_mode == "old":
        event_idx += 29800
    return event_idx


@lru_cache(maxsize=None)
def _event_file_map(test_dir, suffix):
    event_map = {}
    pattern = f"event*-{suffix}.parquet"
    for path in pathlib.Path(test_dir).glob(pattern):
        event_name = path.stem.removesuffix(f"-{suffix}")
        try:
            event_map[int(event_name.removeprefix("event"))] = path
        except ValueError:
            continue
    return event_map


def _read_first_available(group, dataset_names):
    for dataset_name in dataset_names:
        if dataset_name in group:
            return np.asarray(group[dataset_name][:][0])
    raise KeyError(f"None of {dataset_names} found under {group.name}")


def _input_types(data_config):
    inputs = data_config.get("inputs", {})
    if isinstance(inputs, dict):
        return list(inputs)
    return []


def _reconstructable_min_num_hits(data_config):
    if "reconstructable_min_num_hits" in data_config:
        return int(data_config["reconstructable_min_num_hits"])
    return int(data_config.get("particle_min_num_hits", 3))


def _postfilter_hits_per_event(fname, event_ids, key_mode=None):
    event_keys = [str(event_id) for event_id in event_ids]
    if not event_keys:
        return np.array([], dtype=np.int32)

    with h5py.File(fname, "r") as f:
        hit_counts = []
        for event_key in event_keys:
            if key_mode == "old":
                hit_counts.append(np.array(f[event_key]["hits"]["pids"]).shape[0])
            else:
                hit_counts.append(_read_first_available(f[event_key]["targets"], ["particle_key_valid", "particle_hit_valid"]).shape[-1])
    return np.asarray(hit_counts, dtype=np.int32)


def _prefilter_hits_per_event(data_config, event_ids, key_mode=None):
    if len(event_ids) == 0:
        return np.array([], dtype=np.int32)

    test_dir = pathlib.Path(data_config["test_dir"])
    hit_volume_ids = data_config.get("hit_volume_ids")
    event_file_map = _event_file_map(test_dir, "hits")

    hit_counts = []
    for event_id in event_ids:
        event_idx = _event_id_to_event_index(event_id, key_mode=key_mode)
        hits_path = event_file_map[event_idx]
        hits = pd.read_parquet(hits_path, columns=["volume_id"])
        if hit_volume_ids:
            hits = hits[hits["volume_id"].isin(hit_volume_ids)]
        hit_counts.append(len(hits))
    return np.asarray(hit_counts, dtype=np.int32)


def _load_prefilter_truth_particles_for_event(data_config, event_id, key_mode=None):
    test_dir = pathlib.Path(data_config["test_dir"])
    event_idx = _event_id_to_event_index(event_id, key_mode=key_mode)
    parts_path = _event_file_map(test_dir, "parts")[event_idx]
    hits_path = _event_file_map(test_dir, "hits")[event_idx]

    particles = pd.read_parquet(parts_path, columns=["particle_id", "px", "py", "pz"]).copy()
    hits = pd.read_parquet(hits_path, columns=["particle_id", "volume_id"])

    hit_volume_ids = data_config.get("hit_volume_ids")
    if hit_volume_ids:
        hits = hits[hits["volume_id"].isin(hit_volume_ids)]

    particles["particle_pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
    particles["particle_phi"] = np.arctan2(particles["py"], particles["px"])
    particle_p = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
    particles["particle_eta"] = np.arctanh(particles["pz"] / particle_p)

    particles = particles[particles["particle_pt"] > data_config["particle_min_pt"]]
    particles = particles[particles["particle_eta"].abs() < data_config["particle_max_abs_eta"]]

    counts = hits["particle_id"].value_counts()
    keep_particle_ids = counts[counts >= _reconstructable_min_num_hits(data_config)].index.to_numpy()
    particles = particles[particles["particle_id"].isin(keep_particle_ids)].copy()

    event_max_num_particles = data_config["event_max_num_particles"]
    if len(particles) > event_max_num_particles:
        if data_config.get("strict_max_objects", False):
            raise ValueError(f"Event {event_id} has {len(particles)} particles, but limit is {event_max_num_particles}")
        particles = particles.iloc[:event_max_num_particles].copy()

    particles["event_id"] = event_id
    particles["valid"] = True
    particles["reconstructable"] = True
    return particles[["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable"]]


def _attach_eval_particle_ids(fname, tracks, parts, key_mode=None):
    tracks = tracks.copy()
    parts = parts.copy()
    if len(parts) == 0:
        tracks["matched_particle_id"] = np.array([], dtype=np.int64)
        parts["particle_id"] = np.array([], dtype=np.int64)
        return tracks, parts

    parts_particle_ids = np.full(len(parts), -1, dtype=np.int64)
    tracks_particle_ids = np.full(len(tracks), -1, dtype=np.int64)
    particle_ids_by_event = {}

    with h5py.File(fname, "r") as f:
        for event_id, event_parts in parts.groupby("event_id", sort=False):
            event_key = str(event_id)
            if key_mode == "old":
                particle_ids = np.array(f[event_key]["parts"]["pids"])
            else:
                particle_ids = np.array(f[event_key]["targets"]["particle_id"][:][0])
            if len(event_parts) > len(particle_ids):
                raise ValueError(f"Event {event_key} has more parts rows than particle IDs in the eval file.")
            parts_particle_ids[event_parts.index.to_numpy()] = particle_ids[: len(event_parts)]
            particle_ids_by_event[event_key] = parts_particle_ids[event_parts.index.to_numpy()]

    parts["particle_id"] = parts_particle_ids

    for event_id, event_tracks in tracks.groupby("event_id", sort=False):
        matched_pid = event_tracks["matched_pid"].to_numpy(dtype=np.int64)
        event_particle_ids = particle_ids_by_event[str(event_id)]
        valid_match = (matched_pid >= 0) & (matched_pid < len(event_particle_ids))
        matched_particle_ids = np.full(len(event_tracks), -1, dtype=np.int64)
        matched_particle_ids[valid_match] = event_particle_ids[matched_pid[valid_match]]
        tracks_particle_ids[event_tracks.index.to_numpy()] = matched_particle_ids

    tracks["matched_particle_id"] = tracks_particle_ids
    return tracks, parts


def _build_prefilter_truth_reference_parts(data_config, fname, tracks, parts, key_mode=None):
    tracks_with_ids, _parts_with_ids = _attach_eval_particle_ids(fname, tracks, parts, key_mode=key_mode)
    event_ids = np.sort(parts["event_id"].unique())
    truth_parts = []

    for event_id in event_ids:
        event_truth = _load_prefilter_truth_particles_for_event(data_config, event_id, key_mode=key_mode)
        event_tracks = tracks_with_ids[tracks_with_ids["event_id"] == event_id]
        dm_particle_ids = set(event_tracks.loc[event_tracks["eff_dm"], "matched_particle_id"].astype(np.int64).tolist())
        perfect_particle_ids = set(event_tracks.loc[event_tracks["eff_perfect"], "matched_particle_id"].astype(np.int64).tolist())
        dm_particle_ids.discard(-1)
        perfect_particle_ids.discard(-1)
        event_truth["eff_dm"] = event_truth["particle_id"].isin(dm_particle_ids)
        event_truth["eff_perfect"] = event_truth["particle_id"].isin(perfect_particle_ids)
        truth_parts.append(event_truth)

    if not truth_parts:
        return pd.DataFrame(columns=["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable", "eff_dm", "eff_perfect"])
    return pd.concat(truth_parts, ignore_index=True)


def _threshold_tag(track_valid_threshold, iou_threshold):
    track_str = f"{track_valid_threshold:.2f}".replace(".", "p")
    iou_str = f"{iou_threshold:.2f}".replace(".", "p")
    return f"tv{track_str}_iou{iou_str}"


def _as_threshold_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    return [float(value)]


def _resolve_effective_track_threshold(
    track_valid_threshold,
    track_quality_threshold,
    track_validity_mode,
    track_combination_threshold=None,
):
    if track_combination_threshold is not None:
        return float(track_combination_threshold)
    if track_validity_mode == "track_valid_prob_plus_track_quality_score":
        return float(track_valid_threshold + track_quality_threshold)
    if track_validity_mode == "track_valid_prob_times_track_quality_score":
        return float(track_valid_threshold * track_quality_threshold)
    return float(track_valid_threshold)


def _build_threshold_configs(
    track_valid_thresholds,
    track_quality_thresholds,
    iou_thresholds,
    track_validity_mode,
    track_combination_threshold=None,
    require_individual_cuts_for_combination=False,
):
    configs = []
    seen_effective = set()

    for track_valid_threshold in _as_threshold_list(track_valid_thresholds):
        for track_quality_threshold in _as_threshold_list(track_quality_thresholds):
            effective_track_threshold = _resolve_effective_track_threshold(
                track_valid_threshold,
                track_quality_threshold,
                track_validity_mode,
                track_combination_threshold,
            )
            for iou_threshold in _as_threshold_list(iou_thresholds):
                if (
                    track_validity_mode in {"track_valid_prob_plus_track_quality_score", "track_valid_prob_times_track_quality_score"}
                    and not require_individual_cuts_for_combination
                ):
                    dedupe_key = (round(effective_track_threshold, 12), round(iou_threshold, 12))
                    if dedupe_key in seen_effective:
                        continue
                    seen_effective.add(dedupe_key)

                configs.append(
                    {
                        "track_valid_threshold": track_valid_threshold,
                        "track_quality_threshold": track_quality_threshold,
                        "effective_track_threshold": effective_track_threshold,
                        "iou_threshold": iou_threshold,
                    }
                )

    return configs

# ----------------------------------------------------
# Read configuration file information
# ----------------------------------------------------

CONFIG_DIR = pathlib.Path("/share/rcif2/pduckett/hepattn-dq-pr/src/hepattn/experiments/trackml/configs")

tracking_fnames = {
    "MA 600 Strip": "/share/rcif2/pduckett/hepattn-dq-pr/logs/TRK-v8-STRIP-3100-0p3-th0p42-epochs30_20260415-T094006/ckpts/epoch=028-val_loss=0.29386_test_eval.h5",
    "LCA 600 Strip": "/share/rcif2/pduckett/hepattn-dq-pr/logs/TRK-v8-strip-lca-2900-th0p4-0p42-epochs30_20260421-T150024/ckpts/epoch=026-val_loss=0.39419_test_eval.h5",
}

tracking_config_fname = {
    "MA 600 Strip": str(CONFIG_DIR / "tracking-strip.yaml"),
    "LCA 600 Strip": str(CONFIG_DIR / "tracking-strip-lca.yaml"),
}

tracking_params = ["particle_min_pt", "particle_max_abs_eta"]
tracking_configs = {}
for name in tracking_config_fname:
    with pathlib.Path(tracking_config_fname[name]).open() as f:
        fconfig = yaml.safe_load(f)
        print("name: " + fconfig["name"])
        for i in tracking_params:
            print("> " + i + "\t: ", fconfig["data"][i])
    tracking_configs[name] = fconfig

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

tracking_results = {}
num_events = 100
# track_valid_thresholds = [0.6, 0.7, 0.8, 0.9]
# iou_thresholds = [0.6, 0.7, 0.8, 0.9]
track_valid_thresholds =  [0.5, 0.6, 0.7, 0.8, 0.9]
iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
particle_targets = ["pt", "eta", "phi"]
track_quality_thresholds = [0.0]
track_combination_threshold = None
track_validity_mode = "saved_track_valid"
require_individual_cuts_for_combination = False
match_valid_truth_only = False
matching_mode = "auto"
truth_reference_mode = "pre_filter"
# "post_filter": current behavior, denominator uses truth particles after hit filtering
# "pre_filter": denominator uses reconstructable truth particles before hit filtering
# Fake-rate metrics remain track-denominated.

# Eval files written by PredictionWriter already store the targets in the model-aligned
# order when a sorter is used, so the saved ordering is the correct default.
hit_order_by_name = {
    "Paper": "as_saved",
    "Tracking DQ": "as_saved",
    "LCA": "as_saved",
}


def _hit_order_mode_for_model(name):
    if name in hit_order_by_name:
        return hit_order_by_name[name]
    return "as_saved"

threshold_configs = _build_threshold_configs(
    track_valid_thresholds=track_valid_thresholds,
    track_quality_thresholds=track_quality_thresholds,
    iou_thresholds=iou_thresholds,
    track_validity_mode=track_validity_mode,
    track_combination_threshold=track_combination_threshold,
    require_individual_cuts_for_combination=require_individual_cuts_for_combination,
)

truth_reference_results = {}
index_list = None  # ["event_0", ..., "event_99"] or ["29800", ..., "29899"]
for threshold_config in threshold_configs:
    track_valid_threshold = threshold_config["track_valid_threshold"]
    track_quality_threshold = threshold_config["track_quality_threshold"]
    effective_track_threshold = threshold_config["effective_track_threshold"]
    iou_threshold = threshold_config["iou_threshold"]
    threshold_key = (track_valid_threshold, track_quality_threshold, iou_threshold)
    tracking_results[threshold_key] = {}
    truth_reference_results[threshold_key] = {}
    print(
        f"Evaluating threshold combination "
        f"track_valid_threshold={track_valid_threshold:.2f}, "
        f"track_quality_threshold={track_quality_threshold:.2f}, "
        f"effective_track_threshold={effective_track_threshold:.2f}, "
        f"iou_threshold={iou_threshold:.2f}, "
        f"require_individual_cuts_for_combination={require_individual_cuts_for_combination}, "
        f"matching_mode={matching_mode}"
    )
    for name, fname in tracking_fnames.items():
        eta_cut = tracking_configs[name]["data"]["particle_max_abs_eta"]
        pt_cut = tracking_configs[name]["data"]["particle_min_pt"]
        min_num_hits = _reconstructable_min_num_hits(tracking_configs[name]["data"])
        key_mode = "old" if name == "Paper" else None  # None or "old"
        print(f"Loading {name} model with PT > {pt_cut}, |eta| < {eta_cut}, min_num_hits >= {min_num_hits}", f"")
        tracking_results[threshold_key][name] = load_events(
            fname=fname,
            eta_cut=eta_cut,
            index_list=index_list,
            pt_cut=pt_cut,
            randomize=num_events,
            particle_targets=particle_targets,
            regression=False,
            key_mode=key_mode,
            track_valid_threshold=track_valid_threshold,
            track_quality_threshold=track_quality_threshold,
            track_combination_threshold=track_combination_threshold,
            iou_threshold=iou_threshold,
            track_validity_mode=track_validity_mode,
            require_individual_cuts_for_combination=require_individual_cuts_for_combination,
            match_valid_truth_only=match_valid_truth_only,
            hit_order_mode=_hit_order_mode_for_model(name),
            matching_mode=matching_mode,
            min_num_hits=min_num_hits,
        )
        tracks, parts = tracking_results[threshold_key][name]
        if truth_reference_mode == "pre_filter":
            truth_reference_results[threshold_key][name] = _build_prefilter_truth_reference_parts(
                data_config=tracking_configs[name]["data"],
                fname=fname,
                tracks=tracks,
                parts=parts,
                key_mode=key_mode,
            )
        else:
            truth_reference_results[threshold_key][name] = parts.copy()

print("loaded events")

# ----------------------------------------------------
# Efficiency and fake rate plots
# ----------------------------------------------------

single_threshold_combo = len(threshold_configs) == 1
truth_reference_suffix = "" if truth_reference_mode == "post_filter" else f"_{truth_reference_mode}_truth"

for (track_valid_threshold, track_quality_threshold, iou_threshold), threshold_results in tracking_results.items():
    effective_track_threshold = _resolve_effective_track_threshold(
        track_valid_threshold,
        track_quality_threshold,
        track_validity_mode,
        track_combination_threshold,
    )
    threshold_suffix = "" if single_threshold_combo else "_" + _threshold_tag(effective_track_threshold, iou_threshold)
    for qty in particle_targets:
        if qty not in {"pt", "eta", "phi"}:
            continue

        axlist = []
        fig, ax = plt.subplots(ncols=1, figsize=(6, 4), constrained_layout=True)
        axlist.append(ax)

        names = []
        for name, (tracks, parts) in threshold_results.items():
            if name in tracking_fnames:
                names.append(name)
                """Efficiency plots"""
                truth_parts = truth_reference_results[(
                    track_valid_threshold,
                    track_quality_threshold,
                    iou_threshold,
                )][name]
                reconstructable = truth_parts["reconstructable"]
                # double majority
                bin_count, bin_error = binned(
                    truth_parts["eff_dm"][reconstructable],
                    truth_parts["particle_" + qty][reconstructable],
                    qty_bins[qty],
                    underflow=False,
                    overflow=False,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

                # perfect
                bin_count, bin_error = binned(
                    truth_parts["eff_perfect"][reconstructable],
                    truth_parts["particle_" + qty][reconstructable],
                    qty_bins[qty],
                    underflow=False,
                    overflow=False,
                    binomial=False,
                )
                profile_plot(bin_count, bin_error, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

        print("got this far")
        # axis ranges
        ax.set_ylim(0.8, 1.04)
        ax.set_ylabel("Efficiency")
        ax.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")

        for i in axlist:
            i.grid(zorder=0, alpha=0.25, linestyle="--")
            if qty == "pt":
                i.set_xlim([0, 10.5])
                i.set_xticks(np.arange(start=2, stop=11, step=2))
            if qty == "eta":
                i.set_xlim([-4.5, 4.5])
                i.set_xticks(np.arange(start=-4, stop=4.5, step=1))
            if qty == "phi":
                i.set_xlim([-3.5, 3.5])
                i.set_xticks(np.arange(start=-3, stop=3.5, step=1))

        # custom legends
        legend_elements_0 = [Line2D([0], [0], color=training_colours[training], label=training) for training in names]
        leg1_0 = ax.legend(handles=legend_elements_0, frameon=False, loc="upper left")
        ax.add_artist(leg1_0)

        legend_elements_eff = [Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")]
        leg2_0 = ax.legend(handles=legend_elements_eff, frameon=False, loc="upper right")
        ax.add_artist(leg2_0)

        fig.savefig(out_dir + f"{qty}_eff{threshold_suffix}{truth_reference_suffix}.png")
        plt.close(fig)


# ----------------------------------------------------
# Efficiency and fake rate numbers
# ----------------------------------------------------

summary_rows = []

for (track_valid_threshold, track_quality_threshold, iou_threshold), threshold_results in tracking_results.items():
    effective_track_threshold = _resolve_effective_track_threshold(
        track_valid_threshold,
        track_quality_threshold,
        track_validity_mode,
        track_combination_threshold,
    )
    print(
        f"Threshold combination: "
        f"track_valid_threshold={track_valid_threshold:.2f}, "
        f"track_quality_threshold={track_quality_threshold:.2f}, "
        f"effective_track_threshold={effective_track_threshold:.2f}, "
        f"iou_threshold={iou_threshold:.2f}, "
        f"require_individual_cuts_for_combination={require_individual_cuts_for_combination}"
    )
    for name, (tracks, parts) in threshold_results.items():
        print(name)
        event_ids = np.sort(parts["event_id"].unique())
        n_loaded_events = len(event_ids)
        truth_parts = truth_reference_results[(
            track_valid_threshold,
            track_quality_threshold,
            iou_threshold,
        )][name]
        reference_truth_parts = truth_parts[truth_parts.reconstructable]
        reference_particles_per_event = reference_truth_parts.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
        postfilter_valid_parts = parts[parts.valid]
        postfilter_valid_particles_per_event = postfilter_valid_parts.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
        tracks_per_event = tracks.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
        key_mode = "old" if name == "Paper" else None
        prefilter_hits_per_event = _prefilter_hits_per_event(tracking_configs[name]["data"], event_ids, key_mode=key_mode)
        postfilter_hits_per_event = _postfilter_hits_per_event(tracking_fnames[name], event_ids, key_mode=key_mode)
        tgts = reference_truth_parts
        trks = tracks[tracks.reconstructable]
        # compute high pt integrated metrics
        high_pt_parts = tgts[tgts.particle_pt > 1.0]
        high_pt_parts_900 = tgts[tgts.particle_pt > 0.9]
        high_pt_eff = high_pt_parts.eff_dm.mean()
        high_pt_eff_900 = high_pt_parts_900.eff_dm.mean()
        high_pt_tracks = trks[trks.matched_pt > 1.0]
        high_pt_tracks_900 = trks[trks.matched_pt > 0.9]
        high_pt_tracks_600 = trks[trks.matched_pt > 0.6]
        high_pt_fr = (~high_pt_tracks.eff_dm & ~trks.duplicate).mean()
        high_pt_fr_900 = (~high_pt_tracks_900.eff_dm & ~trks.duplicate).mean()
        high_pt_fr_600 = (~high_pt_tracks_600.eff_dm & ~trks.duplicate).mean()

        # compute the overall fake rate
        integrated_fr = (~trks.eff_dm & ~trks.duplicate).mean()
        integrated_eff = tgts.eff_dm.mean()

        # print summary
        print(f"N events: {n_loaded_events}")
        print(
            f"Reference truth particles ({truth_reference_mode}, all loaded events): total={len(reference_truth_parts)}, "
            f"mean/event={reference_particles_per_event.mean():.1f}, std/event={reference_particles_per_event.std():.1f}"
        )
        print(
            f"Post-filter valid truth particles (all loaded events): total={len(postfilter_valid_parts)}, "
            f"mean/event={postfilter_valid_particles_per_event.mean():.1f}, std/event={postfilter_valid_particles_per_event.std():.1f}"
        )
        print(
            f"Valid predicted tracks (all loaded events): total={len(tracks)}, "
            f"mean/event={tracks_per_event.mean():.1f}, std/event={tracks_per_event.std():.1f}"
        )
        print(
            f"Hits before hit filter: total={prefilter_hits_per_event.sum()}, "
            f"mean/event={prefilter_hits_per_event.mean():.1f}, std/event={prefilter_hits_per_event.std():.1f}"
        )
        print(
            f"Hits after hit filter: total={postfilter_hits_per_event.sum()}, "
            f"mean/event={postfilter_hits_per_event.mean():.1f}, std/event={postfilter_hits_per_event.std():.1f}"
        )
        print(f"DM Integrated efficiency: {integrated_eff:.1%}")
        print(f"DM Efficiency for pT > 1.0 GeV: {high_pt_eff:.1%}")
        print(f"DM Efficiency for pT > 0.9 GeV: {high_pt_eff_900:.1%}")
        print()
        print(f"DM Integrated fake rate: {integrated_fr:.1%}")
        print(f"DM Fake rate for pT > 1.0 GeV: {high_pt_fr:.1%}")
        print(f"DM Fake rate for pT > 0.9 GeV: {high_pt_fr_900:.1%}")
        print(f"DM Fake rate for pT > 0.6 GeV: {high_pt_fr_600:.1%}")
        print()
        print(f"Perfect integrated Efficiency: {tgts.eff_perfect.mean():.1%}")
        print(f"Perfect Efficiency for pT > 1.0 GeV: {high_pt_parts.eff_perfect.mean():.1%}")
        print(f"Perfect Efficiency for pT > 0.9 GeV: {high_pt_parts_900.eff_perfect.mean():.1%}")
        print()
        print(f"Duplicate rate: {tracks.duplicate.mean():.1%}")
        print("\n")

        summary_rows.append(
            {
                "model": name,
                "metric_mode": "dm",
                "truth_reference_mode": truth_reference_mode,
                "track_validity_mode": track_validity_mode,
                "match_valid_truth_only": match_valid_truth_only,
                "track_valid_threshold": track_valid_threshold,
                "track_quality_threshold": track_quality_threshold,
                "track_combination_threshold": track_combination_threshold,
                "effective_track_threshold": effective_track_threshold,
                "iou_threshold": iou_threshold,
                "require_individual_cuts_for_combination": require_individual_cuts_for_combination,
                "integrated_efficiency": float(integrated_eff),
                "integrated_fake_rate": float(integrated_fr),
                "n_events": int(n_loaded_events),
                "reference_truth_particles": int(len(reference_truth_parts)),
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_name = "track_eval_threshold_summary.csv"
if truth_reference_mode != "post_filter":
    summary_name = f"track_eval_threshold_summary_{truth_reference_mode}_truth.csv"
summary_path = pathlib.Path(out_dir) / summary_name
summary_df.to_csv(summary_path, index=False)
print(f"Wrote threshold summary to {summary_path}")
