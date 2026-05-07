import math
import pathlib
from functools import lru_cache

import h5py
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import binned, profile_plot
from track_eval_utils import load_events

# Disabled text.usetex: missing texlive font on Nikhef clusters.
plt.rcParams["figure.dpi"] = 400
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["figure.constrained_layout.use"] = True


# ── Event / file utilities ────────────────────────────────────────────────────


def _event_id_to_event_index(event_id, key_mode=None):
    """Convert an event identifier string/int to a numeric index."""
    event_str = str(event_id)
    if event_str.startswith("event_"):
        event_idx = int(event_str.removeprefix("event_"))
    else:
        event_idx = int(event_str)
    # "old" format files are offset by 29800
    if key_mode == "old":
        event_idx += 29800
    return event_idx


@lru_cache(maxsize=None)
def _event_file_map(test_dir, suffix):
    """Build a {event_index: path} map for parquet files matching event*-<suffix>.parquet."""
    event_map = {}
    for path in pathlib.Path(test_dir).glob(f"event*-{suffix}.parquet"):
        event_name = path.stem.removesuffix(f"-{suffix}")
        try:
            event_map[int(event_name.removeprefix("event"))] = path
        except ValueError:
            continue
    return event_map


def _postfilter_hits_per_event(fname, event_ids, key_mode=None):
    """Count hits per event after model-side hit filtering, read from the evaluation HDF5."""
    event_keys = [str(eid) for eid in event_ids]
    if not event_keys:
        return np.array([], dtype=np.int32)
    with h5py.File(fname, "r") as f:
        hit_counts = []
        for event_key in event_keys:
            if key_mode == "old":
                hit_counts.append(np.array(f[event_key]["hits"]["pids"]).shape[0])
            else:
                hit_counts.append(np.array(f[event_key]["targets"]["particle_hit_valid"][:][0]).shape[-1])
    return np.asarray(hit_counts, dtype=np.int32)


def _prefilter_hits_per_event(data_config, event_ids, key_mode=None):
    """Count hits per event before hit filtering, read from raw parquet files."""
    if len(event_ids) == 0:
        return np.array([], dtype=np.int32)
    test_dir = pathlib.Path(data_config["test_dir"])
    hit_volume_ids = data_config.get("hit_volume_ids")
    event_file_map = _event_file_map(test_dir, "hits")
    hit_counts = []
    for event_id in event_ids:
        event_idx = _event_id_to_event_index(event_id, key_mode=key_mode)
        hits = pd.read_parquet(event_file_map[event_idx], columns=["volume_id"])
        if hit_volume_ids:
            hits = hits[hits["volume_id"].isin(hit_volume_ids)]
        hit_counts.append(len(hits))
    return np.asarray(hit_counts, dtype=np.int32)


def _load_prefilter_truth_particles_for_event(data_config, event_id, key_mode=None):
    """Load ground-truth particles for a single event, applying pre-filter acceptance cuts."""
    test_dir = pathlib.Path(data_config["test_dir"])
    event_idx = _event_id_to_event_index(event_id, key_mode=key_mode)
    particles = pd.read_parquet(_event_file_map(test_dir, "parts")[event_idx], columns=["particle_id", "px", "py", "pz"]).copy()
    hits = pd.read_parquet(_event_file_map(test_dir, "hits")[event_idx], columns=["particle_id", "volume_id"])

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
    keep_ids = counts[counts >= data_config["particle_min_num_hits"]].index.to_numpy()
    particles = particles[particles["particle_id"].isin(keep_ids)].copy()

    max_n = data_config["event_max_num_particles"]
    if len(particles) > max_n:
        if data_config.get("strict_max_objects", False):
            raise ValueError(f"Event {event_id} has {len(particles)} particles, but limit is {max_n}")
        particles = particles.iloc[:max_n].copy()

    particles["event_id"] = event_id
    particles["valid"] = True
    particles["reconstructable"] = True
    return particles[["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable"]]


def _attach_eval_particle_ids(fname, tracks, parts, key_mode=None):
    """Resolve eval-file integer slot indices to physical particle IDs for tracks and parts."""
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
    """Build a truth-particle reference DataFrame using pre-filter acceptance as the denominator.

    Pre-filter truth gives a fairer efficiency denominator: it includes all reconstructable
    particles that the model *could* have seen, not just those that survived hit filtering.
    """
    tracks_with_ids, _ = _attach_eval_particle_ids(fname, tracks, parts, key_mode=key_mode)
    truth_parts = []

    for event_id in np.sort(parts["event_id"].unique()):
        event_truth = _load_prefilter_truth_particles_for_event(data_config, event_id, key_mode=key_mode)
        event_tracks = tracks_with_ids[tracks_with_ids["event_id"] == event_id]

        dm_ids = set(event_tracks.loc[event_tracks["eff_dm"], "matched_particle_id"].astype(np.int64).tolist())
        perfect_ids = set(event_tracks.loc[event_tracks["eff_perfect"], "matched_particle_id"].astype(np.int64).tolist())
        dm_ids.discard(-1)
        perfect_ids.discard(-1)

        event_truth["eff_dm"] = event_truth["particle_id"].isin(dm_ids)
        event_truth["eff_perfect"] = event_truth["particle_id"].isin(perfect_ids)
        truth_parts.append(event_truth)

    if not truth_parts:
        return pd.DataFrame(columns=["event_id", "particle_id", "particle_pt", "particle_eta", "particle_phi", "valid", "reconstructable", "eff_dm", "eff_perfect"])
    return pd.concat(truth_parts, ignore_index=True)


# ── Threshold utilities ───────────────────────────────────────────────────────


def _threshold_tag(track_valid_threshold, iou_threshold):
    """Short string identifier for a threshold pair, used in output filenames."""
    tv = f"{track_valid_threshold:.2f}".replace(".", "p")
    iou = f"{iou_threshold:.2f}".replace(".", "p")
    return f"tv{tv}_iou{iou}"


def _as_threshold_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    return [float(value)]


def _build_threshold_configs(track_valid_thresholds, iou_thresholds):
    """Return all (track_valid_threshold, iou_threshold) combinations as a list of dicts."""
    return [
        {"track_valid_threshold": tv, "iou_threshold": iou}
        for tv in _as_threshold_list(track_valid_thresholds)
        for iou in _as_threshold_list(iou_thresholds)
    ]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_all_results(
    tracking_fnames,
    tracking_configs,
    default_track_valid_thresholds,
    default_iou_thresholds,
    tracking_threshold_overrides,
    hit_order_by_name,
    num_events,
    random_seed,
    particle_targets,
    truth_reference_mode,
    match_min_hits,
    match_min_pt,
    event_cache_dir,
    reuse_saved_event_cache,
    write_event_cache,
    n_workers=1,
):
    """Load tracking results for every model × threshold combination.

    Each model uses its own threshold sweep. Per-model overrides for
    track_valid_thresholds and iou_thresholds can be provided via
    tracking_threshold_overrides; models not listed fall back to the
    default_track_valid_thresholds / default_iou_thresholds.

    Returns two dicts keyed by (track_valid_threshold, iou_threshold):
      tracking_results[key][name]       -> (tracks_df, parts_df)
      truth_reference_results[key][name] -> truth_parts_df
    """
    # Build per-model threshold configs, falling back to global defaults.
    model_threshold_configs = {}
    for name in tracking_fnames:
        overrides = tracking_threshold_overrides.get(name, {})
        tvs = overrides.get("track_valid_thresholds", default_track_valid_thresholds)
        ious = overrides.get("iou_thresholds", default_iou_thresholds)
        model_threshold_configs[name] = _build_threshold_configs(tvs, ious)

    # Initialise result dicts for every unique threshold combination across all models.
    all_threshold_keys = {
        (cfg["track_valid_threshold"], cfg["iou_threshold"])
        for cfgs in model_threshold_configs.values()
        for cfg in cfgs
    }
    tracking_results = {key: {} for key in all_threshold_keys}
    truth_reference_results = {key: {} for key in all_threshold_keys}

    for name, fname in tracking_fnames.items():
        data_cfg = tracking_configs[name]["data"]
        eta_cut = data_cfg["particle_max_abs_eta"]
        pt_cut = data_cfg["particle_min_pt"]
        # "old" format applies to the Paper baseline checkpoint
        key_mode = "old" if name == "Paper" else None

        for cfg in model_threshold_configs[name]:
            tv, iou = cfg["track_valid_threshold"], cfg["iou_threshold"]
            threshold_key = (tv, iou)
            print(f"  {name}: tv={tv:.2f}, iou={iou:.2f}, pT > {pt_cut}, |η| < {eta_cut}")

            cache_key = f"{name.replace(' ', '_').lower()}_{_threshold_tag(tv, iou)}"
            tracking_results[threshold_key][name] = load_events(
                fname=fname,
                pt_cut=pt_cut,
                eta_cut=eta_cut,
                index_list=None,
                randomize=num_events,
                random_seed=random_seed,
                particle_targets=particle_targets,
                regression=False,
                key_mode=key_mode,
                track_valid_threshold=tv,
                iou_threshold=iou,
                match_min_hits=match_min_hits,
                match_min_pt=match_min_pt,
                hit_order_mode=hit_order_by_name.get(name, "auto"),
                cache_dir=event_cache_dir,
                cache_key=cache_key,
                reuse_saved=reuse_saved_event_cache,
                write_cache=write_event_cache,
                n_workers=n_workers,
            )
            tracks, parts = tracking_results[threshold_key][name]

            if truth_reference_mode == "pre_filter":
                # Denominator uses truth particles from raw parquet, before hit filtering.
                truth_reference_results[threshold_key][name] = _build_prefilter_truth_reference_parts(
                    data_config=data_cfg,
                    fname=fname,
                    tracks=tracks,
                    parts=parts,
                    key_mode=key_mode,
                )
            else:
                # Denominator uses truth particles after hit filtering (post_filter).
                truth_reference_results[threshold_key][name] = parts.copy()

    print("Loaded all events.")
    return tracking_results, truth_reference_results


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_efficiency(
    tracking_results,
    truth_reference_results,
    tracking_fnames,
    particle_targets,
    qty_bins,
    qty_symbols,
    qty_units,
    training_colours,
    out_dir,
    truth_reference_mode="post_filter",
):
    """Plot DM and perfect-match efficiency vs kinematic quantities for all threshold combinations."""
    single_combo = len(tracking_results) == 1
    truth_suffix = "" if truth_reference_mode == "post_filter" else f"_{truth_reference_mode}_truth"

    for (tv, iou), threshold_results in tracking_results.items():
        threshold_suffix = "" if single_combo else "_" + _threshold_tag(tv, iou)

        for qty in [q for q in particle_targets if q in {"pt", "eta", "phi"}]:
            fig, ax = plt.subplots(figsize=(6, 4))
            names = []

            for name, (_tracks, _parts) in threshold_results.items():
                if name not in tracking_fnames:
                    continue
                names.append(name)

                truth_parts = truth_reference_results[(tv, iou)][name]
                reconstructable = truth_parts["reconstructable"]
                x = truth_parts["particle_" + qty][reconstructable]

                # Double majority efficiency (solid)
                bc, be = binned(truth_parts["eff_dm"][reconstructable], x, qty_bins[qty], underflow=False, overflow=False, binomial=False)
                profile_plot(bc, be, qty_bins[qty], axes=ax, colour=training_colours[name], ls="solid")

                # Perfect-match efficiency (dotted)
                bc, be = binned(truth_parts["eff_perfect"][reconstructable], x, qty_bins[qty], underflow=False, overflow=False, binomial=False)
                profile_plot(bc, be, qty_bins[qty], axes=ax, colour=training_colours[name], ls="dotted")

            ax.set_ylim(0.8, 1.04)
            ax.set_ylabel("Efficiency")
            ax.set_xlabel(rf"Particle ${qty_symbols[qty]}^\mathrm{{True}}$ {qty_units[qty]}")
            ax.grid(zorder=0, alpha=0.25, linestyle="--")

            if qty == "pt":
                ax.set_xlim([0, 10.5])
                ax.set_xticks(np.arange(start=2, stop=11, step=2))
            elif qty == "eta":
                ax.set_xlim([-4.5, 4.5])
                ax.set_xticks(np.arange(start=-4, stop=4.5, step=1))
            elif qty == "phi":
                ax.set_xlim([-3.5, 3.5])
                ax.set_xticks(np.arange(start=-3, stop=3.5, step=1))

            # Two independent legends: one for models (colour), one for efficiency definition (line style)
            model_leg = ax.legend(
                handles=[Line2D([0], [0], color=training_colours[n], label=n) for n in names],
                frameon=False, loc="upper left",
            )
            ax.add_artist(model_leg)
            ax.legend(
                handles=[Line2D([0], [0], color="black", label="DM"), Line2D([0], [0], color="black", ls="dotted", label="Perfect")],
                frameon=False, loc="upper right",
            )

            fig.savefig(out_dir + f"{qty}_eff{threshold_suffix}{truth_suffix}.png")
            plt.close(fig)


# ── Summary metrics ───────────────────────────────────────────────────────────


def print_and_save_summary(
    tracking_results,
    truth_reference_results,
    tracking_configs,
    tracking_fnames,
    truth_reference_mode,
    out_dir,
):
    """Print per-model efficiency / fake-rate metrics and save a summary CSV."""
    summary_rows = []

    for (tv, iou), threshold_results in tracking_results.items():
        print(f"\nThreshold: track_valid={tv:.2f}, iou={iou:.2f}")

        for name, (tracks, parts) in threshold_results.items():
            key_mode = "old" if name == "Paper" else None
            event_ids = np.sort(parts["event_id"].unique())
            n_events = len(event_ids)

            truth_parts = truth_reference_results[(tv, iou)][name]
            ref_parts = truth_parts[truth_parts.reconstructable]

            ref_per_event = ref_parts.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
            valid_parts_per_event = parts[parts.valid].groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
            tracks_per_event = tracks.groupby("event_id").size().reindex(event_ids, fill_value=0).to_numpy()
            pre_hits = _prefilter_hits_per_event(tracking_configs[name]["data"], event_ids, key_mode=key_mode)
            post_hits = _postfilter_hits_per_event(tracking_fnames[name], event_ids, key_mode=key_mode)

            trks = tracks[tracks.reconstructable]

            # DM efficiency at pT thresholds
            eff = {
                "all": ref_parts.eff_dm.mean(),
                "1p0": ref_parts[ref_parts.particle_pt > 1.0].eff_dm.mean(),
                "0p9": ref_parts[ref_parts.particle_pt > 0.9].eff_dm.mean(),
                "0p6": ref_parts[ref_parts.particle_pt > 0.6].eff_dm.mean(),
            }
            # Perfect efficiency at pT thresholds
            perf_eff = {
                "all": ref_parts.eff_perfect.mean(),
                "1p0": ref_parts[ref_parts.particle_pt > 1.0].eff_perfect.mean(),
                "0p9": ref_parts[ref_parts.particle_pt > 0.9].eff_perfect.mean(),
                "0p6": ref_parts[ref_parts.particle_pt > 0.6].eff_perfect.mean(),
            }
            # Fake rate: non-DM, non-duplicate tracks at pT thresholds
            # Denominator is all reconstructable tracks; numerator excludes matched and duplicate.
            fr = {
                "all": (~trks.eff_dm & ~trks.duplicate).mean(),
                "1p0": (~trks[trks.matched_pt > 1.0].eff_dm & ~trks.duplicate).mean(),
                "0p9": (~trks[trks.matched_pt > 0.9].eff_dm & ~trks.duplicate).mean(),
                "0p6": (~trks[trks.matched_pt > 0.6].eff_dm & ~trks.duplicate).mean(),
            }

            print(f"\n  {name}")
            print(f"  N events: {n_events}")
            print(f"  Reference truth particles ({truth_reference_mode}): total={len(ref_parts)}, mean/event={ref_per_event.mean():.3f} ± {ref_per_event.std():.3f}")
            print(f"  Post-filter valid truth particles:  total={len(parts[parts.valid])}, mean/event={valid_parts_per_event.mean():.3f} ± {valid_parts_per_event.std():.3f}")
            print(f"  Valid predicted tracks:             total={len(tracks)}, mean/event={tracks_per_event.mean():.3f} ± {tracks_per_event.std():.3f}")
            print(f"  Hits (pre-filter):  total={pre_hits.sum()}, mean/event={pre_hits.mean():.3f} ± {pre_hits.std():.3f}")
            print(f"  Hits (post-filter): total={post_hits.sum()}, mean/event={post_hits.mean():.3f} ± {post_hits.std():.3f}")
            def _pct(x):
                return f"{x * 100:.3g}%"

            print(f"  DM eff (all / >1.0 / >0.9 / >0.6 GeV): {_pct(eff['all'])} / {_pct(eff['1p0'])} / {_pct(eff['0p9'])} / {_pct(eff['0p6'])}")
            print(f"  Perfect eff (all / >1.0 / >0.9 / >0.6 GeV): {_pct(perf_eff['all'])} / {_pct(perf_eff['1p0'])} / {_pct(perf_eff['0p9'])} / {_pct(perf_eff['0p6'])}")
            print(f"  Fake rate (all / >1.0 / >0.9 / >0.6 GeV): {_pct(fr['all'])} / {_pct(fr['1p0'])} / {_pct(fr['0p9'])} / {_pct(fr['0p6'])}")
            print(f"  Duplicate rate: {_pct(tracks.duplicate.mean())}")

            summary_rows.append({
                "model": name,
                "truth_reference_mode": truth_reference_mode,
                "track_valid_threshold": tv,
                "iou_threshold": iou,
                "n_events": n_events,
                "reference_truth_particles": len(ref_parts),
                "integrated_efficiency": float(eff["all"]),
                "efficiency_pt_gt_1p0": float(eff["1p0"]),
                "efficiency_pt_gt_0p9": float(eff["0p9"]),
                "efficiency_pt_gt_0p6": float(eff["0p6"]),
                "integrated_fake_rate": float(fr["all"]),
                "fake_rate_pt_gt_1p0": float(fr["1p0"]),
                "fake_rate_pt_gt_0p9": float(fr["0p9"]),
                "fake_rate_pt_gt_0p6": float(fr["0p6"]),
                "perfect_efficiency_integrated": float(perf_eff["all"]),
                "perfect_efficiency_pt_gt_1p0": float(perf_eff["1p0"]),
                "perfect_efficiency_pt_gt_0p9": float(perf_eff["0p9"]),
                "perfect_efficiency_pt_gt_0p6": float(perf_eff["0p6"]),
            })

    summary_name = "track_eval_threshold_summary.csv" if truth_reference_mode == "post_filter" else f"track_eval_threshold_summary_{truth_reference_mode}_truth.csv"
    summary_path = pathlib.Path(out_dir) / summary_name
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nWrote threshold summary to {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Model checkpoints ────────────────────────────────────────────────────

    tracking_fnames = {
        "Paper": "",
        "MA 900": "",
        "LCA 900": "",
    }


    tracking_config_fnames = {
        "Paper": "",
        "MA 900": "",
        "LCA 900": "",
    }

    tracking_configs = {}
    for name, cfg_path in tracking_config_fnames.items():
        with pathlib.Path(cfg_path).open() as f:
            cfg = yaml.safe_load(f)
        print(f"  {name}: pT > {cfg['data']['particle_min_pt']}, |η| < {cfg['data']['particle_max_abs_eta']}")
        tracking_configs[name] = cfg

    # ── Visual style ─────────────────────────────────────────────────────────
    training_colours = {
        "Paper": "tab:green",
        "MA 600": "tab:orange",
        "LSCA 600": "tab:blue",
        "MA 900": "tab:red",
        "LCA 900": "tab:orange",
    }

    qty_bins = {
        "pt": np.array([0.6, 0.75, 1.0, 1.5, 2, 3, 4, 6, 10]),
        "eta": np.array([-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]),
        "phi": np.array([-math.pi, -2.36, -1.57, -0.79, 0, 0.79, 1.57, 2.36, math.pi]),
    }
    qty_symbols = {"pt": "p_\\mathrm{T}", "eta": "\\eta", "phi": "\\phi"}
    qty_units = {"pt": "[GeV]", "eta": "", "phi": ""}

    # ── Evaluation parameters ─────────────────────────────────────────────────
    out_dir = ""
    event_cache_dir = pathlib.Path(out_dir) / "event_cache"
    num_events = 100
    random_seed = 0
    particle_targets = ["pt", "eta", "phi"]
    # "post_filter": efficiency denominator uses truth particles after hit filtering
    # "pre_filter":  efficiency denominator uses all reconstructable truth particles (fairer)
    truth_reference_mode = "pre_filter"

    # Number of threads for parallel event loading. Each thread opens its own HDF5 handle.
    # Set to 1 for sequential loading. Values of 4-16 typically give the best speedup.
    n_workers = 8

    # Pre-matching cuts applied to truth particles before track-to-particle assignment.
    # Predicted tracks whose best-matching truth particle fails either cut are treated
    # as unmatched (and counted as fakes if they are otherwise valid).
    # These cuts are intentionally looser than the reconstructability cuts (pt_cut, eta_cut)
    # so that the post-matching reconstructability filter is still the primary acceptance gate.
    match_min_hits = 3    # truth particle must have >= this many hits to be matchable
    match_min_pt = 0.5    # truth particle must have pT >= this [GeV] to be matchable

    # Default threshold sweep applied to all models unless overridden below.
    default_track_valid_thresholds = [0.5, 0.6, 0.7]
    default_iou_thresholds = [0.5, 0.6, 0.7, 0.8]

    # Per-model threshold overrides. Omitted models use the defaults above.
    # Either or both of track_valid_thresholds / iou_thresholds can be specified.
    tracking_threshold_overrides = {
        # "Paper": {"track_valid_thresholds": [0.5], "iou_thresholds": [0.5]},
    }

    # Per-model hit-order handling; "auto" infers the best alignment from file metadata.
    hit_order_by_name = {
        "Paper": "as_saved",
        "Tracking DQ": "as_saved",
        "LCA": "auto",
    }

    # ── Run ───────────────────────────────────────────────────────────────────
    tracking_results, truth_reference_results = load_all_results(
        tracking_fnames=tracking_fnames,
        tracking_configs=tracking_configs,
        default_track_valid_thresholds=default_track_valid_thresholds,
        default_iou_thresholds=default_iou_thresholds,
        tracking_threshold_overrides=tracking_threshold_overrides,
        hit_order_by_name=hit_order_by_name,
        num_events=num_events,
        random_seed=random_seed,
        particle_targets=particle_targets,
        truth_reference_mode=truth_reference_mode,
        match_min_hits=match_min_hits,
        match_min_pt=match_min_pt,
        event_cache_dir=event_cache_dir,
        reuse_saved_event_cache=True,
        write_event_cache=True,
        n_workers=n_workers,
    )

    plot_efficiency(
        tracking_results=tracking_results,
        truth_reference_results=truth_reference_results,
        tracking_fnames=tracking_fnames,
        particle_targets=particle_targets,
        qty_bins=qty_bins,
        qty_symbols=qty_symbols,
        qty_units=qty_units,
        training_colours=training_colours,
        out_dir=out_dir,
        truth_reference_mode=truth_reference_mode,
    )

    print_and_save_summary(
        tracking_results=tracking_results,
        truth_reference_results=truth_reference_results,
        tracking_configs=tracking_configs,
        tracking_fnames=tracking_fnames,
        truth_reference_mode=truth_reference_mode,
        out_dir=out_dir,
    )
