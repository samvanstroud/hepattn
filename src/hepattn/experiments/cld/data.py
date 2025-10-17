from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from scipy.sparse import csr_array, csr_matrix
from torch.utils.data import DataLoader

from hepattn.utils.array_utils import masked_angle_diff_last_axis, masked_diff_last_axis
from hepattn.utils.lrsm_dataset import LRSMDataset

torch.multiprocessing.set_sharing_strategy("file_system")


class CLDDataset(LRSMDataset):
    def __init__(
        self,
        dirpath: str,
        num_samples: int,
        inputs: dict[str, list[str]],
        targets: dict[str, list[str]],
        input_dtype: str = "float32",
        target_dtype: str = "float32",
        input_pad_value: float = 0.0,
        target_pad_value: float = 0.0,
        force_pad_sizes: dict[str, int] | None = None,
        skip_pad_items: list[str] | None = None,
        sampling_seed: int = 42,
        sample_reject_warn_limit: int = 10,
        merge_inputs: dict[str, list[str]] | None = None,
        particle_min_pt: float = 0.01,
        particle_max_abs_eta: float = 4.0,
        include_neutral: bool = True,
        include_charged: bool = True,
        charged_particle_min_num_hits: dict[str, int] | None = None,
        charged_particle_max_num_hits: dict[str, int] | None = None,
        neutral_particle_min_num_hits: dict[str, int] | None = None,
        neutral_particle_max_num_hits: dict[str, int] | None = None,
        particle_cut_veto_min_num_hits: dict[str, int] | None = None,
        particle_hit_min_p_ratio: dict[str, float] | None = None,
        particle_hit_deflection_cuts: dict[str, dict[str, float | int]] | None = None,
        particle_hit_separation_cuts: dict[str, dict[str, float | int]] | None = None,
        particle_min_calib_calo_energy: dict[str, float] | None = None,
        truth_filter_hits: list[str] | None = None,
        calo_energy_thresh: float = 1e-6,
    ):
        if truth_filter_hits is None:
            truth_filter_hits = []
        if particle_hit_min_p_ratio is None:
            particle_hit_min_p_ratio = {}
        if particle_hit_deflection_cuts is None:
            particle_hit_deflection_cuts = {}
        if particle_hit_separation_cuts is None:
            particle_hit_separation_cuts = {}
        if neutral_particle_max_num_hits is None:
            neutral_particle_max_num_hits = {}
        if neutral_particle_min_num_hits is None:
            neutral_particle_min_num_hits = {}
        if charged_particle_max_num_hits is None:
            charged_particle_max_num_hits = {}
        if charged_particle_min_num_hits is None:
            charged_particle_min_num_hits = {}
        if particle_cut_veto_min_num_hits is None:
            particle_cut_veto_min_num_hits = {}
        if particle_min_calib_calo_energy is None:
            particle_min_calib_calo_energy = {}
        if merge_inputs is None:
            merge_inputs = {}

        super().__init__(
            dirpath,
            num_samples,
            inputs,
            targets,
            input_dtype,
            target_dtype,
            input_pad_value,
            target_pad_value,
            force_pad_sizes,
            skip_pad_items,
            sampling_seed,
            sample_reject_warn_limit,
        )

        self.merge_inputs = merge_inputs
        self.particle_min_pt = particle_min_pt
        self.include_neutral = include_neutral
        self.include_charged = include_charged
        self.charged_particle_min_num_hits = charged_particle_min_num_hits
        self.charged_particle_max_num_hits = charged_particle_max_num_hits
        self.neutral_particle_min_num_hits = neutral_particle_min_num_hits
        self.neutral_particle_max_num_hits = neutral_particle_max_num_hits
        self.particle_cut_veto_min_num_hits = particle_cut_veto_min_num_hits
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_hit_min_p_ratio = particle_hit_min_p_ratio
        self.particle_hit_deflection_cuts = particle_hit_deflection_cuts
        self.particle_hit_separation_cuts = particle_hit_separation_cuts
        self.truth_filter_hits = truth_filter_hits
        self.calo_energy_thresh = calo_energy_thresh

        # Setup the number of events that will be used
        event_filenames = list(Path(self.dirpath).rglob("*reco*.npz"))
        num_available_events = len(event_filenames)
        num_requested_events = num_available_events if num_samples == -1 else num_samples
        self.num_samples = min(num_available_events, num_requested_events)

        print(f"Found {num_available_events} available events, {num_requested_events} requested, {self.num_samples} used")

        # Allow us to select events by index
        self.event_filenames = event_filenames[: self.num_samples]

        def event_filenames_to_sample_id(event_filename):
            id_parts = str(event_filename.stem.replace("_condor", "")).split("_")
            job_id = id_parts[-3]
            proc_id = id_parts[-2]
            event_id = id_parts[-1]
            return int(job_id + proc_id.zfill(4) + event_id.zfill(4))

        # Define the sample identifiers unique to each sample, uses the file name
        # Example: reco_p8_ee_tt_ecm365_12012864_7_329 -> 1201286470329
        self.sample_ids = [event_filenames_to_sample_id(f) for f in self.event_filenames]
        self.sample_ids_to_event_filenames = {self.sample_ids[i]: str(self.event_filenames[i]) for i in range(len(self.sample_ids))}

    def load_sample(self, sample_id: int) -> dict[str, np.ndarray] | None:
        """Loads a single CLD event from a preprocessed npz file."""
        event_filename = self.sample_ids_to_event_filenames[sample_id]

        # Load the event, taking care to deal with partially preprocessed / malformed events
        try:
            with np.load(event_filename, allow_pickle=True) as archive:
                event = {key: archive[key] for key in archive.files}
        except EOFError as exception:
            print(f"Encountered exception {exception} while loading sample {sample_id} so skipping it")
            return None

        # Rename from legacy
        aliases = {
            "reco_flow": "pandora",
            "reco_cluster": "topocluster",
            "reco_track": "sitrack",
        }
        for k in list(event.keys()):
            for name, alias in aliases.items():
                if k.startswith(name):
                    event[k.replace(name, alias)] = event[k]

        def convert_mm_to_m(i, p):
            # Convert a spatial coordinate from mm to m inplace
            for coord in ["x", "y", "z"]:
                event[f"{i}.{p}.{coord}_mm"] = event[f"{i}.{p}.{coord}"]
                event[f"{i}.{p}.{coord}"] = 0.001 * event[f"{i}.{p}.{coord}"]

        def add_cylindrical_coords(i, p):
            # Add standard tracking cylindrical coordinates
            event[f"{i}.{p}.r"] = np.sqrt(event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)
            event[f"{i}.{p}.s"] = np.sqrt(event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2 + event[f"{i}.{p}.z"] ** 2)
            event[f"{i}.{p}.theta"] = np.arccos(event[f"{i}.{p}.z"] / event[f"{i}.{p}.s"])

            with np.errstate(invalid="ignore"):
                event[f"{i}.{p}.eta"] = -np.log(np.tan(event[f"{i}.{p}.theta"] / 2))

            event[f"{i}.{p}.abs_eta"] = np.abs(event[f"{i}.{p}.eta"])
            event[f"{i}.{p}.phi"] = np.arctan2(event[f"{i}.{p}.y"], event[f"{i}.{p}.x"])
            event[f"{i}.{p}.rinv"] = 1.0 / event[f"{i}.{p}.r"]
            event[f"{i}.{p}.sinphi"] = np.sin(event[f"{i}.{p}.phi"])
            event[f"{i}.{p}.cosphi"] = np.cos(event[f"{i}.{p}.phi"])
            event[f"{i}.{p}.theta_drad"] = 100 * event[f"{i}.{p}.theta"]
            event[f"{i}.{p}.eta_drad"] = 100 * event[f"{i}.{p}.eta"]
            event[f"{i}.{p}.phi_drad"] = 100 * event[f"{i}.{p}.phi"]

        def add_conformal_coords(i, p):
            # Conformal tracking coordinates
            event[f"{i}.{p}.c"] = 1 / (event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)
            event[f"{i}.{p}.u"] = event[f"{i}.{p}.x"] / (event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)
            event[f"{i}.{p}.v"] = event[f"{i}.{p}.y"] / (event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)

        def add_log_energy(i):
            event[f"{i}.log_energy"] = np.log(1000 * event[f"{i}.energy"])

        # Atomic input types
        trkr_hits = ["vtb", "vte", "itb", "ite", "otb", "ote"]
        calo_hits = ["ecb", "ece", "hcb", "hce", "hco", "msb", "mse"]
        hits = trkr_hits + calo_hits

        trkr_cols = [f"{hit}_col" for hit in trkr_hits]
        calo_cols = [f"{hit}_col" for hit in calo_hits]
        cols = trkr_cols + calo_cols

        calo_cons = [f"{hit}_con" for hit in calo_hits]

        # It is important to do the mm -> m conversion first, so that all other
        # distance fields are also in m, which is required to not to cause nans in the positional encoding
        # Add extra coordinates for tracker + hit positions
        for item in hits + cols:
            convert_mm_to_m(item, "pos")
            add_cylindrical_coords(item, "pos")
            add_conformal_coords(item, "pos")

        # Add extra coords for calo contribution step positions
        for item in calo_cons:
            convert_mm_to_m(item, "step_pos")
            add_cylindrical_coords(item, "step_pos")
            add_conformal_coords(item, "step_pos")

            mask = event[f"{item}.energy"] >= 1e-6
            event[f"{item}.energy"][np.invert(mask)] = 0.0
            event[f"{item}.log_energy"] = np.zeros_like(event[f"{item}.energy"])
            event[f"{item}.log_energy"][mask] = np.log10(1.0 + (event[f"{item}.energy"][mask] * 1e6))

        # Add the log of the energy, useful for training over large dynamic range
        for item in calo_hits:
            add_log_energy(item)

        # Add extra coordinates for tracker collection momenta
        for item in trkr_cols:
            add_cylindrical_coords(item, "mom")

        # Add extra coordinates to the start and and positions and momenta for particles
        add_cylindrical_coords("particle", "vtx")
        add_conformal_coords("particle", "vtx")
        add_cylindrical_coords("particle", "mom")
        add_cylindrical_coords("pandora", "mom")
        add_cylindrical_coords("pandora", "ref")

        event["particle.mom.qopt"] = event["particle.charge"] / event["particle.mom.r"]
        event["pandora.mom.qopt"] = event["pandora.charge"] / event["pandora.mom.r"]

        event["particle.energy"] = np.sqrt(event["particle.mass"] ** 2 + (np.cosh(event["particle.mom.eta"]) * event["particle.mom.r"]) ** 2)

        # Merge inputs, first check all requested merged inputs have the same
        # fields and that the fields are given in the same order
        if self.merge_inputs:
            for merged_input_name, input_names in self.merge_inputs.items():
                merged_input_fields = self.inputs[merged_input_name]

                # Concatenate the fields from all of the inputs that make the merged input
                for field in merged_input_fields:
                    event[f"{merged_input_name}.{field}"] = np.concatenate([event[f"{input_name}.{field}"] for input_name in input_names], axis=-1)

        # Particle count includes invaliud particles, since the linking indices were built
        # before any of these particle selections were made
        event["particle_valid"] = np.full_like(event["particle.PDG"], True, np.bool)
        event["pandora_valid"] = np.full_like(event["pandora.PDG"], True, np.bool)
        event["sitrack_valid"] = np.full_like(event["sitrack.chi2"], True, np.bool)

        num_particles = len(event["particle_valid"])

        particle_hit_masks = [("particle", hit) for hit in hits]
        pandora_hit_masks = [("pandora", hit) for hit in hits]
        sitrack_hit_masks = [("sitrack", hit) for hit in trkr_hits]

        masks = particle_hit_masks + pandora_hit_masks + sitrack_hit_masks

        def load_csr_mask(src, tgt):
            data = (event[f"{src}_to_{tgt}_data"], event[f"{src}_to_{tgt}_indices"], event[f"{src}_to_{tgt}_indptr"])
            shape = tuple(event[f"{src}_to_{tgt}_shape"])
            return csr_matrix(data, shape, dtype=bool)

        # Now we will construct the masks that link particles to hits
        for src, tgt in masks:
            mask_csr = load_csr_mask(src, tgt)
            mask_dense = np.array(mask_csr.todense())
            event[f"{src}_{tgt}_valid"] = mask_dense

        col_fields = [
            "pos.x",
            "pos.y",
            "pos.z",
            "pos.r",
            "pos.theta",
            "pos.phi",
            "pos.sinphi",
            "pos.cosphi",
            "mom.x",
            "mom.y",
            "mom.z",
            "mom.r",
            "mom.theta",
            "mom.phi",
            "mom.rinv",
            "mom.sinphi",
            "mom.cosphi",
        ]
        con_fields = ["energy", "log_energy"]

        particle_trkrhit_fields = {("particle", f"{hit}_col", hit): col_fields for hit in trkr_hits}
        particle_calohit_fields = {("particle", f"{hit}_con", hit): con_fields for hit in calo_hits}

        particle_hit_fields = particle_trkrhit_fields | particle_calohit_fields

        for (src, link, tgt), fields in particle_hit_fields.items():
            src_link_csr = load_csr_mask(src, link)
            link_tgt_csr = load_csr_mask(link, tgt)

            for field in fields:
                link_csr = csr_array(event[f"{link}.{field}"])

                # This performs the (sparse) contraction A[i,j]b[j]C[j,k]
                src_tgt_field_csr = src_link_csr.multiply(link_csr).dot(link_tgt_csr)
                src_tgt_field_dense = np.array(src_tgt_field_csr.todense())
                event[f"{src}_{tgt}.{field}"] = src_tgt_field_dense

        # Calculate the fractional energy contributions for the calo clusters
        for calo_hit in calo_hits:
            # Normalise the energy by the sum of the contributions from all particles
            particle_calo_energy = event[f"particle_{calo_hit}.energy"]
            # This is the total energy each hit recieves from every particle
            particle_total_energy = particle_calo_energy.sum(-2)
            event[f"particle_{calo_hit}.energy_frac"] = np.divide(
                particle_calo_energy,
                particle_total_energy,
                out=np.zeros_like(particle_calo_energy),
                where=particle_total_energy > self.calo_energy_thresh,
            )

        object_names = ["particle", "pandora", "sitrack"]

        # Merge together any masks
        if self.merge_inputs:
            for object_name in object_names:
                for merged_input_name, input_names in self.merge_inputs.items():
                    # TODO: Clean up this hack
                    if object_name == "sitrack" and merged_input_name in {"ecal", "hcal", "muon"}:
                        continue

                    event[f"{object_name}_{merged_input_name}_valid"] = np.concatenate(
                        [event[f"{object_name}_{hit}_valid"] for hit in input_names], axis=-1
                    )

                    if f"{object_name}_{merged_input_name}" in self.targets:
                        for field in self.targets[f"{object_name}_{merged_input_name}"]:
                            event[f"{object_name}_{merged_input_name}.{field}"] = np.concatenate(
                                [event[f"{object_name}_{hit}.{field}"] for hit in input_names], axis=-1
                            )

        # TODO: Clean this up - maybe everything should just be the calibrated energy?
        calo_hit_calibrations = {
            "ecal": 37.0,
            "hcal": 45.0,
        }

        # Now calculate the total energy each particle has in each of the calo hits
        # Need to be careful to do this after we have merged calo hits / on the merged calo hits!
        for calo_hit in ["ecal", "hcal"]:
            # Sum over the hits
            event[f"particle.energy_{calo_hit}"] = event[f"particle_{calo_hit}.energy"].sum(-1)
            event[f"particle.calib_energy_{calo_hit}"] = calo_hit_calibrations[calo_hit] * event[f"particle.energy_{calo_hit}"]

        # Add extra labels for particles
        for object_name in ["particle", "pandora"]:
            event[f"{object_name}.is_charged"] = np.abs(event[f"{object_name}.charge"]) > 0
            event[f"{object_name}.is_neutral"] = ~event[f"{object_name}.is_charged"]

        event["particle.is_primary"] = event["particle.generatorStatus"] == 1
        event["particle.is_secondary"] = event["particle.generatorStatus"] != 1

        event["particle.calib_energy_calo"] = event["particle.calib_energy_ecal"] + event["particle.calib_energy_hcal"]

        # Add one-hot particle class labels
        particle_class_id_to_name = {
            0: "neutral_hadron",
            1: "charged_hadron",
            3: "photon",
            4: "electron",
            5: "muon",
            6: "tau",
            7: "neutrino",
            -1: "other",
        }

        for class_id, class_name in particle_class_id_to_name.items():
            event[f"particle.is_{class_name}"] = np.isclose(event["particle.class"], class_id)

        # Compute angular isolation
        dphi = event["particle.mom.phi"][:, None] - event["particle.mom.phi"][None, :]
        deta = event["particle.mom.eta"][:, None] - event["particle.mom.eta"][None, :]
        isolation = np.sqrt(dphi**2 + deta**2)
        isolation[np.arange(num_particles), np.arange(num_particles)] = np.inf
        event["particle.isolation"] = np.min(isolation, axis=-1)

        # Set which particles we deem to be targets / reconstructable
        particle_cuts = {"min_pt": event["particle.mom.r"] >= self.particle_min_pt}

        # Place a max no. of sihit cut first to get rid of charged looper tracks
        for hit_name, max_num_hits in self.charged_particle_max_num_hits.items():
            particle_cuts[f"before_cut_charged_max_{hit_name}"] = ~(
                event["particle.is_charged"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits)
            )

        # Add the eta cut
        particle_cuts["max_eta"] = np.abs(event["particle.mom.eta"]) <= self.particle_max_abs_eta

        if not self.include_charged:
            particle_cuts["not_charged"] = ~event["particle.is_charged"]

        if not self.include_neutral:
            particle_cuts["not_neutral"] = ~event["particle.is_neutral"]

        # TODO: Clean this up...
        for item_name, min_ratio in self.particle_hit_min_p_ratio.items():
            mask = event[f"particle_{item_name}_valid"]

            px = np.ma.masked_array(event[f"particle_{item_name}.mom.x"], mask=~mask)
            py = np.ma.masked_array(event[f"particle_{item_name}.mom.y"], mask=~mask)
            pz = np.ma.masked_array(event[f"particle_{item_name}.mom.z"], mask=~mask)

            p = np.sqrt(px**2 + py**2 + pz**2)
            item_mean_p_ratio = p / p.mean(-1)[..., None]

            event[f"particle_{item_name}_valid"] = event[f"particle_{item_name}_valid"] & (item_mean_p_ratio >= min_ratio).filled(False)

        # Apply hit cuts based on angular deflection
        # TODO: Clean this up...
        for item_name, cut in self.particle_hit_deflection_cuts.items():
            for _ in range(int(cut["num_passes"])):
                # Indices for sorting based on time
                idx = np.argsort(event[f"{item_name}.time"])

                mask = event[f"particle_{item_name}_valid"][:, idx]

                px = np.ma.masked_array(event[f"particle_{item_name}.mom.x"][:, idx], mask=~mask)
                py = np.ma.masked_array(event[f"particle_{item_name}.mom.y"][:, idx], mask=~mask)
                pz = np.ma.masked_array(event[f"particle_{item_name}.mom.z"][:, idx], mask=~mask)

                # The 3d angle
                angle_diff = masked_angle_diff_last_axis(px, py, pz, ~mask).filled(0.0)
                # Undo the sorting
                angle_diff = angle_diff[:, np.argsort(idx)]

                # The angle in the x-y plane
                zeros = np.ma.masked_array(np.zeros_like(px), mask=~mask)
                angle_diff_xy = masked_angle_diff_last_axis(px, py, zeros, ~mask).filled(0.0)
                angle_diff_xy = angle_diff_xy[:, np.argsort(idx)]

                # The angle in the r-z plane
                pt = np.ma.sqrt(px**2 + py**2)
                angle_diff_rz = masked_angle_diff_last_axis(pt, pz, zeros, ~mask).filled(0.0)
                angle_diff_rz = angle_diff_rz[:, np.argsort(idx)]

                # Requires it to pass one of the cut
                # TODO: Removing the 3d angle and only use the xy and rz angle
                angle_diff_or = (angle_diff <= cut["max_angle"]) | (angle_diff_xy <= cut["max_angle_xy"]) | (angle_diff_rz <= cut["max_angle_rz"])
                event[f"particle_{item_name}_valid"] = event[f"particle_{item_name}_valid"] & angle_diff_or

        # Apply hit cuts based on distance between consecutive hits on particles
        # TODO: Clean this up...
        for item_name, cut in self.particle_hit_separation_cuts.items():
            for _ in range(int(cut["num_passes"])):
                idx = np.argsort(event[f"{item_name}.time"])

                mask = event[f"particle_{item_name}_valid"][:, idx]

                x = np.ma.masked_array(mask * event[f"{item_name}.pos.x"][..., None, :][:, idx], mask=~mask)
                y = np.ma.masked_array(mask * event[f"{item_name}.pos.y"][..., None, :][:, idx], mask=~mask)
                z = np.ma.masked_array(mask * event[f"{item_name}.pos.z"][..., None, :][:, idx], mask=~mask)

                dx = masked_diff_last_axis(x)
                dy = masked_diff_last_axis(y)
                dz = masked_diff_last_axis(z)

                dr = np.ma.sqrt(dx**2 + dy**2 + dz**2).filled(0.0)

                # Undo the sorting
                dr = dr[:, np.argsort(idx)]

                # event[f"particle_valid"] = event[f"particle_valid"] & (dr <= max_dist).all(-1)
                event[f"particle_{item_name}_valid"] = event[f"particle_{item_name}_valid"] & (dr <= cut["max_dist"])

        # Above seems to be tedious, an alternative but not generalised
        num_hit_cut_names = list(dict.fromkeys(self.charged_particle_min_num_hits))
        deflection_cut_names = list(dict.fromkeys(self.particle_hit_deflection_cuts))
        to_merge_name = [name for name in num_hit_cut_names if name in deflection_cut_names]
        particle_hit_valid = [event[f"particle_{hit_name}_valid"] for hit_name in to_merge_name]

        merge_name = set(num_hit_cut_names) - set(deflection_cut_names)
        for name in merge_name:
            event[f"particle_{name}_valid"] = np.concatenate(particle_hit_valid, axis=-1)

        # Now we have built the masks, we can apply hit/counting based cuts
        for hit_name, min_num_hits in self.charged_particle_min_num_hits.items():
            particle_cuts[f"charged_min_{hit_name}"] = ~(event["particle.is_charged"] & (event[f"particle_{hit_name}_valid"].sum(-1) < min_num_hits))

        for hit_name, max_num_hits in self.charged_particle_max_num_hits.items():
            particle_cuts[f"charged_max_{hit_name}"] = ~(event["particle.is_charged"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits))

        for hit_name, min_num_hits in self.neutral_particle_min_num_hits.items():
            particle_cuts[f"neutral_min_{hit_name}"] = ~(event["particle.is_neutral"] & (event[f"particle_{hit_name}_valid"].sum(-1) < min_num_hits))

        for hit_name, max_num_hits in self.neutral_particle_max_num_hits.items():
            particle_cuts[f"neutral_max_{hit_name}"] = ~(event["particle.is_neutral"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits))

        # Apply the particle cuts
        for cut_mask in particle_cuts.values():
            event["particle_valid"] &= cut_mask

        # Apply cut vetos
        for hit_name, min_num_hits in self.particle_cut_veto_min_num_hits.items():
            event["particle_valid"] = event["particle_valid"] | (event[f"particle_{hit_name}_valid"].sum(-1) > min_num_hits)

        # Remove any mask slots for invalid particles
        for input_name in self.inputs:
            event[f"particle_{input_name}_valid"] &= event["particle_valid"][:, np.newaxis]

            # Zero out any invalid particle slots for the mask regression
            if f"particle_{input_name}" in self.targets:
                for field in self.targets[f"particle_{input_name}"]:
                    event[f"particle_{input_name}.{field}"] *= event["particle_valid"][:, np.newaxis].astype(np.float32)

        # If specified, mark any noise hits as hits that should be dropped
        item_cuts = {}
        for input_name in self.truth_filter_hits:
            # Get hits that are not noise
            mask = event[f"particle_{input_name}_valid"].any(-2)

            if input_name not in item_cuts:
                item_cuts[input_name] = mask
            else:
                item_cuts[input_name] &= mask

        # Do truth hit filtering if specified
        for input_name, mask in item_cuts.items():
            # First drop hits from inputs
            for field in self.inputs[input_name]:
                event[f"{input_name}.{field}"] = event[f"{input_name}.{field}"][mask]

            # Also drop hits from the target masks
            if f"particle_{input_name}" in self.targets:
                event[f"particle_{input_name}_valid"] = event[f"particle_{input_name}_valid"][:, mask]

                if f"particle_{input_name}" in self.targets:
                    for field in self.targets[f"particle_{input_name}"]:
                        event[f"particle_{input_name}.{field}"] = event[f"particle_{input_name}.{field}"][:, mask]

        # Event level info
        event["event_num_particles"] = event["particle_valid"].sum()

        for input_name in self.inputs:
            event[f"event_num_{input_name}"] = ~np.isnan(event[f"{input_name}.type"]).sum()

        # Check that there are no noise hits if we specified this
        for input_name in self.truth_filter_hits:
            if f"particle_{input_name}" in self.targets:
                assert np.all(event[f"particle_{input_name}_valid"].sum(-2) > 0)

        # Calculate some truth particle summary stats
        for hit in hits + list(self.merge_inputs.keys()):
            event[f"particle.num_{hit}"] = event[f"particle_{hit}_valid"].sum(-1)

        # Remove invalid particle slots
        # Assue that particle axis is always 0, make a copy as we will also change particle_valid
        particle_valid = np.copy(event["particle_valid"])
        for target_name, fields in self.targets.items():
            if "particle" in target_name:
                event[f"{target_name}_valid"] = event[f"{target_name}_valid"][particle_valid, ...]
                for field in fields:
                    event[f"{target_name}.{field}"] = event[f"{target_name}.{field}"][particle_valid, ...]

        for input_name, fields in self.inputs.items():
            event[f"{input_name}_valid"] = ~np.isnan(event[f"{input_name}.type"])

            for field in fields:
                event[f"{input_name}_{field}"] = event[f"{input_name}.{field}"]

        # Now pick out the targets
        for target_name, fields in self.targets.items():
            for field in fields:
                event[f"{target_name}_{field}"] = event[f"{target_name}.{field}"]

        # Add any metadata
        event["sample_id"] = sample_id

        return event


class CLDDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        batch_size: int = 1,
        test_dir: str | None = None,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dset = CLDDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset = CLDDataset(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = CLDDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, dataset: CLDDataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset)
