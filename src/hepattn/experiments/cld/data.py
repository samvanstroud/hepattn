from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule, seed_everything
from scipy.sparse import csr_array, csr_matrix
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hepattn.utils.array_utils import masked_angle_diff_last_axis, masked_diff_last_axis
from hepattn.utils.tensor_utils import pad_to_size

torch.multiprocessing.set_sharing_strategy("file_system")


class CLDDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        merge_inputs: dict[str, list[str]] | None = None,
        num_events: int = -1,
        particle_min_pt: float = 0.1,
        include_neutral: bool = True,
        include_charged: bool = True,
        charged_particle_min_num_hits: dict[str, int] | None = None,
        charged_particle_max_num_hits: dict[str, int] | None = None,
        neutral_particle_min_num_hits: dict[str, int] | None = None,
        neutral_particle_max_num_hits: dict[str, int] | None = None,
        particle_cut_veto_min_num_hits: dict[str, int] | None = None,
        particle_max_abs_eta: float = 4.0,
        particle_hit_min_p_ratio: dict[str, float] | None = None,
        particle_hit_deflection_cuts: dict[str, dict] | None = None,
        particle_hit_separation_cuts: dict[str, dict] | None = None,
        particle_min_calib_calo_energy: dict[str, float] | None = None,
        truth_filter_hits: list[str] | None = None,
        event_max_num_particles: int = 256,
        calo_energy_thresh: float = 1e-6,
        random_seed: int = 42,
        precision: str | int = 16,
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
        super().__init__()

        self.dirpath = dirpath
        self.inputs = inputs
        self.targets = targets
        self.merge_inputs = merge_inputs
        self.num_events = num_events
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
        self.event_max_num_particles = event_max_num_particles
        self.calo_energy_thresh = calo_energy_thresh
        self.random_seed = random_seed
        self.precision = precision
        self.precision_type = {
            "16": torch.float16,
            "bf16": torch.bfloat16,
            "32": torch.float32,
            "64": torch.float64,
        }[str(precision)]

        # Global random state initialisation
        np.random.default_rng(42)
        seed_everything(42, workers=True)

        # Setup the number of events that will be used
        event_filenames = list(Path(self.dirpath).rglob("*reco*.npz"))
        num_available_events = len(event_filenames)
        num_requested_events = num_available_events if num_events == -1 else num_events
        self.num_events = min(num_available_events, num_requested_events)

        print(f"Found {num_available_events} available events, {num_requested_events} requested, {self.num_events} used")

        # Allow us to select events by index
        self.event_filenames = event_filenames[: self.num_events]

        def event_filenames_to_sample_id(event_filename):
            id_parts = str(event_filename.stem).split("_")
            job_id = id_parts[-3]
            proc_id = id_parts[-2]
            event_id = id_parts[-1]
            return int(job_id + proc_id.zfill(4) + event_id.zfill(4))

        # Define the sample identifiers unique to each sample, uses the file name
        # Example: reco_p8_ee_tt_ecm365_12012864_7_329 -> 1201286470329
        self.sample_ids = np.array([event_filenames_to_sample_id(f) for f in self.event_filenames], dtype=np.int64)
        self.sample_ids_to_event_filenames = {self.sample_ids[i]: str(self.event_filenames[i]) for i in range(len(self.sample_ids))}
        self.event_filenames_to_sample_ids = {v: k for k, v in self.sample_ids_to_event_filenames.items()}

    def __len__(self):
        return int(self.num_events)

    def load_event(self, sample_id):
        """Loads a single CLD event from a preprocessed npz file."""
        event_filename = self.sample_ids_to_event_filenames[sample_id]

        # Load the event
        with np.load(event_filename, allow_pickle=True) as archive:
            event = {key: archive[key] for key in archive.files}

        def convert_mm_to_m(i, p):
            # Convert a spatial coordinate from mm to m inplace
            for coord in ["x", "y", "z"]:
                event[f"{i}.{p}.{coord}_mm"] = event[f"{i}.{p}.{coord}"]
                event[f"{i}.{p}.{coord}"] *= 0.001

        def add_cylindrical_coords(i, p):
            # Add standard tracking cylindrical coordinates
            event[f"{i}.{p}.r"] = np.sqrt(event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2)
            event[f"{i}.{p}.s"] = np.sqrt(event[f"{i}.{p}.x"] ** 2 + event[f"{i}.{p}.y"] ** 2 + event[f"{i}.{p}.z"] ** 2)
            event[f"{i}.{p}.theta"] = np.arccos(event[f"{i}.{p}.z"] / event[f"{i}.{p}.s"])
            event[f"{i}.{p}.eta"] = -np.log(np.tan(event[f"{i}.{p}.theta"] / 2))
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
        cons = calo_cons

        items = hits + cols + cons

        # Make sure everything is sorted in time
        item_orderings = {}
        for item in items:
            if f"{item}.time" in event:
                item_orderings[item] = np.argsort(event[f"{item}.time"])

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
        particle_pos_fields = ["vtx"]
        particle_mom_fields = ["mom"]

        for field in particle_pos_fields:
            convert_mm_to_m("particle", field)
            add_cylindrical_coords("particle", field)
            add_conformal_coords("particle", field)

        for field in particle_mom_fields:
            add_cylindrical_coords("particle", field)

        event["particle.mom.qopt"] = event["particle.charge"] / event["particle.mom.r"]

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
        num_particles = len(event["particle_valid"])

        particle_hit_masks = [("particle", hit) for hit in hits]
        masks = particle_hit_masks

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

        # Merge together any masks
        if self.merge_inputs:
            for merged_input_name, input_names in self.merge_inputs.items():
                event[f"particle_{merged_input_name}_valid"] = np.concatenate([event[f"particle_{hit}_valid"] for hit in input_names], axis=-1)

                if f"particle_{merged_input_name}" in self.targets:
                    for field in self.targets[f"particle_{merged_input_name}"]:
                        event[f"particle_{merged_input_name}.{field}"] = np.concatenate(
                            [event[f"particle_{hit}.{field}"] for hit in input_names], axis=-1
                        )

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
        event["particle.isCharged"] = np.abs(event["particle.charge"]) > 0
        event["particle.isNeutral"] = ~event["particle.isCharged"]

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
                event["particle.isCharged"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits)
            )

        # Add the eta cut
        particle_cuts["max_eta"] = np.abs(event["particle.mom.eta"]) <= self.particle_max_abs_eta

        if not self.include_charged:
            particle_cuts["not_charged"] = ~event["particle.isCharged"]

        if not self.include_neutral:
            particle_cuts["not_neutral"] = ~event["particle.isNeutral"]

        for item_name, min_ratio in self.particle_hit_min_p_ratio.items():
            mask = event[f"particle_{item_name}_valid"]

            px = np.ma.masked_array(event[f"particle_{item_name}.mom.x"], mask=~mask)
            py = np.ma.masked_array(event[f"particle_{item_name}.mom.y"], mask=~mask)
            pz = np.ma.masked_array(event[f"particle_{item_name}.mom.z"], mask=~mask)

            p = np.sqrt(px**2 + py**2 + pz**2)
            item_mean_p_ratio = p / p.mean(-1)[..., None]

            event[f"particle_{item_name}_valid"] &= (item_mean_p_ratio >= min_ratio).filled(False)

        # Apply hit cuts based on angular deflection
        for item_name, cut in self.particle_hit_deflection_cuts.items():
            for _ in range(cut["num_passes"]):
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
                event[f"particle_{item_name}_valid"] &= angle_diff_or

        # Apply hit cuts based on distance between consecutive hits on particles
        for item_name, cut in self.particle_hit_separation_cuts.items():
            for _ in range(cut["num_passes"]):
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
                event[f"particle_{item_name}_valid"] &= dr <= cut["max_dist"]

        # Merge the hits if they do not appear in the angular deflection and hit separation cuts
        # all_cut_names = set(self.charged_particle_min_num_hits) | set(self.charged_particle_max_num_hits)
        # cut_names = set(self.particle_hit_deflection_cuts) | set(self.particle_hit_separation_cuts)
        # # Pick out the field that needs to be merged
        # merge_name = all_cut_names - cut_names

        # Pick out the input names beside the ones needed to be merged
        # to_merge = {}
        # for merged_input_name, input_names in self.merge_inputs.items():
        #     if merged_input_name in merge_name:
        #         continue
        #     for names in input_names:
        #         to_merge[names] = merged_input_name

        # # Pick out the merged field name by matching the input names then merge
        # for merge_name in to_merge:
        #     input_names = self.merge_inputs.get(merge_name, [])
        #     merged_input_name = list(dict.fromkeys(to_merge.get(hit_name) for hit_name in input_names))
        #     particle_hit_valid = [event[f"particle_{hit_name}_valid"] for hit_name in merged_input_name]
        #     event[f"particle_{merge_name}_valid"] = np.concatenate(particle_hit_valid, axis=-1)

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
            particle_cuts[f"charged_min_{hit_name}"] = ~(event["particle.isCharged"] & (event[f"particle_{hit_name}_valid"].sum(-1) < min_num_hits))

        for hit_name, max_num_hits in self.charged_particle_max_num_hits.items():
            particle_cuts[f"charged_max_{hit_name}"] = ~(event["particle.isCharged"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits))

        for hit_name, min_num_hits in self.neutral_particle_min_num_hits.items():
            particle_cuts[f"neutral_min_{hit_name}"] = ~(event["particle.isNeutral"] & (event[f"particle_{hit_name}_valid"].sum(-1) < min_num_hits))

        for hit_name, max_num_hits in self.neutral_particle_max_num_hits.items():
            particle_cuts[f"neutral_max_{hit_name}"] = ~(event["particle.isNeutral"] & (event[f"particle_{hit_name}_valid"].sum(-1) > max_num_hits))

        # Apply the particle cuts
        for cut_mask in particle_cuts.values():
            event["particle_valid"] &= cut_mask

        # Apply cut vetos
        for hit_name, min_num_hits in self.particle_cut_veto_min_num_hits.items():
            event["particle_valid"] |= event[f"particle_{hit_name}_valid"].sum(-1) > min_num_hits

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

        # Pick out the inputs that have actually been requested
        inputs = {}
        targets = {}
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = ~np.isnan(event[f"{input_name}.type"])
            targets[f"{input_name}_valid"] = inputs[f"{input_name}_valid"]

            for field in fields:
                inputs[f"{input_name}_{field}"] = event[f"{input_name}.{field}"]

        # Now pick out the targets
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = event[f"{target_name}_valid"]
            for field in fields:
                targets[f"{target_name}_{field}"] = event[f"{target_name}.{field}"]

        # Add any metadata
        targets["sample_id"] = self.event_filenames_to_sample_ids[str(event_filename)]

        return inputs, targets

    def __getitem__(self, idx):
        inputs, targets = self.load_event(self.sample_ids[idx])

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        inputs_out = {}
        targets_out = {}
        for input_name, fields in self.inputs.items():
            inputs_out[f"{input_name}_valid"] = torch.from_numpy(inputs[f"{input_name}_valid"]).bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets_out[f"{input_name}_valid"] = inputs_out[f"{input_name}_valid"]
            for field in fields:
                inputs_out[f"{input_name}_{field}"] = torch.from_numpy(inputs[f"{input_name}_{field}"]).to(self.precision_type).unsqueeze(0)

        # Convert the targets
        for target_name, fields in self.targets.items():
            targets_out[f"{target_name}_valid"] = torch.from_numpy(targets[f"{target_name}_valid"]).bool().unsqueeze(0)
            for field in fields:
                targets_out[f"{target_name}_{field}"] = torch.from_numpy(targets[f"{target_name}_{field}"]).to(self.precision_type).unsqueeze(0)

        # Convert the metedata
        targets_out["sample_id"] = torch.tensor([targets["sample_id"]], dtype=torch.int64)

        return inputs_out, targets_out


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along the a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class CLDCollator:
    def __init__(self, dataset_inputs, dataset_targets, max_num_obj):
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_targets
        self.max_num_obj = max_num_obj

    def __call__(self, batch):
        inputs, targets = zip(*batch, strict=False)

        hit_max_sizes = {}
        for input_name in self.dataset_inputs:
            hit_max_sizes[input_name] = max(event[f"{input_name}_valid"].shape[-1] for event in inputs)

        batched_inputs = {}
        batched_targets = {}
        for input_name, fields in self.dataset_inputs.items():
            k = f"{input_name}_valid"
            batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)

            # Some tasks might require to know hit padding info for loss masking
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{input_name}_{field}"
                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)

        for target_name, fields in self.dataset_targets.items():
            if target_name == "particle":
                size = (self.max_num_obj,)
            else:
                hit = target_name.split("_")[1]
                size = (self.max_num_obj, hit_max_sizes[hit])

            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)
            batched_inputs[k] = batched_targets[k]

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, 0.0)

        # Batch the metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)

        return batched_inputs, batched_targets


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
        if stage in {"fit", "test"}:
            self.train_dset = CLDDataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs)

        if stage == "fit":
            self.val_dset = CLDDataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit":
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = CLDDataset(dirpath=self.test_dir, num_events=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: CLDDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=CLDCollator(dataset.inputs, dataset.targets, dataset.event_max_num_particles),
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
