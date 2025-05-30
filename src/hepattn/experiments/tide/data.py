from pathlib import Path

import h5py
import numpy as np
import torch
import awkward as ak
import random
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from hepattn.utils.tensor_utils import pad_to_size

# ruff: noqa


def wrap_phi(x: np.ndarray) -> np.ndarray:
    """Correctly wraps an input tensor of pi angles so they lie in [0, 2pi]."""
    x = x + (x < -np.pi).astype(np.float32) * 2 * np.pi
    x = x - (x > np.pi).astype(np.float32) * 2 * np.pi
    return x


def theta_to_eta(theta: np.ndarray, theta_clip=0.1) -> np.ndarray:
    """Converts theta to eta, clipping theta in the case that it is 0 or pi."""
    theta = np.clip(theta, theta_clip,  np.pi - theta_clip)
    return -np.log(np.tan(theta / 2))



class ROIDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        num_samples: int,
        inputs: dict,
        targets: dict,
        sampling_seed: int = 42,
        track_min_pt: float = 1.0,
        track_max_d0: float = 100.0,
        track_max_z0: float = 1000.0,
        track_min_num_hits: dict[str, int] = {"pix": 1, "sct": 2},
        roi_max_energy: float = 1e12,
        roi_max_abs_eta: float = 4.0,
        roi_min_num_tracks: int = 1,
        roi_max_num_tracks: int = 32,
        roi_max_num_dropped_tracks: int = 16,
        selection_pass_rate: float = 0.2,
        precision: str = "single",
    ):
        """ """
        super().__init__()

        self.sampling_seed = sampling_seed
        self.dirpath = Path(dirpath)
        self.num_samples = num_samples
        self.inputs = inputs
        self.targets = targets
        self.track_min_pt = track_min_pt
        self.track_max_d0 = track_max_d0
        self.track_max_z0 = track_max_z0
        self.track_min_num_hits = track_min_num_hits
        self.roi_max_energy = roi_max_energy
        self.roi_max_abs_eta = roi_max_abs_eta
        self.roi_min_num_tracks = roi_min_num_tracks
        self.roi_max_num_tracks = roi_max_num_tracks
        self.roi_max_num_dropped_tracks = roi_max_num_dropped_tracks
        self.selection_pass_rate = selection_pass_rate
        self.precision = precision

        self.precision_type = {
            "half": torch.float16,
            "single": torch.float32,
            "double": torch.float64,
        }[precision]

        self.dirpath = Path(self.dirpath)

        np.random.seed(self.sampling_seed)

        file_paths = list(Path(self.dirpath).glob("*.h5"))

        # Maps the ROI id to the h5 file it is in
        self.roi_id_to_file_path = {}

        # Map the ROI id to the dataset index and vice-versa
        self.idx_to_roi_id = {}
        self.roi_id_to_idx = {}

        # ROIs that are known to fail the selection
        self.rejected_roi_ids = []
        # ROIs that are known to pass the selection
        self.accepted_roi_ids = []
        # ROIs that have not yet been evaluated on the selection
        self.unevaluated_roi_ids = []

        total_num_rois = 0

        for file_path in file_paths:
            with h5py.File(file_path, "r") as file:
                roi_ids = list(file.keys())
                total_num_rois += len(roi_ids)
                print(f"Found {len(roi_ids)} ROIs in {file_path}, total of {total_num_rois}")

                for roi_id in roi_ids:
                    self.roi_id_to_file_path[roi_id] = file_path
                    self.unevaluated_roi_ids.append(roi_id)

                if total_num_rois >= int(self.num_samples / self.selection_pass_rate):
                    print(f"Read sufficient ROIs given the assumed selection pass rate.")
                    break

    def __len__(self) -> int:
        """Returns the number of samples / ROIs that are available in the dataset after all cuts have been applied."""
        return int(self.num_samples)

    def load_roi(self, roi_id):
        with h5py.File(self.roi_id_to_file_path[roi_id], "r") as file:
            roi = {"roi_valid": np.array([True])}

            # Load some ROI info so we can check if the ROI passes basic cuts before proceeding
            for field in ["id", "energy", "eta", "phi", "mass"]:
                roi[f"roi_{field}"] = file[f"{roi_id}/roi_{field}"][:]

            # Apply any ROI-based cuts
            roi["roi_valid"] &= roi["roi_energy"] <= self.roi_max_energy
            roi["roi_valid"] &= np.abs(roi["roi_eta"]) <= self.roi_max_abs_eta

            # If the ROI has failed the selection, return nothing
            if not roi["roi_valid"][0]:
                return None

            # Apply any hit based cuts

            # Apply any track based cuts
            for track in ["sudo", "sisp", "reco"]:
                # Convert pT to GeV
                roi[f"{track}_pt"] = file[f"{roi_id}/{track}_pt"][:] / 1000.0
                roi[f"{track}_d0"] = file[f"{roi_id}/{track}_d0"][:]
                roi[f"{track}_z0"] = file[f"{roi_id}/{track}_z0"][:]
                roi[f"{track}_valid"] = np.full_like(roi[f"{track}_pt"], True, dtype=bool)

            # Apply pT and d0 cut to pseudotracks
            roi["sudo_valid"] &= roi["sudo_pt"] >= self.track_min_pt
            roi["sudo_valid"] &= roi["sudo_d0"] <= self.track_max_d0
            roi["sudo_valid"] &= roi["sudo_z0"] <= self.track_max_z0

            # Calculate the ROI reference point
            for coord in ["vx", "vy", "vz", "d0", "z0"]:
                roi[f"sisp_{coord}"] = file[f"{roi_id}/sisp_{coord}"][:]
                roi[f"roi_{coord}"] = np.array([np.median(roi[f"sisp_{coord}"])])

            roi["roi_theta"] = 2 * np.arctan(np.exp(-roi[f"roi_eta"]))

            for track in ["sudo", "sisp", "reco"]:
                # Load in track fields
                for field in ["pt", "eta", "phi", "z0", "d0", "vx", "vy", "vz", "q", "origin"]:
                    roi[f"{track}_{field}"] = file[f"{roi_id}/{track}_{field}"][:]

                # Make extra track fields
                roi[f"{track}_px"] = roi[f"{track}_pt"] * np.cos(roi[f"{track}_phi"])
                roi[f"{track}_py"] = roi[f"{track}_pt"] * np.sin(roi[f"{track}_phi"])
                roi[f"{track}_pz"] = roi[f"{track}_pt"] * np.sinh(roi[f"{track}_eta"])
                roi[f"{track}_theta"] = 2 * np.arctan(np.exp(-roi[f"{track}_eta"]))

                # Make track fields that are in the ROI frame
                roi[f"{track}_deta"] = roi[f"{track}_eta"] - roi["roi_eta"]
                roi[f"{track}_dtheta"] = roi[f"{track}_theta"] - roi["roi_theta"]
                roi[f"{track}_dphi"] = wrap_phi(roi[f"{track}_phi"] - roi["roi_phi"])
                roi[f"{track}_dz0"] = roi[f"{track}_z0"] - roi["roi_z0"]
                
                # Convert the track pT from MeV to GeV
                roi[f"{track}_pt"] *= 0.001
                roi[f"{track}_qopt"] = roi[f"{track}_q"] / roi[f"{track}_pt"]

                # Create scaled fields that are O(1)
                field_scalings = {"deta": 100.0, "dphi": 100.0, "dtheta": 100.0, "qopt": 10.0, "dz0": 0.1, "d0": 1.0}
                for field, scaling in field_scalings.items():
                    roi[f"{track}_scaled_{field}"] = scaling * roi[f"{track}_{field}"]

            def load_csr_matrix(index_field, data_field, dtype):
                # Load the CSR data
                csr_data = file[f"{roi_id}/{data_field}_data"][:]
                csr_indices = file[f"{roi_id}/{index_field}_indices"][:]
                csr_indptr = file[f"{roi_id}/{index_field}_indptr"][:]
                csr_shape = tuple(file[f"{roi_id}/{index_field}_shape"][:])
                csr_mat = csr_matrix((csr_data, csr_indices, csr_indptr), csr_shape, dtype=dtype)

                # Convert the mask to a dense mask and keep it
                return np.array(csr_mat.todense())

            for track in ["sudo", "sisp", "reco"]:
                for hit in ["pix", "sct"]:
                    # Load the mask
                    roi[f"{track}_{hit}_valid"] = load_csr_matrix(f"{track}_{hit}_valid", f"{track}_{hit}_valid", bool)

                    # Remove tracks that failed the selection from the mask
                    roi[f"{track}_{hit}_valid"] &= roi[f"{track}_valid"][..., None]
                    roi[f"{track}_num_{hit}"] = roi[f"{track}_{hit}_valid"].sum(-1)

                    # Remove tracks that dont have enough hits
                    roi[f"{track}_valid"] &= roi[f"{track}_num_{hit}"] >= self.track_min_num_hits[hit]

                    # Remove tracks that failed the selection from then mask
                    roi[f"{track}_{hit}_valid"] &= roi[f"{track}_valid"][..., None]

                # Apply any ROI based cut that requres the track content
                roi[f"roi_num_{track}"] = roi[f"{track}_valid"].sum(-1)
                roi["roi_valid"] &= roi[f"roi_num_{track}"] >= self.roi_min_num_tracks
                roi["roi_valid"] &= roi[f"roi_num_{track}"] <= self.roi_max_num_tracks

            # If the ROI has failed the selection, return nothing
            if not roi["roi_valid"][0]:
                return None

            # Load the track-hit regression targets
            for field in ["loc_x", "loc_y", "phi", "theta", "energy"]:
                try:
                    roi[f"sudo_pix_{field}"] = load_csr_matrix(f"sudo_pix_valid", f"sudo_pix_{field}", np.float32)
                except ValueError as e:
                    print(e)
                    return None

            for hit in ["pix", "sct"]:
                # Load in hit fields
                for suffix in ["", "_mod", "_mod_norm"]:
                    item = hit + suffix
                    for coord in ["x", "y", "z"]:
                        # Convert mm to m
                        roi[f"{item}_{coord}"] = file[f"{roi_id}/{item}_{coord}"][:] / 1000.0

                    # Add extra hit fields
                    roi[f"{item}_r"] = np.sqrt(roi[f"{item}_x"] ** 2 + roi[f"{item}_y"] ** 2)
                    roi[f"{item}_s"] = np.sqrt(roi[f"{item}_x"] ** 2 + roi[f"{item}_y"] ** 2 + roi[f"{item}_z"] ** 2)
                    roi[f"{item}_theta"] = np.arccos(roi[f"{item}_z"] / roi[f"{item}_s"])
                    roi[f"{item}_eta"] = theta_to_eta(roi[f"{item}_theta"])
                    roi[f"{item}_phi"] = np.arctan2(roi[f"{item}_y"], roi[f"{item}_x"])

                # Add the ROI fields onto the hit fields
                for field in ["theta", "eta", "phi", "vx", "vy", "vz", "z0", "d0"]:
                    roi[f"{hit}_roi_{field}"] = np.full_like(roi[f"{hit}_x"], roi[f"roi_{field}"][0])

                # Add the relative ROI coordinates
                for suffix in ["", "_mod", "_mod_norm"]:
                    item = hit + suffix
                    roi[f"{item}_rel_z"] = roi[f"{item}_z"] - (roi[f"roi_z0"] / 1000.0)
                    roi[f"{item}_rel_s"] = np.sqrt(roi[f"{item}_x"] ** 2 + roi[f"{item}_y"] ** 2 + roi[f"{item}_rel_z"] ** 2)
                    roi[f"{item}_rel_theta"] = np.arccos(roi[f"{item}_rel_z"] / roi[f"{item}_rel_s"])
                    roi[f"{item}_rel_eta"] = theta_to_eta(roi[f"{item}_rel_theta"])

                    roi[f"{item}_dtheta"] = roi[f"{item}_rel_theta"] - roi["roi_theta"]
                    roi[f"{item}_deta"] = roi[f"{item}_rel_eta"] - roi["roi_eta"]
                    roi[f"{item}_dphi"] = wrap_phi(roi[f"{item}_phi"] - roi["roi_phi"])

                # Add the module local coordinates
                for coord in ["x", "y"]:
                    roi[f"{hit}_mod_loc_{coord}"] = file[f"{roi_id}/{hit}_mod_loc_{coord}"][:]

                # Add conformal coordinates
                roi[f"{hit}_u"] = roi[f"{hit}_x"] / (roi[f"{hit}_x"]**2 + roi[f"{hit}_y"]**2)
                roi[f"{hit}_v"] = roi[f"{hit}_y"] / (roi[f"{hit}_x"]**2 + roi[f"{hit}_y"]**2)

                # Mark the hits as valid inputs
                roi[f"{hit}_valid"] = np.full_like(roi[f"{hit}_x"], True)

            # Add the charge and log charge for the pixel
            roi[f"pix_charge"] = file[f"{roi_id}/pix_charge"][:]
            roi[f"pix_log_charge"] = np.log10(1.0 + np.clip(roi[f"pix_charge"], a_min=0.0, a_max=1e12))

            # Pixel specific fields
            for field in ["lshift", "pitches"]:
                roi[f"pix_{field}"] = file[f"{roi_id}/pix_{field}"][:]

            # Load the pixel charge matrices
            roi["pix_charge_matrix"] = load_csr_matrix("pix_charge_matrix", "pix_charge_matrix", np.float32)

            # Charge can have a large dynamic range, so take the log
            # We need to clamp since sometimes noise means cells have a readout < 0
            roi["pix_log_charge_matrix"] = np.log10(1.0 + np.clip(roi["pix_charge_matrix"], a_min=0.0, a_max=1e12))

            # SCT specific fields
            for field in ["side", "width"]:
                roi[f"sct_{field}"] = file[f"{roi_id}/sct_{field}"][:]
            
            # Check the number of tracks that were dropped
            num_tracks_pre_cuts = len(roi[f"{track}_valid"])
            num_tracks_post_cuts = roi[f"{track}_valid"].sum()
            num_dropped_tracks = num_tracks_pre_cuts - num_tracks_post_cuts

            # Drop the ROI if we no longer have enough tracks after cuts
            if num_tracks_post_cuts < self.roi_min_num_tracks:
                return None
            
            # Drop any ROIs that dropped more tracks than allowed
            if num_dropped_tracks > self.roi_max_num_dropped_tracks:
                return None

            # Drop any invalid track slots
            for track in ["sudo", "sisp", "reco"]:
                track_valid = roi[f"{track}_valid"]
                for target_name, fields in self.targets.items():
                    if track in target_name:
                        roi[f"{target_name}_valid"] = roi[f"{target_name}_valid"][track_valid, ...]
                        for field in fields:
                            roi[f"{target_name}_{field}"] = roi[f"{target_name}_{field}"][track_valid, ...]

            # If we got to here, the ROI passes the selection, so return the ROI
            return roi

    def __getitem__(self, idx):
        # Attempt to load the ROI with the given ID
        # First check if an ROI has already been evaluated and assigned to this index
        if idx in self.idx_to_roi_id:
            # If its been assigned to an index, we know it passes the selection, and so we can go ahead and load it
            roi = self.load_roi(self.idx_to_roi_id[idx])
        else:
            roi = None

            # Keep trying ROI IDs from the set of unevaluated ROI IDs until we eventually get an ROI that
            # passes the selection
            num_attempts = 1
            while roi is None:
                # Check we still have some ROIs left
                if not len(self.unevaluated_roi_ids) > 0:
                    raise StopIteration("Ran out of ROIs that pass the selection.")

                # Randomly sample an ID from the set of unevaluated IDs
                roi_id = random.choice(self.unevaluated_roi_ids)

                # Load this ROI ID and see if it passes the selection
                roi = self.load_roi(roi_id)

                # This ROI ID has been evaluated on the selection now
                self.unevaluated_roi_ids.remove(roi_id)
                num_attempts += 1

                # If the ROI passes the selection, add it to the accepted list and map it to a dataset index
                if roi is not None:
                    self.accepted_roi_ids.append(roi_id)
                    self.idx_to_roi_id[idx] = roi_id
                    self.roi_id_to_idx[roi_id] = idx
                # If the ROI fails the selection, add it to the rejected list and try again
                else:
                    self.rejected_roi_ids.append(roi_id)

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        inputs = {}
        targets = {}
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = torch.from_numpy(roi[f"{input_name}_valid"]).bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets[f"{input_name}_valid"] = inputs[f"{input_name}_valid"]
            for field in fields:
                inputs[f"{input_name}_{field}"] = torch.from_numpy(roi[f"{input_name}_{field}"]).to(self.precision_type).unsqueeze(0)

        # Convert the targets
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = torch.from_numpy(roi[f"{target_name}_valid"]).bool().unsqueeze(0)
            for field in fields:
                targets[f"{target_name}_{field}"] = torch.from_numpy(roi[f"{target_name}_{field}"]).to(self.precision_type).unsqueeze(0)

        # Convert the metedata
        targets["sample_id"] = torch.tensor(roi["roi_id"], dtype=torch.int64)

        return inputs, targets


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Takes a list of tensors, pads them to a given size, and then concatenates them along the a new dimension at zero."""
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class ROICollator:
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

                # If the field is a vector we need to adjust the target size accordingly
                # TODO: Make this nicer
                if next(iter(inputs))[k].ndim == 3:
                    size = (hit_max_sizes[input_name], next(iter(inputs))[k].shape[-1])
                else:
                    size = (hit_max_sizes[input_name],)

                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], size, 0.0)

        for target_name, fields in self.dataset_targets.items():
            if len(target_name.split("_")) == 1:
                size = (self.max_num_obj,)
            else:
                hit = target_name.split("_")[1]
                size = (self.max_num_obj, hit_max_sizes[hit])

            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)

        # Batch the metadata
        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)

        return batched_inputs, batched_targets


class ROIDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        **kwargs,
    ):
        """Lightning data module. Will iterate over the given directories
        and read preprocessed awkward parquet files until the desired number
        samples are obtained for each dataset.

        Parameters
        ----------
        batch_dimension : int
            Number of samples to read in a minibatch.
        train_dir : str
            Training data directory.
        val_dir : str
            Validation data directory.
        test_dir : str
            Test data directory.
        num_workers : int
            Number of workers / threads too use to read batches.
        num_train " int
            Target number of training samples to load.
        num_val " int
            Target number of training samples to load.
        num_test " int
            Target number of training samples to load.
        """
        super().__init__()

        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        # Create training and validation datasets
        if stage == "fit":
            self.train_dset = ROIDataset(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dset = ROIDataset(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit":
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ROIDataset(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ROIDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=ROICollator(dataset.inputs, dataset.targets, dataset.roi_max_num_tracks),
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dset, stage="fit", shuffle=False)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dset, stage="test", shuffle=False)
