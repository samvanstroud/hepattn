from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from hepattn.utils.tensor_utils import pad_to_size


def pad_and_concat(items: list[Tensor], target_size: tuple[int], pad_value) -> Tensor:
    """Pad and stack tensors for batching.

    Each tensor in `items` is padded to `target_size` (with a leading batch dim of 1)
    using the shared `pad_to_size` helper, then concatenated along the new batch
    dimension. This keeps the original tensors unchanged and produces a single
    batched Tensor suitable for models or loss computation.
    """
    return torch.cat([pad_to_size(item, (1, *target_size), pad_value) for item in items], dim=0)


class AtlasMuonDataset(Dataset):
    """Dataset for ATLAS muon events stored in compound HDF5 files.

    This dataset maps a single integer event index to the corresponding
    hit and track arrays stored in pre-chunked HDF5 files. It converts raw
    numpy arrays into the dictionary-of-tensors format expected by the
    training pipeline, including boolean valid masks and per-field tensors.

    Args:
        dirpath (str): Directory containing HDF5 chunks and `metadata.yaml`.
        inputs (dict): Mapping of input groups to field names (e.g. {'hit': [...]}).
        targets (dict): Mapping of target groups to field names (e.g. {'particle': [...]}).
        num_events (int): Number of events to expose (-1 means use all available).
        event_max_num_particles (int): Fixed slot size for particle-level tensors.
        dummy_testing (bool): If True, filters out invalid hits for quick tests.

    Attributes:
        metadata (dict): Loaded metadata describing features and file chunks.
        file_indices, row_indices (np.ndarray): Efficient index arrays to locate events.
        hit_features, track_features (list): Ordered feature names used to map columns.
    """

    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        event_max_num_particles: int = 6,  # Typically fewer tracks per event in muon data
        dummy_testing: bool = False,
        dummy_data: bool = False,
    ):
        super().__init__()
        # Set the global random sampling seed
        self.sampling_seed = 42

        self.dirpath = Path(dirpath)
        self.inputs = inputs
        self.targets = targets
        self.dummy_testing = dummy_testing
        self.dummy_data = dummy_data
        self.event_max_num_particles = event_max_num_particles

        # If dummy_data mode, skip loading metadata and set dummy values
        if self.dummy_data:
            self.num_events = 100 if num_events == -1 else num_events
            self.hit_features = []
            self.track_features = []
            print(f"Created ATLAS muon dummy dataset with {self.num_events:,} events")
            return

        # Load metadata describing feature names, event mapping and file chunks
        # (this metadata was prepared when creating the dataset on disk).
        with (self.dirpath / "metadata.yaml").open() as f:
            self.metadata = yaml.safe_load(f)

        self.hit_features = self.metadata["hit_features"]
        self.track_features = self.metadata["track_features"]

        # Load efficient index arrays
        self.file_indices = np.load(self.dirpath / "event_file_indices.npy")
        self.row_indices = np.load(self.dirpath / "event_row_indices.npy")

        # Calculate number of events to use
        num_events_available = len(self.row_indices)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available."
            raise ValueError(msg)

        if num_events == -1:
            num_events = num_events_available

        if num_events == 0:
            raise ValueError("num_events must be greater than 0")

        self.num_events = num_events

        self.event_max_num_particles = event_max_num_particles

        # Informational print used during debugging/setup
        print(f"Created ATLAS muon dataset with {self.num_events:,} events")

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        # Return dummy data for CI testing if flag is set
        if self.dummy_data:
            return self._generate_dummy_data(idx)

        inputs = {}
        targets = {}

        # Load the event
        hits, particles, num_hits, num_tracks = self.load_event(idx)

        # Build the input hit tensors using the same naming structure as TrackML.
        # For each registered input feature group (e.g. 'hit'), we create a
        # `{feature}_valid` mask and one tensor per field describing the feature.
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]
            for field in fields:
                inputs[f"{feature}_{field}"] = torch.from_numpy(hits[field]).unsqueeze(0)

        # Build the targets that describe which particle slots are occupied.
        # The dataset uses a fixed-size particle dimension (`event_max_num_particles`)
        # so we mark unused slots as False and only True up to `num_tracks`.
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_tracks] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        message = f"Event {idx} has {num_tracks} particles, but limit is {self.event_max_num_particles}"
        assert num_tracks <= self.event_max_num_particles, message

        # Build the particle regression targets
        particle_ids = torch.from_numpy(particles["particle_id"])
        # print("Particle IDs:", particle_ids)

        # Fill unused particle slots with a sentinel (-999) so comparisons are safe
        # and produce `particle_hit_valid` masks by comparing particle IDs per-hit.
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - len(particle_ids))]).type(torch.int32)
        hit_particle_ids = torch.from_numpy(hits["spacePoint_truthLink"])

        # Create the mask targets
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)
        targets["hit_on_valid_particle"] = torch.from_numpy(hits["on_valid_particle"]).unsqueeze(0)
        # Add sample ID (helps tracking predictions back to input event)
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)

        # Build per-particle regression targets (e.g. kinematics).
        # Empty particle slots are filled with NaNs to avoid contributing to
        # regression losses.
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Null target/particle slots are filled with nans
                x = torch.full((self.event_max_num_particles,), torch.nan)
                if field in particles:
                    x[:num_tracks] = torch.from_numpy(particles[field][: self.event_max_num_particles])

                targets[f"particle_{field}"] = x.unsqueeze(0)
        return inputs, targets

    def load_event(self, idx):
        """Load a single event from compound HDF5 files using count-based slicing.

        Raises:
            RuntimeError: If loading the event from the HDF5 file fails.
        """
        # Get file and row info using efficient indexing
        file_idx = self.file_indices[idx]
        row_idx = self.row_indices[idx]  # This is the row within the compound arrays

        # Get chunk info
        chunk = self.metadata["event_mapping"]["chunk_summary"][file_idx]

        # Load from HDF5 file
        h5_file_path = self.dirpath / chunk["h5_file"]

        try:
            # Open the HDF5 file and slice the pre-packed compound arrays.
            # The on-disk layout stores fixed-size rows where only the first
            # `num_hits/num_tracks` entries are valid for the event.
            with h5py.File(h5_file_path, "r") as f:
                num_hits = f["num_hits"][row_idx]
                num_tracks = f["num_tracks"][row_idx]

                # Use count-based slicing to extract only the valid portion.
                hits_array = f["hits"][row_idx, :num_hits]  # [num_hits, num_hit_features]
                tracks_array = f["tracks"][row_idx, :num_tracks]  # [num_tracks, num_track_features]

        except OSError as e:
            raise RuntimeError(f"Failed to load event {idx} from HDF5 file {h5_file_path}: {e}") from e

        # Post-processing: convert the raw arrays into named numpy arrays and
        # apply unit/scaling transforms expected by downstream code.
        # Convert hits array to dictionary
        hits_dict = {}
        for i, feature_name in enumerate(self.hit_features):
            hits_dict[feature_name] = hits_array[:, i]
            if np.isnan(hits_dict[feature_name]).any():
                print(f"WARNING: NaN values found in hits for feature '{feature_name}'")
            if np.isinf(hits_dict[feature_name]).any():
                print(f"WARNING: Inf values found in hits for feature '{feature_name}'")
            if hits_dict[feature_name].size == 0:
                print(f"WARNING: Empty hits array for feature '{feature_name}'")
        # Some scaling: convert units to the conventions used by the model.
        # e.g. positions from mm -> m, times to a smaller scale, covariances to
        # appropriate units to keep magnitudes numerically stable.
        hits = {
            "spacePoint_globEdgeHighX": hits_dict["spacePoint_globEdgeHighX"] * 0.001,
            "spacePoint_globEdgeHighY": hits_dict["spacePoint_globEdgeHighY"] * 0.001,
            "spacePoint_globEdgeHighZ": hits_dict["spacePoint_globEdgeHighZ"] * 0.001,
            "spacePoint_globEdgeLowX": hits_dict["spacePoint_globEdgeLowX"] * 0.001,
            "spacePoint_globEdgeLowY": hits_dict["spacePoint_globEdgeLowY"] * 0.001,
            "spacePoint_globEdgeLowZ": hits_dict["spacePoint_globEdgeLowZ"] * 0.001,
            # 'spacePoint_time': hits_dict['spacePoint_time'] ,
            "spacePoint_time": hits_dict["spacePoint_time"] * 0.00001,
            "spacePoint_driftR": hits_dict["spacePoint_driftR"],
            # Add covariance information
            "spacePoint_covXX": hits_dict["spacePoint_covXX"] * 0.000001,
            "spacePoint_covXY": hits_dict["spacePoint_covXY"] * 0.000001,
            "spacePoint_covYX": hits_dict["spacePoint_covYX"] * 0.000001,
            "spacePoint_covYY": hits_dict["spacePoint_covYY"] * 0.000001,
            # Add detector information
            "spacePoint_channel": hits_dict["spacePoint_channel"] * 0.001,
            "spacePoint_layer": hits_dict["spacePoint_layer"],
            "spacePoint_stationPhi": hits_dict["spacePoint_stationPhi"],
            "spacePoint_stationEta": hits_dict["spacePoint_stationEta"],
            "spacePoint_technology": hits_dict["spacePoint_technology"],
            "spacePoint_stationIndex": hits_dict["spacePoint_stationIndex"] * 0.1,
            # Add truth information
            "spacePoint_truthLink": hits_dict["spacePoint_truthLink"],
        }
        # Add derived hit fields (vectorized numpy operations) used as additional
        # geometric features (radius, spherical distance, angles, eta/phi).
        hits["r"] = np.sqrt(hits["spacePoint_globEdgeLowX"] ** 2 + hits["spacePoint_globEdgeLowY"] ** 2)

        hits["s"] = np.sqrt(hits["spacePoint_globEdgeLowX"] ** 2 + hits["spacePoint_globEdgeLowY"] ** 2 + hits["spacePoint_globEdgeLowZ"] ** 2)

        hits["theta"] = np.arccos(np.clip(hits["spacePoint_globEdgeLowZ"] / hits["s"], -1, 1))
        hits["phi"] = np.arctan2(hits["spacePoint_globEdgeLowY"], hits["spacePoint_globEdgeLowX"])

        hits["eta"] = -np.log(np.tan(hits["theta"] / 2.0))

        hits["on_valid_particle"] = hits["spacePoint_truthLink"] >= 0

        # Convert tracks array (per-event particles) into a dictionary keyed by
        # the `self.track_features` list. These are used to produce particle
        # regression targets later in `__getitem__`.
        tracks_dict = {}
        for i, feature_name in enumerate(self.track_features):
            tracks_dict[feature_name] = tracks_array[:, i]

        # For debugging/testing: optionally drop invalid hits so downstream
        # shapes are smaller and easier to inspect.
        if self.dummy_testing:
            for k in hits:
                hits[k] = hits[k][hits["on_valid_particle"]]
            num_hits = np.sum(hits["on_valid_particle"])

        # Build the particle-level dictionary returned to the caller. Note that
        # `particle_id` is derived from the hit truth links and provides a
        # compact list of particle indices present in the event.
        particles = {
            "particle_id": np.unique(hits["spacePoint_truthLink"][hits["on_valid_particle"]]),  # Sequential IDs
            "truthMuon_pt": tracks_dict["truthMuon_pt"],
            "truthMuon_eta": tracks_dict["truthMuon_eta"],
            "truthMuon_phi": tracks_dict["truthMuon_phi"],
            "truthMuon_q": tracks_dict["truthMuon_q"],
            "truthMuon_qpt": tracks_dict["truthMuon_q"] / tracks_dict["truthMuon_pt"],
        }
        return hits, particles, num_hits, num_tracks

    def _generate_dummy_data(self, idx):
        """Generate completely random dummy data for CI testing."""
        inputs = {}
        targets = {}

        # Create random number generator
        rng = np.random.default_rng(self.sampling_seed + idx)

        # Generate random number of hits (between 10 and 100)
        num_hits = rng.integers(10, 101)

        # Always generate max particles to avoid NaN cost matrix issues
        # (matching fails with scipy when there are NaN entries)
        num_particles = self.event_max_num_particles

        # Build the input hits with random data
        for feature, fields in self.inputs.items():
            inputs[f"{feature}_valid"] = torch.full((num_hits,), True).unsqueeze(0)
            targets[f"{feature}_valid"] = inputs[f"{feature}_valid"]

            for field in fields:
                # Generate random normal data for all fields
                data = rng.standard_normal(num_hits)
                inputs[f"{feature}_{field}"] = torch.from_numpy(data).unsqueeze(0).to(torch.float32)

        # Build the targets for whether a particle slot is used or not
        targets["particle_valid"] = torch.full((self.event_max_num_particles,), False)
        targets["particle_valid"][:num_particles] = True
        targets["particle_valid"] = targets["particle_valid"].unsqueeze(0)

        # Build dummy particle IDs
        particle_ids = torch.arange(num_particles, dtype=torch.long)
        particle_ids = torch.cat([particle_ids, -999 * torch.ones(self.event_max_num_particles - num_particles)])

        # Assign random particle IDs to hits
        hit_particle_ids = torch.randint(0, num_particles, (num_hits,))

        # Create the mask targets
        targets["particle_hit_valid"] = (particle_ids.unsqueeze(-1) == hit_particle_ids.unsqueeze(-2)).unsqueeze(0)

        # Create the hit filter targets (random boolean)
        targets["hit_on_valid_particle"] = torch.randint(0, 2, (num_hits,), dtype=torch.bool).unsqueeze(0)

        # Add sample ID
        targets["sample_id"] = torch.tensor([idx], dtype=torch.int32)

        # Build the regression targets
        if "particle" in self.targets:
            for field in self.targets["particle"]:
                # Generate random particle data
                x = torch.full((self.event_max_num_particles,), torch.nan)
                data = rng.standard_normal(num_particles)
                x[:num_particles] = torch.from_numpy(data).float()
                targets[f"particle_{field}"] = x.unsqueeze(0)

        return inputs, targets


class AtlasMuonCollator:
    """Collate function to batch variable-length event dictionaries.

    Pads variable-length hit and particle dimensions to create consistent
    batched tensors. The collator expects dataset `inputs` and `targets`
    metadata (mappings of group -> fields) to know which tensors to pad.

    Args:
        dataset_inputs (dict): Mapping of input groups to field names.
        dataset_targets (dict): Mapping of target groups to field names.
        max_num_obj (int): Maximum number of particle slots for padding.
    """

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
            # Create `{feature}_valid` mask batch (bool), padded to the maximum
            # number of hits in this minibatch.
            batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), False)
            # Some tasks might require to know hit padding info for loss masking
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{input_name}_{field}"
                batched_inputs[k] = pad_and_concat([i[k] for i in inputs], (hit_max_sizes[input_name],), 0.0)
        if "particle_hit_valid" in targets[0]:
            size = (self.max_num_obj, hit_max_sizes["hit"])
            batched_targets["particle_hit_valid"] = pad_and_concat([t["particle_hit_valid"] for t in targets], size, False)

        for target_name, fields in self.dataset_targets.items():
            if target_name == "particle":
                size = (self.max_num_obj,)

            elif target_name == "hit":
                size = (hit_max_sizes[target_name],)
            k = f"{target_name}_valid"
            batched_targets[k] = pad_and_concat([t[k] for t in targets], size, False)

            for field in fields:
                k = f"{target_name}_{field}"
                batched_targets[k] = pad_and_concat([t[k] for t in targets], size, torch.nan)

        batched_targets["sample_id"] = torch.cat([t["sample_id"] for t in targets], dim=-1)

        return batched_inputs, batched_targets


class AtlasMuonDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for the ATLAS muon datasets.

    Encapsulates creation of `AtlasMuonDataset` for train/val/test splits and
    exposes `train_dataloader`, `val_dataloader`, and `test_dataloader`.
    It centralises DataLoader configuration such as `batch_size`, number of
    workers, `prefetch_factor`, and `pin_memory`.

    Args (constructor):
        train_dir, val_dir, test_dir (str): Directories containing dataset chunks.
        num_workers (int): Number of DataLoader worker processes.
        num_train, num_val, num_test (int): Number of events to load per split.
        batch_size (int): Batch size for DataLoaders.
        pin_memory (bool): Whether to pin memory on CUDA transfer.
        hit_eval_* (str|None): Optional hit-evaluation dataset paths.
        **kwargs: Passed through to `AtlasMuonDataset` (e.g. inputs/targets).
    """

    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = True,
        batch_size: int = 100,
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dataset = AtlasMuonDataset(
                dirpath=self.train_dir,
                num_events=self.num_train,
                **self.kwargs,
            )

        if stage in {"fit", "validate"}:
            self.val_dataset = AtlasMuonDataset(
                dirpath=self.val_dir,
                num_events=self.num_val,
                **self.kwargs,
            )
        # Only print train/val dataset details when actually training (global rank 0)
        if stage == "fit" and self.trainer is not None and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dataset):,} events")
            print(f"Created validation dataset with {len(self.val_dataset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dataset = AtlasMuonDataset(
                dirpath=self.test_dir,
                num_events=self.num_test,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dataset):,} events")

    def get_dataloader(self, stage: str, dataset: AtlasMuonDataset, shuffle: bool, prefetch_factor: int = 8):
        # Set prefetch_factor to None when num_workers=0 to avoid ValueError
        actual_prefetch_factor = None if self.num_workers == 0 else prefetch_factor

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=AtlasMuonCollator(dataset.inputs, dataset.targets, dataset.event_max_num_particles),
            sampler=None,
            num_workers=self.num_workers,
            prefetch_factor=actual_prefetch_factor,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self, shuffle=False):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=shuffle)
