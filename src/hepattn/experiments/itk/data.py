from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hepattn.utils.tensor_utils import pad_to_size


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class ITkDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        targets: dict,
        num_events: int = -1,
        hit_regions: list | None = None,
        particle_min_pt: float = 1.0,
        particle_max_abs_eta: float = 2.5,
        particle_min_num_hits: dict[str, int] | None = None,
        event_max_num_particles=1000,
        hit_eval_path: str | None = None,
        append_hit_eval_output: bool = False,
        apply_hit_eval_pred: bool = False,
    ):
        if particle_min_num_hits is None:
            particle_min_num_hits = {"pixel": 3, "strip": 6}
        super().__init__()

        if append_hit_eval_output:
            assert hit_eval_path is not None, "A hit eval path must be specified to append hit filter outputs."

        if apply_hit_eval_pred:
            assert hit_eval_path is not None, "A hit eval path must be specified to apply hit filtering."

        # Global random state initialisation
        np.random.default_rng(42)

        # Get a list of event names
        event_names = [Path(file).stem.replace("-parts", "") for file in Path(dirpath).glob("*-parts.parquet")]

        # Calculate the number of events that will actually be used
        num_events_available = len(event_names)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)
        if num_events_available == 0:
            msg = f"No events found in {dirpath}"
            raise ValueError(msg)
        if num_events < 0:
            num_events = num_events_available

        # Metadata
        self.dirpath = Path(dirpath)
        self.hit_eval_path = hit_eval_path
        self.apply_hit_eval_pred = apply_hit_eval_pred
        self.append_hit_eval_output = append_hit_eval_output
        self.inputs = inputs
        self.targets = targets
        self.num_events = num_events
        self.event_names = event_names[:num_events]

        # Sample ID is an integer that can uniquely identify each event/sample, used for picking out events during eval etc
        self.sample_ids = np.array([int(name.split("event")[-1]) for name in self.event_names], dtype=np.int64)
        self.sample_ids_to_event_names = {self.sample_ids[i]: str(self.event_names[i]) for i in range(len(self.sample_ids))}
        self.event_names_to_sample_ids = {v: k for k, v in self.sample_ids_to_event_names.items()}

        # Setup hit eval file if specified
        if self.hit_eval_path:
            print(f"Using hit eval dataset {self.hit_eval_path}")

        # Hit level cuts
        self.hit_regions = hit_regions

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_hits = particle_min_num_hits

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles

    def __len__(self):
        return int(self.num_events)

    def load_event(self, sample_id):
        event_name = self.sample_ids_to_event_names[sample_id]

        # Load in event data
        hit_names = list(self.inputs.keys())

        # Load the particles
        particles = pd.read_parquet(self.dirpath / Path(f"{event_name}-parts.parquet"))

        # Load the hits
        hits = {}
        for hit_name in hit_names:
            hits[hit_name] = pd.read_parquet(self.dirpath / Path(f"{event_name}-{hit_name}.parquet"))

        cartesian_fields = {
            "pixel": ["x", "y", "z", "cluster_x", "cluster_y", "cluster_z"],
            "strip": ["x", "y", "z", "cluster_x_1", "cluster_y_1", "cluster_z_1", "cluster_x_2", "cluster_y_2", "cluster_z_2"],
        }

        # Scale the input coordinates to in meters so they are ~ 1
        for hit in hit_names:
            for field in cartesian_fields[hit]:
                hits[hit][field] *= 0.01

        charge_fields = {
            "pixel": ["charge_count"],
            "strip": ["charge_count_1", "charge_count_2"],
        }

        # Provide the logarithm of the charge as the raw charge can be O(several thousand)
        for hit in hit_names:
            for field in charge_fields[hit]:
                # Make sure the charge count is not negative, or this will cause nans from the log
                assert np.all(hits[hit][field] >= 0.0)
                hits[hit][f"log_{field}"] = np.log(hits[hit][field] + 1.0)

        for k in hit_names:
            # Only include hits from the specified detector regions
            if self.hit_regions:
                hits[k] = hits[k][hits[k]["region"].isin(self.hit_regions)]

            # Add extra hit fields
            hits[k]["r"] = np.sqrt(hits[k]["x"] ** 2 + hits[k]["y"] ** 2)
            hits[k]["s"] = np.sqrt(hits[k]["x"] ** 2 + hits[k]["y"] ** 2 + hits[k]["z"] ** 2)
            hits[k]["theta"] = np.arccos(hits[k]["z"] / hits[k]["s"])
            hits[k]["phi"] = np.arctan2(hits[k]["y"], hits[k]["x"])
            hits[k]["eta"] = -np.log(np.tan(hits[k]["theta"] / 2))
            hits[k]["u"] = hits[k]["x"] / (hits[k]["x"] ** 2 + hits[k]["y"] ** 2)
            hits[k]["v"] = hits[k]["y"] / (hits[k]["x"] ** 2 + hits[k]["y"] ** 2)

        # Add extra particle fields
        particles["p"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2 + particles["pz"] ** 2)
        particles["pt"] = np.sqrt(particles["px"] ** 2 + particles["py"] ** 2)
        particles["qopt"] = particles["charge"] / particles["pt"]
        particles["eta"] = np.arctanh(particles["pz"] / particles["p"])
        particles["theta"] = np.arccos(particles["pz"] / particles["p"])
        particles["phi"] = np.arctan2(particles["py"], particles["px"])
        particles["costheta"] = np.cos(particles["theta"])
        particles["sintheta"] = np.sin(particles["theta"])
        particles["cosphi"] = np.cos(particles["phi"])
        particles["sinphi"] = np.sin(particles["phi"])

        # Apply particle level cuts based on particle fields
        particles = particles[particles["pt"] >= self.particle_min_pt]
        particles = particles[particles["eta"].abs() <= self.particle_max_abs_eta]

        # Remove particles that have no truth link record
        particles = particles[particles["particle_id"] != 0]

        # Apply particle cut based on hit content
        for hit in hit_names:
            hit_counts = hits[hit]["particle_id"].value_counts()
            keep_particle_ids = hit_counts[hit_counts >= self.particle_min_num_hits[hit_name]].index.to_numpy()
            particles = particles[particles["particle_id"].isin(keep_particle_ids)]

        for hit in hit_names:
            # Mark which hits are on a valid / reconstructable particle, for the hit filter
            hits[hit]["on_valid_particle"] = hits[hit]["particle_id"].isin(particles["particle_id"])

            # If a hit eval file was specified, read in the predictions from it
            if self.hit_eval_path:
                with h5py.File(self.hit_eval_path, "r") as hit_eval_file:
                    # Append the hit filter logit scores if specified
                    if self.append_hit_eval_output:
                        hits[hit]["filter_logit"] = hit_eval_file[f"{sample_id}/outputs/final/{hit}_filter/{hit}_logit"][0]

                    # If true, we drop hits based on the hit eval prediction, i.e. the pre-decided cut of the model
                    if self.apply_hit_eval_pred:
                        # The dataset has shape (1, num_hits)
                        hit_filter_pred = hit_eval_file[f"{sample_id}/preds/final/{hit}_filter/{hit}_on_valid_particle"][0]
                        hits[hit] = hits[hit][hit_filter_pred]

        # Pack everything togrther
        # First the hit fields
        inputs = {}
        for hit, fields in self.inputs.items():
            inputs[f"{hit}_valid"] = np.ones(len(hits[hit]), dtype=bool)

            for field in fields:
                inputs[f"{hit}_{field}"] = hits[hit][field].to_numpy()

        # Hit filter target
        targets = {}
        for hit in self.inputs:
            targets[f"{hit}_valid"] = np.ones(len(hits[hit]), dtype=bool)
            targets[f"{hit}_on_valid_particle"] = hits[hit]["on_valid_particle"].to_numpy(dtype=bool)

        # Now build and add the masks
        particle_ids = particles["particle_id"].to_numpy(dtype=np.int64)
        for hit in self.inputs:
            hit_particle_ids = hits[hit]["particle_id"].to_numpy(dtype=np.int64)
            targets[f"particle_{hit}_valid"] = particle_ids[:, None] == hit_particle_ids[None, :]

        # Now the particle fields
        if "particle" in self.targets:
            targets["particle_valid"] = particles["pt"].to_numpy() >= self.particle_min_pt
            for field in self.targets["particle"]:
                targets[f"particle_{field}"] = particles[field].to_numpy()

        # Add metadata
        targets["sample_id"] = sample_id

        return inputs, targets

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        # First do the inputs / hits
        inputs_out = {}
        targets_out = {}
        for input_name, fields in self.inputs.items():
            inputs_out[f"{input_name}_valid"] = torch.from_numpy(inputs[f"{input_name}_valid"]).bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets_out[f"{input_name}_valid"] = inputs_out[f"{input_name}_valid"]
            for field in fields:
                inputs_out[f"{input_name}_{field}"] = torch.from_numpy(inputs[f"{input_name}_{field}"]).half().unsqueeze(0)

        # Now pack the targets
        target_shapes = {
            "pixel": (-1,),
            "strip": (-1,),
            "particle": (self.event_max_num_particles,),
            "particle_pixel": (self.event_max_num_particles, -1),
            "particle_strip": (self.event_max_num_particles, -1),
        }

        for target_name, fields in self.targets.items():
            target_valid = torch.from_numpy(targets[f"{target_name}_valid"]).bool()
            target_valid = pad_to_size(target_valid, target_shapes[target_name], False)
            targets_out[f"{target_name}_valid"] = target_valid.unsqueeze(0)

            for field in fields:
                target_field = torch.from_numpy(targets[f"{target_name}_{field}"])
                if torch.is_floating_point(target_field):
                    target_field = pad_to_size(target_field, target_shapes[target_name], 0.0).half()
                else:
                    target_field = pad_to_size(target_field, target_shapes[target_name], False).bool()
                targets_out[f"{target_name}_{field}"] = target_field.unsqueeze(0)

        # Now the metadata
        targets_out["sample_id"] = torch.tensor([targets["sample_id"]], dtype=torch.int64)

        return inputs_out, targets_out


class ITkDataModule(LightningDataModule):
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
        hit_eval_train: str | None = None,
        hit_eval_val: str | None = None,
        hit_eval_test: str | None = None,
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
        self.pin_memory = pin_memory
        self.hit_eval_train = hit_eval_train
        self.hit_eval_val = hit_eval_val
        self.hit_eval_test = hit_eval_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dset = ITkDataset(dirpath=self.train_dir, num_events=self.num_train, hit_eval_path=self.hit_eval_train, **self.kwargs)

        if stage == "fit":
            self.val_dset = ITkDataset(dirpath=self.val_dir, num_events=self.num_val, hit_eval_path=self.hit_eval_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ITkDataset(dirpath=self.test_dir, num_events=self.num_test, hit_eval_path=self.hit_eval_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ITkDataset, shuffle: bool):
        return DataLoader(
            dataset=dataset,
            batch_size=None,
            collate_fn=None,
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
