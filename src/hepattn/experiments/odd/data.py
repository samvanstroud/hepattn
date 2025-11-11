from pathlib import Path

import numpy as np
import pandas as pd
import torch
import awkward as ak

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class ODDEventDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        num_events: int = -1,
        particle_min_pt: float = 1.0,
        particle_max_abs_eta: float = 4.0,
        event_max_num_particles: int = 10_000_000,
    ):
        super().__init__()

        # Get a list of event names, using truth particles as the reference
        self.sample_id_file_paths = {}
        for path in Path(dirpath).rglob("particles_event*.parquet"):
            sample_id = int(path.stem.replace("particles_event_", ""))
            self.sample_id_file_paths[sample_id] = path

        # Calculate the number of events that will actually be used
        sample_ids = list(self.sample_id_file_paths.keys())
        num_events_available = len(sample_ids)

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
        self.num_events = num_events
        self.sample_ids = sample_ids[:num_events]

        # Sample ID is an integer that can uniquely identify each event/sample, used for picking out events during eval etc
        #self.sample_ids = np.array([int(name.split("event")[-1]) for name in self.event_names], dtype=np.int64)
        #self.sample_ids_to_event_names = {self.sample_ids[i]: str(self.event_names[i]) for i in range(len(self.sample_ids))}
        #self.event_names_to_sample_ids = {v: k for k, v in self.sample_ids_to_event_names.items()}

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta


        # Event level cuts
        self.event_max_num_particles = event_max_num_particles

    def __len__(self):
        return int(self.num_events)

    def load_event(self, sample_id):
        particle_path = self.sample_id_file_paths[sample_id]
        sihit_path = Path(str(particle_path).replace("particles", "tracker_hits"))
        calohits_path = Path(str(particle_path).replace("particles", "calo_hits"))
        tracks_path = Path(str(particle_path).replace("particles", "tracks"))

        particles = ak.from_parquet(particle_path)[0]
        sihits = ak.from_parquet(sihit_path)[0]
        calohits = ak.from_parquet(calohits_path)[0]
        tracks = ak.from_parquet(tracks_path)[0]

        inputs = {}
        targets = {}

        for field in sihits.fields:
            inputs[f"sihit_{field}"] = ak.to_torch(sihits[field])

        for field in calohits.fields:
            if field in {"detector", "contrib_particle_ids", "contrib_energies", "contrib_times"}:
                continue

            inputs[f"calohit_{field}"] = ak.to_torch(calohits[field])




        for field in particles.fields:
            targets[f"particle_{field}"] = ak.to_torch(particles[field])

        targets["particle_pt"] = torch.sqrt(targets["particle_py"]**2 + targets["particle_py"]**2)

        for field in particles.fields:
            if field == "event_id": continue
            targets[f"particle_{field}"] = targets[f"particle_{field}"][targets[f"particle_pt"] >= 0.5]


        targets["particle_valid"] = torch.full_like(targets["particle_particle_id"], True)
        targets["particle_sihit_valid"] = targets["particle_particle_id"][:, None] == inputs["sihit_particle_id"][None, :]

        # For the calohits we have to build the mask using awkward arrays since calo hits can be shared
        particle_calohit_assoc = targets["particle_particle_id"][:, None, None] == calohits["contrib_particle_ids"][None, :, :]
        particle_calohit_valid = ak.any(particle_calohit_assoc, axis=-1)
        particle_calohit_energy = ak.sum(calohits["contrib_energies"][None, :, :] * particle_calohit_assoc, axis=-1)
        particle_calohit_time = ak.sum(calohits["contrib_times"][None, :, :] * particle_calohit_assoc, axis=-1)
        

        targets["particle_calohit_valid"] = ak.to_torch(particle_calohit_valid)
        targets["particle_calohit_energy"] = ak.to_torch(particle_calohit_energy)

        # Add metadata
        targets["sample_id"] = torch.tensor(sample_id)

        return inputs, targets

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        inputs_out = {k: v.unsqueeze(0) for k, v in inputs.items()}
        targets_out = {k: v.unsqueeze(0) for k, v in targets.items()}

        return inputs_out, targets_out


class ODDEventDataModule(LightningDataModule):
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
        self.kwargs = kwargs

    def setup(self, stage: str):
        if stage in {"fit", "test"}:
            self.train_dset = ODDEventDataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs)

        if stage == "fit":
            self.val_dset = ODDEventDataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ODDEventDataset(dirpath=self.test_dir, num_events=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ODDEventDataset, shuffle: bool):
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
