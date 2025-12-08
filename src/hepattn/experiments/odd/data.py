from pathlib import Path

import awkward as ak
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class ODDDataset(IterableDataset):
    def __init__(
        self,
        dirpath: str,
        num_events: int = -1,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_min_num_sihits: int = 6,
        particle_include_charged: bool = True,
        particle_include_neutral: bool = True,
        event_max_num_particles: int = 10_000_000,
        return_calohits: bool = True,
        return_tracks: bool = True,
    ):
        super().__init__()

        # Get a list of event names, using truth particles as the reference
        self.sample_id_file_paths = {}
        for path in Path(dirpath).rglob("particles_event*.parquet"):
            sample_id = int(path.stem.replace("particles_event_", ""))
            self.sample_id_file_paths[sample_id] = path

        # Calculate the number of events that will actually be used
        sample_ids = set(self.sample_id_file_paths.keys())

        # If we are requesting calohits, the calohit data for each event used must be available
        if return_calohits:
            sample_ids &= {
                int(path.stem.replace("calo_hits_event_", ""))
                for path in Path(dirpath).rglob("calo_hits_event*.parquet")
            }

        # If we are requesting tracks, the track data for each event used must be available
        if return_tracks:
            sample_ids &= {
                int(path.stem.replace("tracks_event_", ""))
                for path in Path(dirpath).rglob("tracks_event*.parquet")
            }

        sample_ids = list(sample_ids)
        num_events_available = len(sample_ids)

        if num_events > num_events_available:
            msg = f"Requested {num_events} events, but only {num_events_available} are available in the directory {dirpath}."
            raise ValueError(msg)
        if num_events_available == 0:
            msg = f"No events found in {dirpath}"
            raise ValueError(msg)
        if num_events < 0:
            num_events = num_events_available

        print(f"Found {num_events_available} available events, using {num_events} events.")

        # Sample ID is an integer that can uniquely identify each event/sample, used for picking out events during eval etc
        self.dirpath = Path(dirpath)
        self.num_events = num_events
        self.sample_ids = sample_ids[:num_events]

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_min_num_sihits = particle_min_num_sihits
        self.particle_include_charged = particle_include_charged
        self.particle_include_neutral = particle_include_neutral

        # Event level cuts
        self.event_max_num_particles = event_max_num_particles

        # Whether to return calo/ACTS track collections
        self.return_calohits = return_calohits
        self.return_tracks = return_tracks

    def __len__(self):
        return int(self.num_events)

    def load_event(self, sample_id):
        # Define the paths for the different objects in the event
        particle_path = self.sample_id_file_paths[sample_id]
        sihit_path = Path(str(particle_path).replace("particles", "tracker_hits"))

        # Read in the data
        # TODO: Check if we can skip any of these e.g. skip calo if only doing tracking
        particles = ak.from_parquet(particle_path)[0]
        sihits = ak.from_parquet(sihit_path)[0]

        # Now we build the data tensors
        inputs = {}
        targets = {}

        # Particle fields are not jagged, so can convert straight to tensor
        for field in particles.fields:
            # TODO: Handle the fact that particles can have multiple parents?
            if field == "parent_id":
                continue

            dtype = torch.int64 if field == "particle_id" else torch.float32
            targets[f"particle_{field}"] = ak.to_torch(particles[field]).to(dtype)

        targets["particle_event_id"] = targets["particle_event_id"].expand_as(
            targets["particle_particle_id"]
        )

        # Add extra particle fields
        targets["particle_pt"] = torch.sqrt(
            targets["particle_px"] ** 2 + targets["particle_py"] ** 2
        )
        targets["particle_p"] = torch.sqrt(
            targets["particle_px"] ** 2
            + targets["particle_py"] ** 2
            + targets["particle_pz"] ** 2
        )
        targets["particle_qopt"] = targets["particle_charge"] / targets["particle_pt"]
        targets["particle_eta"] = torch.arctanh(
            targets["particle_pz"] / targets["particle_p"]
        )
        targets["particle_theta"] = torch.arccos(
            targets["particle_pz"] / targets["particle_p"]
        )
        targets["particle_phi"] = torch.arctan2(
            targets["particle_py"], targets["particle_px"]
        )
        targets["particle_charged"] = targets["particle_charge"] != 0
        targets["particle_neutral"] = ~targets["particle_charged"]

        # Calculate the particle cuts
        particle_valid = targets["particle_pt"] >= self.particle_min_pt
        particle_valid = particle_valid & (
            torch.abs(targets["particle_eta"]) <= self.particle_max_abs_eta
        )

        if not self.particle_include_neutral:
            particle_valid = particle_valid & (~targets["particle_neutral"])

        if not self.particle_include_charged:
            particle_valid = particle_valid & (~targets["particle_charged"])

        # Apply the particle cut
        for k in targets:
            if k.split("_")[0] != "particle":
                continue

            targets[k] = targets[k][particle_valid]

        # Apply event level cut
        if particle_valid.sum() > self.event_max_num_particles:
            return None

        targets["particle_valid"] = torch.full_like(
            targets["particle_pt"], True, dtype=torch.bool
        )

        # For now, all sihit fields are not jagged, so can convert straight to tensor
        for field in sihits.fields:
            dtype = torch.int64 if field == "particle_id" else torch.float32
            inputs[f"sihit_{field}"] = ak.to_torch(sihits[field]).to(dtype)

        # Scale coords from mm to m
        for k in ("x", "y", "z"):
            inputs[f"sihit_{k}"] = inputs[f"sihit_{k}"] / 1000.0

        # Add extra sihit fields
        inputs["sihit_r"] = torch.sqrt(inputs["sihit_x"] ** 2 + inputs["sihit_y"] ** 2)
        inputs["sihit_s"] = torch.sqrt(
            inputs["sihit_x"] ** 2 + inputs["sihit_y"] ** 2 + inputs["sihit_z"] ** 2
        )
        inputs["sihit_eta"] = torch.arctanh(inputs["sihit_z"] / inputs["sihit_s"])
        inputs["sihit_theta"] = torch.arccos(inputs["sihit_z"] / inputs["sihit_s"])
        inputs["sihit_phi"] = torch.arctan2(inputs["sihit_y"], inputs["sihit_x"])

        inputs["sihit_valid"] = torch.full_like(
            inputs["sihit_x"], True, dtype=torch.bool
        )
        targets["sihit_valid"] = inputs["sihit_valid"]
        targets["particle_sihit_valid"] = (
            targets["particle_particle_id"][:, None]
            == inputs["sihit_particle_id"][None, :]
        )
        targets["particle_num_sihits"] = targets["particle_sihit_valid"].float().sum(
            -1
        )

        # Have to now perform the particle cuts that depend on constituents
        particle_valid = targets["particle_num_sihits"] >= self.particle_min_num_sihits

        # Apply the constituent based particle cuts
        for k in targets:
            if k.split("_")[0] != "particle":
                continue

            targets[k] = targets[k][particle_valid]

        # Return the calorimeter hit info if requested
        if self.return_calohits:
            calohits_path = Path(str(particle_path).replace("particles", "calo_hits"))

            if not calohits_path.exists():
                print(
                    f"Calo hits data not found at {calohits_path}, so skipping event {sample_id}"
                )

            calohits = ak.from_parquet(calohits_path)[0]

            # Convert the calo hit fields to tensors, skip the jagged fields which we will handle separately
            for field in calohits.fields:
                if field in {
                    "detector",
                    "contrib_particle_ids",
                    "contrib_energies",
                    "contrib_times",
                }:
                    continue
                inputs[f"calohit_{field}"] = ak.to_torch(calohits[field]).to(
                    torch.float32
                )

            # Scale coords from mm to m
            for k in ("x", "y", "z"):
                inputs[f"calohit_{k}"] = inputs[f"calohit_{k}"] / 1000.0

            # For the calohits we have to build the mask using awkward arrays since calo hits can be shared
            particle_calohit_assoc = (
                targets["particle_particle_id"][:, None, None]
                == calohits["contrib_particle_ids"][None, :, :]
            )
            particle_calohit_valid = ak.any(particle_calohit_assoc, axis=-1)
            particle_calohit_energy = ak.sum(
                calohits["contrib_energies"][None, :, :] * particle_calohit_assoc,
                axis=-1,
            )
            particle_calohit_time = ak.sum(
                calohits["contrib_times"][None, :, :] * particle_calohit_assoc, axis=-1
            )

            inputs["calohit_valid"] = torch.full_like(
                inputs["calohit_x"], True, dtype=torch.bool
            )
            targets["calohit_valid"] = inputs["calohit_valid"]
            targets["particle_calohit_valid"] = ak.to_torch(particle_calohit_valid)
            targets["particle_calohit_energy"] = ak.to_torch(particle_calohit_energy)
            targets["particle_calohit_time"] = ak.to_torch(particle_calohit_time)

        # Return ACTS track info if requested
        if self.return_tracks:
            tracks_path = Path(str(particle_path).replace("particles", "tracks"))

            if not tracks_path.exists():
                print(
                    f"Calo hits data not found at {tracks_path}, so skipping event {sample_id}"
                )

            tracks = ak.from_parquet(tracks_path)[0]

            # Read the tracking info
            for field in tracks.fields:
                if field == "hit_ids":
                    continue
                targets[f"track_{field}"] = ak.to_torch(tracks[field])

            targets["track_valid"] = torch.full_like(
                targets["track_phi"], True, dtype=torch.bool
            )

            # Now add the track masks
            hit_id = ak.from_numpy(np.arange(len(inputs["sihit_valid"])))
            track_sihit_valid = ak.to_torch(
                ak.any(tracks["hit_ids"][:, None] == hit_id[None, :], axis=2)
            )

            targets["track_sihit_valid"] = track_sihit_valid

        # Add metadata
        targets["sample_id"] = torch.tensor(sample_id)

        return inputs, targets

    def __iter__(self):
        # Support multi-worker loading by splitting sample_ids across workers
        worker_info = get_worker_info()
        if worker_info is None:
            sample_ids = self.sample_ids
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(
                np.ceil(len(self.sample_ids) / float(num_workers))
            )
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.sample_ids))
            sample_ids = self.sample_ids[start:end]

        for sample_id in sample_ids:
            event = self.load_event(sample_id)
            # Skip the event if load_event yields None
            if event is None:
                continue

            inputs, targets = event

            # Apply particle padding
            for k, v in targets.items():
                if k.startswith("particle_"):
                    x = v

                    pad_shape = (
                        self.event_max_num_particles - x.size(0),
                        *x.shape[1:],
                    )
                    pad_tensor = x.new_full(
                        pad_shape, False if x.dtype is torch.bool else 0
                    )

                    targets[k] = torch.cat([x, pad_tensor], dim=0)

            # Add dummy batch dimension
            # TODO: Support batch size > 1
            inputs_out = {k: v.unsqueeze(0) for k, v in inputs.items()}
            targets_out = {k: v.unsqueeze(0) for k, v in targets.items()}

            yield inputs_out, targets_out


class ODDDataModule(LightningDataModule):
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
            self.train_dset = ODDDataset(
                dirpath=self.train_dir, num_events=self.num_train, **self.kwargs
            )

        if stage == "fit":
            self.val_dset = ODDDataset(
                dirpath=self.val_dir, num_events=self.num_val, **self.kwargs
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert (
                self.test_dir is not None
            ), "No test file specified, see --data.test_dir"
            self.test_dset = ODDDataset(
                dirpath=self.test_dir, num_events=self.num_test, **self.kwargs
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

    def get_dataloader(self, stage: str, dataset: ODDDataset, shuffle: bool):
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
