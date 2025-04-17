from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from numpy import ndarray as A
from torch.utils.data import DataLoader, Dataset

import hepattn.experiments.trackml.cluster_features as ecf
from hepattn.utils.masks import build_target_masks

ADDITIONAL_EXATRKX_FEATURES = [
    "u",
    "v",
    "charge_frac",
    "leta",
    "lphi",
    "lx",
    "ly",
    "lz",
    "geta",
    "gphi",
]


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class TrackMLDataset(Dataset):
    def __init__(
        self,
        dirpath: str,
        inputs: dict,
        labels: dict,
        trainer: L.Trainer,
        use_exatrkx_inputs: bool = False,
        volume_ids: list | None = None,
        num_samples: int = -1,
        minimum_pt: float = 1,
        max_abs_eta: float = 2.5,
        num_objects: int = 2000,
        min_hits: int = 3,
    ):
        super().__init__()

        self.sampling_seed = 42
        np.random.seed(self.sampling_seed)  # noqa: NPY002

        # trainer info for the scheduling
        self.trainer = trainer

        # input stuff
        self.dirpath = dirpath
        p = Path(self.dirpath).glob("event*-truth.csv.gz")
        self.files = [
            x
            for x in p
            if is_valid_file(x)
            and is_valid_file(str(x).replace("truth", "hits"))
            and is_valid_file(str(x).replace("truth", "particles"))
            and is_valid_file(str(x).replace("truth", "cells"))
        ]
        assert len(self.files) > 0
        self.num_samples = self.get_num_samples(num_samples)

        # other properties
        self.inputs = inputs
        self.labels = labels
        self.volume_ids = volume_ids
        self.minimum_pt = minimum_pt
        self.max_abs_eta = max_abs_eta
        self.min_hits = min_hits
        self.num_objects = num_objects
        self.use_exatrkx_inputs = use_exatrkx_inputs

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, idx):
        """Use .unsqueeze(0) to add in the dummy batch dimension (length 1 always)"""
        inputs = {}
        labels = {}

        hits, particles = self.load_event(idx)

        inputs["hit"] = torch.from_numpy(hits[self.inputs["hit"]].values).unsqueeze(0).half()

        # set class labels for the object queries
        class_labels = torch.ones(self.num_objects)
        class_labels[: len(particles)] = 0
        labels["object_class"] = class_labels.long().unsqueeze(0)

        # get masks
        pids = torch.from_numpy(particles["particle_id"].values)
        if len(pids) > self.num_objects:
            pids = pids[: self.num_objects]
        pids = torch.cat([pids, -1 * torch.ones(self.num_objects - len(pids))])
        labels["masks"] = build_target_masks(
            pids.unsqueeze(0),
            torch.from_numpy(hits["particle_id"].values).unsqueeze(0),
        )

        # regression targets
        for label in self.labels["truth"]:
            tgt = torch.full((self.num_objects,), torch.nan)  # number of reconstructed tracks
            tgt[: len(particles)] = torch.from_numpy(particles[label].to_numpy()[: self.num_objects])
            labels[label] = tgt.unsqueeze(0)

        # for pos enc
        inputs["phi"] = torch.from_numpy(hits["phi"].values).unsqueeze(0).half()
        inputs["theta"] = torch.from_numpy(hits["theta"].values).unsqueeze(0).half()
        inputs["r"] = torch.from_numpy(hits["r"].values).unsqueeze(0).half()

        # target for hit classifier, don't change name or things will break
        labels["hit"] = {}
        labels["hit"]["tgt_pid"] = torch.from_numpy(hits["tgt_pid"].values).unsqueeze(0)
        labels["hit"]["hit_tgt"] = labels["hit"]["tgt_pid"] != 0

        return inputs, labels

    def add_info(self, hits, parts):
        parts["p"] = np.sqrt(parts.px**2 + parts.py**2 + parts.pz**2)
        parts["pt"] = np.sqrt(parts.px**2 + parts.py**2)
        parts["eta"] = np.arctanh(parts.pz / parts.p)
        parts["theta"] = np.arccos(parts["pz"] / parts["p"])
        parts["phi"] = np.arctan2(parts.py, parts.px)
        parts["costheta"] = np.cos(parts["theta"])
        parts["sintheta"] = np.sin(parts["theta"])
        parts["cosphi"] = np.cos(parts["phi"])
        parts["sinphi"] = np.sin(parts["phi"])

        # add cylindrical hit coords
        hits["r"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2)
        hits["s"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + hits["z"] ** 2)
        hits["theta"] = np.arccos(hits["z"] / hits["s"])
        hits["eta"] = -np.log(np.tan(hits["theta"] / 2))

        return hits, parts

    def kinematic_selection(self, particles):
        # keep high pt central particles only
        particles = particles[particles["pt"] > self.minimum_pt]  # GeV
        return particles[particles["eta"].abs() < self.max_abs_eta]

    def calc_eta(self, r: A, z: A) -> A:
        """Compute pseudorapidity (spatial)."""
        theta = np.arctan2(r, z)
        return -np.log(np.tan(theta / 2.0))

    def set_unreconstructable_targets(self, hits, particles):
        # keep only target particles with at least 3 hits (i.e. keep only "reconstructable" truth targets)
        counts = hits.particle_id.value_counts()
        keep_particle_ids = counts[counts >= self.min_hits].index.to_numpy()
        particles = particles[particles.particle_id.isin(keep_particle_ids)]
        # set the particle id of hits from particles which are not kept to zero
        hits.loc[~hits.particle_id.isin(keep_particle_ids), "tgt_pid"] = 0
        return hits, particles

    def get_event_id_from_batch_idx(self, batch_idx) -> str:
        path = self.files[batch_idx]
        fname = Path(path).name
        return fname.split("-")[0]

    def load_event(self, idx):
        truth_fname = f = self.files[idx]
        hits_fname = Path(str(f).replace("-truth", "-hits"))
        particles_fname = Path(str(f).replace("-truth", "-particles"))

        # load data
        truth = pd.read_csv(truth_fname, engine="pyarrow")[["hit_id", "particle_id", "weight"]]
        hits = pd.read_csv(
            hits_fname,
            engine="pyarrow",
            dtype={"x": np.float32, "y": np.float32, "z": np.float32},
        )
        particles = pd.read_csv(particles_fname, engine="pyarrow")
        assert (truth.index == hits.index).all()

        # add hit info
        hits["particle_id"] = truth["particle_id"]  # used for evaluation, don't modify
        hits["tgt_pid"] = truth["particle_id"]  # used for training, can modify targets

        # only include hits from the specified volumes
        # pix barrel: 8, pix endcap: 7, 9
        # https://competitions.codalab.org/competitions/20112
        if self.volume_ids:
            hits = hits[hits["volume_id"].isin(self.volume_ids)]
            truth = truth[truth["hit_id"].isin(hits["hit_id"])]

        # add additional input information about cluster shapes etc
        if self.use_exatrkx_inputs:
            cells_info_path = Path(str(f).replace("-truth", "-cells"))
            if is_valid_file(cells_info_path):
                cells = pd.read_csv(cells_info_path, engine="pyarrow")
                if self.volume_ids:
                    allowed_values = list(hits["hit_id"].unique())
                    cells = cells[cells["hit_id"].isin(allowed_values)]
                assert hits["hit_id"].nunique() == cells["hit_id"].nunique(), (
                    "load event just before different number of unique hit ids in hits & cells"
                )
                detector_config = cells_info_path.parent.parent / "detectors.csv"
                hits = ecf.append_cell_features(hits, cells, detector_config)
            else:
                msg = f"failed to find cells info for {f!s}"
                raise ValueError(msg)

        # hardcoded input scaling
        hits.x *= 0.01
        hits.y *= 0.01
        hits.z *= 0.01

        hits, particles = self.add_info(hits, particles)
        particles = self.kinematic_selection(particles)
        hits, particles = self.set_unreconstructable_targets(hits, particles)

        # set hit targets
        # get indices of hits associated to particles after filtering particles
        valid_idx = hits.particle_id.isin(particles.particle_id)

        # if the hit is not in the filtered particles, set id to zero but don't remove
        hits.loc[~valid_idx, "tgt_pid"] = 0

        # sort hits by phi
        hits["phi"] = np.arctan2(hits["y"], hits["x"])
        hits = hits.sort_values("phi")

        # sanity checks
        assert len(particles) != 0, "no particles remaining; loosen selection!"
        assert len(hits) != 0, "no hits remaining; loosen selection!"
        assert particles.particle_id.nunique() == len(particles), "non-unique particle ids"

        return hits, particles

    def get_num_samples(self, num_samples_requested: int):
        num_samples_available = len(self.files)

        # not enough samples
        if num_samples_requested > num_samples_available:
            msg = f"Requested {num_samples_requested:,} samples, but only {num_samples_available:,} are available in the directory {self.dirpath}."
            raise ValueError(msg)

        # use all samples
        if num_samples_requested < 0:
            return num_samples_available

        # use requested samples
        return num_samples_requested


class TrackMLDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        pin_memory: bool = False,
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
        if self.trainer.is_global_zero:
            print("-" * 80)

        # create training and validation datasets
        if stage in {"fit", "test"}:
            self.train_dset = TrackMLDataset(
                dirpath=self.train_dir,
                num_samples=self.num_train,
                trainer=self.trainer,
                **self.kwargs,
            )

        if stage == "fit":
            self.val_dset = TrackMLDataset(
                dirpath=self.val_dir,
                num_samples=self.num_val,
                trainer=self.trainer,
                **self.kwargs,
            )

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = TrackMLDataset(
                dirpath=self.test_dir,
                num_samples=self.num_test,
                trainer=self.trainer,
                **self.kwargs,
            )
            print(f"Created test dataset with {len(self.test_dset):,} events")

        if self.trainer.is_global_zero:
            print("-" * 80, "\n")

    def get_dataloader(self, stage: str, dataset: TrackMLDataset, shuffle: bool):  # noqa: ARG002
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
