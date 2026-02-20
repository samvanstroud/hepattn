from pathlib import Path
from collections import OrderedDict
from time import perf_counter

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class ODDDataset(Dataset):
    CALO_SUBSYSTEMS = ("ecb", "ece", "hcb", "hce")
    CALO_SUBSYSTEM_DETECTOR_IDS = {
        "ecb": np.array([10], dtype=np.int64),
        "ece": np.array([9, 11], dtype=np.int64),
        "hcb": np.array([13], dtype=np.int64),
        "hce": np.array([12, 14], dtype=np.int64),
    }
    CALO_ECAL_DETECTOR_IDS = np.array([9, 10, 11], dtype=np.int64)
    CALO_HCAL_DETECTOR_IDS = np.array([12, 13, 14], dtype=np.int64)
    CALO_SUBSYSTEM_CALIBRATION = {
        "ecb": 37.5,
        "ece": 38.7,
        "hcb": 45.0,
        "hce": 46.9,
    }
    CALO_GROUP_DETECTOR_IDS = {
        "ecalhits": CALO_ECAL_DETECTOR_IDS,
        "hcalhits": CALO_HCAL_DETECTOR_IDS,
    }
    PARTICLE_HIT_CUT_CLASS_TO_MASK = {
        "all": "particle_valid",
        "charged_hadron": "particle_is_charged_hadron",
        "neutral_hadron": "particle_is_neutral_hadron",
        "electron": "particle_is_electron",
        "photon": "particle_is_photon",
        "muon": "particle_is_muon",
        "tau": "particle_is_tau",
        "neutrino": "particle_is_neutrino",
        "other": "particle_is_other",
    }
    PARTICLE_HIT_CUT_KEYS = (
        "min_num_sihit",
        "min_num_ecal",
        "min_num_hcal",
    )
    PARTICLE_HIT_CUT_DEFAULTS = {
        "min_num_sihit": 0,
        "min_num_ecal": 0,
        "min_num_hcal": 0,
    }

    def __init__(
        self,
        dirpath: str,
        num_events: int = -1,
        particle_min_pt: float = 0.5,
        particle_max_abs_eta: float = 4.0,
        particle_hit_cuts: dict[str, dict[str, int]] | None = None,
        particle_include_charged: bool = True,
        particle_include_neutral: bool = True,
        return_calohits: bool = True,
        return_tracks: bool = True,
        event_type: str = "ttbar",
        build_calohit_associations: bool = True,
        debug: bool = False,
    ):
        super().__init__()

        self.dirpath = Path(dirpath)
        dataset_prefix = f"{event_type}_pu200"
        self.collection_dirs = {
            "particles": self.dirpath / f"{dataset_prefix}_particles",
            "tracker_hits": self.dirpath / f"{dataset_prefix}_tracker_hits",
            "calo_hits": self.dirpath / f"{dataset_prefix}_calo_hits",
            "tracks": self.dirpath / f"{dataset_prefix}_tracks",
        }

        self.particle_hit_cuts = self._normalize_particle_hit_cuts(particle_hit_cuts)
        self._requires_calohits_for_hit_cuts = any(
            cuts["min_num_ecal"] > 0 or cuts["min_num_hcal"] > 0
            for cuts in self.particle_hit_cuts.values()
        )

        required_collections = {"particles", "tracker_hits"}
        if return_calohits or self._requires_calohits_for_hit_cuts:
            required_collections.add("calo_hits")
        if return_tracks:
            required_collections.add("tracks")

        missing_dirs = [name for name in sorted(required_collections) if not self.collection_dirs[name].is_dir()]
        if missing_dirs:
            msg = (
                f"Missing required dataset directories for '{event_type}': {missing_dirs}. "
                f"Expected these under {self.dirpath}."
            )
            raise ValueError(msg)

        # Use particle shards as the reference and keep only shard names that are
        # available in every required collection.
        shared_shard_names = self._get_shared_shard_names(required_collections)
        self.particle_shard_paths = [self.collection_dirs["particles"] / name for name in shared_shard_names]
        if not self.particle_shard_paths:
            msg = f"No shared parquet shards found for '{event_type}' under {self.dirpath}"
            raise ValueError(msg)

        # Cache parquet row-group metadata and a few decoded row-groups.
        # Must be initialised before counting events in shards.
        self._row_group_starts: dict[str, np.ndarray] = {}
        self._row_group_cache: OrderedDict[tuple[str, int], ak.Array] = OrderedDict()
        self._row_group_cache_size = 8

        # Build a dense sample index: each sample_id maps to (shard index, row/event index in shard).
        self.sample_index = []
        for shard_idx, particle_path in enumerate(self.particle_shard_paths):
            num_events_in_shard = self._get_num_events_in_shard(particle_path)
            self.sample_index.extend((shard_idx, event_idx) for event_idx in range(num_events_in_shard))

        num_events_available = len(self.sample_index)

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
        self.num_events = num_events
        self.sample_ids = list(range(num_events))

        # Particle level cuts
        self.particle_min_pt = particle_min_pt
        self.particle_max_abs_eta = particle_max_abs_eta
        self.particle_include_charged = particle_include_charged
        self.particle_include_neutral = particle_include_neutral

        # Whether to return calo/ACTS track collections
        self.return_calohits = return_calohits
        self.return_tracks = return_tracks
        self.event_type = event_type
        self.build_calohit_associations = build_calohit_associations
        self.debug = debug

    @staticmethod
    def _coerce_non_negative_int(name: str, value) -> int:
        if isinstance(value, bool):
            msg = f"Cut '{name}' must be numeric, got bool."
            raise TypeError(msg)
        as_float = float(value)
        if as_float < 0:
            msg = f"Cut '{name}' must be non-negative, got {value}."
            raise ValueError(msg)
        as_int = int(as_float)
        if as_int != as_float:
            msg = f"Cut '{name}' must be an integer-like value, got {value}."
            raise ValueError(msg)
        return as_int

    @classmethod
    def _normalize_particle_hit_cuts(cls, particle_hit_cuts: dict[str, dict[str, int]] | None) -> dict[str, dict[str, int]]:
        if particle_hit_cuts is None:
            return {}
        if not isinstance(particle_hit_cuts, dict):
            msg = "particle_hit_cuts must be a dict like {'electron': {'min_num_sihit': 6, ...}}."
            raise TypeError(msg)

        normalized: dict[str, dict[str, int]] = {}
        for particle_type, cut_cfg in particle_hit_cuts.items():
            class_key = str(particle_type).strip().lower()
            if class_key not in cls.PARTICLE_HIT_CUT_CLASS_TO_MASK:
                valid = ", ".join(sorted(set(cls.PARTICLE_HIT_CUT_CLASS_TO_MASK)))
                msg = f"Unknown particle_hit_cuts class '{particle_type}'. Valid classes: {valid}"
                raise ValueError(msg)
            if not isinstance(cut_cfg, dict):
                msg = f"particle_hit_cuts['{particle_type}'] must be a dict."
                raise TypeError(msg)

            mask_key = cls.PARTICLE_HIT_CUT_CLASS_TO_MASK[class_key]
            if mask_key not in normalized:
                normalized[mask_key] = cls.PARTICLE_HIT_CUT_DEFAULTS.copy()

            for cut_name, cut_value in cut_cfg.items():
                cut_key = str(cut_name).strip().lower()
                if cut_key not in cls.PARTICLE_HIT_CUT_KEYS:
                    valid = ", ".join(sorted(cls.PARTICLE_HIT_CUT_KEYS))
                    msg = f"Unknown cut key '{cut_name}' for '{particle_type}'. Valid keys: {valid}"
                    raise ValueError(msg)
                cut_threshold = cls._coerce_non_negative_int(cut_name, cut_value)
                normalized[mask_key][cut_key] = max(normalized[mask_key][cut_key], cut_threshold)

        return {
            mask_key: cfg
            for mask_key, cfg in normalized.items()
            if any(cfg[cut_key] > 0 for cut_key in cls.PARTICLE_HIT_CUT_KEYS)
        }

    def __len__(self):
        return int(self.num_events)

    def _get_shared_shard_names(self, required_collections: set[str]) -> list[str]:
        shared_names = {path.name for path in self.collection_dirs["particles"].glob("*.parquet")}
        for collection in required_collections:
            if collection == "particles":
                continue
            shared_names &= {path.name for path in self.collection_dirs[collection].glob("*.parquet")}
        return sorted(shared_names)

    def _get_row_group_starts(self, path: Path) -> np.ndarray:
        path_key = str(path)
        if path_key in self._row_group_starts:
            return self._row_group_starts[path_key]

        parquet_file = pq.ParquetFile(path)
        num_row_groups = parquet_file.metadata.num_row_groups
        row_group_sizes = np.fromiter(
            (parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups)),
            dtype=np.int64,
            count=num_row_groups,
        )

        row_group_starts = np.zeros(num_row_groups + 1, dtype=np.int64)
        row_group_starts[1:] = np.cumsum(row_group_sizes)
        self._row_group_starts[path_key] = row_group_starts
        return row_group_starts

    def _get_num_events_in_shard(self, path: Path) -> int:
        row_group_starts = self._get_row_group_starts(path)
        return int(row_group_starts[-1])

    def _get_collection_path(self, collection: str, particle_path: Path) -> Path:
        return self.collection_dirs[collection] / particle_path.name

    def _get_row_group_lookup(self, path: Path, event_idx: int) -> tuple[int, int]:
        row_group_starts = self._get_row_group_starts(path)
        num_rows = int(row_group_starts[-1])
        if event_idx < 0 or event_idx >= num_rows:
            msg = f"Event index {event_idx} is out of range [0, {num_rows - 1}] for {path}"
            raise ValueError(msg)

        row_group_idx = int(np.searchsorted(row_group_starts, event_idx, side="right") - 1)
        row_in_group_idx = int(event_idx - row_group_starts[row_group_idx])
        return row_group_idx, row_in_group_idx

    def _get_row_group_array(self, path: Path, row_group_idx: int) -> ak.Array:
        key = (str(path), row_group_idx)
        if key in self._row_group_cache:
            self._row_group_cache.move_to_end(key)
            return self._row_group_cache[key]

        row_group_array = ak.from_parquet(path, row_groups=[row_group_idx])
        self._row_group_cache[key] = row_group_array
        self._row_group_cache.move_to_end(key)

        if len(self._row_group_cache) > self._row_group_cache_size:
            self._row_group_cache.popitem(last=False)

        return row_group_array

    def _read_event_from_file(self, path: Path, event_idx: int) -> ak.Record:
        row_group_idx, row_in_group_idx = self._get_row_group_lookup(path, event_idx)
        row_group_array = self._get_row_group_array(path, row_group_idx)
        return row_group_array[row_in_group_idx]

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[ODDDataset] {message}", flush=True)

    @staticmethod
    def _record_to_tensors(
        record: ak.Record,
        prefix: str,
        *,
        int_fields: set[str] | None = None,
        skip_fields: set[str] | None = None,
        default_dtype: torch.dtype | None = torch.float32,
    ) -> dict[str, torch.Tensor]:
        int_fields = set() if int_fields is None else set(int_fields)
        skip_fields = set() if skip_fields is None else set(skip_fields)
        tensors: dict[str, torch.Tensor] = {}

        for field in record.fields:
            if field in skip_fields:
                continue

            tensor = ak.to_torch(record[field])
            if field in int_fields:
                tensor = tensor.to(torch.int64)
            elif default_dtype is not None:
                tensor = tensor.to(default_dtype)
            tensors[f"{prefix}_{field}"] = tensor

        return tensors

    @staticmethod
    def _scale_xyz_inplace(tensors: dict[str, torch.Tensor], prefix: str, scale: float) -> None:
        for axis in ("x", "y", "z"):
            key = f"{prefix}_{axis}"
            if key in tensors:
                tensors[key] = tensors[key] * scale

    @staticmethod
    def _add_cylindrical_coords_inplace(tensors: dict[str, torch.Tensor], prefix: str) -> None:
        x = tensors[f"{prefix}_x"]
        y = tensors[f"{prefix}_y"]
        z = tensors[f"{prefix}_z"]
        r = torch.sqrt(x**2 + y**2)
        s = torch.sqrt(r**2 + z**2).clamp_min(1e-12)
        cos_theta = (z / s).clamp(-1.0, 1.0)

        tensors[f"{prefix}_r"] = r
        tensors[f"{prefix}_s"] = s
        tensors[f"{prefix}_eta"] = torch.arctanh(z / s)
        tensors[f"{prefix}_theta"] = torch.arccos(cos_theta)
        tensors[f"{prefix}_phi"] = torch.arctan2(y, x)

    @staticmethod
    def _valid_mask_like(values: torch.Tensor) -> torch.Tensor:
        return torch.full_like(values, True, dtype=torch.bool)

    @staticmethod
    def _apply_mask_to_prefixed_keys(
        tensors: dict[str, torch.Tensor],
        prefix: str,
        mask: torch.Tensor,
        *,
        skip_prefixes: tuple[str, ...] = (),
    ) -> None:
        for key in list(tensors.keys()):
            if not key.startswith(prefix):
                continue
            if any(key.startswith(skip) for skip in skip_prefixes):
                continue
            tensors[key] = tensors[key][mask]

    @staticmethod
    def _lookup_row_indices(reference_ids: np.ndarray, query_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if reference_ids.size == 0 or query_ids.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(query_ids.size, dtype=bool)

        sort_idx = np.argsort(reference_ids)
        sorted_reference_ids = reference_ids[sort_idx]

        search_idx = np.searchsorted(sorted_reference_ids, query_ids)
        in_bounds = search_idx < sorted_reference_ids.size
        matched = np.zeros(query_ids.shape[0], dtype=bool)
        matched[in_bounds] = sorted_reference_ids[search_idx[in_bounds]] == query_ids[in_bounds]

        if not np.any(matched):
            return np.zeros(0, dtype=np.int64), matched

        row_indices = sort_idx[search_idx[matched]]
        return row_indices, matched

    @staticmethod
    def _build_csr_components(
        num_rows: int,
        num_cols: int,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        col_indices = np.asarray(col_indices, dtype=np.int64)

        if row_indices.size == 0:
            indptr = np.zeros(num_rows + 1, dtype=np.int64)
            indices = np.zeros(0, dtype=np.int64)
            shape = np.array([num_rows, num_cols], dtype=np.int64)
            return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(shape)

        # Build canonical CSR with optimized sparse backend:
        # deduplicates duplicate entries and stores row-sorted indices.
        values = np.ones(row_indices.shape[0], dtype=np.uint8)
        csr = sp.coo_matrix((values, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.uint8).tocsr()
        csr.sum_duplicates()
        indptr = np.array(csr.indptr, dtype=np.int64, copy=True)
        indices = np.array(csr.indices, dtype=np.int64, copy=True)
        shape = np.array([num_rows, num_cols], dtype=np.int64)
        return torch.from_numpy(indptr), torch.from_numpy(indices), torch.from_numpy(shape)

    @staticmethod
    def _empty_csr_components(num_rows: int, num_cols: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        empty = np.zeros(0, dtype=np.int64)
        return ODDDataset._build_csr_components(num_rows, num_cols, empty, empty)

    @staticmethod
    def _float32_bincount(
        indices: np.ndarray,
        *,
        minlength: int,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        counts = np.bincount(indices, weights=weights, minlength=minlength)
        return counts.astype(np.float32, copy=False)

    @staticmethod
    def _get_calohit_detector_ids(calohits: ak.Record) -> np.ndarray:
        return np.asarray(ak.to_numpy(calohits["detector"]), dtype=np.int64).reshape(-1)

    def _build_particle_sihit_csr(
        self,
        particle_ids: torch.Tensor,
        sihit_particle_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        num_sihits = int(sihit_particle_ids.numel())
        if num_particles == 0 or num_sihits == 0:
            return self._empty_csr_components(num_particles, num_sihits)

        particle_ids_np = particle_ids.cpu().numpy()
        sihit_particle_ids_np = sihit_particle_ids.cpu().numpy()

        row_indices, matched = self._lookup_row_indices(particle_ids_np, sihit_particle_ids_np)
        if row_indices.size == 0:
            return self._empty_csr_components(num_particles, num_sihits)

        col_indices = np.nonzero(matched)[0]
        return self._build_csr_components(num_particles, num_sihits, row_indices, col_indices)

    def _build_particle_calohit_csr(self, particle_ids: torch.Tensor, calohits) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        num_calohits = int(len(calohits["contrib_particle_ids"]))
        if num_particles == 0 or num_calohits == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if contrib_counts.sum() == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        flat_contrib_particle_ids = ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1))
        calohit_indices = np.repeat(np.arange(num_calohits, dtype=np.int64), contrib_counts)

        particle_ids_np = particle_ids.cpu().numpy()
        row_indices, matched = self._lookup_row_indices(particle_ids_np, flat_contrib_particle_ids)
        if row_indices.size == 0:
            return self._empty_csr_components(num_particles, num_calohits)

        col_indices = calohit_indices[matched]
        return self._build_csr_components(num_particles, num_calohits, row_indices, col_indices)

    def _build_track_sihit_csr(self, track_hit_ids, num_sihits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tracks = int(len(track_hit_ids))
        if num_tracks == 0 or num_sihits == 0:
            return self._empty_csr_components(num_tracks, num_sihits)

        row_counts = ak.to_numpy(ak.num(track_hit_ids, axis=1))
        if row_counts.sum() == 0:
            return self._empty_csr_components(num_tracks, num_sihits)

        row_indices = np.repeat(np.arange(num_tracks, dtype=np.int64), row_counts)
        col_indices = ak.to_numpy(ak.flatten(track_hit_ids, axis=1)).astype(np.int64, copy=False)

        valid = (col_indices >= 0) & (col_indices < num_sihits)
        row_indices = row_indices[valid]
        col_indices = col_indices[valid]
        return self._build_csr_components(num_tracks, num_sihits, row_indices, col_indices)

    def _build_particle_targets(self, particles: ak.Record) -> dict[str, torch.Tensor]:
        targets = self._record_to_tensors(
            particles,
            "particle",
            int_fields={"particle_id"},
            skip_fields={"parent_id"},
            default_dtype=torch.float32,
        )
        targets["particle_event_id"] = targets["particle_event_id"].expand_as(targets["particle_particle_id"])

        pt = torch.sqrt(targets["particle_px"] ** 2 + targets["particle_py"] ** 2)
        p = torch.sqrt(targets["particle_px"] ** 2 + targets["particle_py"] ** 2 + targets["particle_pz"] ** 2).clamp_min(1e-12)

        targets["particle_pt"] = pt
        targets["particle_p"] = p
        targets["particle_qopt"] = targets["particle_charge"] / pt.clamp_min(1e-6)
        targets["particle_eta"] = torch.arctanh(targets["particle_pz"] / p)
        targets["particle_theta"] = torch.arccos((targets["particle_pz"] / p).clamp(-1.0, 1.0))
        targets["particle_phi"] = torch.arctan2(targets["particle_py"], targets["particle_px"])
        targets["particle_d0"] = (
            -targets["particle_vx"] * targets["particle_py"] + targets["particle_vy"] * targets["particle_px"]
        ) / pt.clamp_min(1e-6)
        targets["particle_z0"] = targets["particle_vz"]
        targets["particle_charged"] = targets["particle_charge"] != 0
        targets["particle_neutral"] = ~targets["particle_charged"]
        self._add_particle_classes_inplace(targets)
        return targets

    @staticmethod
    def _add_particle_classes_inplace(targets: dict[str, torch.Tensor]) -> None:
        pdg_key = "particle_pdg_id" if "particle_pdg_id" in targets else "particle_pdgid"
        pdg_id = targets[pdg_key].to(torch.int64)
        abs_pdg_id = torch.abs(pdg_id)

        class_masks: dict[str, torch.Tensor] = {
            "is_photon": abs_pdg_id == 22,
            "is_electron": abs_pdg_id == 11,
            "is_muon": abs_pdg_id == 13,
            "is_tau": abs_pdg_id == 15,
            "is_neutrino": (abs_pdg_id == 12) | (abs_pdg_id == 14) | (abs_pdg_id == 16) | (abs_pdg_id == 18),
        }

        is_known_nonhadron = torch.zeros_like(abs_pdg_id, dtype=torch.bool)
        for mask in class_masks.values():
            is_known_nonhadron |= mask

        class_masks["is_charged_hadron"] = (~is_known_nonhadron) & targets["particle_charged"]
        class_masks["is_neutral_hadron"] = (~is_known_nonhadron) & targets["particle_neutral"]

        class_assignment = {
            "is_neutral_hadron": 0,
            "is_charged_hadron": 1,
            "is_photon": 3,
            "is_electron": 4,
            "is_muon": 5,
            "is_tau": 6,
            "is_neutrino": 7,
        }
        class_id = torch.full_like(abs_pdg_id, -1, dtype=torch.int64)
        for class_name, class_value in class_assignment.items():
            class_id[class_masks[class_name]] = class_value

        is_other = torch.ones_like(abs_pdg_id, dtype=torch.bool)
        for class_name in class_assignment:
            is_other &= ~class_masks[class_name]
        class_masks["is_other"] = is_other

        targets["particle_class"] = class_id
        for class_name, class_mask in class_masks.items():
            targets[f"particle_{class_name}"] = class_mask

    def _build_particle_kinematic_mask(self, targets: dict[str, torch.Tensor]) -> torch.Tensor:
        particle_valid = targets["particle_pt"] >= self.particle_min_pt
        particle_valid = particle_valid & (torch.abs(targets["particle_eta"]) <= self.particle_max_abs_eta)

        if not self.particle_include_neutral:
            particle_valid = particle_valid & (~targets["particle_neutral"])
        if not self.particle_include_charged:
            particle_valid = particle_valid & (~targets["particle_charged"])
        return particle_valid

    def _build_particle_hit_cut_mask(
        self,
        targets: dict[str, torch.Tensor],
        particle_num_sihits: torch.Tensor,
        particle_num_calo_hit_fields: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        particle_valid = torch.ones_like(targets["particle_valid"], dtype=torch.bool)
        num_ecalhits = (
            particle_num_calo_hit_fields["particle_num_ecalhits"]
            if particle_num_calo_hit_fields is not None
            else torch.zeros_like(particle_num_sihits)
        )
        num_hcalhits = (
            particle_num_calo_hit_fields["particle_num_hcalhits"]
            if particle_num_calo_hit_fields is not None
            else torch.zeros_like(particle_num_sihits)
        )

        for class_mask_key, cuts in self.particle_hit_cuts.items():
            class_mask = targets[class_mask_key].to(torch.bool)
            class_valid = torch.ones_like(class_mask)
            for cut_key, counts in (
                ("min_num_sihit", particle_num_sihits),
                ("min_num_ecal", num_ecalhits),
                ("min_num_hcal", num_hcalhits),
            ):
                threshold = cuts[cut_key]
                if threshold > 0:
                    class_valid &= counts >= threshold

            particle_valid &= (~class_mask) | class_valid

        return particle_valid

    def _build_particle_sihit_fields(
        self,
        particle_ids: torch.Tensor,
        sihit_particle_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indptr, indices, shape = self._build_particle_sihit_csr(
            particle_ids,
            sihit_particle_ids,
        )
        num_sihits = torch.diff(indptr).to(torch.float32)
        return indptr, indices, shape, num_sihits

    def _build_sihit_inputs(self, sihits: ak.Record) -> dict[str, torch.Tensor]:
        inputs = self._record_to_tensors(
            sihits,
            "sihit",
            int_fields={"particle_id"},
            default_dtype=torch.float32,
        )
        self._scale_xyz_inplace(inputs, "sihit", scale=1e-3)
        self._add_cylindrical_coords_inplace(inputs, "sihit")
        inputs["sihit_valid"] = self._valid_mask_like(inputs["sihit_x"])
        return inputs

    def _read_calohits(self, particle_path: Path, event_idx: int) -> ak.Record:
        t_calo_read = perf_counter()
        calohits_path = self._get_collection_path("calo_hits", particle_path)
        calohits = self._read_event_from_file(calohits_path, event_idx)
        self._debug(f"read calohits in {perf_counter() - t_calo_read:.3f}s")
        return calohits

    def _build_particle_num_calo_hit_fields(self, particle_ids: torch.Tensor, calohits: ak.Record) -> dict[str, torch.Tensor]:
        particle_calohit_indptr, particle_calohit_indices, _ = self._build_particle_calohit_csr(particle_ids, calohits)
        num_calohits = torch.diff(particle_calohit_indptr).to(torch.float32)
        num_particles = int(num_calohits.numel())
        if num_particles == 0 or particle_calohit_indices.numel() == 0:
            empty_counts = {
                f"particle_num_{group_name}": torch.zeros(num_particles, dtype=torch.float32)
                for group_name in self.CALO_GROUP_DETECTOR_IDS
            }
            return {
                "particle_num_calohits": num_calohits,
                **empty_counts,
            }

        indptr_np = particle_calohit_indptr.cpu().numpy().astype(np.int64, copy=False)
        indices_np = particle_calohit_indices.cpu().numpy().astype(np.int64, copy=False)
        row_indices = np.repeat(np.arange(num_particles, dtype=np.int64), np.diff(indptr_np))

        linked_detector_ids = self._get_calohit_detector_ids(calohits)[indices_np]
        grouped_counts = {
            f"particle_num_{group_name}": torch.from_numpy(
                self._float32_bincount(
                    row_indices[np.isin(linked_detector_ids, detector_ids)],
                    minlength=num_particles,
                )
            )
            for group_name, detector_ids in self.CALO_GROUP_DETECTOR_IDS.items()
        }

        return {
            "particle_num_calohits": num_calohits,
            **grouped_counts,
        }

    def _build_particle_num_calo_hit_fields_if_available(
        self,
        particle_ids: torch.Tensor,
        calohits: ak.Record | None,
    ) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None]:
        if calohits is None:
            return None, None

        fields = self._build_particle_num_calo_hit_fields(particle_ids, calohits)
        return fields, fields["particle_num_calohits"]

    def _build_particle_calo_energy_fields(self, particle_ids: torch.Tensor, calohits: ak.Record) -> dict[str, torch.Tensor]:
        num_particles = int(particle_ids.numel())
        particle_energy_raw_np = {
            key: np.zeros(num_particles, dtype=np.float32)
            for key in (*self.CALO_SUBSYSTEMS, "calo_sum")
        }

        contrib_counts = ak.to_numpy(ak.num(calohits["contrib_particle_ids"], axis=1))
        if num_particles > 0 and contrib_counts.sum() > 0:
            flat_contrib_particle_ids = ak.to_numpy(ak.flatten(calohits["contrib_particle_ids"], axis=1))
            flat_contrib_energies = ak.to_numpy(ak.flatten(calohits["contrib_energies"], axis=1)).astype(np.float32, copy=False)
            flat_contrib_detector = np.repeat(self._get_calohit_detector_ids(calohits), contrib_counts)

            particle_ids_np = particle_ids.cpu().numpy()
            row_indices, matched = self._lookup_row_indices(particle_ids_np, flat_contrib_particle_ids)
            if row_indices.size > 0:
                matched_energies = flat_contrib_energies[matched]
                matched_detectors = flat_contrib_detector[matched]
                particle_energy_raw_np["calo_sum"] = self._float32_bincount(
                    row_indices,
                    minlength=num_particles,
                    weights=matched_energies,
                )
                for subsystem, detector_ids in self.CALO_SUBSYSTEM_DETECTOR_IDS.items():
                    subsystem_mask = np.isin(matched_detectors, detector_ids)
                    if not np.any(subsystem_mask):
                        continue

                    particle_energy_raw_np[subsystem] = self._float32_bincount(
                        row_indices[subsystem_mask],
                        minlength=num_particles,
                        weights=matched_energies[subsystem_mask],
                    )

        particle_energy_raw = {key: torch.from_numpy(values) for key, values in particle_energy_raw_np.items()}
        particle_energy_raw["ecal"] = particle_energy_raw["ecb"] + particle_energy_raw["ece"]
        particle_energy_raw["hcal"] = particle_energy_raw["hcb"] + particle_energy_raw["hce"]

        particle_energy_calib = {
            f"{subsystem}_calib": particle_energy_raw[subsystem] * scale
            for subsystem, scale in self.CALO_SUBSYSTEM_CALIBRATION.items()
        }
        particle_energy_calib["ecal_calib"] = particle_energy_calib["ecb_calib"] + particle_energy_calib["ece_calib"]
        particle_energy_calib["hcal_calib"] = particle_energy_calib["hcb_calib"] + particle_energy_calib["hce_calib"]
        particle_energy_calib["calo_calib"] = particle_energy_calib["ecal_calib"] + particle_energy_calib["hcal_calib"]

        out = {
            f"particle_energy_{subsystem}": particle_energy_raw[subsystem]
            for subsystem in (*self.CALO_SUBSYSTEMS, "ecal", "hcal")
        }
        out["particle_energy_calo_sum"] = particle_energy_raw["calo_sum"]
        out.update({f"particle_energy_{name}": values for name, values in particle_energy_calib.items()})
        return out

    @staticmethod
    def _build_calohit_contrib_energy_sum(calohits: ak.Record) -> torch.Tensor:
        contrib_energy_sum = ak.to_numpy(
            ak.sum(calohits["contrib_energies"], axis=1, mask_identity=False)
        ).astype(np.float32, copy=False)
        return torch.from_numpy(contrib_energy_sum)

    def _add_calohits(
        self,
        inputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        particle_path: Path,
        event_idx: int,
        calohits: ak.Record | None = None,
    ) -> None:
        if calohits is None:
            calohits = self._read_calohits(particle_path, event_idx)

        inputs.update(
            self._record_to_tensors(
                calohits,
                "calohit",
                skip_fields={"detector", "contrib_particle_ids", "contrib_energies", "contrib_times"},
                default_dtype=torch.float32,
            )
        )
        inputs["calohit_detector"] = ak.to_torch(calohits["detector"]).to(torch.int64)
        inputs["calohit_contrib_energy_sum"] = self._build_calohit_contrib_energy_sum(calohits)
        self._scale_xyz_inplace(inputs, "calohit", scale=1e-3)
        inputs["calohit_valid"] = self._valid_mask_like(inputs["calohit_x"])
        targets["calohit_valid"] = inputs["calohit_valid"]
        targets.update(self._build_particle_num_calo_hit_fields(targets["particle_particle_id"], calohits))
        targets.update(self._build_particle_calo_energy_fields(targets["particle_particle_id"], calohits))

        if not self.build_calohit_associations:
            self._debug("skipping calo association build (build_calohit_associations=False)")
            return

        self._debug(
            f"building calo associations: n_particles={targets['particle_particle_id'].size(0)} "
            f"n_calohits={inputs['calohit_x'].size(0)}"
        )
        t_calo_assoc = perf_counter()
        particle_calohit_indptr, particle_calohit_indices, particle_calohit_shape = self._build_particle_calohit_csr(
            targets["particle_particle_id"],
            calohits,
        )
        self._debug(
            f"built calo CSR in {perf_counter() - t_calo_assoc:.3f}s "
            f"shape=({int(particle_calohit_shape[0])}, {int(particle_calohit_shape[1])}) nnz={particle_calohit_indices.numel()}"
        )

        targets["particle_calohit_indptr"] = particle_calohit_indptr
        targets["particle_calohit_indices"] = particle_calohit_indices
        targets["particle_calohit_shape"] = particle_calohit_shape

    def _add_tracks(self, inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], particle_path: Path, event_idx: int) -> None:
        t_track_read = perf_counter()
        tracks_path = self._get_collection_path("tracks", particle_path)
        tracks = self._read_event_from_file(tracks_path, event_idx)
        self._debug(f"read tracks in {perf_counter() - t_track_read:.3f}s")

        targets.update(
            self._record_to_tensors(
                tracks,
                "track",
                skip_fields={"hit_ids"},
                default_dtype=None,
            )
        )
        targets["track_valid"] = self._valid_mask_like(targets["track_phi"])

        t_track_assoc = perf_counter()
        track_sihit_indptr, track_sihit_indices, track_sihit_shape = self._build_track_sihit_csr(
            tracks["hit_ids"],
            num_sihits=len(inputs["sihit_valid"]),
        )
        self._debug(
            f"built track_sihit CSR in {perf_counter() - t_track_assoc:.3f}s "
            f"shape=({int(track_sihit_shape[0])}, {int(track_sihit_shape[1])}) nnz={track_sihit_indices.numel()}"
        )
        targets["track_sihit_indptr"] = track_sihit_indptr
        targets["track_sihit_indices"] = track_sihit_indices
        targets["track_sihit_shape"] = track_sihit_shape

    @staticmethod
    def _pad_particle_targets_inplace(targets: dict[str, torch.Tensor]) -> None:
        particle_keys = [
            k
            for k in targets
            if k.startswith("particle_") and not k.startswith("particle_sihit_") and not k.startswith("particle_calohit_")
        ]
        if not particle_keys:
            return

        pad_size = max(int(targets[k].size(0)) for k in particle_keys)
        for key in particle_keys:
            values = targets[key]
            extra = pad_size - int(values.size(0))
            if extra <= 0:
                continue

            pad_shape = (extra, *values.shape[1:])
            pad_value = False if values.dtype is torch.bool else 0
            targets[key] = torch.cat([values, values.new_full(pad_shape, pad_value)], dim=0)

    def load_event(self, sample_id):
        t0 = perf_counter()
        if sample_id < 0 or sample_id >= len(self.sample_index):
            msg = f"sample_id {sample_id} is out of range [0, {len(self.sample_index) - 1}]"
            raise ValueError(msg)

        shard_idx, event_idx = self.sample_index[sample_id]
        particle_path = self.particle_shard_paths[shard_idx]
        sihit_path = self._get_collection_path("tracker_hits", particle_path)
        self._debug(f"sample_id={sample_id} shard_idx={shard_idx} event_idx={event_idx} file={particle_path.name}")

        # Read in the data
        t_read = perf_counter()
        particles = self._read_event_from_file(particle_path, event_idx)
        sihits = self._read_event_from_file(sihit_path, event_idx)
        self._debug(f"read particles+sihits in {perf_counter() - t_read:.3f}s")

        targets = self._build_particle_targets(particles)
        particle_valid = self._build_particle_kinematic_mask(targets)
        self._apply_mask_to_prefixed_keys(targets, "particle_", particle_valid)
        self._debug(f"after kinematic cuts: n_particles={targets['particle_particle_id'].size(0)}")

        targets["particle_valid"] = self._valid_mask_like(targets["particle_pt"])
        inputs = self._build_sihit_inputs(sihits)
        targets["sihit_valid"] = inputs["sihit_valid"]

        t_assoc = perf_counter()
        particle_sihit_indptr, particle_sihit_indices, particle_sihit_shape, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )

        calohits = self._read_calohits(particle_path, event_idx) if self.return_calohits or self._requires_calohits_for_hit_cuts else None
        particle_num_calo_hit_fields, particle_num_calohits = self._build_particle_num_calo_hit_fields_if_available(
            targets["particle_particle_id"],
            calohits,
        )

        # Apply constituent based particle cuts
        particle_valid = self._build_particle_hit_cut_mask(
            targets,
            particle_num_sihits,
            particle_num_calo_hit_fields,
        )

        self._apply_mask_to_prefixed_keys(
            targets,
            "particle_",
            particle_valid,
            skip_prefixes=("particle_sihit_", "particle_calohit_"),
        )
        particle_sihit_indptr, particle_sihit_indices, particle_sihit_shape, particle_num_sihits = self._build_particle_sihit_fields(
            targets["particle_particle_id"],
            inputs["sihit_particle_id"],
        )
        particle_num_calo_hit_fields, particle_num_calohits = self._build_particle_num_calo_hit_fields_if_available(
            targets["particle_particle_id"],
            calohits,
        )
        if particle_num_calo_hit_fields is not None:
            targets.update(particle_num_calo_hit_fields)

        constituent_debug = (
            f"after constituent cuts: n_particles={targets['particle_particle_id'].size(0)} "
            f"n_sihits={inputs['sihit_particle_id'].size(0)}"
        )
        if particle_num_calohits is not None:
            constituent_debug += (
                f" n_calohits={int(particle_num_calohits.sum().item())}"
            )
            if particle_num_calo_hit_fields is not None:
                constituent_debug += (
                    f" n_ecalhits={int(particle_num_calo_hit_fields['particle_num_ecalhits'].sum().item())}"
                    f" n_hcalhits={int(particle_num_calo_hit_fields['particle_num_hcalhits'].sum().item())}"
                )
        self._debug(
            constituent_debug
        )

        self._debug(
            f"built particle_sihit CSR in {perf_counter() - t_assoc:.3f}s "
            f"shape=({int(particle_sihit_shape[0])}, {int(particle_sihit_shape[1])}) nnz={particle_sihit_indices.numel()}"
        )
        targets["particle_sihit_indptr"] = particle_sihit_indptr
        targets["particle_sihit_indices"] = particle_sihit_indices
        targets["particle_sihit_shape"] = particle_sihit_shape
        targets["particle_num_sihits"] = particle_num_sihits

        # Return the calorimeter hit info if requested
        if self.return_calohits:
            self._add_calohits(inputs, targets, particle_path, event_idx, calohits=calohits)

        # Return ACTS track info if requested
        if self.return_tracks:
            self._add_tracks(inputs, targets, particle_path, event_idx)

        # Add metadata
        targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)
        self._debug(f"load_event finished in {perf_counter() - t0:.3f}s")

        return inputs, targets

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())

        if idx < 0:
            idx += len(self.sample_ids)
        if idx < 0 or idx >= len(self.sample_ids):
            msg = f"Dataset index {idx} out of range for length {len(self.sample_ids)}"
            raise IndexError(msg)

        sample_id = self.sample_ids[idx]
        inputs, targets = self.load_event(sample_id)

        self._pad_particle_targets_inplace(targets)

        # Add dummy batch dimension
        # TODO: Support batch size > 1
        inputs_out = {k: v.unsqueeze(0) for k, v in inputs.items()}
        targets_out = {k: v.unsqueeze(0) for k, v in targets.items()}

        return inputs_out, targets_out


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
            self.train_dset = ODDDataset(dirpath=self.train_dir, num_events=self.num_train, **self.kwargs)

        if stage == "fit":
            self.val_dset = ODDDataset(dirpath=self.val_dir, num_events=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit" and self.trainer.is_global_zero:
            print(f"Created training dataset with {len(self.train_dset):,} events")
            print(f"Created validation dataset with {len(self.val_dset):,} events")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dset = ODDDataset(dirpath=self.test_dir, num_events=self.num_test, **self.kwargs)
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
