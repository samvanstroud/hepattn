from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import yaml
from tqdm import tqdm

from .jet_helper import JetHelper, compute_jets
from .matching import match_jets_all_ev, match_particles_all_ev
from .reader import (
    load_hgpflow_target,
    load_pred_hgpflow,
    load_pred_mlpf,
    load_pred_mpflow,
    load_truth_clic,
)


class NetworkType(Enum):
    HGPFLOW = "hgpflow"
    HGPFLOW_TARGET = "hgpflow_target"
    MLPLF = "mlpf"
    MPFLOW = "mpflow"


@dataclass
class NetworkConfig:
    name: str
    path: str | Path
    network_type: NetworkType
    num_events: int | None = None
    ind_threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Network path {self.path} does not exist.")
        if self.num_events is not None and self.num_events < 0:
            raise ValueError("num_events must be a non-negative integer or None.")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            path=data["path"],
            network_type=NetworkType(data["network_type"]),
            num_events=data.get("num_events"),
            ind_threshold=data.get("ind_threshold", 0.5),
        )


@dataclass
class PerformanceConfig:
    truth_path: str | Path
    networks: list[NetworkConfig]

    def __post_init__(self):
        if isinstance(self.truth_path, str):
            self.truth_path = Path(self.truth_path)
        if not self.truth_path.exists():
            raise FileNotFoundError(f"Truth path {self.truth_path} does not exist.")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        networks = [NetworkConfig.from_dict(net) for net in data["networks"]]
        return cls(
            truth_path=data["truth_path"],
            networks=networks,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Self:
        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML configuration file {yaml_path} does not exist.")
        with yaml_path.open() as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)


class Performance:
    def __init__(
        self,
        config: PerformanceConfig,
    ):
        self.config = config
        self.truth_dict = load_truth_clic(config.truth_path)

        self.data = {}
        for net_config in config.networks:
            net_name = net_config.name
            pred_path = net_config.path
            num_events = net_config.num_events
            match net_config.network_type:
                case NetworkType.HGPFLOW:
                    self.data[net_name] = load_pred_hgpflow(
                        pred_path,
                        threshold=net_config.ind_threshold,
                        num_events=num_events,
                    )
                case NetworkType.HGPFLOW_TARGET:
                    self.data[net_name] = load_hgpflow_target(
                        pred_path,
                        threshold=net_config.ind_threshold,
                        num_events=num_events,
                    )
                case NetworkType.MLPLF:
                    self.data[net_name] = load_pred_mlpf(
                        pred_path,
                    )
                case NetworkType.MPFLOW:
                    self.data[net_name] = load_pred_mpflow(
                        pred_path,
                        threshold=net_config.ind_threshold,
                        num_events=num_events,
                    )

    def reorder_and_find_intersection(self):
        self.common_event_numbers = self.truth_dict["event_number"]
        for net_dict in self.data.values():
            self.common_event_numbers = np.intersect1d(self.common_event_numbers, net_dict["event_number"])

        print("common event count:", len(self.common_event_numbers))

        # order them according to truth (we don't need to order self.truth_dict then)
        truth_mask = np.isin(self.truth_dict["event_number"], self.common_event_numbers)
        self.common_event_numbers = self.truth_dict["event_number"][truth_mask]

        # filter truth
        mask = np.isin(self.truth_dict["event_number"], self.common_event_numbers)
        if not mask.all():
            for var in tqdm(
                self.truth_dict.keys(),
                desc="Filtering truth...",
                total=len(self.truth_dict.keys()),
            ):
                self.truth_dict[var] = self.truth_dict[var][mask]

        # filter and reorder networks
        for net_name, net_dict in self.data.items():
            positions = np.array([np.where(net_dict["event_number"] == x)[0][0] for x in self.common_event_numbers]).astype(int)
            for var in tqdm(
                net_dict.keys(),
                desc=f"Filtering and reordering {net_name}...",
                total=len(net_dict.keys()),
            ):
                net_dict[var] = net_dict[var][positions]

    def compute_jets(self, radius=0.7, algo="genkt", n_procs=0):
        jet_obj = JetHelper(radius=radius, algo=algo)

        print("truth")
        self.truth_dict["truth_jets"] = compute_jets(
            jet_obj,
            self.truth_dict["particle_pt"],
            self.truth_dict["particle_eta"],
            self.truth_dict["particle_phi"],
            self.truth_dict["particle_e"],
            fourth_name="E",
            n_procs=n_procs,
        )

        print("pandora")
        self.truth_dict["pandora_jets"] = compute_jets(
            jet_obj,
            self.truth_dict["pandora_pt"],
            self.truth_dict["pandora_eta"],
            self.truth_dict["pandora_phi"],
            self.truth_dict["pandora_e"],
            fourth_name="E",
            n_procs=n_procs,
        )

        for net_config in self.config.networks:
            net_name = net_config.name
            net_dict = self.data[net_name]
            print(f"Computing jets for {net_name}...")
            net_dict["jets"] = compute_jets(
                jet_obj,
                net_dict["pt"],
                net_dict["eta"],
                net_dict["phi"],
                net_dict["mass"],
                fourth_name="mass",
                n_procs=n_procs,
            )
            if net_config.network_type in {NetworkType.HGPFLOW, NetworkType.MPFLOW}:
                print(f"Computing proxy jets for {net_name}...")
                net_dict["proxy_jets"] = compute_jets(
                    jet_obj,
                    net_dict["proxy_pt"],
                    net_dict["proxy_eta"],
                    net_dict["proxy_phi"],
                    net_dict["mass"],
                    fourth_name="mass",
                    n_procs=n_procs,
                )

    def hung_match_jets(
        self,
    ):
        """Match truth jets with the PF jets."""
        self.truth_dict["matched_pandora_jets"] = match_jets_all_ev(self.truth_dict["self.truth_jets"], self.truth_dict["pandora_jets"])
        for net_dict in self.data.values():
            net_dict["matched_jets"] = match_jets_all_ev(self.truth_dict["self.truth_jets"], net_dict["jets"])
            if "proxy_jets" in net_dict:
                net_dict["matched_proxy_jets"] = match_jets_all_ev(self.truth_dict["self.truth_jets"], net_dict["proxy_jets"])

    def hung_match_particles(self, flatten=False, return_unmatched=False):
        """Match truth particles with the PF particles."""
        for net_config in self.config.networks:
            net_name = net_config.name
            net_dict = self.data[net_name]
            if net_config.network_type in {NetworkType.HGPFLOW, NetworkType.MPFLOW}:
                net_dict["matched_proxy_particles"] = match_particles_all_ev(
                    (
                        self.truth_dict["particle_pt"],
                        self.truth_dict["particle_eta"],
                        self.truth_dict["particle_phi"],
                        self.truth_dict["particle_class"],
                    ),
                    (
                        net_dict["proxy_pt"],
                        net_dict["proxy_eta"],
                        net_dict["proxy_phi"],
                        net_dict["class"],
                    ),
                    flatten,
                    return_unmatched,
                )
            net_dict["matched_particles"] = match_particles_all_ev(
                (
                    self.truth_dict["particle_pt"],
                    self.truth_dict["particle_eta"],
                    self.truth_dict["particle_phi"],
                    self.truth_dict["particle_class"],
                ),
                (
                    net_dict["pt"],
                    net_dict["eta"],
                    net_dict["phi"],
                    net_dict["class"],
                ),
                flatten,
                return_unmatched,
            )
        if "pandora_pt" in self.truth_dict:
            self.truth_dict["matched_pandora_particles"] = match_particles_all_ev(
                (
                    self.truth_dict["particle_pt"],
                    self.truth_dict["particle_eta"],
                    self.truth_dict["particle_phi"],
                    self.truth_dict["particle_class"],
                ),
                (
                    self.truth_dict["pandora_pt"],
                    self.truth_dict["pandora_eta"],
                    self.truth_dict["pandora_phi"],
                    self.truth_dict["pandora_class"],
                ),
                flatten,
                return_unmatched,
            )
