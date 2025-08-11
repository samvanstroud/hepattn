from copy import deepcopy
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
    PANDORA = "Pandora"
    HGPFLOW = "hgpflow"
    HGPFLOW_PROXY = "hgpflow_proxy"
    HGPFLOW_TARGET = "hgpflow_target"
    MLPLF = "mlpf"
    MPFLOW = "mpflow"
    MPFLOW_PROXY = "mpflow_proxy"


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
        self.config = deepcopy(config)
        self.truth_dict, pandora_dict = load_truth_clic(config.truth_path)
        self.data = {}

        for net_config in config.networks:
            net_name = net_config.name
            pred_path = net_config.path
            num_events = net_config.num_events
            match net_config.network_type:
                case NetworkType.HGPFLOW | NetworkType.HGPFLOW_PROXY:
                    self.data[net_name] = load_pred_hgpflow(
                        pred_path,
                        threshold=net_config.ind_threshold,
                        num_events=num_events,
                        return_proxy=net_config.network_type == NetworkType.HGPFLOW_PROXY,
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
                case NetworkType.MPFLOW | NetworkType.MPFLOW_PROXY:
                    self.data[net_name] = load_pred_mpflow(
                        pred_path,
                        threshold=net_config.ind_threshold,
                        num_events=num_events,
                        return_proxy=net_config.network_type == NetworkType.MPFLOW_PROXY,
                    )
        # HACK: add pandora dict to data and network configs  # noqa: FIX004
        self.data["Pandora"] = pandora_dict
        self.config.networks.append(
            NetworkConfig(
                name="Pandora",
                path=config.truth_path,
                network_type=NetworkType.PANDORA,
            )
        )

        self.network_names = [net.name for net in self.config.networks]
        # Initialize flags
        self._events_reordered: bool
        self._jets_computed: bool
        self._jets_matched: bool
        self._particles_matched: bool
        self.n_events: int = 0
        self.reset()

    def reset(self):
        self._events_reordered = False
        self._jets_computed = False
        self._jets_matched = False
        self._particles_matched = False

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
        self.n_events = len(self.common_event_numbers)
        self._events_reordered = True

    def compute(self):
        """Compute jets, match jets, and compute event features."""
        assert self._events_reordered, "Events must be reordered before computing jets."
        self.compute_jets()
        self.hung_match_jets()
        self.hung_match_particles(flatten=True, return_unmatched=False)
        self.compute_event_features()
        self.compute_jet_res_features()

    def compute_jets(self, radius=0.7, algo="genkt", n_procs=0):
        assert self._events_reordered, "Events must be reordered before computing jets."
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

        for net_config in self.config.networks:
            net_name = net_config.name
            net_dict = self.data[net_name]
            net_type = net_config.network_type
            print(f"Computing jets for {net_name}...")
            kwargs = {}
            if net_type == NetworkType.PANDORA:
                kwargs["fourths"] = net_dict["e"]
                kwargs["fourth_name"] = "E"
            else:
                kwargs["fourths"] = net_dict["mass"]
                kwargs["fourth_name"] = "mass"
            net_dict["jets"] = compute_jets(
                jet_obj,
                net_dict["pt"],
                net_dict["eta"],
                net_dict["phi"],
                n_procs=n_procs,
                **kwargs,
            )
        self._jets_computed = True

    def hung_match_jets(
        self,
    ):
        """Match truth jets with the PF jets."""
        assert self._jets_computed, "Jets must be computed before matching."

        for net_dict in self.data.values():
            net_dict["matched_jets"] = match_jets_all_ev(self.truth_dict["truth_jets"], net_dict["jets"])
        self._jets_matched = True

    def hung_match_particles(self, flatten=False, return_unmatched=False):
        """Match truth particles with the PF particles."""
        assert self._events_reordered, "Events must be reordered before matching particles."
        for net_config in self.config.networks:
            net_name = net_config.name
            net_dict = self.data[net_name]
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
        self._particles_matched = True

    def compute_met_ht(self, pt, phi):
        """Calculate missing transverse energy (MET) and total transverse energy (HT)."""
        met_x = np.zeros(self.n_events)
        met_y = np.zeros(self.n_events)
        met = np.zeros(self.n_events)
        ht = np.zeros(self.n_events)
        for i in range(self.n_events):
            met_x[i] = -np.sum(pt[i] * np.cos(phi[i]))
            met_y[i] = -np.sum(pt[i] * np.sin(phi[i]))
            met[i] = np.sqrt(met_x[i] ** 2 + met_y[i] ** 2)
            ht[i] = np.sum(pt[i])
        return met_x, met_y, met, ht

    def compute_nconst(self, particle_class):
        """Calculate the number of constituents."""
        nconst_ch = np.zeros(self.n_events)
        nconst_neut = np.zeros(self.n_events)
        for i in range(self.n_events):
            nconst_ch[i] = np.sum(particle_class[i] <= 2)  # charged particles
            nconst_neut[i] = np.sum(particle_class[i] > 2)  # neutral particles
        return nconst_ch, nconst_neut

    def compute_event_features(self):
        """Calculate event features."""
        assert self._events_reordered, "Events must be reordered before calculating event features."
        # Truth features
        truth_met_x, truth_met_y, truth_met, truth_ht = self.compute_met_ht(
            self.truth_dict["particle_pt"],
            self.truth_dict["particle_phi"],
        )
        truth_nconst_ch, truth_nconst_neut = self.compute_nconst(self.truth_dict["particle_class"])
        self.truth_dict["met_x"] = truth_met_x
        self.truth_dict["met_y"] = truth_met_y
        self.truth_dict["met"] = truth_met
        self.truth_dict["ht"] = truth_ht
        self.truth_dict["nconst_ch"] = truth_nconst_ch
        self.truth_dict["nconst_neut"] = truth_nconst_neut

        # Networks features
        for net_config in self.config.networks:
            net_name = net_config.name
            met_x, met_y, met, ht = self.compute_met_ht(
                self.data[net_name]["pt"],
                self.data[net_name]["phi"],
            )
            nconst_ch, nconst_neut = self.compute_nconst(self.data[net_name]["class"])
            self.data[net_name]["met_x"] = met_x
            self.data[net_name]["met_y"] = met_y
            self.data[net_name]["met"] = met
            self.data[net_name]["ht"] = ht
            self.data[net_name]["nconst_ch"] = nconst_ch
            self.data[net_name]["nconst_neut"] = nconst_neut
            for var in ["met_x", "met_y", "met", "ht", "nconst_ch", "nconst_neut"]:
                if var in {"nconst_ch", "nconst_neut"}:
                    self.data[net_name][f"{var}_res"] = self.data[net_name][var] - self.truth_dict[var]
                else:
                    self.data[net_name][f"{var}_res"] = (self.data[net_name][var] - self.truth_dict[var]) / (
                        self.truth_dict[var] + 1e-8
                    )  # Avoid division by zero

    def compute_jet_residual_dict(self, ref_jets, reco_jets, dr_cut=0.1, leading_n_jets=999, pt_min=10, eta_max=2.5):
        """Args:
        matched_jets: {name: (truth, reco), ...].
        """
        residual_dict = {
            "pt": [],
            "pt_rel": [],
            "eta": [],
            "phi": [],
            "dR": [],
            "ref_pt": [],
            "ref_eta": [],
            "nconst": [],
            "e": [],
            "e_rel": [],
            "ref_e": [],
        }
        ref_count = 0
        matched_count = 0
        for ev_i in range(len(ref_jets)):
            ref_jets_ev, reco_jets_ev = ref_jets[ev_i], reco_jets[ev_i]
            for j_i, (ref_j, reco_j) in enumerate(zip(ref_jets_ev, reco_jets_ev, strict=False)):
                dr = ref_j.delta_r(reco_j)
                if dr < dr_cut and ref_j.pt > pt_min and abs(ref_j.eta) < eta_max:
                    residual_dict["pt"].append(reco_j.pt - ref_j.pt)
                    residual_dict["pt_rel"].append(residual_dict["pt"][-1] / ref_j.pt)
                    residual_dict["e"].append(reco_j.e - ref_j.e)
                    residual_dict["e_rel"].append(residual_dict["e"][-1] / ref_j.e)
                    residual_dict["eta"].append(reco_j.eta - ref_j.eta)
                    residual_dict["phi"].append(reco_j.phi - ref_j.phi)
                    residual_dict["dR"].append(dr)
                    residual_dict["ref_pt"].append(ref_j.pt)
                    residual_dict["ref_eta"].append(ref_j.eta)
                    residual_dict["ref_e"].append(ref_j.e)
                    residual_dict["nconst"].append(reco_j.n_constituents - ref_j.n_constituents)
                    matched_count += 1
                ref_count += 1

                if j_i == leading_n_jets - 1:
                    break

        f_matched = matched_count / ref_count
        residual_dict["f_matched"] = f_matched

        for var in residual_dict:
            residual_dict[var] = np.array(residual_dict[var])

        return residual_dict

    def compute_jet_res_features(self, dr_cut=0.1, leading_n_jets=999, pt_min=10, eta_max=2.5):
        assert self._jets_matched, "Jets must be matched before computing residuals."
        """Calculate jet residual features."""
        for net in self.config.networks:
            net_name = net.name
            self.data[net_name]["jet_residuals"] = self.compute_jet_residual_dict(
                self.data[net_name]["matched_jets"][0],
                self.data[net_name]["matched_jets"][1],
                dr_cut=dr_cut,
                leading_n_jets=leading_n_jets,
                pt_min=pt_min,
                eta_max=eta_max,
            )
