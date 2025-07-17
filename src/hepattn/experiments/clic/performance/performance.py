from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .jet_helper import JetHelper, compute_jets
from .reader import (
    load_hgpflow_target,
    load_pred_hgpflow,
    load_pred_mlpf,
    load_pred_mpflow,
    load_truth_clic,
)
from .utils import delta_r


class Performance:
    def __init__(
        self,
        truth_path: str,
        networks: dict[str, str],
        num_events: dict[str, int] | int | None = None,
        ind_threshold: float = 0.65,
        topo: bool = False,
        truth_event_number_offset: int = 0,
    ):
        self.topo = topo
        self.truth_event_number_offset = truth_event_number_offset
        self.truth_dict = load_truth_clic(truth_path, truth_event_number_offset)
        if num_events is None:
            num_events = defaultdict(lambda: None)
        elif isinstance(num_events, int):
            # if num_events is an int, we assume it is the same for all networks
            # and convert it to a dict with the network names as keys
            num_events = dict.fromkeys(networks.keys(), num_events)

        self.data = {}
        for net_name, pred_path in networks.items():
            if net_name == "hgpflow":
                self.data[net_name] = load_pred_hgpflow(pred_path, threshold=ind_threshold, num_events=num_events[net_name])
            elif net_name == "mlpf":
                self.data[net_name] = load_pred_mlpf(pred_path, truth_event_number_offset=truth_event_number_offset)
            elif net_name == "hgpflow_target":
                self.data[net_name] = load_hgpflow_target(pred_path, num_events=num_events[net_name])
            elif net_name == "mpflow":
                self.data[net_name] = load_pred_mpflow(pred_path, threshold=ind_threshold, num_events=num_events[net_name])
        # self.reorder_and_find_intersection()

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

        # filter and reorder hgpflow
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

        for net_name, net_dict in self.data.items():
            print(f"{net_name} jets")
            net_dict["jets"] = compute_jets(
                jet_obj,
                net_dict["pt"],
                net_dict["eta"],
                net_dict["phi"],
                net_dict["mass"],
                fourth_name="mass",
                n_procs=n_procs,
            )
            if net_name in {"hgpflow", "mpflow"}:
                net_dict["proxy_jets"] = compute_jets(
                    jet_obj,
                    net_dict["proxy_pt"],
                    net_dict["proxy_eta"],
                    net_dict["proxy_phi"],
                    net_dict["mass"],
                    fourth_name="mass",
                    n_procs=n_procs,
                )

    def match_jets_single_ev(self, ref_jets, comp_jets):
        n_ref_jets = len(ref_jets)
        n_comp_jets = len(comp_jets)

        if n_ref_jets == 0 or n_comp_jets == 0:
            return [[], []]

        dr_matrix = np.zeros((n_ref_jets, n_comp_jets))
        for i in range(n_ref_jets):
            for j in range(n_comp_jets):
                dr_matrix[i, j] = ref_jets[i].delta_r(comp_jets[j])

        row_indices, col_indices = linear_sum_assignment(dr_matrix, maximize=False)
        ref_jets_matched = [ref_jets[i] for i in row_indices]
        comp_jets_matched = [comp_jets[i] for i in col_indices]

        # sort both by pt of ref_jet
        sorted_idx = np.argsort([j.pt for j in ref_jets_matched])[::-1]
        ref_jets_matched = [ref_jets_matched[i] for i in sorted_idx]
        comp_jets_matched = [comp_jets_matched[i] for i in sorted_idx]

        return ref_jets_matched, comp_jets_matched

    def match_jets_all_ev(self, ref_jets, comp_jets):
        ref_jets_matched, comp_jets_matched = [], []
        for _ev_i, (ref_jets_ev, comp_jets_ev) in enumerate(
            tqdm(zip(ref_jets, comp_jets, strict=False), total=len(ref_jets), desc="Matching jets...")
        ):
            ref_jets_ev_matched, comp_jets_ev_matched = self.match_jets_single_ev(ref_jets_ev, comp_jets_ev)
            ref_jets_matched.append(ref_jets_ev_matched)
            comp_jets_matched.append(comp_jets_ev_matched)

        return ref_jets_matched, comp_jets_matched

    def match_jets(self):
        self.truth_dict["matched_pandora_jets"] = self.match_jets_all_ev(self.truth_dict["truth_jets"], self.truth_dict["pandora_jets"])
        for net_dict in self.data.values():
            net_dict["matched_jets"] = self.match_jets_all_ev(self.truth_dict["truth_jets"], net_dict["jets"])
            if "proxy_jets" in net_dict:
                net_dict["matched_proxy_jets"] = self.match_jets_all_ev(self.truth_dict["truth_jets"], net_dict["proxy_jets"])

    def hung_match_ev(self, ref_particles, comp_particles, return_unmatched=False):
        ref_pt, ref_eta, ref_phi, ref_cl = ref_particles
        comp_pt, comp_eta, comp_phi, comp_cl = comp_particles

        cost_delpt_sq = (np.expand_dims(ref_pt, axis=1) - np.expand_dims(comp_pt, axis=0)) ** 2
        cost_delpt_sq_by_pt_sq = cost_delpt_sq / np.expand_dims(ref_pt, axis=1) ** 2
        cost_deltar = delta_r(  # deltaR(
            np.expand_dims(ref_eta, axis=1),
            np.expand_dims(comp_eta, axis=0),
            np.expand_dims(ref_phi, axis=1),
            np.expand_dims(comp_phi, axis=0),
        )
        cost = np.sqrt(cost_delpt_sq_by_pt_sq + cost_deltar**2)

        ref_ch_mask = ref_cl <= 2
        comp_ch_mask = comp_cl <= 2

        # charged
        masked_cost = cost[np.ix_(ref_ch_mask, comp_ch_mask)]
        row_i, col_i = linear_sum_assignment(masked_cost, maximize=False)
        row_indices = np.arange(len(ref_pt))[ref_ch_mask][row_i]
        col_indices = np.arange(len(comp_pt))[comp_ch_mask][col_i]

        # neutral
        masked_cost = cost[np.ix_(~ref_ch_mask, ~comp_ch_mask)]
        row_i, col_i = linear_sum_assignment(masked_cost, maximize=False)
        row_indices = np.concatenate([row_indices, np.arange(len(ref_pt))[~ref_ch_mask][row_i]])
        col_indices = np.concatenate([col_indices, np.arange(len(comp_pt))[~comp_ch_mask][col_i]])

        ref_matched_dict = {
            "pt": ref_pt[row_indices],
            "eta": ref_eta[row_indices],
            "phi": ref_phi[row_indices],
            "class": ref_cl[row_indices],
        }
        comp_matched_dict = {
            "pt": comp_pt[col_indices],
            "eta": comp_eta[col_indices],
            "phi": comp_phi[col_indices],
            "class": comp_cl[col_indices],
        }

        ref_unmatched_dict = None
        comp_unmatched_dict = None
        if return_unmatched:
            ref_unmatched_dict = {
                "pt": np.delete(ref_pt, row_indices),
                "eta": np.delete(ref_eta, row_indices),
                "phi": np.delete(ref_phi, row_indices),
                "class": np.delete(ref_cl, row_indices),
            }
            comp_unmatched_dict = {
                "pt": np.delete(comp_pt, col_indices),
                "eta": np.delete(comp_eta, col_indices),
                "phi": np.delete(comp_phi, col_indices),
                "class": np.delete(comp_cl, col_indices),
            }

        return (
            ref_matched_dict,
            comp_matched_dict,
            ref_unmatched_dict,
            comp_unmatched_dict,
        )

    def hung_match_all_ev(self, ref_particles, comp_particles, flatten=False, return_unmatched=False):
        rp_pt, rp_eta, rp_phi, rp_cl = ref_particles
        cp_pt, cp_eta, cp_phi, cp_cl = comp_particles

        ref_particles_matched = {"pt": [], "eta": [], "phi": [], "class": []}
        comp_particles_matched = {"pt": [], "eta": [], "phi": [], "class": []}
        ref_particles_unmatched = {"pt": [], "eta": [], "phi": [], "class": []}
        comp_particles_unmatched = {"pt": [], "eta": [], "phi": [], "class": []}

        for i in tqdm(range(len(ref_particles[0])), desc="Matching particles..."):
            (
                ref_particles_ev_matched,
                comp_particles_ev_matched,
                ref_particles_ev_unmatched,
                comp_particles_ev_unmatched,
            ) = self.hung_match_ev(
                (rp_pt[i], rp_eta[i], rp_phi[i], rp_cl[i]),
                (cp_pt[i], cp_eta[i], cp_phi[i], cp_cl[i]),
                return_unmatched,
            )

            for key in ref_particles_matched:
                ref_particles_matched[key].append(ref_particles_ev_matched[key])
                comp_particles_matched[key].append(comp_particles_ev_matched[key])
                if return_unmatched:
                    ref_particles_unmatched[key].append(ref_particles_ev_unmatched[key])
                    comp_particles_unmatched[key].append(comp_particles_ev_unmatched[key])

        if flatten:
            for key in ref_particles_matched:
                ref_particles_matched[key] = np.hstack(ref_particles_matched[key])
                comp_particles_matched[key] = np.hstack(comp_particles_matched[key])
                if return_unmatched:
                    ref_particles_unmatched[key] = np.hstack(ref_particles_unmatched[key])
                    comp_particles_unmatched[key] = np.hstack(comp_particles_unmatched[key])

        if return_unmatched:
            return (
                ref_particles_matched,
                comp_particles_matched,
                ref_particles_unmatched,
                comp_particles_unmatched,
            )
        return ref_particles_matched, comp_particles_matched

    def hung_match_particles(self, flatten=False, return_unmatched=False):
        for net_name, net_dict in self.data.items():
            if net_name in {"hgpflow", "mpflow"}:
                net_dict["matched_proxy_particles"] = self.hung_match_all_ev(
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
            net_dict["matched_particles"] = self.hung_match_all_ev(
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
            self.truth_dict["matched_pandora_particles"] = self.hung_match_all_ev(
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
