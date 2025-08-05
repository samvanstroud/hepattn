import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .utils import delta_r


def match_jets_single_ev(ref_jets, comp_jets):
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


def match_jets_all_ev(ref_jets, comp_jets):
    ref_jets_matched, comp_jets_matched = [], []
    for _ev_i, (ref_jets_ev, comp_jets_ev) in enumerate(tqdm(zip(ref_jets, comp_jets, strict=False), total=len(ref_jets), desc="Matching jets...")):
        ref_jets_ev_matched, comp_jets_ev_matched = match_jets_single_ev(ref_jets_ev, comp_jets_ev)
        ref_jets_matched.append(ref_jets_ev_matched)
        comp_jets_matched.append(comp_jets_ev_matched)

    return ref_jets_matched, comp_jets_matched


def match_particles_single_ev(ref_particles, comp_particles, return_unmatched=False):
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


def match_particles_all_ev(ref_particles, comp_particles, flatten=False, return_unmatched=False):
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
        ) = match_particles_single_ev(
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
