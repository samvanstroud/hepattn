# ruff: noqa: FIX004

import gc

import numpy as np
import uproot
from tqdm import tqdm

from .helper_dicts import class_mass_dict, pdgid_class_dict


def load_pred_hgpflow(pred_path, threshold=0.5, num_events=None):
    tree = uproot.open(pred_path)["event_tree"]

    vars_to_load = ["pred_ind", "proxy_pt", "proxy_eta", "proxy_phi", "hgpflow_pt", "hgpflow_eta", "hgpflow_phi", "hgpflow_class"]

    mask = np.array([np.array(x > threshold) for x in tree["pred_ind"].array(library="np", entry_stop=num_events)], dtype=object)

    hgpflow_dict = {}
    for var in tqdm(vars_to_load, desc="Loading HGPFlow predictions...", total=len(vars_to_load)):
        new_var = var.replace("hgpflow_", "")
        hgpflow_dict[new_var] = np.array(
            [x[m] for x, m in zip(tree[var].array(library="np", entry_stop=num_events), mask, strict=False)], dtype=object
        )
    # hgpflow_dict["event_number"] = tree["event_number"].array(library="np", entry_stop=num_events).astype(int) - 55_000 # HACK

    # HACK: to fix event numbers we just arange them manually
    hgpflow_dict["event_number"] = np.arange(len(hgpflow_dict["pt"]))  # HACK

    # compute mass and energy
    for k in ["mass", "e", "charge"]:
        hgpflow_dict[f"{k}"] = np.empty_like(hgpflow_dict["pt"], dtype=object)

    for i, cls in tqdm(enumerate(hgpflow_dict["class"]), desc="Computing HGPFlow mass...", total=len(hgpflow_dict["class"])):
        hgpflow_dict["mass"][i] = np.array([class_mass_dict[x] for x in cls])
        p = hgpflow_dict["pt"][i] * np.cosh(hgpflow_dict["eta"][i])
        hgpflow_dict["e"][i] = np.sqrt(p**2 + hgpflow_dict["mass"][i] ** 2)

        hgpflow_dict["charge"][i] = np.full_like(hgpflow_dict["pt"][i], 0)
        hgpflow_dict["charge"][i][cls <= 2] = 1

    return hgpflow_dict


def load_pred_mpflow(pred_path, threshold=0.5, num_events=None):
    tree = uproot.open(pred_path)["event_tree"]

    vars_to_load = ["pred_ind", "proxy_pt", "proxy_eta", "proxy_phi", "mpflow_pt", "mpflow_eta", "mpflow_phi", "mpflow_class"]

    mask = np.array([np.array(x > threshold) for x in tree["pred_ind"].array(library="np", entry_stop=num_events)])  # , dtype=object)

    mpflow_dict = {}
    for var in tqdm(vars_to_load, desc="Loading mpflow predictions...", total=len(vars_to_load)):
        new_var = var.replace("mpflow_", "")
        mpflow_dict[new_var] = np.array(
            [x[m] for x, m in zip(tree[var].array(library="np", entry_stop=num_events), mask, strict=False)], dtype=object
        )
    mpflow_dict["event_number"] = tree["event_number"].array(library="np", entry_stop=num_events).astype(int)  # - 643_00 # HACK

    # compute mass and energy
    for k in ["mass", "e", "charge"]:
        mpflow_dict[f"{k}"] = np.empty_like(mpflow_dict["pt"], dtype=object)

    for i, cls in tqdm(enumerate(mpflow_dict["class"]), desc="Computing mpflow mass...", total=len(mpflow_dict["class"])):
        mpflow_dict["mass"][i] = np.array([class_mass_dict[x] for x in cls])
        p = mpflow_dict["pt"][i] * np.cosh(mpflow_dict["eta"][i])
        mpflow_dict["e"][i] = np.sqrt(p**2 + mpflow_dict["mass"][i] ** 2)

        mpflow_dict["charge"][i] = np.full_like(mpflow_dict["pt"][i], 0)
        mpflow_dict["charge"][i][cls <= 2] = 1

    return mpflow_dict


def load_pred_mlpf(pred_path, truth_event_number_offset):
    class_remap = {
        1.0: 0,  # ch_had
        2.0: 3,  # neut had
        3.0: 4,  # photon
        4.0: 1,  # electron
        5.0: 2,  # muon
    }

    tree = uproot.open(pred_path)["parts"]
    vars_to_load = ["pred_pt", "pred_phi", "pred_eta", "pred_cl", "pred_e"]

    mlpf_class = tree["pred_cl"].array(library="np")
    mlpf_mask_cl = np.array([x != 0 for x in mlpf_class], dtype=object)

    mlpf_dict = {}
    for var in tqdm(vars_to_load, desc="Loading MLPF predictions...", total=len(vars_to_load)):
        new_var = var.replace("pred_", "")
        mlpf_dict[new_var] = np.array([x[m] for x, m in zip(tree[var].array(library="np"), mlpf_mask_cl, strict=False)], dtype=object)

    # class remapping
    mlpf_dict["class"] = np.array([np.array([class_remap[x] for x in cls]) for cls in mlpf_dict["cl"]], dtype=object)

    # charge (0 for netural particles, 1 for charged particles)
    mlpf_dict["charge"] = np.empty_like(mlpf_dict["pt"])
    for i, cls in tqdm(enumerate(mlpf_dict["class"]), desc="Computing MLPF charge...", total=len(mlpf_dict["class"])):
        mlpf_dict["charge"][i] = np.full_like(cls, 0)
        mlpf_dict["charge"][i][cls <= 2] = 1

    # compute event number
    mlpf_dict["event_id"] = tree["event_id"].array(library="np")
    # mlpf_dict['file_id'] = tree['file_id'].array(library='np')
    # mlpf_dict['file_id'] = mlpf_dict['file_id'] - mlpf_dict['file_id'].min()
    # mlpf_dict['event_number'] = mlpf_dict['event_id'] + mlpf_dict['file_id'] * num_ev_in_one_file + truth_event_number_offset

    mlpf_dict["event_number"] = mlpf_dict["event_id"] + truth_event_number_offset

    # compute mass
    mlpf_dict["mass"] = np.empty_like(mlpf_dict["pt"])
    for i, cls in tqdm(enumerate(mlpf_dict["class"]), desc="Computing MLPF mass...", total=len(mlpf_dict["class"])):
        mlpf_dict["mass"][i] = np.array([class_mass_dict[x] for x in cls])

    return mlpf_dict


def load_truth_clic(truth_path, event_number_offset=0):
    scale_e_pt = 1
    # pt_min_gev = 0.1
    print("\033[96m" + f"E, pT will be scaled by {scale_e_pt}" + "\033[0m")

    tree = uproot.open(truth_path)["events"]
    n_events = tree.num_entries

    truth_dict = {}
    vars_to_load = [
        "particle_pt",
        "particle_eta",
        "particle_phi",
        "particle_e",
        "particle_pdg",
        "particle_gen_status",
        "pandora_e",
        "pandora_eta",
        "pandora_phi",
        "pandora_pt",
        "pandora_pdg",
    ]

    for var in tqdm(vars_to_load, desc="Reading truth tree...", total=len(vars_to_load)):
        truth_dict[var] = tree[var].array(library="np")

    n_tracks = tree["ntrack_pt"].array(library="np")
    n_topos = tree["ntopo_e"].array(library="np")
    # filter out events with no tracks and no topoclusters
    track_topo_mask = (n_tracks > 0) | (n_topos > 0)
    print(f"Number of events with at least one track or topocluster: {np.sum(track_topo_mask)} out of {n_events}")
    print(np.argwhere(~track_topo_mask))
    for var in vars_to_load:
        truth_dict[var] = truth_dict[var][track_topo_mask]
    n_events = len(truth_dict["particle_pt"])
    print(f"Number of events after filtering: {n_events}")
    # MeV to GeV scaling not needed (already in GeV)

    # particle class and charge
    truth_dict["particle_class"] = np.empty_like(truth_dict["particle_pdg"])
    truth_dict["particle_charge"] = np.empty_like(truth_dict["particle_pdg"])
    for i, pdgid in tqdm(enumerate(truth_dict["particle_pdg"]), desc="Computing particle class...", total=n_events):
        truth_dict["particle_class"][i] = np.array([pdgid_class_dict.get(x, 4) for x in pdgid])
        truth_dict["particle_charge"][i] = np.array([1 if x <= 2 else 0 for x in truth_dict["particle_class"][i]])

    # pandora class and charge
    truth_dict["pandora_class"] = np.empty_like(truth_dict["pandora_pdg"])
    truth_dict["pandora_charge"] = np.empty_like(truth_dict["pandora_pdg"])
    for i, pdgid in tqdm(enumerate(truth_dict["pandora_pdg"]), desc="Computing pandora class...", total=n_events):
        truth_dict["pandora_class"][i] = np.array([pdgid_class_dict.get(x, 4) for x in pdgid])
        truth_dict["pandora_charge"][i] = np.array([1 if x <= 2 else 0 for x in truth_dict["pandora_class"][i]])

    # delete unnecessary variables
    vars_to_delete = ["particle_pdg", "pandora_pdg"]
    for var in vars_to_delete:
        del truth_dict[var]
    gc.collect()

    # fiducial cuts
    for i in tqdm(range(n_events), desc="Applying fiducial cuts...", total=n_events):
        # on particles (gen_status=1)
        mask = truth_dict["particle_gen_status"][i] == 1  # * (truth_dict['particle_pt'][i] >= pt_min_gev)
        for var in ["particle_pt", "particle_eta", "particle_phi", "particle_e", "particle_class", "particle_gen_status"]:
            truth_dict[var][i] = truth_dict[var][i][mask]

    if "event_number" in tree:
        truth_dict["event_number"] = tree["event_number"].array(library="np").astype(int)
    else:
        truth_dict["event_number"] = np.arange(len(truth_dict["particle_pt"])) + event_number_offset

    return truth_dict


def load_hgpflow_target(target_path, drop_res=True, num_events=None, event_number_offset=0):
    tree = uproot.open(target_path)["EventTree"]
    vars_to_load = ["particle_pt", "particle_eta", "particle_phi", "particle_e", "particle_pdgid"]

    hgpflow_target_dict_tmp = {}
    data = tree.arrays(vars_to_load, library="np", entry_stop=num_events)
    for var in tqdm(vars_to_load, desc="Loading HGPflow target (segmented)...", total=len(vars_to_load)):
        new_var = var.replace("particle_", "")
        hgpflow_target_dict_tmp[new_var] = data[var]

    # filter out the residual particles
    if drop_res:  # will be dafault, here it is just for debugging
        mask = np.array([np.array([pdgid_class_dict[xx] for xx in x]) <= 4 for x in hgpflow_target_dict_tmp["pdgid"]], dtype=object)
        for key, val in hgpflow_target_dict_tmp.items():
            if key == "event_number":
                continue
            hgpflow_target_dict_tmp[key] = np.array([x[m] for x, m in zip(val, mask, strict=False)], dtype=object)

    # unique_sorted_ev_num = np.sort(np.unique(hgpflow_target_dict_tmp['event_number']))
    # hgpflow_target_dict = {}
    # for key, val in tqdm(hgpflow_target_dict_tmp.items(), desc="Merging HGPflow target...", total=len(hgpflow_target_dict_tmp)):
    #     hgpflow_target_dict[key] = []
    #     for ev_num in unique_sorted_ev_num:
    #         mask = hgpflow_target_dict_tmp['event_number'] == ev_num
    #         hgpflow_target_dict[key].append(np.hstack(val[mask]))
    #     hgpflow_target_dict[key] = np.array(hgpflow_target_dict[key], dtype=object)
    # hgpflow_target_dict['event_number'] = unique_sorted_ev_num
    hgpflow_target_dict = hgpflow_target_dict_tmp

    # compute class
    hgpflow_target_dict["class"] = np.empty_like(hgpflow_target_dict["pdgid"])
    for i, pdgid in tqdm(enumerate(hgpflow_target_dict["pdgid"]), desc="Computing HGPFlow target class...", total=len(hgpflow_target_dict["pdgid"])):
        hgpflow_target_dict["class"][i] = np.array([pdgid_class_dict[x] for x in pdgid])

    # compute mass
    hgpflow_target_dict["mass"] = np.empty_like(hgpflow_target_dict["pt"])
    for i, pdgid in tqdm(enumerate(hgpflow_target_dict["pdgid"]), desc="Computing HGPFlow target mass...", total=len(hgpflow_target_dict["pdgid"])):
        hgpflow_target_dict["mass"][i] = np.array([class_mass_dict[pdgid_class_dict[x]] for x in pdgid])

    # if 'event_number' in tree.keys():
    #     truth_dict['event_number'] = tree['event_number'].array(library='np').astype(int)
    # else:
    hgpflow_target_dict["event_number"] = np.arange(len(hgpflow_target_dict["pt"])) + event_number_offset
    return hgpflow_target_dict
