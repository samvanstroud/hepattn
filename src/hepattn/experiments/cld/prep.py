import os
import time
from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from particle.pdgid import is_hadron
from scipy.sparse import csr_matrix

# Overall process:
# Read in specified items and format into regular arrays
# Build masks between the obejcts using the different linking systems
# Use the masks to build any extra needed masks
# Peform particle cuts based on particle properties
# Perform particle cuts based on hit content
# Account for any cut particles in the masks
# Apply item and field name aliasing
# Save the items we want to keep

###############################################################
# Specify the names of items we want to save
###############################################################
item_names = [
    "MCParticles",
    "VXDTrackerHits",
    "VXDEndcapTrackerHits",
    "ITrackerHits",
    "ITrackerEndcapHits",
    "OTrackerHits",
    "OTrackerEndcapHits",
    "ECALBarrel",
    "ECALEndcap",
    "HCALBarrel",
    "HCALEndcap",
    "HCALOther",
    "MUON",
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
    "HCalRingCollection",
    "YokeBarrelCollection",
    "YokeEndcapCollection",
    "ECalBarrelCollectionContributions",
    "ECalEndcapCollectionContributions",
    "HCalBarrelCollectionContributions",
    "HCalEndcapCollectionContributions",
    "HCalRingCollectionContributions",
    "YokeBarrelCollectionContributions",
    "YokeEndcapCollectionContributions",
]

###############################################################
# Masks that we will have to build from the raw data
###############################################################

# Specify the masks/links to build that use bidirectional truth links
relations_links = {
    "VXDTrackerHitRelations": [
        ("VXDTrackerHits", "VertexBarrelCollection"),
    ],
    "VXDEndcapTrackerHitRelations": [("VXDEndcapTrackerHits", "VertexEndcapCollection")],
    "InnerTrackerBarrelHitsRelations": [("ITrackerHits", "InnerTrackerBarrelCollection")],
    "InnerTrackerEndcapHitsRelations": [("ITrackerEndcapHits", "InnerTrackerEndcapCollection")],
    "OuterTrackerBarrelHitsRelations": [("OTrackerHits", "OuterTrackerBarrelCollection")],
    "OuterTrackerEndcapHitsRelations": [("OTrackerEndcapHits", "OuterTrackerEndcapCollection")],
    "CalohitMCTruthLink": [
        ("ECALBarrel", "MCParticles"),
        ("ECALEndcap", "MCParticles"),
        ("HCALBarrel", "MCParticles"),
        ("HCALEndcap", "MCParticles"),
        ("HCALOther", "MCParticles"),
        ("MUON", "MCParticles"),
    ],
    "RelationCaloHit": [
        ("ECALBarrel", "ECalBarrelCollection"),
        ("ECALEndcap", "ECalEndcapCollection"),
        ("HCALBarrel", "HCalBarrelCollection"),
        ("HCALEndcap", "HCalEndcapCollection"),
        ("HCALOther", "HCalRingCollection"),
    ],
    "RelationMuonHit": [
        ("MUON", "ECalBarrelCollection"),
        ("MUON", "ECalEndcapCollection"),
        ("MUON", "HCalBarrelCollection"),
        ("MUON", "HCalEndcapCollection"),
        ("MUON", "HCalRingCollection"),
        ("MUON", "YokeBarrelCollection"),
        ("MUON", "YokeEndcapCollection"),
    ],
}

# Specify masks/links which use start/end indexing to specify linkage
start_end_links = [
    ("ECalBarrelCollection", "ECalBarrelCollectionContributions"),
    ("ECalEndcapCollection", "ECalEndcapCollectionContributions"),
    ("HCalBarrelCollection", "HCalBarrelCollectionContributions"),
    ("HCalEndcapCollection", "HCalEndcapCollectionContributions"),
    ("HCalRingCollection", "HCalRingCollectionContributions"),
    ("YokeBarrelCollection", "YokeBarrelCollectionContributions"),
    ("YokeEndcapCollection", "YokeEndcapCollectionContributions"),
]

# Specify the mask/links which use particle based links
particle_links = [
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
    "ECalBarrelCollectionContributions",
    "ECalEndcapCollectionContributions",
    "HCalBarrelCollectionContributions",
    "HCalEndcapCollectionContributions",
    "HCalRingCollectionContributions",
    "YokeEndcapCollectionContributions",
    "YokeBarrelCollectionContributions",
]

###############################################################
# Which masks we have to build by glueing together the other masks
###############################################################

mask_joins = [
    ("VXDTrackerHits", "VertexBarrelCollection", "MCParticles"),
    ("VXDEndcapTrackerHits", "VertexEndcapCollection", "MCParticles"),
    ("ITrackerHits", "InnerTrackerBarrelCollection", "MCParticles"),
    ("ITrackerEndcapHits", "InnerTrackerEndcapCollection", "MCParticles"),
    ("OTrackerHits", "OuterTrackerBarrelCollection", "MCParticles"),
    ("OTrackerEndcapHits", "OuterTrackerEndcapCollection", "MCParticles"),
    ("ECALBarrel", "ECalBarrelCollection", "ECalBarrelCollectionContributions"),
    ("ECALEndcap", "ECalEndcapCollection", "ECalEndcapCollectionContributions"),
    ("HCALBarrel", "HCalBarrelCollection", "HCalBarrelCollectionContributions"),
    ("HCALEndcap", "HCalEndcapCollection", "HCalEndcapCollectionContributions"),
    ("HCALOther", "HCalRingCollection", "HCalRingCollectionContributions"),
    ("MUON", "YokeBarrelCollection", "YokeBarrelCollectionContributions"),
    ("MUON", "YokeEndcapCollection", "YokeEndcapCollectionContributions"),
]

###############################################################
# Object groupings used for counting number of hits
###############################################################

sihits = [
    "VXDTrackerHits",
    "VXDEndcapTrackerHits",
    "ITrackerHits",
    "ITrackerEndcapHits",
    "OTrackerHits",
    "OTrackerEndcapHits",
]

calohits = [
    "ECALBarrel",
    "ECALEndcap",
    "HCALBarrel",
    "HCALEndcap",
    "HCALOther",
]

###############################################################
# Aliases for the items
###############################################################

item_aliases = {
    "MCParticles": "particle",
    "VXDTrackerHits": "vtb",  # Vertex tracker barrel
    "VXDEndcapTrackerHits": "vte",  # Vertex tracker endcap
    "ITrackerHits": "itb",  # Inner tracker barrel, etc
    "ITrackerEndcapHits": "ite",
    "OTrackerHits": "otb",
    "OTrackerEndcapHits": "ote",
    "ECALBarrel": "ecb",  # Electronic calorimeter barrel
    "ECALEndcap": "ece",
    "HCALBarrel": "hcb",
    "HCALEndcap": "hce",
    "HCALOther": "hco",  # Hadronic calorimeter other
    "MUON": "muon",
    "MUONBarrel": "msb",
    "MUONEndcap": "mse",
    "VertexBarrelCollection": "vtb_col",
    "VertexEndcapCollection": "vte_col",
    "InnerTrackerBarrelCollection": "itb_col",
    "InnerTrackerEndcapCollection": "ite_col",
    "OuterTrackerBarrelCollection": "otb_col",
    "OuterTrackerEndcapCollection": "ote_col",
    "ECalBarrelCollection": "ecb_col",
    "ECalEndcapCollection": "ece_col",
    "HCalBarrelCollection": "hcb_col",
    "HCalEndcapCollection": "hce_col",
    "HCalRingCollection": "hco_col",
    "YokeBarrelCollection": "msb_col",
    "YokeEndcapCollection": "mse_col",
    "ECalBarrelCollectionContributions": "ecb_con",
    "ECalEndcapCollectionContributions": "ece_con",
    "HCalBarrelCollectionContributions": "hcb_con",
    "HCalEndcapCollectionContributions": "hce_con",
    "HCalRingCollectionContributions": "hco_con",
    "YokeBarrelCollectionContributions": "msb_con",
    "YokeEndcapCollectionContributions": "mse_con",
}

###############################################################
# Aliases for the fields
###############################################################

field_aliases = {
    "position.x": "pos.x",
    "position.y": "pos.y",
    "position.z": "pos.z",
    "vertex.x": "vtx.x",
    "vertex.y": "vtx.y",
    "vertex.z": "vtx.z",
    "endpoint.x": "end.x",
    "endpoint.y": "end.y",
    "endpoint.z": "end.z",
    "momentumAtEndpoint.x": "end_mom.x",
    "momentumAtEndpoint.y": "end_mom.y",
    "momentumAtEndpoint.z": "end_mom.z",
    "momentum.x": "mom.x",
    "momentum.y": "mom.y",
    "momentum.z": "mom.z",
    "pathLegth": "pathlen",
    "stepPosition.x": "step_pos.x",
    "stepPosition.y": "step_pos.y",
    "stepPosition.z": "step_pos.z",
}

###############################################################
# Output items
###############################################################

output_items = [
    "particle",
    "vtb",
    "vte",
    "itb",
    "ite",
    "otb",
    "ote",
    "ecb",
    "ece",
    "hcb",
    "hce",
    "hco",
    "msb",
    "mse",
    "vtb_col",
    "vte_col",
    "itb_col",
    "ite_col",
    "otb_col",
    "ote_col",
    "ecb_col",
    "ece_col",
    "hcb_col",
    "hce_col",
    "hco_col",
    "msb_col",
    "mse_col",
    "ecb_con",
    "ece_con",
    "hcb_con",
    "hce_con",
    "hco_con",
    "msb_con",
    "mse_con",
]

# Specify which masks we actually want to save
output_masks = [
    ("vtb_col", "vtb"),
    ("vte_col", "vte"),
    ("itb_col", "itb"),
    ("ite_col", "ite"),
    ("otb_col", "otb"),
    ("ote_col", "ote"),
    ("ecb_col", "ecb"),
    ("ece_col", "ece"),
    ("hcb_col", "hcb"),
    ("hce_col", "hce"),
    ("hco_col", "hco"),
    ("msb_col", "msb"),
    ("mse_col", "mse"),
    ("ecb_con", "ecb"),
    ("ece_con", "ece"),
    ("hcb_con", "hcb"),
    ("hce_con", "hce"),
    ("hco_con", "hco"),
    ("msb_con", "msb"),
    ("mse_con", "mse"),
    ("ecb_con", "ecb_col"),
    ("ece_con", "ece_col"),
    ("hcb_con", "hcb_col"),
    ("hce_con", "hce_col"),
    ("hco_con", "hco_col"),
    ("msb_con", "msb_col"),
    ("mse_con", "mse_col"),
    ("particle", "vtb_col"),
    ("particle", "vte_col"),
    ("particle", "itb_col"),
    ("particle", "ite_col"),
    ("particle", "otb_col"),
    ("particle", "ote_col"),
    ("particle", "ecb_con"),
    ("particle", "ece_con"),
    ("particle", "hcb_con"),
    ("particle", "hce_con"),
    ("particle", "hco_con"),
    ("particle", "msb_con"),
    ("particle", "mse_con"),
    ("particle", "vtb"),
    ("particle", "vte"),
    ("particle", "itb"),
    ("particle", "ite"),
    ("particle", "otb"),
    ("particle", "ote"),
    ("particle", "ecb"),
    ("particle", "ece"),
    ("particle", "hcb"),
    ("particle", "hce"),
    ("particle", "hco"),
    ("particle", "msb"),
    ("particle", "mse"),
]


non_hadron_pdgid_to_class = {
    22: 3,  # Photon
    11: 4,  # Electron
    12: 7,  # Neutrino
    13: 5,  # Muon
    14: 7,  # Neutrino
    15: 6,  # Tau
    16: 7,  # Neutrino
}


def get_particle_class(pid, charge):
    if is_hadron(pid):
        if charge == 0:
            # Neutral hadron
            return 0
        # Charged hadron
        return 1
    if np.abs(pid) in non_hadron_pdgid_to_class:
        return non_hadron_pdgid_to_class[np.abs(pid)]
    return -1


def preprocess_event(events, event_idx, namecodes, min_pt, verbose):
    items = {}

    # First build the items by combinging them into properly formatted awkward arrays
    for item_name in item_names:
        # Convert to sane format
        x = events[item_name].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]
        x = ak.zip({field.replace(f"{item_name}.", ""): x[field] for field in x.fields}, depth_limit=1)
        items[item_name] = x

    # Add in particle classes
    pids = items["MCParticles"]["PDG"]
    charges = items["MCParticles"]["charge"]

    separator = "\n" + "=" * 64 + "\n"

    # This is actually quite slow since we make use of scikit-hep particle, so we do it in preprocessing
    # Would be better if we can get a comprehensive list of every hadron pdgid
    items["MCParticles"]["class"] = np.array([get_particle_class(pids[i], charges[i]) for i in range(len(pids))])

    # Attach an index before we do any cuts, this will allow us to make use of the daughters/parents index
    # even after cuts have been made
    items["MCParticles"]["index"] = np.arange(len(items["MCParticles"]["PDG"]))

    if verbose:
        print(separator)
        for item_name in item_names:
            print(f"Loaded item {item_name} of type {items[item_name].type}")

    # Now build the masks that link the various items together
    masks = {}

    # First build the masks that use bidirectional truth links
    for relation, links in relations_links.items():
        link_src_cid = ak.to_numpy(
            events[f"_{relation}_from/_{relation}_from.collectionID"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]
        )
        link_src_idx = ak.to_numpy(events[f"_{relation}_from/_{relation}_from.index"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])

        link_tgt_cid = ak.to_numpy(events[f"_{relation}_to/_{relation}_to.collectionID"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])
        link_tgt_idx = ak.to_numpy(events[f"_{relation}_to/_{relation}_to.index"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])

        for src, tgt in links:
            link_mask = (namecodes[src] == link_src_cid) & (namecodes[tgt] == link_tgt_cid)
            num_src = len(items[src])
            num_tgt = len(items[tgt])

            mask = np.full((num_src, num_tgt), False)
            mask[link_src_idx[link_mask], link_tgt_idx[link_mask]] = True
            masks[src, tgt] = mask

    # Now build masks that use start/end indices
    for src, tgt in start_end_links:
        # The source has the indices that index into the target
        start_idx = ak.to_numpy(events[f"{src}/{src}.contributions_begin"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])
        end_idx = ak.to_numpy(events[f"{src}/{src}.contributions_end"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])

        num_src = len(items[src])
        num_tgt = len(items[tgt])

        tgt_idx = np.arange(num_tgt)
        mask = (tgt_idx[None, :] >= start_idx[:, None]) & (tgt_idx[None, :] <= end_idx[:, None])
        masks[src, tgt] = mask

    # Now build the masks that use a single particle link
    for src in particle_links:
        tgt_cids = ak.to_numpy(events[f"_{src}_particle/_{src}_particle.collectionID"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])
        tgt_idxs = ak.to_numpy(events[f"_{src}_particle/_{src}_particle.index"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0])

        tgt = "MCParticles"

        # Check that the particle links are indeed all mcparticles
        assert np.all(tgt_cids == namecodes[tgt])

        num_src = len(items[src])
        num_tgt = len(items[tgt])

        mask = np.full((num_src, num_tgt), False)
        mask[np.arange(num_src), tgt_idxs] = True
        masks[src, tgt] = mask

    if verbose:
        print(separator)
        for (src, tgt), mask in masks.items():
            print(f"Built mask for {src} -> {tgt} of shape {mask.shape}")

    # Join together existing masks to make new masks
    for src, link, tgt in mask_joins:
        src_link_mask = masks[src, link]
        link_tgt_mask = masks[link, tgt]

        src_link_mask_sparse = csr_matrix(src_link_mask)
        link_tgt_mask_sparse = csr_matrix(link_tgt_mask)
        src_tgt_mask_sparse = (src_link_mask_sparse @ link_tgt_mask_sparse).astype(bool)

        masks[src, tgt] = np.array(src_tgt_mask_sparse.todense())

        if verbose:
            print(f"Linked {src} to {tgt} using {link}")

    # Count number of hits on a particle, and vice-versa, which is needed for hit based cuts
    item_counts = {}
    for src, tgt in masks:
        mask = masks[src, tgt]
        item_counts[f"{src}_num_{tgt}"] = mask.sum(-1)
        item_counts[f"{tgt}_num_{src}"] = mask.sum(-2)

    particle_pt = np.sqrt(items["MCParticles"]["momentum.x"] ** 2 + items["MCParticles"]["momentum.y"] ** 2)

    # Now apply any cuts to the items
    # Name some cuts for convenience
    particle_cuts = {
        # Used to remove generator particles
        "Status 0": items["MCParticles"]["generatorStatus"] == 0,
        "Status 1": items["MCParticles"]["generatorStatus"] == 1,
        "Status 2": items["MCParticles"]["generatorStatus"] == 2,
        # Apply pT cut, min_pt is given in MeV, momentum.t is in GeV
        "Good Momentum": particle_pt >= min_pt * 0.001,
        # Remove particles which are beam background
        # TODO: Need to check if we should include this instead, might be the case we should include, see:
        # https://indico.cern.ch/event/656491/contributions/2939124/attachments/1629649/2597052/Pairs_voutsi.pdf
        "Electron Beam Remenant": (items["MCParticles"]["momentum.y"] == 0) & (items["MCParticles"]["PDG"] == 11),
        "Photon Beam Remenant": (items["MCParticles"]["momentum.y"] == 0) & (items["MCParticles"]["PDG"] == 22),
    }

    particle_cuts["Not Beam Remnant"] = ~(particle_cuts["Electron Beam Remenant"] | particle_cuts["Photon Beam Remenant"])
    particle_cuts["Good Status"] = particle_cuts["Status 0"] | particle_cuts["Status 1"] | particle_cuts["Status 2"]

    particle_cuts["Is Charged"] = items["MCParticles"]["charge"] != 0
    particle_cuts["Is Neutral"] = ~particle_cuts["Is Charged"]

    particle_cuts["No SiHits"] = np.sum([item_counts[f"MCParticles_num_{hit}"] for hit in sihits], axis=0) == 0
    particle_cuts["No CaloHits"] = np.sum([item_counts[f"MCParticles_num_{hit}"] for hit in calohits], axis=0) == 0

    particle_cuts["Charged has SiHits"] = ~(particle_cuts["Is Charged"] & particle_cuts["No SiHits"])
    particle_cuts["Neutral has CaloHits"] = ~(particle_cuts["Is Neutral"] & particle_cuts["No CaloHits"])

    particle_cut_final = np.full_like(items["MCParticles"]["PDG"], True, np.bool)

    applied_particle_cuts = [
        "Good Status",
        "Not Beam Remnant",
        "Good Momentum",
        "Charged has SiHits",
        "Neutral has CaloHits",
    ]

    if verbose:
        print(separator)
        print("Applying particle cuts:")

    for cut_name in applied_particle_cuts:
        cut_mask = particle_cuts[cut_name]
        cut_size = np.sum(cut_mask)
        pre_cut_size = np.sum(particle_cut_final)
        particle_cut_final = particle_cut_final & cut_mask
        post_cut_size = np.sum(particle_cut_final)

        if verbose:
            print(f"Applying cut {cut_name}: size {cut_size}, pre {pre_cut_size}, post {post_cut_size}")

    # Keep only particles that have status codes 1 or 2, and that are not beam remenants
    item_cuts = {"MCParticles": particle_cut_final}

    # Apply cuts to the items
    for item_name in item_names:
        if item_name in item_cuts:
            items[item_name] = items[item_name][item_cuts[item_name]]

    # Apply cuts to the masks
    for src, tgt in masks:
        if src in item_cuts:
            masks[src, tgt] = masks[src, tgt][item_cuts[src], :]
        if tgt in item_cuts:
            masks[src, tgt] = masks[src, tgt][:, item_cuts[tgt]]

    # Split the muon into barrel and endcap regions, first the items
    muon_in_barrel = masks["MUON", "YokeBarrelCollection"].any(-1)
    muon_in_endcap = masks["MUON", "YokeEndcapCollection"].any(-1)

    items["MUONBarrel"] = items["MUON"][muon_in_barrel]
    items["MUONEndcap"] = items["MUON"][muon_in_endcap]

    if verbose:
        print(f"Split MUON {len(items['MUON'])} into MUONBarrel {len(items['MUONBarrel'])} and MUONEndcap {len(items['MUONEndcap'])}")

    # Now split the muon masks
    mask_keys = list(masks.keys())
    for src, tgt in mask_keys:
        if src == "MUON":
            masks["MUONBarrel", tgt] = masks[src, tgt][muon_in_barrel, :]
            masks["MUONEndcap", tgt] = masks[src, tgt][muon_in_endcap, :]

        if tgt == "MUON":
            masks[src, "MUONBarrel"] = masks[src, tgt][:, muon_in_barrel]
            masks[src, "MUONEndcap"] = masks[src, tgt][:, muon_in_endcap]

    # Alias items and their fields
    aliased_items = {}
    for item_name in items:
        aliased_item_name = item_aliases.get(item_name, item_name)

        aliased_items[aliased_item_name] = {}

        for field_name in items[item_name].fields:
            aliased_field_name = field_aliases.get(field_name, field_name)

            aliased_items[aliased_item_name][aliased_field_name] = items[item_name][field_name]

            if verbose:
                print(f"Aliased item {item_name}/{field_name} -> {aliased_item_name}/{aliased_field_name}")

    if verbose:
        print(separator)

    # Alias the masks
    aliased_masks = {}
    for src, tgt in masks:
        # For now just assume all items are using an alias
        aliased_masks[item_aliases[src], item_aliases[tgt]] = masks[src, tgt]

        if verbose:
            print(f"Aliased mask ({src}, {tgt}) -> ({item_aliases[src]}, {item_aliases[tgt]})")

    if verbose:
        print(separator)

    field_blacklist = [
        "contributions_begin",
        "contributions_end",
    ]

    # Now save the output items
    data_out = {}
    for item_name in output_items:
        for field in aliased_items[item_name]:
            if field in field_blacklist:
                continue

            output_item = ak.to_numpy(aliased_items[item_name][field])

            if output_item.dtype == np.float64:
                output_item = output_item.astype(np.float32)

            if output_item.dtype == np.int64:
                output_item = output_item.astype(np.int32)

            data_out[f"{item_name}.{field}"] = output_item

            if verbose:
                print(f"Saved item {item_name}.{field}, {output_item.shape}, {output_item.dtype}")

    # Now save the output masks, in sparse format to save space
    for src, tgt in output_masks:
        if (src, tgt) not in aliased_masks:
            aliased_masks[src, tgt] = aliased_masks[tgt, src].T

        mask_csr = csr_matrix(aliased_masks[src, tgt], dtype=bool)

        data_out[f"{src}_to_{tgt}_data"] = mask_csr.data
        data_out[f"{src}_to_{tgt}_indices"] = mask_csr.indices
        data_out[f"{src}_to_{tgt}_indptr"] = mask_csr.indptr
        data_out[f"{src}_to_{tgt}_shape"] = np.array(mask_csr.shape)

    # Cast any uints to ints
    for k, v in data_out.items():
        if v.dtype == np.uint64:
            data_out[k] = v.astype(np.int64)

    return data_out


def preprocess_file(
    in_dir: Path,
    out_dir: Path,
    filename: str,
    min_pt: float = 10.0,
    verbose: bool = False,
):
    in_file_path = in_dir / Path(filename).with_suffix(".root")

    print("=" * 100 + f"\nReading events from: {in_file_path}\n" + "=" * 100)
    file = uproot.open(in_file_path)

    # First read the metadata that is gives the codes to map between tree names and collections IDs
    metadata = file["podio_metadata;1"]

    codenames = {}
    namecodes = {}
    for code, name in zip(metadata["events___idTable/m_collectionIDs"].array()[0], metadata["events___idTable/m_names"].array()[0], strict=False):
        codenames[code] = name
        namecodes[name] = code

    # Get the event numbers that will be used to identify each event
    events_key = next(k for k in file if "events" in k)
    events = file[events_key]
    event_numbers = ak.to_numpy(ak.flatten(events["EventHeader/EventHeader.eventNumber"].array()))

    assert len(event_numbers) == 1000, f"Found less than 1000 events in {in_file_path}"

    # Make the directory to store events from this file in
    out_folder = Path(out_dir / Path(filename))
    out_folder.mkdir(parents=True, exist_ok=True)

    event_names = [f"{filename}_{event_number}" for event_number in event_numbers]
    completed_event_names = [f.stem for f in out_folder.glob("*.npz")]
    uncompleted_event_names = list(set(event_names) - set(completed_event_names))
    num_completed_events = len(completed_event_names)
    print(f"Found {len(completed_event_names)} completed events in {out_folder}\n" + "=" * 100)

    # Iterate over each event in the file
    for event_name in uncompleted_event_names:
        t0 = time.time()

        event_number = int(event_name.split("_")[-1])
        event = preprocess_event(events, event_number, namecodes, min_pt, verbose)
        out_event_path = out_folder / Path(event_name).with_suffix(".npz")
        np.savez(out_event_path, **event)

        dt = time.time() - t0
        num_particles = len(event["particle.PDG"])
        size_mb = sum(value.nbytes for value in event.values()) / (1024 * 1024)
        num_completed_events += 1
        print(
            f"Prepped event {event_name} ({num_completed_events}/{len(event_names)}) to {out_event_path}, "
            f"num_particles={num_particles}, size={size_mb:.2f}Mb, time={dt:.2f}s"
        )

    print("=" * 100 + f"\nPreprocessed events in {in_file_path} and saved them to {out_folder}\n" + "=" * 100)


def preprocess_files(in_dir: str, out_dir: str, overwrite: bool, parallel: bool = False):
    """Preprpocess root files into parquet files.

    Parameters
    ----------
    in_dir : str
        Directory of input root files
    out_dir : str
        Directory of where to save output numpy files
    """
    num_events_per_file = 1000

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    filenames = [path.stem for path in in_dir.glob("*.root")]

    print("=" * 100)
    print(f"Found {len(filenames)} files in {in_dir}")

    completed_filenames = [filename for filename in filenames if len(list((out_dir / Path(filename)).glob("*.npz"))) == num_events_per_file]

    # Determine which output folders have 1000 events in, and so are complete

    uncompleted_filenames = list(set(filenames) - set(completed_filenames))

    target_filenames = filenames if overwrite else uncompleted_filenames

    print(f"Found {len(completed_filenames)} completed files in {out_dir}")
    print(f"Set to prep {len(uncompleted_filenames)} files")

    if parallel:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        ntasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        target_filenames = [f for i, f in enumerate(target_filenames) if i % ntasks == task_id]
        print(f"Task {task_id} has been allocated {len(target_filenames)} files")

    print("=" * 100)

    for filename in target_filenames:
        preprocess_file(in_dir, out_dir, filename)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert root TTree files to numpy binary files")

    parser.add_argument("-i", "--in_dir", dest="in_dir", type=str, required=True, help="Input directory containing ROOT files")
    parser.add_argument("-o", "--out_dir", dest="out_dir", type=str, required=True, help="Output directory for parquet files")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing events or not.")
    parser.add_argument("--min_pt", type=float, required=False, default=10, help="Minimum pT cut to apply on particles, in MeV")
    parser.add_argument("--verbose", action="store_true", help="Whether to print extra info or not.")
    parser.add_argument("--parallel", action="store_true", help="Whether the script will be run on a SLURM array.")

    args = parser.parse_args()

    preprocess_files(
        args.in_dir,
        args.out_dir,
        args.overwrite,
        min_pt=args.min_pt,
        verbose=args.verbose,
        parallel=args.parallel,
    )
