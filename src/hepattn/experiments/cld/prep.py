import time
from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from particle.pdgid import is_hadron

# Specify the names of items / objects we want to save
item_names = [
    "MCParticles",
    "VXDTrackerHits",
    "VXDEndcapTrackerHits",
    "ITrackerHits",
    "ITrackerEndcapHits",
    "OTrackerHits",
    "OTrackerEndcapHits",
    "VertexBarrelCollection",
    "VertexEndcapCollection",
    "InnerTrackerBarrelCollection",
    "InnerTrackerEndcapCollection",
    "OuterTrackerBarrelCollection",
    "OuterTrackerEndcapCollection",
    "ECALBarrel",
    "ECALEndcap",
    "HCALBarrel",
    "HCALEndcap",
    "HCALOther",
    "MUON",
    "ECalBarrelCollection",
    "ECalEndcapCollection",
    "HCalBarrelCollection",
    "HCalEndcapCollection",
    "HCalRingCollection",
    "ECalBarrelCollectionContributions",
    "ECalEndcapCollectionContributions",
    "HCalBarrelCollectionContributions",
    "HCalEndcapCollectionContributions",
    "HCalRingCollectionContributions",
]

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
}

# Specify the mask/links to build between these items and particles
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
]

# Specify which masks to create by joining together existing masks
mask_joins = [
    ("VXDTrackerHits", "VertexBarrelCollection", "MCParticles"),
    ("VXDEndcapTrackerHits", "VertexEndcapCollection", "MCParticles"),
    ("ITrackerHits", "InnerTrackerBarrelCollection", "MCParticles"),
    ("ITrackerEndcapHits", "InnerTrackerEndcapCollection", "MCParticles"),
    ("OTrackerHits", "OuterTrackerBarrelCollection", "MCParticles"),
    ("OTrackerEndcapHits", "OuterTrackerEndcapCollection", "MCParticles"),
]

object_aliases = {
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
    "InnerTrackerBarrelCollection": "itb_particle",
    "InnerTrackerEndcapCollection": "ite_particle",
    "OuterTrackerBarrelCollection": "otb_particle",
    "OuterTrackerEndcapCollection": "ote_particle",
}

field_aliases = {
    "position": "pos",
    "vertex": "vtx.pos",
    "endpoint": "end.pos",
    # Aliases are applied in order, so string supersets must go first
    "momentumAtEndpoint": "end.mom",
    "momentum": "vtx.mom",
}

# Specify which items we actually want to save
output_items = [
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
]

# Specify which masks we actually want to save
output_masks = [
    ("VXDTrackerHits", "MCParticles"),
    ("VXDEndcapTrackerHits", "MCParticles"),
    ("ITrackerHits", "MCParticles"),
    ("ITrackerEndcapHits", "MCParticles"),
    ("OTrackerHits", "MCParticles"),
    ("OTrackerEndcapHits", "MCParticles"),
    ("ECALBarrel", "MCParticles"),
    ("ECALEndcap", "MCParticles"),
    ("HCALBarrel", "MCParticles"),
    ("HCALEndcap", "MCParticles"),
    ("HCALOther", "MCParticles"),
    ("MUON", "MCParticles"),
]

# Specify which items we actually want to save
aliased_output_items = [
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
    "muon",
]

aliased_output_hits = [
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
    "muon",
]

# Specify which masks we actually want to save
aliased_output_masks = [
    ("vtb", "particle"),
    ("vte", "particle"),
    ("itb", "particle"),
    ("ite", "particle"),
    ("otb", "particle"),
    ("ote", "particle"),
    ("ecb", "particle"),
    ("ebe", "particle"),
    ("hcb", "particle"),
    ("hce", "particle"),
    ("hco", "particle"),
    ("muon", "particle"),
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

max_num_hits = 15000


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


def prep_event(events, event_idx, namecodes, min_pt):  # noqa: C901, PLR0912, PLR0914, PLR0915
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

    # This is actually quite slow since we make use of scikit-hep particle, so we do it in preprocessing
    # Would be better if we can get a comprehensive list of every hadron pdgid
    items["MCParticles"]["class"] = np.array([get_particle_class(pids[i], charges[i]) for i in range(len(pids))])

    # Now build the masks that link the various items together
    masks = {}

    # First build the masks that use bidirectional truth links
    for relation, links in relations_links.items():
        link_src_cid = events[f"_{relation}_from/_{relation}_from.collectionID"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]
        link_src_idx = events[f"_{relation}_from/_{relation}_from.index"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]

        link_tgt_cid = events[f"_{relation}_to/_{relation}_to.collectionID"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]
        link_tgt_idx = events[f"_{relation}_to/_{relation}_to.index"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]

        for src, tgt in links:
            link_mask = (namecodes[src] == link_src_cid) & (namecodes[tgt] == link_tgt_cid)
            num_src = len(items[src])
            num_tgt = len(items[tgt])

            mask = np.full((num_src, num_tgt), False)
            mask[link_src_idx[link_mask], link_tgt_idx[link_mask]] = True
            masks[src, tgt] = mask

    # Now build the masks that use a single particle link
    for src in particle_links:
        tgt_cids = events[f"_{src}_particle/_{src}_particle.collectionID"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]
        tgt_idxs = events[f"_{src}_particle/_{src}_particle.index"].array(entry_start=event_idx, entry_stop=event_idx + 1)[0]

        tgt = "MCParticles"

        # Check that the particle links are indeed all mcparticles
        assert np.all(tgt_cids == namecodes[tgt])

        num_src = len(items[src])
        num_tgt = len(items[tgt])

        mask = np.full((num_src, num_tgt), False)
        mask[np.arange(num_src), tgt_idxs] = True
        masks[src, tgt] = mask

    # Join together existing masks to make new masks
    for src, link, tgt in mask_joins:
        src_link_mask = masks[src, link]
        link_tgt_mask = masks[link, tgt]

        masks[src, tgt] = np.dot(src_link_mask, link_tgt_mask)

    items["MCParticles"]["momentum.t"] = np.sqrt(items["MCParticles"]["momentum.x"] ** 2 + items["MCParticles"]["momentum.y"] ** 2)

    # Now apply any cuts to the items
    # Name some cuts for convenience
    particle_cuts = {
        "Status 0": items["MCParticles"]["generatorStatus"] == 0,
        "Status 1": items["MCParticles"]["generatorStatus"] == 1,
        "Status 2": items["MCParticles"]["generatorStatus"] == 2,
        "10 MeV": items["MCParticles"]["momentum.t"] >= min_pt * 0.001,  # Minimum pT cut is given in MeV
        "Electron Beam Remenant": (items["MCParticles"]["momentum.y"] == 0) & (items["MCParticles"]["PDG"] == 11),
        "Photon Beam Remenant": (items["MCParticles"]["momentum.y"] == 0) & (items["MCParticles"]["PDG"] == 22),
    }
    particle_cuts["Beam Remenant"] = particle_cuts["Electron Beam Remenant"] | particle_cuts["Photon Beam Remenant"]
    particle_cuts["Good Status"] = particle_cuts["Status 0"] | particle_cuts["Status 1"] | particle_cuts["Status 2"]

    # Keep only particles that have status codes 1 or 2, and that are not beam remenants
    item_cuts = {"MCParticles": particle_cuts["Good Status"] & particle_cuts["10 MeV"] & ~particle_cuts["Beam Remenant"]}

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

    # Keep only the hits / items we want to save
    data_out = {}
    for item_name in output_items:
        for field in items[item_name].fields:
            data_out[f"{item_name}.{field}"] = ak.to_numpy(items[item_name][field])

    # Keep only the masks we want to save
    # Save them in sparse format to save space
    for src, tgt in output_masks:
        data_out[f"{src}_to_{tgt}_idxs"] = np.argwhere(masks[src, tgt])

    for k, v in data_out.items():
        if v.dtype == np.uint64:
            data_out[k] = v.astype(np.int64)

    # Build they key alias map
    key_alias_map = {}
    for key in data_out:
        aliased_key = key

        for name, alias in object_aliases.items():
            aliased_key = aliased_key.replace(name, alias)

        for name, alias in field_aliases.items():
            aliased_key = aliased_key.replace(name, alias)

        key_alias_map[key] = aliased_key

    # Apply the alias map
    data_out_aliased = {}
    for key, value in data_out.items():
        data_out_aliased[key_alias_map[key]] = value

    data_out = ak.Array({key: [value] for key, value in data_out_aliased.items()})

    return data_out


def preprocess(in_dir: str, out_dir: str, overwrite: bool, min_pt: float = 10.0, max_num_particles: int = 1000):
    """Preprpocess root files into parquet files.

    Parameters
    ----------
    in_dir : str
        Directory of input root files
    out_dir : str
        Directory of where to save output parquet files
    overwrite : bool
        Whether to overwrite existing output files or not, by default false
    """

    # Iterate over all of the files ending in .root in the input directory
    for in_file in Path(in_dir).iterdir():
        if Path(in_file).suffix != ".root":
            continue

        file = uproot.open(in_file)

        print(f"\nReading events from: {in_file}\n")

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

        for event_idx, event_number in enumerate(event_numbers):
            t0 = time.time()

            # Determine the name of the output event for this file, skip if it exists and overwrite is false
            event_name = in_file.stem.replace("_condor", "") + f"_{str(event_number + 1).zfill(8)}_{int(min_pt)}_{max_num_particles}"
            output_file = Path(out_dir) / Path(f"{event_name}.parquet")

            if output_file.is_file() and not overwrite:
                print(f"Skipping {output_file} as it already exists")
                continue

            # Process the event to get the hit / particle data and the links between them
            event_data = prep_event(events, event_idx, namecodes, min_pt)

            num_particles = len(event_data["particle.PDG"][0])
            num_hits = sum(len(event_data[f"{k}.type"][0]) for k in aliased_output_hits)

            if num_particles > max_num_particles:
                print(f"Skipping {in_file} as has {num_particles} partcles")
                continue

            if num_hits > max_num_hits:
                print(f"Skipping {in_file} as has {num_hits} hits")
                continue

            # Save the processed event data as a dict of numpy arrays
            ak.to_parquet(event_data, output_file)

            dt = time.time() - t0
            print(f"Prepped {output_file}: {num_hits} hits, {num_particles} particles, took {dt:.3f}s")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert root TTree files to binary parquet files")

    parser.add_argument("-i", "--in_dir", dest="in_dir", type=str, required=True, help="Input directory containing ROOT files")
    parser.add_argument("-o", "--out_dir", dest="out_dir", type=str, required=True, help="Output directory for parquet files")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing events or not.")
    parser.add_argument("--min_pt", type=float, help="Minimum pT cut to apply on particles, in MeV")
    parser.add_argument(
        "--max_num_particles", type=int, required=False, default=1000, help="Maximum number of particles in the event for it to be saved accepted."
    )

    args = parser.parse_args()

    preprocess(args.in_dir, args.out_dir, args.overwrite, args.min_pt, args.max_num_particles)
