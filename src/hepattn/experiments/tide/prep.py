from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import h5py
import numba
import numpy as np
import uproot
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm


# ruff: noqa

###############################################################
# Define item hit field aliases
###############################################################

# Cluster fields that are common between pixel and SCT
clus_field_aliases = {
    "x": "x",
    "y": "y",
    "z": "z",
    "layer": "layer",
    "bec": "bec",
    "charge": "charge",
    "module_global_x": "mod_x",
    "module_global_y": "mod_y",
    "module_global_z": "mod_z",
    "module_normal_x": "mod_norm_x",
    "module_normal_y": "mod_norm_y",
    "module_normal_z": "mod_norm_z",
    "module_local_x": "mod_loc_x",
    "module_local_y": "mod_loc_y",
}

# Pixel specific fields
pix_field_aliases = clus_field_aliases | {
    "LorentzShift": "lshift",
    "NN_vectorOfPitchesY": "pitches",
}

# SCT specific fields
sct_field_aliases = clus_field_aliases | {
    "side": "side",
    "SiWidth": "width",
    # "rdo_strip": "strip",
    # "rdo_groupsize": "groupsize",
}

###############################################################
# Define track-hit object field aliases
###############################################################

sudo_pix_field_aliases = {
    "NN_positions_indexX": "loc_x",
    "NN_positions_indexY": "loc_y",
    "NN_phi": "phi",
    "NN_theta": "theta",
    "sihit_energyDeposits": "energy",
    "sihit_meanTimes": "time",
    "sihit_startPosXs": "mod_x0",
    "sihit_startPosYs": "mod_y0",
    "sihit_startPosZs": "mod_z0",
    "sihit_endPosXs": "mod_x1",
    "sihit_endPosYs": "mod_y1",
    "sihit_endPosZs": "mod_z1",
}

###############################################################
# Track field aliases
###############################################################

# Track fields that are common betweeen all track collections
trk_field_aliases = {
    "pt": "pt",
    "eta": "eta",
    "phi": "phi",
    "d0": "d0",
    "z0": "z0",
    "vx": "vx",
    "vy": "vy",
    "vz": "vz",
    "charge": "q",
    "origin": "origin",
}

# Pseudotrack specific fields
sudo_field_aliases = trk_field_aliases | {
    "BHadronPt": "bhad_pt",
    "isComplete": "complete",
    "hasReco": "has_reco",
    "hasSiSP": "has_sisp",
}

# Standard reconstruction track fields
reco_field_aliases = trk_field_aliases

# SiSp track fields
sisp_field_aliases = trk_field_aliases

# ROI level fields
roi_field_aliases = {
    "e": "energy",
    "eta": "eta",
    "phi": "phi",
    "m": "mass",
}

###############################################################
# Define which masks should be built
###############################################################

masks = [
    ("pseudotracks", "pixel"),
    ("pseudotracks", "strip"),
    ("sisptracks", "pixel"),
    ("sisptracks", "strip"),
    ("recotracks", "pixel"),
    ("recotracks", "strip"),
]

###############################################################
# Deine the object name / item aliases
###############################################################

item_aliases = {
    "pixel": "pix",
    "strip": "sct",
    "pseudotracks": "sudo",
    "recotracks": "reco",
    "sisptracks": "sisp",
    "roi": "roi",
}

item_fields = {
    "pixel": pix_field_aliases,
    "strip": sct_field_aliases,
    "pseudotracks": sudo_field_aliases,
    "sisptracks": sisp_field_aliases,
    "recotracks": reco_field_aliases,
    "roi": roi_field_aliases,
}

###############################################################
# Which items will be saved
###############################################################

output_items = [
    "pix",
    "sct",
    "sudo",
    "sisp",
    "reco",
    "roi",
]

###############################################################
# Which masks will be saved
###############################################################

output_masks = [
    ("sudo", "pix"),
    ("sudo", "sct"),
    ("sisp", "pix"),
    ("sisp", "sct"),
    ("reco", "pix"),
    ("reco", "sct"),
]


@numba.njit()
def build_track_hit_mask_bcodes(tracks_bcode, hits_bcodes):
    track_hit_mask = np.zeros(shape=(len(tracks_bcode), len(hits_bcodes)), dtype=np.bool_)

    for track_idx, track_bcode in enumerate(tracks_bcode):
        for hit_idx, hit_bcodes in enumerate(hits_bcodes):
            for hit_bcode in hit_bcodes:
                if hit_bcode == track_bcode:
                    track_hit_mask[track_idx, hit_idx] = True

    return track_hit_mask


@numba.njit()
def build_track_hit_mask_ids(tracks_hits_ids, hits_id):
    track_hit_mask = np.zeros(shape=(len(tracks_hits_ids), len(hits_id)), dtype=np.bool_)

    for track_idx, track_hits_ids in enumerate(tracks_hits_ids):
        for hit_idx, hit_id in enumerate(hits_id):
            for track_hit_id in track_hits_ids:
                if track_hit_id == hit_id:
                    track_hit_mask[track_idx, hit_idx] = True

    return track_hit_mask


def build_track_hit_field(tracks_bcode, hits_bcodes, hits_fields):
    track_hit_field = np.zeros(shape=(len(tracks_bcode), len(hits_bcodes)), dtype=np.float32)

    for track_idx, track_bcode in enumerate(tracks_bcode):
        for hit_idx, hit_bcodes in enumerate(hits_bcodes):
            for bcode_idx, hit_bcode in enumerate(hit_bcodes):
                if hit_bcode == track_bcode:
                    track_hit_field[track_idx, hit_idx] = hits_fields[hit_idx][bcode_idx]

    return track_hit_field


def preprocess_file(
    input_dir: Path,
    output_dir: Path,
    file_name: str,
    verbose: bool = False,
    overwrite: bool = False,
):
    in_file_path = input_dir / Path(f"{file_name}.root")
    out_file_path = output_dir / Path(f"{file_name}.h5")
    out_file_path_tmp = output_dir / Path(f"{file_name}.tmp")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Skip the file if it has already been preprocessed
    if out_file_path.is_file() and not overwrite:
        print(f"Found existing completed file {out_file_path} and overwrite is false, so skipping")
        return

    # Check if there is an existing tmp file which has been partially preprocessed
    if out_file_path_tmp.is_file():
        # Check which ROIs have already been prepped
        print(f"Found existing uncompleted file {out_file_path_tmp}")
        with h5py.File(out_file_path_tmp, "r") as out_file:
            completed_roi_ids = {int(roi_id) for roi_id in out_file}
    else:
        print(f"Did not find existing uncompleted file {out_file_path_tmp}")
        completed_roi_ids = set()

    in_file = uproot.open(in_file_path)["hadronic_roi_tree"]

    print(f"Reading fields from {in_file_path}")
    max_num_rois = -1

    # Load the file into memory
    data = {}
    for key in tqdm(in_file.keys()):
        # Split up the clusters into pixels and strips
        if key.startswith("cluster_"):
            is_pixel = in_file["cluster_isPixel"].array(entry_stop=max_num_rois)
            data[key.replace("cluster_", "pixel_")] = in_file[key].array(entry_stop=max_num_rois)[is_pixel]
            data[key.replace("cluster_", "strip_")] = in_file[key].array(entry_stop=max_num_rois)[~is_pixel]
        else:
            data[key] = in_file[key].array(entry_stop=max_num_rois)

    print(f"Done reading fields from {in_file_path}")

    # user.srettie.42156221.EXT1._000193.tree.tmp -> 000193
    file_id = str(file_name.replace("user.srettie.42156221.EXT1._", "").replace(".tree", "").replace("hadronic_rois", ""))

    # Maps the ROI id to the ROI index in the input file
    roi_ids_idxs = {int(file_id + str(roi_idx).zfill(6)): roi_idx for roi_idx in range(len(data["roi_e"]))}
    roi_ids = set(roi_ids_idxs.keys())
    uncompleted_roi_ids = roi_ids - completed_roi_ids

    target_roi_ids = roi_ids if overwrite else uncompleted_roi_ids

    print(f"Completed {len(completed_roi_ids)} of {len(roi_ids)} ROIs")
    print(f"Preprocessing {len(uncompleted_roi_ids)} ROIs")

    # Now start processing the ROIs
    # Write to a tmp file incase the process ends before we are finished
    with h5py.File(out_file_path_tmp, "w") as out_file:
        for roi_id in tqdm(target_roi_ids):
            roi_idx = roi_ids_idxs[roi_id]
            roi_data = {"roi_id": np.array([roi_id], dtype=np.int64)}

            # Build the objects
            for item_name, field_aliases in item_fields.items():
                for field_name, field_alias in field_aliases.items():
                    item_alias = item_aliases[item_name]
                    roi_data[f"{item_alias}_{field_alias}"] = ak.to_numpy(data[f"{item_name}_{field_name}"][roi_idx]).astype(np.float32)

            # Build the masks
            for track, hit in masks:
                # For pseduotracks we use the barcode so the mask will definitely match
                # the assignment that is sued for the sudo-pix fields
                if track == "pseudotracks":
                    track_hit_mask = build_track_hit_mask_bcodes(data[f"{track}_barcode"][roi_idx], data[f"{hit}_sihit_barcodes"][roi_idx])
                else:
                    track_hit_mask = build_track_hit_mask_ids(data[f"{track}_to_cluster_ids"][roi_idx], data[f"{hit}_id"][roi_idx])

                # Convert to a sparse binary matrix
                mask_csr = csr_matrix(track_hit_mask, dtype=bool)

                # Save under the item name aliases
                track_alias = item_aliases[track]
                hit_alias = item_aliases[hit]

                # Save the mask in sparse format to save space
                roi_data[f"{track_alias}_{hit_alias}_valid_data"] = mask_csr.data
                roi_data[f"{track_alias}_{hit_alias}_valid_indices"] = mask_csr.indices
                roi_data[f"{track_alias}_{hit_alias}_valid_indptr"] = mask_csr.indptr
                roi_data[f"{track_alias}_{hit_alias}_valid_shape"] = np.array(mask_csr.shape)

            # We built the sudo pix fields using the valid mask, since if we just used the track-hit fields,
            # track-hit fields that had a value exactly equal to zero would get marked as an invalid, and the resulting
            # sparse matrix of values would have a shape different shape to the sparse valid mask
            sudo_pix_valid = build_track_hit_mask_bcodes(data[f"pseudotracks_barcode"][roi_idx], data[f"pixel_sihit_barcodes"][roi_idx])
            sudo_idx, pix_idx = np.nonzero(sudo_pix_valid)

            # We built the sudo pix fields using the valid mask, since if we just used the track-hit fields,
            # track-hit fields that had a value exactly equal to zero would get marked as an invalid, and the resulting
            # sparse matrix of values would have a shape different shape to the sparse valid mask
            sudo_pix_valid = build_track_hit_mask_bcodes(data[f"pseudotracks_barcode"][roi_idx], data[f"pixel_sihit_barcodes"][roi_idx])
            sudo_idx, pix_idx = np.nonzero(sudo_pix_valid)

            # Build the track-hit fields
            for field_name, field_alias in sudo_pix_field_aliases.items():
                sudo_pix_field = build_track_hit_field(
                    data["pseudotracks_barcode"][roi_idx], data["pixel_sihit_barcodes"][roi_idx], data[f"pixel_{field_name}"][roi_idx]
                )
                # Use the valid mask indices to prevent field values equal to zero from being marked as empty/sparse
                sudo_pix_field_csr = csr_matrix(
                    (sudo_pix_field[sudo_idx, pix_idx], (sudo_idx, pix_idx)),
                    shape=sudo_pix_valid.shape,
                    dtype=np.float32,
                )

                roi_data[f"sudo_pix_{field_alias}_data"] = sudo_pix_field_csr.data

                # The number of indices need not matche exctly since we can have true zeros in the mask
                assert np.all(sudo_pix_field_csr.shape == roi_data["sudo_pix_valid_shape"])

            # Build the pixel charge matrix in sparse format
            pixel_charge_mats = ak.to_numpy(data["pixel_NN_matrixOfCharge"][roi_idx])
            pixel_charge_mats_csr = csr_matrix(pixel_charge_mats)

            roi_data["pix_charge_matrix_data"] = pixel_charge_mats_csr.data
            roi_data["pix_charge_matrix_indices"] = pixel_charge_mats_csr.indices
            roi_data["pix_charge_matrix_indptr"] = pixel_charge_mats_csr.indptr
            roi_data["pix_charge_matrix_shape"] = np.array(pixel_charge_mats_csr.shape)

            # Now save the data to the file
            roi_group = out_file.create_group(str(roi_id))
            for key, value in roi_data.items():
                roi_group.create_dataset(key, data=value)

                if verbose:
                    print(key, value.shape, value.dtype)

    # Rename the temp file to the final output file name now we are done
    out_file_path_tmp.rename(out_file_path)
    print(f"Preprocessed and saved {file_name}")


def preprocess_files(input_dir: str, output_dir: str, overwrite: bool, parallel: bool = False, **kwargs):
    """Preprpocess root files into parquet files.

    Parameters
    ----------
    input_dir : str
        Directory of input root files
    output_dir : str
        Directory of where to save output numpy files
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Get all available input root files
    input_file_names = [path.stem for path in input_dir.glob("*.root")]

    print(f"Found {len(input_file_names)} files in {input_dir}")

    # If overwrite is true, we have to preprocess all available files, otherwise
    # we just have to prep unprepped files
    completed_file_names = [path.stem for path in input_dir.glob("*.h5")]
    uncompleted_file_names = list(set(input_file_names) - set(completed_file_names))
    target_file_names = input_file_names if overwrite else uncompleted_file_names

    print(f"Found {len(completed_file_names)} completed files in {output_dir}")
    print(f"Set to preprocess {len(target_file_names)} files")

    # Select just a subset of the files if we are in a SLURM array
    if parallel:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        target_file_names = [f for i, f in enumerate(target_file_names) if i % num_tasks == task_id]
        print(f"Task {task_id} has been allocated {len(target_file_names)} files")

    # Now preprocess the files
    for file_name in target_file_names:
        preprocess_file(input_dir, output_dir, file_name, overwrite=overwrite)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert root TTree files to numpy binary files")

    parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True, help="Input directory containing ROOT files")
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, required=True, help="Output directory for parquet files")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing events or not.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print extra info or not.")
    parser.add_argument("--parallel", action="store_true", help="Whether the script will be run on a SLURM array.")

    args = parser.parse_args()

    preprocess_files(
        args.input_dir,
        args.output_dir,
        args.overwrite,
        verbose=args.verbose,
        parallel=args.parallel,
    )
