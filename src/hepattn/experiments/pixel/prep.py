import os
from argparse import ArgumentParser
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot
import yaml


def preprocess_file(
    input_dir: Path,
    output_dir: Path,
    file_name: str,
    config: dict,
    verbose: bool = False,
    overwrite: bool = False,
):
    in_file_path = input_dir / Path(file_name + config["input_file_extension"])
    out_file_path = output_dir / Path(f"{file_name}.h5")
    out_file_path_tmp = output_dir / Path(f"{file_name}.tmp")

    # Skip the file if it has already been preprocessed
    if out_file_path.is_file() and not overwrite:
        print(f"Found existing completed file {out_file_path} and overwrite is false, so skipping")
        return

    # Check if there is an existing tmp file which has been partially preprocessed
    if out_file_path_tmp.is_file():
        print(f"Found existing uncompleted file {out_file_path_tmp}")
    else:
        print(f"Did not find existing uncompleted file {out_file_path_tmp}")

    in_file = uproot.open(in_file_path)[config["tree_name"]]

    max_num_clusters = -1

    # Dimensions are event, cluster, particles on cluster
    cluster_barcode = in_file[config["fields_prefix"] + "." + "NN_barcode"].array(entry_stop=max_num_clusters)

    # Flatten the event and cluster dimensions
    cluster_multiplcity = ak.to_numpy(ak.flatten(ak.count(cluster_barcode, axis=-1), axis=1))
    print(f"Preproessing {len(cluster_multiplcity)} clusters from {in_file_path}")

    # Build mask which will be true for clusters we want to retain
    # Remove clusters with no particles on them
    cluster_mask = cluster_multiplcity > 0
    rng_generator = np.random.default_rng()

    for multiplicity, subsample_frac in config["cluster_multiplicity_subsample_frac"].items():
        multiplicity_mask = cluster_multiplcity == multiplicity
        subsample_mask = rng_generator.uniform(size=len(cluster_mask)) > float(subsample_frac)

        # Remove clusters that have this multiplicity but fail the subsample mask
        cluster_mask &= ~(multiplicity_mask & subsample_mask)

    cluster_fields = config["cluster_fields"]
    particle_fields = config["particle_fields"]
    pixel_fields = config["pixel_fields"]

    fields = cluster_fields | particle_fields | pixel_fields

    with h5py.File(out_file_path_tmp, "w") as out_file:
        # Save the cluster multiplicity for easy selection
        out_file.create_dataset("cluster_multiplicity", data=cluster_multiplcity[cluster_mask])

        # Make and save a unique cluster ID
        file_id = str(in_file_path).split("EXT0._")[-1].split(".")[0][2:]
        cluster_ids = [int(file_id + str(i).zfill(8)) for i in range(len(cluster_multiplcity[cluster_mask]))]
        out_file.create_dataset("cluster_id", data=cluster_ids, dtype=np.int64)

        for field, alias in fields.items():
            field_data = in_file[config["fields_prefix"] + "." + field].array(entry_stop=max_num_clusters)

            # Flatten out the event dimension and apply the mask
            field_data = ak.flatten(field_data, axis=1)

            # Apply the mask to discard unwanted clusters
            field_data = field_data[cluster_mask]

            # If the field is a per-cluster field, we can just save it as a regular array dataset
            if field in cluster_fields:
                field_data = ak.to_numpy(field_data)
                out_file.create_dataset(f"cluster_{alias}", data=field_data)

            # If the field is a particle field, we need to save it as a variable length dataset
            if field in particle_fields:
                # TODO: This is very slow, can it be faster?
                field_data = ak.to_list(field_data)
                field_data = [np.array(field_data[i]) for i in range(len(field_data))]
                out_file.create_dataset(f"particle_{alias}", data=field_data, dtype=h5py.vlen_dtype(np.float32))

            # Pixels are also variable length
            if field in pixel_fields:
                field_data = [ak.to_numpy(field_data[i]) for i in range(len(field_data))]
                out_file.create_dataset(f"pixel_{alias}", data=field_data, dtype=h5py.vlen_dtype(np.float32))

            if verbose:
                print(f"Preprocessed {field} of shape {out_file[alias].shape} and dtype {out_file[alias].dtype}")

    # Rename the temp file to the final output file name now we are done
    out_file_path_tmp.rename(out_file_path)
    print(f"Preprocessed and saved {out_file_path}")


def preprocess_files(config_path: str, overwrite: bool, parallel: bool = False, verbose: bool = False):
    """Preprpocess root files into parquet files.

    Parameters
    ----------
    input_dir : str
        Directory of input root files
    output_dir : str
        Directory of where to save output numpy files
    """
    with Path.open(config_path) as f:
        config = yaml.safe_load(f)["preprocessing"]

    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])

    # Get all available input root files
    input_file_names = [path.stem for path in input_dir.glob("*" + config["input_file_extension"])]

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
        preprocess_file(input_dir, output_dir, file_name, config, overwrite=overwrite, verbose=verbose)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert ROOT files into slimmed HDF5 files.")

    parser.add_argument("-c", "--config_path", dest="config_path", type=str, required=True, help="Preprocessing config path.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing events or not.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print extra info or not.")
    parser.add_argument("--parallel", action="store_true", help="Whether the script will be run on a SLURM array.")

    args = parser.parse_args()

    preprocess_files(
        args.config_path,
        args.overwrite,
        verbose=args.verbose,
        parallel=args.parallel,
    )
