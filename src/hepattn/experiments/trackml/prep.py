from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from hepattn.experiments.trackml import cluster_features

# A script for preprocessing TrackML CSV files into parquet binary files


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def preprocess(in_dir: str, out_dir: str, overwrite: bool):
    """Preprocess csv files into parquet files, and separate pixel and sct hits.

    Parameters
    ----------
    in_dir : str
        Directory of input root files
    out_dir : str
        Directory of where to save output parquet files
    overwrite : bool
        Whether to overwrite existing output files or not, by default false
    """

    # Load in the detector geometry config file
    detector_config_path = Path(in_dir).parent / "detectors.csv"
    assert detector_config_path.is_file(), f"Missing detector config at {detector_config_path}"

    for truth_in_path in Path(in_dir).glob("event*-truth.csv.gz"):
        event_name = str(truth_in_path.name).replace("-truth.csv.gz", "")

        # Other input files along with the particles input file that we require
        parts_in_path = Path(in_dir) / Path(event_name + "-particles.csv.gz")
        hits_in_path = Path(in_dir) / Path(event_name + "-hits.csv.gz")
        cells_in_path = Path(in_dir) / Path(event_name + "-cells.csv.gz")

        # Check that all of the input files required exist, otherwise skip this event
        if not is_valid_file(truth_in_path):
            print(f"Skipping {event_name} as found no corresponding truth input file")
            continue

        if not is_valid_file(parts_in_path):
            print(f"Skipping {event_name} as found no corresponding parts input file")
            continue

        if not is_valid_file(hits_in_path):
            print(f"Skipping {event_name} as found no corresponding hits input file")
            continue

        if not is_valid_file(cells_in_path):
            print(f"Skipping {event_name} as found no corresponding cells input file")
            continue

        # Define the output file paths
        parts_out_path = Path(out_dir) / Path(event_name + "-parts.parquet")
        hits_out_path = Path(out_dir) / Path(event_name + "-hits.parquet")

        # Check if we have already prepped the event
        if parts_out_path.exists() and hits_out_path.exists() and (overwrite is False):
            print(f"Skipping {event_name} as found existing parts and hits output files")
            continue

        # Load the input information
        parts = pd.read_csv(parts_in_path)
        truth = pd.read_csv(truth_in_path)
        cells = pd.read_csv(cells_in_path)
        hits = pd.read_csv(hits_in_path)

        assert (truth.index == hits.index).all()

        # Add truth particle info to hits
        hits["particle_id"] = truth["particle_id"]

        # Add truth hit-particle intersection/momentum info
        for field in ["tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"]:
            hits[field] = truth[field]

        # Add the charge cluster info to the hits
        hits = cluster_features.append_cell_features(hits, cells, detector_config_path)

        # Save the output binary files
        parts.to_parquet(parts_out_path)
        hits.to_parquet(hits_out_path)

        print(f"Prepped {event_name}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prep TrackML CSV files")
    parser.add_argument("-i", "--in_dir", dest="in_dir", type=str, required=True, help="Input directory containing csv files")
    parser.add_argument("-o", "--out_dir", dest="out_dir", type=str, required=True, help="Where to save output binary files")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    preprocess(args.in_dir, args.out_dir, args.overwrite)
