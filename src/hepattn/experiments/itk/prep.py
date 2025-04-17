from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

# A script for preprocessing ITk CSV files into parquet binary files


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

    for parts_in_path in Path(in_dir).glob("**/*-particles.csv"):
        event_name = parts_in_path.stem.replace("-particles", "")
        truth_in_path = Path(in_dir) / Path(event_name + "-truth.csv")
        if not truth_in_path.exists():
            print(f"Skipping {event_name} as found no corresponding truth input file")
            continue

        parts_out_path = Path(out_dir) / Path(event_name + "-parts.parquet")
        pixel_out_path = Path(out_dir) / Path(event_name + "-pixel.parquet")
        strip_out_path = Path(out_dir) / Path(event_name + "-strip.parquet")

        if parts_out_path.exists() & pixel_out_path.exists() & strip_out_path.exists() & (overwrite is False):
            print(f"Skipping {event_name} as found existing truth, pixel and strip output files")
            continue

        # Load the particles and hits information
        parts = pd.read_csv(parts_in_path, dtype={"particle_id": np.int64})
        truth = pd.read_csv(truth_in_path, dtype={"tgt_id": np.int64})

        # Rename hit fields to be compatible with TrackML
        truth = truth.rename(columns={"tgt_pid": "particle_id"})

        # Rename particle fields to be compatible with TrackML
        parts = parts.rename(columns={"q": "charge", "particle_type": "pdgId", "nhits": "num_clusters"})

        # ITk gives things in MeV, convert to GeV to be consistent with TrackML
        for field in ["px", "py", "pz", "pt"]:
            parts[field] /= 1000

        # Separate out pixel and strip hits
        pixel = truth[truth["hardware"] == "PIXEL"].copy()
        strip = truth[truth["hardware"] == "STRIP"].copy()

        # Have to drop non-numeric fields
        pixel = pixel.drop(columns=["hardware"])
        strip = strip.drop(columns=["hardware"])

        # Drop the duplicate pixel fields
        pixel = pixel.drop(columns=[*list(filter(lambda col: col.endswith("_2"), pixel.columns)), "particle_id"])
        pixel = pixel.rename(columns={col: col.replace("_1", "") for col in filter(lambda col: col.endswith("_1"), pixel.columns)})

        # Save the output binary files
        parts.to_parquet(parts_out_path)
        pixel.to_parquet(pixel_out_path)
        strip.to_parquet(strip_out_path)

        print(f"Prepped {event_name}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prep ITk CSV files")
    parser.add_argument("-i", "--in_dir", dest="in_dir", type=str, required=True, help="Input directory containing csv files")
    parser.add_argument("-o", "--out_dir", dest="out_dir", type=str, required=True, help="Where to save output binary files")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    preprocess(args.in_dir, args.out_dir, args.overwrite)
