import argparse
from pathlib import Path

import numpy as np


def main(top_dir):
    # find all dirs which have a subdir "times"
    time_dirs = [d for d in top_dir.glob("**/times") if d.is_dir()]

    for time_dir in time_dirs:
        # find all .npy files in the dir with "times" in the name
        npy_times = list(time_dir.glob("*times.npy"))

        # check there is only one
        if len(npy_times) != 1:
            raise ValueError(f"Expected only one times.npy file in {time_dir}, found {len(npy_times)}")
        times_path = npy_times[0]

        # load it
        name = str(times_path.name).split("_times")[0]
        times = np.load(times_path)
        print(f"{name:<40} mean time: {times.mean():.1f} ± {times.std():.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Report timing statistics")
    parser.add_argument("path", type=str, help="Path to the top directory")
    args = parser.parse_args()
    main(Path(args.path))
