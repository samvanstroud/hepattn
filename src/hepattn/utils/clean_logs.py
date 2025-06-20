import argparse
import shutil
from pathlib import Path


def delete_dirs_without_subdir(path, subdir, dry_run):
    folder_path = Path(path)
    count = 0

    for directory in folder_path.iterdir():
        if not directory.is_dir():
            continue

        # Check if directory contains only yaml files
        files = list(directory.glob("*"))
        if not files or not all(file.suffix == ".yaml" for file in files):
            continue

        # Delete directory if it doesn't contain the required subdirectory
        # (indicates that no checkpoints were saved)
        if not (directory / subdir).is_dir():
            print(f"Deleting directory: {directory}")
            if not dry_run:
                shutil.rmtree(directory)
            count += 1

    print(f"Deleted {count} directories")


def main(args=None):
    parser = argparse.ArgumentParser(description="Delete directories which do not contain a specified subdirectory")
    parser.add_argument("--path", required=True, type=str, help="Path to the folder to clean")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--subdir", default="ckpts", type=str, help="Name of the subdirectory to check for in each directory")
    args = parser.parse_args(args)

    delete_dirs_without_subdir(**vars(args))
    if args.dry_run:
        print("Dry run enabled. No directories were deleted.")


if __name__ == "__main__":
    main()
