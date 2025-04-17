#!/bin/bash

#SBATCH --job-name=trackml-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out
#SBATCH --error=/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out

# Used for preprocessing raw TrackML samples into binary parquet files used for training

# Move to workdir
cd /share/rcifdata/maxhart/hepattn-test/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
IN_DIR="/share/rcifdata/maxhart/data/trackml/raw/train/"
OUT_DIR="/share/rcifdata/maxhart/data/trackml/prepped/train/"

# Python command that will be run
PYTORCH_CMD="python src/hepattn/experiments/trackml/prep.py --in_dir $IN_DIR --out_dir $OUT_DIR"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn-test/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
