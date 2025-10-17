#!/bin/bash

#SBATCH --job-name=cld-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/slurm_logs/slurm-%j.%x.out
#SBATCH --array 0-4

# Used for preprocessing raw CLD samples into binary parquet files used for training

# Move to workdir
cd /share/rcifdata/maxhart/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
#IN_DIR="/share/rcifdata/maxhart/data/cld/raw/val/"
IN_DIR="/share/rcif2/maxhart/data/cld/test/raw/"
OUT_DIR="/share/rcif2/maxhart/data/cld/test/prepped/"

# Python command that will be run
# Note we specify a minimum pT cut, particles below this will be removed
PYTORCH_CMD="python src/hepattn/experiments/cld/prep.py --in_dir $IN_DIR --out_dir $OUT_DIR --min_pt 10 --parallel"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --bind /share/rcifdata/maxhart,/share/lustre/maxhart,/share/rcif2/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"