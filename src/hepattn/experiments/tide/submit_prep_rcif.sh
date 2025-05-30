#!/bin/bash

#SBATCH --job-name=tide-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out
#SBATCH --array 0-8

# Used for preprocessing raw tide samples into binary parquet files used for training

# Move to workdir
cd /share/rcifdata/maxhart/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
#IN_DIR="/share/rcifdata/maxhart/data/tide/raw/val/"
IN_DIR="/share/lustre/maxhart/data/ambi/user.srettie.800030.flatpT_Zprime_Extended.e7954_s3582_r12643_20241122_nom_with_rois_EXT1/"
OUT_DIR="/share/rcifdata/maxhart/data/ambi/tmp"

# Python command that will be run
# Note we specify a minimum pT cut, particles below this will be removed
PYTORCH_CMD="python src/hepattn/experiments/tide/prep.py -i $IN_DIR -o $OUT_DIR --parallel"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --bind /share/rcifdata/maxhart,/share/lustre/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
