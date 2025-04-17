#!/bin/bash

#SBATCH --job-name=itk-prep
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/itk/slurm_logs/slurm-%j.%x.out

# Used for preprocessing raw ITk samples into binary parquet files used for training

# Move to workdir
cd /share/rcifdata/maxhart/hepattn-test/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the preprocessing
echo "Running preprocessing script..."

# Change these to wherever your data is, or get access to them
IN_DIR="/share/rcifdata/maxhart/data/itk/ATLAS-P2-RUN4-03-00-00_Rel.24_ttbar_uncorr_pu200_dumpv5_acorn2.0.0_acorn_data_reading_output_trainset/"
OUT_DIR="/share/rcifdata/maxhart/data/itk/train/"

# Python command that will be run
PYTORCH_CMD="python src/hepattn/experiments/itk/prep.py --in_dir $IN_DIR --out_dir $OUT_DIR"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn-test/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
