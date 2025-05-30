#!/bin/bash

#SBATCH --job-name=trackml-train
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out


# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# Print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Move to workdir
cd /share/rcifdata/maxhart/hepattn-test/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

# Python command that will be run
# PYTORCH_CMD="python src/hepattn/experiments/trackml/run_tracking.py fit --config src/hepattn/experiments/trackml/configs/tracking.yaml"

# To do filtering instead
PYTORCH_CMD="python src/hepattn/experiments/trackml/run_filtering.py fit --config src/hepattn/experiments/trackml/configs/filtering.yaml"

# Do testing instead
#PYTORCH_CMD="python src/hepattn/experiments/trackml/run_filtering.py test --config /share/rcifdata/maxhart/hepattn-test/hepattn/logs/ec_eta4_20250409-T184858/config.yaml --ckpt_path /share/rcifdata/maxhart/hepattn-test/hepattn/logs/ec_eta4_20250409-T184858/ckpts/epoch=029-val_loss=0.05526.ckpt"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn-test/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"

# Interactive shell command
# apptainer shell --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn-test/hepattn/pixi.sif