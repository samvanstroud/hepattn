#!/bin/bash

#SBATCH --job-name=itk-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/itk/slurm_logs/slurm-%j.%x.out


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
echo "nvidia-smi:"
nvidia-smi

# Move to workdir
cd /share/rcifdata/maxhart/hepattn/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the training
echo "Running training script..."

# Python command that will be run
# PYTORCH_CMD="python src/hepattn/experiments/itk/run_tracking.py fit --config src/hepattn/experiments/itk/configs/tracking.yaml"

PYTORCH_CMD="python src/hepattn/experiments/itk/run_filtering.py fit --config src/hepattn/experiments/itk/configs/filtering_pixel.yaml"

# PYTORCH_CMD="python src/hepattn/experiments/itk/run_filtering.py test --config /share/rcifdata/maxhart/hepattn/logs/ITk_pixel_region135_eta4_pt1_20250422-T132414/config.yaml --ckpt_path /share/rcifdata/maxhart/hepattn/logs/ITk_pixel_region135_eta4_pt1_20250422-T132414/ckpts/epoch=029-val_loss=0.40065.ckpt"

# PYTORCH_CMD="python src/hepattn/experiments/itk/run_filtering.py fit --config /share/rcifdata/maxhart/hepattn/logs/pixel_region135_eta4_pt1_20250414-T145756/config.yaml --ckpt_path /share/rcifdata/maxhart/hepattn/logs/pixel_region135_eta4_pt1_20250414-T145756/ckpts/epoch=028-val_loss=0.40953.ckpt"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
