#!/bin/bash

#SBATCH --job-name=tide-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out
#SBATCH --error=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/tide/slurm_logs/slurm-%j.%x.out


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
export TMPDIR=/share/rcifdata/maxhart/tmp/

# Run the training
echo "Running training script..."

# Python command that will be run
PYTORCH_CMD="python src/hepattn/experiments/tide/main.py fit --config src/hepattn/experiments/tide/configs/regression.yaml "
# PYTORCH_CMD="python src/hepattn/experiments/tide/main.py fit --config /share/rcifdata/maxhart/hepattn/logs/TIDE_1M_100_32trk_F32_20250517-T092110/config.yaml --ckpt_path /share/rcifdata/maxhart/hepattn/logs/TIDE_1M_100_32trk_F32_20250517-T092110/ckpts/epoch=001-train_loss=73.99285.ckpt"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
