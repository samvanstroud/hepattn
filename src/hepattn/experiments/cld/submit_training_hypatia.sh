#!/bin/bash

#SBATCH --job-name=cld-training
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --output=/share/rcifdata/maxhart/hepattn/src/hepattn/experiments/cld/slurm_logs/slurm-%j.%x.out


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
CONFIG_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_TRKECALHCAL_16_96_TF_charged_10MeV_F16_scaled_20250526-T120023/config.yaml"
CKPT_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_TRKECALHCAL_16_96_TF_charged_10MeV_F16_scaled_20250526-T120023/ckpts/epoch=009-train_loss=1.45212.ckpt"
PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config $CONFIG_PATH --ckpt_path $CKPT_PATH"
# PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config src/hepattn/experiments/cld/configs/tracking.yaml "

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"