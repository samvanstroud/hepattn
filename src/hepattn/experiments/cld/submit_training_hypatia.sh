#!/bin/bash

#SBATCH --job-name=cld-training
#SBATCH -p GPU
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32
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
#CONFIG_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_5_320_10MeV_neutrals_F16_tight_20250719-T101521/config.yaml"
#CKPT_PATH="/share/rcifdata/maxhart/hepattn/logs/CLD_5_320_10MeV_neutrals_F16_tight_20250719-T101521/ckpts/epoch=001-train_loss=41.42195.ckpt"
#PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config $CONFIG_PATH --ckpt_path $CKPT_PATH"
PYTORCH_CMD="python src/hepattn/experiments/cld/main.py fit --config src/hepattn/experiments/cld/configs/base.yaml"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcifdata/maxhart,/share/lustre/maxhart/ /share/rcifdata/maxhart/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
