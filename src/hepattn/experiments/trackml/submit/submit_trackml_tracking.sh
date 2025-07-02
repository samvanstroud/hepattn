#!/bin/bash

#SBATCH --job-name=trackml-hepattnbasic-train-standardpe
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1         # must match number of devices
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/share/rcifdata/pduckett/hepattn-basic/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out


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
cd /share/rcifdata/pduckett/hepattn-basic/src/hepattn/experiments/trackml/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

# Python command that will be run
#PYTORCH_CMD="python run_filtering.py fit --config configs/filtering.yaml"
PYTORCH_CMD="python run_tracking.py fit --config configs/tracking-lite-standard-pe.yaml --trainer.devices 1"

# Do testing instead
#PYTORCH_CMD="python run_filtering.py test --config /share/rcifdata/svanstroud/hepattn/logs/ec_eta4_20250409-T184858/config.yaml --ckpt_path /share/rcifdata/svanstroud/hepattn/logs/ec_eta4_20250409-T184858/ckpts/epoch=029-val_loss=0.05526.ckpt"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="srun apptainer run --nv --bind /share/rcifdata/ /share/rcifdata/svanstroud/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
