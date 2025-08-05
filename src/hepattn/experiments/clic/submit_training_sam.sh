#!/bin/bash

#SBATCH --job-name=clic-train
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2        # must match number of devices
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/share/rcifdata/svanstroud/slurm_logs/slurm-%j.%x.out

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
cd /share/rcifdata/svanstroud/hepattn/src/hepattn/experiments/clic/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR=/var/tmp/

# Run the training
echo "Running training script..."

# Python command that will be run
CONFIG_PATH="configs/base.yaml"
PYTORCH_CMD="python main.py fit --config $CONFIG_PATH"

# Pixi command that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
# Add srun in front of apptainer command for multiple gpus training
APPTAINER_CMD="srun apptainer run --nv --bind /share/  /share/rcifdata/svanstroud/hepattn/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
