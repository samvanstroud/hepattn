#!/bin/bash

# Job name
#SBATCH --job-name=trackml

# choose the GPU queue
#SBATCH -p GPU

# requesting one node
#SBATCH --nodes=1
####SBATCH --exclusive

# keep environment variables
#SBATCH --export=ALL

# requesting 4 V100 GPU
# (remove the "v100:" if you don't care what GPU)
#SBATCH --gres=gpu:a100:1

# note! this needs to match --trainer.devices!
#SBATCH --ntasks-per-node=1

# number of cpus per task
# useful if you don't have exclusive access to the node
#SBATCH --cpus-per-task=15

# request enough memory
#SBATCH --mem=150G

# Change log names; %j gives job id, %x gives job name
#SBATCH --output=/home/xucapsva/slurm_logs/slurm-%j.%x.out

# Comet variables
echo "Setting comet experiment key"
timestamp=$( date +%s )
COMET_EXPERIMENT_KEY=$timestamp
echo $COMET_EXPERIMENT_KEY
echo "COMET_WORKSPACE"
echo $COMET_WORKSPACE

# print host info
echo "Hostname: $(hostname)"
echo "CPU count: $(cat /proc/cpuinfo | awk '/^processor/{print $3}' | tail -1)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# move to workdir
cd /home/xucapsva/hepattn/src/hepattn/experiments/trackml/
echo "Moved dir, now in: ${PWD}"

# set tmpdir
export TMPDIR=/var/tmp/

echo "nvidia-smi:"
nvidia-smi

# run the training
echo "Running training script..."

# train hit filter tracking model
PYTORCH_CMD="python hit_filter.py fit --config hit_filter.yaml"
PIXI_CMD="pixi run $PYTORCH_CMD"
APPTAINER_CMD="apptainer run --nv /home/xucapsva/hepattn/pixi.sif $PIXI_CMD"

echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
