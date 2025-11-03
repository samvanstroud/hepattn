#!/bin/bash

#SBATCH --job-name=attn-method-trackml-ma-eta2p5-pt600-scaleup-loss-weights
#SBATCH -p GPU
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1         # must match number of devices
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --output=/share/rcif2/pduckett/hepattn-experiment/src/hepattn/experiments/trackml/slurm_logs/slurm-%j.%x.out


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
cd /share/rcif2/pduckett/hepattn-experiment/src/hepattn/experiments/trackml/
echo "Moved dir, now in: ${PWD}"

# Set tmpdir
export TMPDIR="/share/rcif2/pduckett/tmp/$SLURM_JOB_ID/"
mkdir -p "$TMPDIR"

echo "nvidia-smi:"
nvidia-smi

# Run the training
echo "Running training script..."

# Python command that will be run
config=/share/rcif2/pduckett/hepattn-experiment/src/hepattn/experiments/trackml/configs/scale-up-simplified-lossweights.yaml
PYTORCH_CMD="python run_tracking.py fit --config $config --trainer.devices 1"

# Pixi commnand that runs the python command inside the pixi env
PIXI_CMD="pixi run $PYTORCH_CMD"

# Apptainer command that runs the pixi command inside the pixi apptainer image
APPTAINER_CMD="apptainer run --nv --bind /share/rcif2/ /share/rcif2/pduckett/hepattn-experiment/pixi.sif $PIXI_CMD"

# Run the final command
echo "Running command: $APPTAINER_CMD"
$APPTAINER_CMD
echo "Done!"
