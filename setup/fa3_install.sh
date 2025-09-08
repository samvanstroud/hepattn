#!/bin/bash
#SBATCH --job-name=fa3-install
#SBATCH --gpus=1                  # this also allocates 72 CPU cores and 115GB memory per gpu
#SBATCH --cpus-per-task 16
#SBATCH --mem=150G
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/u5ar/mhart.u5ar/hepattn/setup/slurm_log.out

# Move to project directory
cd /home/u5ar/mhart.u5ar/hepattn

# Setup the environment (non-interactive)
pixi install -e isambard
eval "$(pixi shell-hook -e isambard)"

# Move to parent dir
cd ..

# Clean up any old builds
rm -rf flash-attention

# Clone and build
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper/

MAX_JOBS=16 python setup.py install
