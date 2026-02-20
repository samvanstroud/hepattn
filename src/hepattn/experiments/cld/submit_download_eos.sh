#!/bin/bash

#SBATCH --job-name=cld-download
#SBATCH -p RCIF
#SBATCH --export=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/share/rcifdata/maxhart/hepattn-test/hepattn/src/hepattn/experiments/cld/slurm_logs/slurm-%j.%x.out

# Used for downloading raw CLD samples from EOS
# TODO

# Move to download directory
cd /share/rcifdata/maxhart/data/cld/raw
echo "Moved dir, now in: ${PWD}"
