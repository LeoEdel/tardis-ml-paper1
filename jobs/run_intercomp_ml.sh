#!/bin/bash

#SBATCH --job-name='intercomp_ml'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --partition=standard
# SBATCH --gpus=1

echo "Starting job $SLURM_JOB_ID at `date`"

# Run python script
python ../pyroutine/intercomp_ml.py

