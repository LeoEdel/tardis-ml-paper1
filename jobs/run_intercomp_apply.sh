#!/bin/bash

#SBATCH --account=nn2993k
#SBATCH --job-name='intercomp_apply'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=standard
# SBATCH --gpus=1

echo "Starting job $SLURM_JOB_ID at `date`"

# Run python script
python ../pyroutine/intercomp_apply.py

