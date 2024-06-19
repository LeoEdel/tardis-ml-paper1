#!/bin/bash

#SBATCH -A project_465000269
#SBATCH --job-name='fig06'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=standard-g
# SBATCH --mem=0


echo "Starting job $SLURM_JOB_ID at `date`"
echo "From script $(basename $BASH_SOURCE)"


export PATH="/users/edelleo1/my_envs/TG/bin:$PATH"  # GPU
#export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Library for GPU usage
module load rocm

# Run python script
python ../pyroutine/plot_SIV.py

