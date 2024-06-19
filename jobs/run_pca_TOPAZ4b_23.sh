#!/bin/bash

#SBATCH -A project_465000269
#SBATCH --job-name='PCA_4b'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --partition=small
#SBATCH --mem=0


echo "Starting job $SLURM_JOB_ID at `date`"
echo "From script $(basename $BASH_SOURCE)"


# export PATH="/users/edelleo1/my_envs/TG/bin:$PATH"  # GPU
export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Library for GPU usage
# module load rocm


# Run python script
python ../pyroutine/run_pca_TOPAZ4b.py 

