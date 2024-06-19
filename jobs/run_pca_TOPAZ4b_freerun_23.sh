#!/bin/bash

#SBATCH -A project_465000269
#SBATCH --job-name='PCA_FreeRun_Topaz'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --mem=0


echo "Starting job $SLURM_JOB_ID at `date`"
echo "From script $(basename $BASH_SOURCE)"


# export PATH="/users/edelleo1/my_envs/TG/bin:$PATH"  # GPU
export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Library for GPU usage
# module load rocm


# Run python script
# python ../pyroutine/run_pca_TOPAZ4b_noSIT.py 
python ../pyroutine/run_pca_TOPAZ4b_freerun.py 

