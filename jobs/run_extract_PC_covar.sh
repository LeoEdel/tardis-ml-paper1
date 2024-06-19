#!/bin/bash

#SBATCH -A project_465000269
#SBATCH --job-name='PC-covar'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=standard
# SBATCH --mem=0

# SBATCH --time=06:00:00
# SBATCH --partition=small
# SBATCH --mem=0


echo "Starting job $SLURM_JOB_ID at `date`"
echo "From script $(basename $BASH_SOURCE)"

# export PATH="/users/edelleo1/my_envs/TG/bin:$PATH"  # GPU
export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Library for GPU usage
# module load rocm


# Run python script
python ../pyroutine/extract_PC_covar_TOPAZ4b_FR_2000-2010.py $1

