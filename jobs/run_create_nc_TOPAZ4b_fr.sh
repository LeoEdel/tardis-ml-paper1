#!/bin/bash

#SBATCH -A project_465000269
#SBATCH --job-name='nc_sithick_TP_11'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --partition=small
#SBATCH --mem=0


echo "Starting job $SLURM_JOB_ID at `date`"
echo "From script $(basename $BASH_SOURCE)"

export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Run python script
python ../pyroutine/create_nc_TOPAZ4b_FR.py $1 

