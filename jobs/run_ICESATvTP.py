#!/bin/bash

#SBATCH -A project_465000269
#SBATCH --job-name='ICESATvTP'
#SBATCH --ntasks=1
#  SBATCH --nodes=3
## The following line can be omitted to run on CPU alone
# SBATCH --partition=accel --gpus=1
#SBATCH --time=02:00:00
#SBATCH --partition=standard
# SBATCH --gpus=1

echo "Starting job $SLURM_JOB_ID at `date`"


# Purge modules and load tensorflow
# module purge
# module --force purge
# module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
# module load Anaconda3/2019.07
# List loaded modules for reproducibility
# module list

# conda init bash
# source activate tardis-env
# conda info --envs

# Run python script
python ../pyroutine/ICESAT_vs_TOPAZ.py 

