#!/bin/bash

#SBATCH --job-name='apply_ml'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
## The following line can be omitted to run on CPU alone
#SBATCH --partition=standard-g
#SBATCH --gpus=1

echo "Starting job $SLURM_JOB_ID at `date`"

export PATH="/users/edelleo1/my_envs/TG/bin:$PATH"  # GPU
# export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Library for GPU usage
module load rocm

# Run python script
# python ../pyroutine/apply_global_to_2000_2010.py
python ../pyroutine/apply_LSTM_to_past.py
