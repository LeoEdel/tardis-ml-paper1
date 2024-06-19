#!/bin/bash

#SBATCH --job-name='train_glstm'
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
## The following line can be omitted to run on CPU alone
#SBATCH --partition=standard-g
#SBATCH --gpus=1

echo "Starting job $SLURM_JOB_ID at `date`"
# echo 'No GPU to check if callback are working\n\n'

export PATH="/users/edelleo1/my_envs/TG/bin:$PATH"  # GPU
# export PATH="/users/edelleo1/my_envs/TaGpu/bin:$PATH"  # CPU

# Library for GPU usage
module load rocm

# Run python script
python ../pyroutine/build_ml_LSTM.py $1

