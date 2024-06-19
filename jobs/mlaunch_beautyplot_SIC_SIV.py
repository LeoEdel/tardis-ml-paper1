#!/usr/bin/env python

# python slurm script to submit multiple jobs and to pass them arguments

import os
import numpy as np

years = np.arange(1992, 2023)  #1992, 2023)

for yr in years:
    print(f'\nSubmitting job for {yr}')
    os.system(f'sbatch run_007_SIC_SIV_BP.sh {yr}')
