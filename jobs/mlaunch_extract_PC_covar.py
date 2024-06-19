#!/usr/bin/env python

# python slurm script to submit multiple jobs and to pass them arguments

import os

# import config to get covariables list
# covar_fields = ['siconc', 'sisnthick', 'zos', 'vxsi', 'vysi']
covar_fields = ['zos']


for covar in covar_fields:
    print(f'\nSubmitting job for {covar}')
    os.system(f'sbatch run_extract_PC_covar.sh {covar}')
    # os.system(f'sbatch quick_try.sh {covar}')
