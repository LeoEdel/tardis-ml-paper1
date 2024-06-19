#!/usr/bin/env python

# python slurm script to submit multiple jobs and to pass them arguments

import os

# import config to get covariables list
covar_fields = ['sithick', 'siconc', 'sisnthick', 'zos', 'vxsi', 'vysi']
# covar_fields = ['sithick']


for covar in covar_fields[:]:
    print(f'\nSubmitting job for {covar}')
    os.system(f'sbatch run_create_nc_TOPAZ4b_fr.sh {covar}')
    # os.system(f'sbatch quick_try.sh {covar}')
