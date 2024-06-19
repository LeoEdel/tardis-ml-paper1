#!/usr/bin/env python

# python slurm script to submit multiple jobs and to pass them arguments

import os
import time

# import config to get forcings list
forcing_fields = ['2T', 'MSL', '10V', '10U', 'SSR', 'STR'] # 'TP'
forcing_fields = ['MSL', '10V', '10U', 'SSR', 'STR'] 
# forcing_fields = ['2T', '10U', 'TP', 'SSR', 'STR'] 


for forc in forcing_fields:
    print(f'\nSubmitting job for {forc}')
    os.system(f'sbatch run_extract_PC_forcings.sh {forc}')
    time.sleep(120)
    # os.system(f'sbatch quick_try.sh {forc}')
