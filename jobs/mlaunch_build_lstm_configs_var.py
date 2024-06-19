#!/usr/bin/env python

'''python slurm script to submit multiple jobs (.sh) and to pass them arguments
made to build ML algorithm for each config files (.yaml) present in a define subfolder

will call the ML algorithm
LSTM    -> build_ml_lstm
'''

import os
import glob
import time

path_to_cfiles = '/users/edelleo1/tardis/tardis-ml/config/for_paper_3/*full*.yaml'
cfiles = glob.glob(path_to_cfiles, recursive=False)
cfiles.sort()

# import pdb; pdb.set_trace()

for cfile in cfiles[1:]:
    print(f'Sbatching {cfile}')
    time.sleep(120)
    os.system(f'sbatch run_build_lstm.sh {cfile}')
