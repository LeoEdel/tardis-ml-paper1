'''Create .netcdf from TOPAZ4b FreeRun
for any variable in TOPAZ daily .nc files
'''

import os
import sys
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
from glob import glob
import datetime
from datetime import timedelta

from src.data_preparation import load_data
from src.feature_extraction import extract_pca
from src.utils import load_config
from src.utils import tardisml_utils

rootdir = tardisml_utils.get_rootdir()

# Path to config file
file_config = '../config/config_default_2023.yaml'

nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)


# -------------------------------------------------
# selection files (last year included)

# years = [2011, 2022]
years = [2010, 2022]

## years = [1999, 2010]
## years = [1991, 1998]
extend_3m = True  # extend -/+ 3 months
# years = [1998, 2011]
# years = [1991, 1999]


listfile_na = sorted(glob(os.path.join(rootdir + freerun_dir,'*.nc')))
listyear = [os.path.basename(name)[:4] for name in listfile_na]  # read years
# selection corresponding years
listfile_na = [listfile_na[idx] for idx, name in enumerate(listyear) if int(name)>=years[0] and int(name)<=years[-1]]

# -------------------------------------------------

# for target_field in covar_fields:

# target_field = 'siconc'   # 'sithick' 'zos'
target_field = sys.argv[1]



print(f'\n\nWorking on {target_field}')

# load Topaz 4b Free Run
print('Loading Topaz4b FreeRun files...')
nc_sel_na, chrono_na = extract_pca.load_TOPAZ(listfile_na, target_field=target_field, lim_idm=lim_idm, lim_jdm=lim_jdm)

## Only keep from 1st october of first year to 31 march of last year
if extend_3m:
    first_oct = datetime.datetime(years[0], 10, 1)
    last_march = datetime.datetime(years[-1], 3, 31)
    idx_start = np.where(chrono_na == first_oct)[0][0]
    idx_end = np.where(chrono_na == last_march)[0][0] + 1  # to include the last day of march

    nc_sel_na = nc_sel_na.isel(time=slice(idx_start, idx_end))

# save .nc - time serie SIT
ofile = f'{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun_x3m.nc'
ofile_X = os.path.join(rootdir, pca_dir, ofile)

nc_sel_na.to_netcdf(ofile_X)
print(f'Saved as {ofile_X}')