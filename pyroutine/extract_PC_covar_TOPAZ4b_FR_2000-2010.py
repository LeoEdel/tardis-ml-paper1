# extract sit from TOPAZ4b FreeRun

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
# file_config = '../config/config_default_2023.yaml'
file_config = '../config/config_default_2023_N16.yaml'
# file_config = '../config/config_default_2023_N24.yaml'


nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)


# -------------------------------------------------
# selection files (last year included)

years = [2011, 2022]
# years = [1999, 2010]
# years = [1991, 1998]
extend_3m = True  # extend -/+ 3 months
# years = [1998, 2011]
# years = [1991, 1999]


listfile_na = sorted(glob(os.path.join(rootdir + freerun_dir,'*.nc')))
listyear = [os.path.basename(name)[:4] for name in listfile_na]  # read years
# selection corresponding years
listfile_na = [listfile_na[idx] for idx, name in enumerate(listyear) if int(name)>=years[0] and int(name)<=years[-1]]

# -------------------------------------------------

target_field = sys.argv[1]  # select variables from command line

print(f'\n\nWorking on {target_field}')

# load Topaz 4b Free Run
ofile = f'{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun.nc'
full_filename = os.path.join(rootdir, pca_dir, ofile)

# if exists already: use the file
if os.path.exists(full_filename):
    nc_sel_na, _ = load_data.load_nc(full_filename, f'{target_field}', True)

# otherwise: extract from TOPAZ daily .nc and save
else:
    print('Loading Topaz4b FreeRun files...')
    nc_sel_na, chrono_na = extract_pca.load_TOPAZ(listfile_na, target_field=target_field, lim_idm=lim_idm, lim_jdm=lim_jdm)

    if extend_3m:
        first_oct = datetime.datetime(years[0], 12, 1)
        last_march = datetime.datetime(years[-1], 3, 31)
        idx_start = np.where(chrono_na == first_oct)[0][0]
        idx_end = np.where(chrono_na == last_march)[0][0]
        
        nc_sel_na = nc_sel_na.isel(time=slice(idx_start, idx_end))
    # save .nc 
    nc_sel_na.to_netcdf(full_filename)


# --------- TOPAZ4b FREERUN ------------------
print('Loading PCA from 2014-2022...')


data_kind = "covariable"
n_components = load_config.get_n_components(data_kind, file_config)
filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_4b23_2014_2022_FreeRun.pkl")
pca_na = load_data.load_pca(filename)
# -------------------------------------------------

# apply PCA of 2010-2020 to 2000-2010
print(f'Applying PCA 2014-2022 on {years[0]}-{years[-1]}...')

Xna = nc_sel_na

maskok = (np.isfinite(Xna)).all(dim='time')
maskok1d = maskok.stack(z=('y','x'))
PCs_na = extract_pca.pca_to_PC(pca_na, Xna, maskok1d)

# -------------------------------------------------
# save PC for 2000-2010 as .pkl file
print('Saving files...')
ofile = f'PC_{target_field}_{n_components}N_{years[0]}_{years[-1]}_FreeRun.pkl'
ofile_PC = os.path.join(rootdir, pca_dir, ofile)

extract_pca.save_pca(ofile_PC, PCs_na)
