'''Extract PC for TOPAZ4b FreeRun
from PCA of 2014-2022
'''

import os
import sys
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
from glob import glob
from datetime import datetime
from datetime import timedelta

from src.data_preparation import load_data
from src.feature_extraction import extract_pca
from src.utils import load_config
from src.utils import tardisml_utils

rootdir = tardisml_utils.get_rootdir()

# Path to config file
# file_config = '../config/config_default_2023.yaml'
file_config = '../config/config_default_2023_N24.yaml'


nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

# target_field = 'siconc'

# -------------------------------------------------
# selection files (last year included)

# years = [2011, 2022]
# years = [1999, 2010] 
years = [1991, 1998]
adjSIC = True  ## to use SIT adjusted with SIC >15%



listfile_na = sorted(glob(os.path.join(rootdir + freerun_dir,'*.nc')))
listyear = [os.path.basename(name)[:4] for name in listfile_na]  # read years
# selection corresponding years
listfile_na = [listfile_na[idx] for idx, name in enumerate(listyear) if int(name)>=years[0] and int(name)<=years[-1]]

# -------------------------------------------------

data_kind = "nosit"
n_components = load_config.get_n_components(data_kind, file_config)

# load Topaz 4b Free Run
print('Loading Topaz4b FreeRun files...')

nc_file = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun.nc")
if adjSIC:
    nc_file = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun_adjSIC.nc")

if os.path.exists(nc_file):  # load dataset
    nc_sel_na, chrono_na = load_data.load_nc(nc_file, f'{target_field}', True)
else:
    nc_sel_na, chrono_na = extract_pca.load_TOPAZ(listfile_na, target_field=target_field, lim_idm=lim_idm, lim_jdm=lim_jdm)

# save .nc - time serie SIT
#ofile = f'{target_field}_TOPAZ4b_FR_{years[0]}_{years[-1]}.nc'
#ofile_X = os.path.join(rootdir, pca_dir, ofile)
#nc_sel_na.to_netcdf(ofile_X)

# --------- TOPAZ4b FREERUN ------------------
print('Loading PCA from 2014-2022...')

data_kind = "nosit"
n_components = load_config.get_n_components(data_kind, file_config)

filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_4b23_2014_2022_FreeRun.pkl")
if adjSIC:
    filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_4b23_2014_2022_FreeRun_adjSIC.pkl")

pca_na = load_data.load_pca(filename)

# ----------- Extracting PCA ---------------------
print(f'Applying PCA 2014-2022 on {years[0]}-{years[-1]}...')

Xna = nc_sel_na

maskok = (np.isfinite(Xna)).all(dim='time')
maskok1d = maskok.stack(z=('y','x')).compute()
PCs_na = extract_pca.pca_to_PC(pca_na, Xna, maskok1d)
           
# -------------------------------------------------
# save pca for 2000-2010 as .pkl file

str_na = 'FreeRun'  # 'noSITass' or 'FreeRun'
if adjSIC:
    str_na = 'FreeRun_adjSIC'  # 'noSITass' or 'FreeRun'

print(f'Saving PC {years[0]}_{years[-1]} as .pkl...')
ofile = f'PC_{target_field}_{n_components}N_{years[0]}_{years[-1]}_{str_na}.pkl'
ofile_PC = os.path.join(rootdir, pca_dir, ofile)

extract_pca.save_pca(ofile_PC, PCs_na)
