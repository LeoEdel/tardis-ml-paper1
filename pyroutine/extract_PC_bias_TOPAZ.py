'''extract sit for SIT bias from TOPAZ4 
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
file_config = '../config/config_default_2023_bias_N24.yaml'


nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)


# -------------------------------------------------
# selection files (last year included)

years = [2011, 2022]
adjSIC = True  # False  ## to use SIT adjusted with SIC >15%


# ------------  Import TOPAZ ------------

data_kind = "withsit"
n_components = load_config.get_n_components(data_kind, file_config)  # get number of PC

print('Loading TOPAZ datasets...')

suffix = '_FreeRun'  # get filename TOPAZ without assimilation
if adjSIC:
    suffix += '_adjSIC'
filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}{suffix}.nc")
X_na, chrono_na = load_data.load_nc(filename, f'{target_field}', True)  # load netcdf

# get filename TOPAZ with assimilation
filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}.nc")
if adjSIC:
    filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}_adjSIC.nc")

X_a, chrono_a = load_data.load_nc(filename, f'{target_field}', True)  # load netcdf

# ------------  Compute bias ------------
nc_sel_e = X_a - X_na

# Both TOPAZ versions should span from 2011-2022. Just a check to keep only common times
chrono_e = chrono_na.merge(chrono_a)


# --------- TOPAZ4b FREERUN ------------------
print('Loading PCA of SIT bias from 2014-2022...')

data_kind = "withsit"
n_components = load_config.get_n_components(data_kind, file_config)
filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_SITerr23_2014_2022.pkl")
if adjSIC:
    filename = os.path.join(rootdir,pca_dir,f"pca_{target_field}_{n_components}N_SITerr23_2014_2022_adjSIC.pkl")
    
pca_e = load_data.load_pca(filename)
# -------------------------------------------------

# apply PCA of 2010-2020 to 2000-2010
print(f'Applying PCA 2014-2022 on {years[0]}-{years[-1]}...')

Xe = nc_sel_e

maskok = (np.isfinite(Xe)).all(dim='time')
maskok1d = maskok.stack(z=('y','x'))
PCs_e = extract_pca.pca_to_PC(pca_e, Xe, maskok1d)

# -------------------------------------------------
print(f'Saving PC {years[0]}_{years[-1]} as .pkl...')
ofile = f'PC_SITbias_{n_components}N_{years[0]}_{years[-1]}.pkl'
if adjSIC:
    ofile = f'PC_SITbias_{n_components}N_{years[0]}_{years[-1]}_adjSIC.pkl'
    
ofile_PC = os.path.join(rootdir, pca_dir, ofile)

extract_pca.save_pca(ofile_PC, PCs_e)
