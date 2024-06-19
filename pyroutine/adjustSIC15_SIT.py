'''If SIC_freerun < 15%
we consider SIC = 0 && SIT = 0

>> should remove all the same bias in SIT that could perturb the correction by ML

Apply to SIT for TOPAZ with assimilation and freerun
Save new file with the suffix '_adjSIC' for 'Adjusted SIC'
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


# Path to config file
rootdir = tardisml_utils.get_rootdir()
file_config = '../config/config_default_2023.yaml'
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)


years = [2011,2022]
# years = [1999,2010]
# years = [1991,1998]


# -------------- Import datasets -------------------
print('Loading SIT TOPAZ datasets...')

target_field = 'sithick'
filename = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun.nc")
sit_na, chrono_na = load_data.load_nc(filename, f'{target_field}', True)

# with assimilation
filename = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4b23_{years[0]}_{years[-1]}.nc")
sit_a, chrono_a = load_data.load_nc(filename, f'{target_field}', True)


print('Loading SIC TOPAZ Freerun...')
filename = os.path.join(rootdir, pca_dir, f"siconc_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun.nc")
sic_na, chrono_na = load_data.load_nc(filename, 'siconc', True)

# with assimilation
# filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}.nc")
# sic_a, chrono_a = load_data.load_nc(filename, f'{target_field}', True)


# ---------- Apply correction with SIC  ----------------
print('Apply correction...')
## If SIC_freerun < 15% and not on land
## we consider SIC = 0 && SIT = 0

sic_threshold = 0.15

# Try on reduced dataset:
# sit_a_adj = sit_a.isel(time=slice(0, 10)).where((sic_na.isel(time=slice(0, 10))>0.15) | (sic_na.isel(time=slice(0, 10)).isnull()), 0)
# sit_na_adj = sit_na.isel(time=slice(0, 10)).where((sic_na.isel(time=slice(0, 10))>0.15) | (sic_na.isel(time=slice(0, 10)).isnull()), 0)
# sic_na_adj = sic_na.isel(time=slice(0, 10)).where((sic_na.isel(time=slice(0, 10))>0.15) | (sic_na.isel(time=slice(0, 10)).isnull()), 0)

sit_a_adj = sit_a.where((sic_na>0.15) | (sic_na.isnull()), 0)
sit_na_adj = sit_na.where((sic_na>0.15) | (sic_na.isnull()), 0)
sic_na_adj = sic_na.where((sic_na>0.15) | (sic_na.isnull()), 0)

# ---------- Save adjusted datasets  ----------------
print('Save .nc...')

filename_na = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun_adjSIC.nc")
sit_na_adj.to_netcdf(filename_na)
print(f'{filename_na}')

# with assimilation
filename_a = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4b23_{years[0]}_{years[-1]}_adjSIC.nc")
sit_a_adj.to_netcdf(filename_a)
print(f'{filename_a}')

# SIC without SIC < 15%
ofile_sic_na = os.path.join(rootdir, pca_dir, f"siconc_TOPAZ4b23_{years[0]}_{years[-1]}_FreeRun_adjSIC.nc")
sic_na_adj.to_netcdf(ofile_sic_na)
print(f'{ofile_sic_na}')

