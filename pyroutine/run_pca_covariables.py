#!/usr/bin/env python
# coding: utf-8
'''Compute and extract the PCA of the covariables for the training period (2014-2022)
covariables are all variables from TOPAZ4b (version WITHOUT assimilation=freerun)
Save as .pkl

then '/pyroutine/extract_covar_TOPAZ4b_FR_2000-2010.py' can be used to extract PCA
from any other periods:
2011 - 2022: development algo
1999 - 2010: application
1991 - 1998: application ++

'''

import os
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
from glob import glob
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data_preparation import load_data
from src.data_preparation import mdl_dataset_prep
from src.feature_extraction import extract_pca
from src.visualization import visualize_pca
from src.utils import load_config
from src.utils import tardisml_utils

rootdir = tardisml_utils.get_rootdir()


# Path to config file
file_config = '../config/config_default_2023.yaml'

# Path to template file
file_template = '../config/template_name.yaml'

# config
template = yaml.load(open(file_template),Loader=yaml.FullLoader)
load_config.update_config(file_config, verbose=True)
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

config = yaml.load(open(file_config), Loader=yaml.FullLoader) # just to get forcing_mean_days

# save for plot
PCA_co = {}
EOF2d_co = {}
PCs_co = {}

# REMINDER to understand how data is stored: go read /scratch/project_465000269/edelleo1/Leo/Jiping_2023/info.txt 
# years = [2011, 2022]  # nosit_dir: stored in nosit_dir but freerun
years = [2014, 2022]  # just on training period

# years = [1999, 2010]  # nosit_dir or freerun_dir. stored according to the name (=correctly)
# years = [1991, 1998]  # nosit_dir: stored in nosit_dir but freerun

# we want to save .netcdf of full period
# we want to extract the PCA on (full period - test period) and still have PCA values for all period
# <<<<<<<<<

data_kind = "covariable"
n_components = load_config.get_n_components(data_kind, file_config)
# suffix = '' # on whole dataset
suffix = '_FreeRun' # on whole dataset


listfile_co = sorted(glob(os.path.join(rootdir + nosit_dir,'*.nc')))
# listfile_co = sorted(glob(os.path.join(rootdir + freerun_dir,'*.nc')))
listyear = [os.path.basename(name)[:4] for name in listfile_co]  # read years
# selection corresponding years
listfile_co = [listfile_co[idx] for idx, name in enumerate(listyear) if int(name)>=years[0] and int(name)<=years[-1]]
# print(rootdir + nosit_dir)

# load dataset
nc, chrono_co = extract_pca.load_TOPAZ(listfile_co)
# nc, chrono_co = extract_pca.load_TOPAZ(listfile_co[:test_limit])

for covar in covar_fields:
    print(f'\n{covar}')
    
    # select variable and area
    print('Variable selection...')
    nc_sel_co = nc[covar]
    print('Spatial selection...')
    nc_sel_co = nc_sel_co.isel(y=slice(*lim_jdm),x=slice(*lim_idm))
    print('Done!')


    # Compute and save PCA
    print(f'Compute PCA over {years[0]}_{years[-1]}...')
    ofile_pca = os.path.join(rootdir, pca_dir, f'pca_{covar}_{n_components}N_4b23_{years[0]}_{years[-1]}{suffix}.pkl')
#     ofile_X = os.path.join(rootdir, pca_dir, f"{covar}_TOPAZ4b23_{years[0]}_{years[-1]}{suffix}.nc")
    mu_co, X_co, X1d_nonan, pca_co, maskok = extract_pca.compute_pca_TOPAZ(nc_sel_co, n_components, lim_idm, lim_jdm, ofile_pca, ofile_X='')
    
    PCA_co[covar] = pca_co

    # Compute EOF
    print('EOF2D')
    _, EOF2d_co[covar] = extract_pca.compute_eof(n_components, X_co, pca_co, maskok)
    
    # todo : split test/evaluation
    PCs_co[covar] = xr.DataArray(pca_co.transform(X1d_nonan), dims=['time','comp'])

    
# ------------ PLOT ------------

for field in covar_fields:
    filename = f'{rootdir}{fig_dir}{field}_PCA{n_components}_noSITass_EOF_PC_{years[0]}_{years[-1]}{suffix}.png'
    visualize_pca.plot_eof2D(chrono_co, EOF2d_co[field], PCs_co[field], field, ofile=filename)

    filename = f'{rootdir}{fig_dir}{field}_PCA_{years[0]}_{years[-1]}{suffix}.png'
    visualize_pca.plot_pca_variance(n_components, PCA_co[field], field, False, filename)    

# almost same as plot_pca_variance
# filename = f"{rootdir}{fig_dir}covar_Ncomp_variance.png"
# visualize_pca.plot_ncomp_var(PCA_co, covar_fields, True) # , filename)    
    
print('\nPCA extracted for all variables!')

