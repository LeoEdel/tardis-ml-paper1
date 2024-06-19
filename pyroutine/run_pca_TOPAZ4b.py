'''Only used to get timeline of variables for TOPAZ4b (with assimilation) as .nc files
Put all daily files as one .nc.

Also compute the PCA and EOF of TOPAZ4b (with assimilation), but they will not be used further, except for visual comparison.
'''

import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import yaml
import pickle as pkl
from datetime import datetime
from datetime import timedelta
from glob import glob

from src.data_preparation import load_data
from src.data_preparation import mdl_dataset_prep
from src.feature_extraction import extract_pca
from src.visualization import visualize_pca
from src.utils import load_config

from src.utils import tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# ----------------------
#       Parameters
# ----------------------

years = [2011, 2022]


# Path to config file
# file_config = '../config/config_default_2023.yaml'
file_config = '../config/config_default_2023_N16.yaml'

file_template = '../config/template_name.yaml'

template = yaml.load(open(file_template),Loader=yaml.FullLoader)
load_config.update_config(file_config, verbose=True)
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, freerun_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

# ------------ PCA of SIT with sit assimilation ------------
# <<<<<<<<<<<<< TO REMOVE !! <<<<<<<<<<<<<<<<<<
# target_field = 'siconc'
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

data_kind = "withsit"
n_components = load_config.get_n_components(data_kind, file_config)
#file_save = os.path.join(rootdir, pca_dir, f'pca_{target_field}_{n_components}N_SITass.pkl')
suffix = '' # on whole dataset

listfile_a = sorted(glob(os.path.join(rootdir + withsit_dir,'*.nc')))
listyear = [os.path.basename(name)[:4] for name in listfile_a]  # read years
# selection corresponding years
listfile_a = [listfile_a[idx] for idx, name in enumerate(listyear) if int(name)>=years[0] and int(name)<=years[-1]]

full_filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}{suffix}.nc")

# -------------------------------------------------
## load dataset
if os.path.exists(full_filename):
    nc_sel_a, chrono_a = load_data.load_nc(full_filename, f'{target_field}', True)
else:
    nc_sel_a, chrono_a = extract_pca.load_TOPAZ(listfile_a, f'{target_field}', lim_idm, lim_jdm)
# nc_sel_a, chrono_a = extract_pca.load_TOPAZ(listfile_a[:test_limit], f'{target_field}', lim_idm, lim_jdm)

# -------------------------------------------------
# split dataset train/eval/test
ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(nc_sel_a.shape[0], train_p=1, val_p=0)
nc_sel_a = nc_sel_a[ntest+nval:]  # only training dataset ntrain
chrono_a = chrono_a[ntest+nval:]
# suffix = '_train'

first_year = chrono_a.iloc[0].date.year  # 2014
last_year = chrono_a.iloc[-1].date.year  # 2022

# -------------------------------------------------
# Compute and save PCA
ofile_pca = os.path.join(rootdir, pca_dir, f'pca_{target_field}_{n_components}N_4b23_{first_year}_{last_year}{suffix}.pkl')
ofile_X = '' # os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}{suffix}.nc")
mu_a, X_a, X1d_nonan, pca_a, maskok = extract_pca.compute_pca_TOPAZ(nc_sel_a, n_components, lim_idm, lim_jdm, ofile_pca, ofile_X)

# Compute EOF
EOF1d, EOF2d_a = extract_pca.compute_eof(n_components, X_a, pca_a, maskok)
PCs_a = xr.DataArray(pca_a.transform(X1d_nonan), dims=['time','comp'])

# ------------ PLOT ------------

filename = f'{rootdir}{fig_dir}{target_field}_PCA{n_components}_SITass_EOF_PC_{first_year}_{last_year}{suffix}.png'
visualize_pca.plot_save_eof(chrono_a, n_components, n_components, EOF2d_a, PCs_a, target_field, 0, ofile=filename)

# Visualize cumulative explained variance of the first axe
filename = f'{rootdir}{fig_dir}{target_field}_PCA{n_components}_Xa_cumvar_{first_year}_{last_year}{suffix}.png'
visualize_pca.plot_pca_variance(n_components, pca_a, target_field, True, filename)















