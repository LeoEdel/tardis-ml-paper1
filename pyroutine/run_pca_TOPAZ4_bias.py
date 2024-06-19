'''Import .nc for TOPAZ with and without assimilation
Substract SIT (with - without)
Extract PCA on test period (2014-2022)
Save PCA as.pkl file and SIT values as .nc, both over test period
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
adjSIC = True  # False  ## to use SIT adjusted with SIC >15%


# Path to config file
file_config = '../config/config_default_2023.yaml'
file_config = '../config/config_default_2023_bias_N16.yaml'
file_config = '../config/config_default_2023_bias_N24.yaml'



file_template = '../config/template_name.yaml'

template = yaml.load(open(file_template),Loader=yaml.FullLoader)
load_config.update_config(file_config, verbose=True)
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)


# ------------  Import TOPAZ ------------

data_kind = "withsit"
n_components = load_config.get_n_components(data_kind, file_config)

print('Loading TOPAZ datasets...')

suffix = '_FreeRun'  # without assimilation
if adjSIC:
    suffix += '_adjSIC'
filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}{suffix}.nc")
X_na, chrono_na = load_data.load_nc(filename, f'{target_field}', True)

# with assimilation
filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}.nc")
if adjSIC:
    filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b23_{years[0]}_{years[-1]}_adjSIC.nc")
    
X_a, chrono_a = load_data.load_nc(filename, f'{target_field}', True)


# ------------  Compute bias ------------
nc_sel_e = X_a - X_na

# Both TOPAZ versions should span from 2011-2022. Just a check to keep only common times
chrono_e = chrono_na.merge(chrono_a)

# split dataset train/eval/test
ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(nc_sel_e.shape[0], train_p=1, val_p=0)
nc_sel_e = nc_sel_e[ntest+nval:]  # only training dataset ntrain
chrono_e = chrono_e[ntest+nval:]
suffix = ''
if adjSIC:
    suffix += '_adjSIC'

first_year = chrono_e.iloc[0].date.year  # 2014
last_year = chrono_e.iloc[-1].date.year  # 2022

# ------------ Compute and save PCA ------------
ofile_pca = os.path.join(rootdir, pca_dir, f'pca_{target_field}_{n_components}N_SITerr23_{first_year}_{last_year}{suffix}.pkl')
# ofile_pca = ''
# ofile_X = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4err23_{first_year}_{last_year}{suffix}.nc")
ofile_X = ''

mu_e, X_e, X1d_nonan, pca_e, maskok = extract_pca.compute_pca_TOPAZ(nc_sel_e, n_components, lim_idm, lim_jdm, ofile_pca, ofile_X)

# Compute EOF
EOF1d, EOF2d_e = extract_pca.compute_eof(n_components, X_e, pca_e, maskok)
PCs_e = xr.DataArray(pca_e.transform(X1d_nonan), dims=['time','comp'])

# ------------ PLOT ------------

filename = f'{rootdir}{fig_dir}{target_field}_PCA{n_components}_SITerr23_EOF_PC_{years[0]}_{years[-1]}{suffix}.png'
# visualize_pca.plot_save_eof(chrono_e, n_components, n_components, EOF2d_e, PCs_e, target_field, 0, ofile=filename)
visualize_pca.plot_save_eof_annotation(chrono_e, EOF2d_e, PCs_e, pca_e.explained_variance_ratio_.cumsum() , ofile=filename)


# Visualize cumulative explained variance of the first axes
filename = f'{rootdir}{fig_dir}{target_field}_PCA{n_components}_Xe23_cumvar_{years[0]}_{years[-1]}{suffix}.png'
visualize_pca.plot_pca_variance(n_components, pca_e, target_field, True, filename)







