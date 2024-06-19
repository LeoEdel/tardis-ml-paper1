'''get PCA of sea ice age (average over the column of ice) 'sia'
'''

import os
import netCDF4 as nc4
import xarray as xr
import pandas as pd

from src.data_preparation import mdl_dataset_prep
from src.feature_extraction import extract_pca
from src.visualization import visualize_pca

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# Parameters for PCA
n_components = 8 # 4,8,12,24  # could get config but just easier that way

# years = [2011, 2022]
suffix = ''

# -----------------------------------
# load sia
path = f'{rootdir}Leo/sia/'
file = 'Topaz_arctic25km_sea_ice_age_v2p1_20110101_20221231.nc'


nc = nc4.Dataset(f'{path}{file}', mode='r')
# sia = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sia']  # for averaged sea ice age
sia = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['fyi'][:,:,:,-1]  # for sea ice age == 1
chrono = pd.DataFrame({'date':pd.to_datetime(sia['time'].to_numpy())})


# split dataset train/eval/test
ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(sia.shape[0], train_p=1, val_p=0)
sia = sia.isel(time=slice(ntest+nval,None))  # only training dataset ntrain
chrono = chrono.iloc[ntest+nval:]
# suffix = '_train'

first_year = chrono.iloc[0].date.year  # 2014
last_year = chrono.iloc[-1].date.year  # 2022

pca_dir = f'{rootdir}Leo/results/pca_i100-550_j150-629/'
fig_dir = 'Leo/results/figures_pca/'

# -----------------------------------
# Compute and save PCA
# ofile_pca = os.path.join(rootdir, pca_dir, f'pca_sia_{n_components}N_{first_year}_{last_year}{suffix}.pkl')  # for averaged sea ice age
ofile_pca = os.path.join(rootdir, pca_dir, f'pca_fyi1_{n_components}N_{first_year}_{last_year}{suffix}.pkl')  # for sea ice age == 1

mu_a, X_a, X1d_nonan, pca_a, maskok = extract_pca.compute_pca_TOPAZ(sia, 
                                                                    n_components, None, None, 
                                                                    ofile_pca)

# Compute EOF
EOF1d, EOF2d_a = extract_pca.compute_eof(n_components, X_a, pca_a, maskok)

# todo : split test/evaluation
PCs_a = xr.DataArray(pca_a.transform(X1d_nonan), dims=['time','comp'])


# ------------ PLOT ------------

filename = f'{rootdir}{fig_dir}sia_PCA{n_components}_EOF_PC_{first_year}_{last_year}{suffix}.png'
visualize_pca.plot_save_eof(chrono, n_components, n_components, EOF2d_a, PCs_a, 'sia', 0, ofile=filename)

# Visualize cumulative explained variance of the first axe
filename = f'{rootdir}{fig_dir}sia_PCA{n_components}_cumvar_{first_year}_{last_year}{suffix}.png'
visualize_pca.plot_pca_variance(n_components, pca_a, 'sia', True, filename)