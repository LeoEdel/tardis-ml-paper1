'''Load Sea Ice Age for 2000-2011
and extract PC from the PCA and EOF of the SIA of the period 2011-2019
'''

import os
import numpy as np
import netCDF4 as nc4
import xarray as xr
import pandas as pd

from src.feature_extraction import extract_pca
from src.data_preparation import load_data
from src.visualization import visualize_pca

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# -----------------------------------
# load sia
path = f'{rootdir}Leo/sia/'


file = 'Topaz_arctic25km_sea_ice_age_v2p1_20110101_20221231.nc'
years = [2011, 2022]

# file = 'Topaz_arctic25km_sea_ice_age_v2p1_19981001_20110331.nc'
# years = [1999, 2010]

# file = 'Topaz_arctic25km_sea_ice_age_v2p1_19911001_19990331.nc'
# years = [1991, 1998]

nc = nc4.Dataset(f'{path}{file}', mode='r')
# sia = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sia']  # for averaged sea ice age
sia = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['fyi'][:,:,:,-1]  # for sea ice age == 1
chrono = pd.DataFrame({'date':pd.to_datetime(sia['time'].to_numpy())})



# Parameters for PCA
n_components = 8  # could get config but just easier that way

pca_dir = f'{rootdir}Leo/results/pca_i100-550_j150-629/'
fig_dir = 'Leo/results/figures_pca/'

# --------- SIA 2011-2020 ------------------
print('Loading PCA from 2014-2022...')


# data_kind = "covariable"
# n_components = load_config.get_n_components(data_kind, file_config)
# n_components = 8
# ifile = f'pca_sia_{n_components}N_2014_2022.pkl'  # for averaged sea ice age
ifile = f'pca_fyi1_{n_components}N_2014_2022.pkl'  # for sea ice age == 1



filename = os.path.join(rootdir, pca_dir, ifile)
pca_sa = load_data.load_pca(filename)

# -------------------------------------------------

# apply PCA of 2010-2020 to 2000-2010
print(f'Applying PCA 2014-2022 on {years[0]}-{years[-1]}...')

Xna = sia

maskok = (np.isfinite(Xna)).all(dim='time')
maskok1d = maskok.stack(z=('y','x'))
PCs_sia = extract_pca.pca_to_PC(pca_sa, Xna, maskok1d)


# -------------------------------------------------
# save pca for 2000-2010 as .pkl file
print('Saving PC files...')
# ofile = f'PC_sia_{n_components}N_{years[0]}_{years[-1]}.pkl'  # for averaged sea ice age
ofile = f'PC_fyi1_{n_components}N_{years[0]}_{years[-1]}.pkl'  # for sea ice age == 1

ofile_PC = os.path.join(rootdir, pca_dir, ofile)

extract_pca.save_pca(ofile_PC, PCs_sia)


