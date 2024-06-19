'''Load forcings for 2000-2011
and extract PC from PCA and EOF of same forcings on the period 2011-2019
'''
import os
import sys
import xarray as xr

from src.utils import reload_config

from src.utils import tardisml_utils
from src.data_preparation import load_data
from src.feature_extraction import extract_pca


file_config = '../config/config_default_2023.yaml'
# file_config = '../config/for_paper/config_forcings_2.yaml'

rootdir = tardisml_utils.get_rootdir()

config = reload_config.Config(file_config, rootdir=rootdir, verbose=1)

config.forcing_fields = [sys.argv[1] + f'_mean{config.forcing_mean_days}d']
# ----------------------------------------------------
# import .netcdf 2000- 2011
# ----------------------------------------------------

train = False
apply91 = False  # True

if train:
    years = [2011, 2022]
elif not train and not apply91:
    years = [1999, 2010]
elif not train and apply91:
    years = [1991, 1998]   
    
print(f'\nLoading forcings .pkl {years[0]}_{years[-1]}...')
    
forcings, Nf, chrono = load_data.load_forcing(config.forcing_fields, config.forcing_bdir, return_chrono=True, train=train, apply91=apply91)

# ----------------------------------------------------
# import PCA 2014-2022 (training period)
# ----------------------------------------------------

data_kind = "fo"
field_str = '-'.join(sorted([item.split('_')[0] for item in config.forcing_fields]))
filename = os.path.join(config.rootdir, config.pca_dir, f"pca_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_2014_2022.pkl")

print('Loading pca for forcing 2014-2022...')
pca = load_data.load_pca(filename)

# ----------------------------------------------------
# extract PCA
# ----------------------------------------------------

print(f'Projecting PCA 2014-2022 on {years[0]}-{years[-1]}...')


maskok = load_data.load_land_mask((100,550), (150,629), rootdir, 'Leo/results/pca_i100-550_j150-629')
maskok1d = maskok.stack(z=('y','x'))

# for each forcing:
# inverse latitude
for forcing in config.forcing_fields:
     forcings[forcing][:] = forcings[forcing][:][::-1]

forcings2d = forcings.copy()
# stack lat and lon dimensions
# apply mask to exclude values not over sea-ice
# ---------- apply mask ---------- 
print('\tApply land/ocean mask...')
for forcing in config.forcing_fields:
    tmp2D = xr.DataArray(forcings2d[forcing].reshape(forcings2d[forcing].shape[0], -1), dims=('time', 'z'))
    tmp2D_nonan = tmp2D.where(maskok1d, drop=True)
    forcings2d[forcing] = tmp2D_nonan # .to_numpy()

print('Retrieve PCs and EOFs')
PCs = dict()
# EOF2d = dict()
for forcing in config.forcing_fields:
    X = forcings2d[forcing]
    PCs[forcing] = xr.DataArray(pca[forcing].transform(X), dims=['time','comp'])

    

# ----------------------------------------------------
# save PC apply to (years[0] - years[-1])
# ----------------------------------------------------


print('Saving files...')

ofile = f'PC_{field_str}_{config.n_comp[data_kind]}N_{config.forcing_mean_days}d_{years[0]}_{years[-1]}.pkl'
ofile_PC = os.path.join(config.rootdir, config.pca_dir, ofile)

extract_pca.save_pca(ofile_PC, PCs)