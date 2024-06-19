

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

from src.utils import modif_plot
import src.utils.load_config as load_config
import src.data_preparation.load_data as load_data
import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# -------------- load --------------------------
# Path to config file
file_config = '../config/data_proc_full.yaml'
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

# load datasets from both TOPAZ versions
print('Loading TOPAZ datasets...')
filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4c.nc")
X_f = load_data.load_nc(filename, f'{target_field}', True)
chrono_f = pd.DataFrame({'date':pd.to_datetime(X_f['time'].to_numpy())})

filename = os.path.join(rootdir, pca_dir, f"{target_field}_TOPAZ4b.nc")
X_a = load_data.load_nc(filename, f'{target_field}', True)
chrono_a = pd.DataFrame({'date':pd.to_datetime(X_a['time'].to_numpy())})

# --------------- get bias ---------------------

# TOPAZ4b contains all 2010, TOPAZ4c contains only the ~2 last months
chrono_e = chrono_f.merge(chrono_a)  # only keep common times between 2 versions of TOPAZ

Xfm = X_f.mean(dim=('y','x')).compute()
Xam = X_a.mean(dim=('y','x')).compute()
Xem = Xam - Xfm

# get CS2SMOS
odir = '/nird/projects/nird/NS2993K/Leo/SIT_observations/CS2SMOS/results/'
# ofile = f'CS2SMOS_SIT_mean_SIC15_20101115_20210415.nc'
ofile = f'CS2SMOS_SIT_mean_20101115_20210415.nc'
nc = xr.open_mfdataset(f'{odir}{ofile}', combine='nested', concat_dim='time')
cs2 = nc['analysis_sea_ice_thickness']


# ----------- plot -------------------

fig, (ax, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(16, 8))
ax.plot(chrono_f, Xfm, label='TOPAZ4c')
ax.plot(chrono_a, Xam, label='TOPAZ4b')
ax.plot(cs2.time, cs2, marker='.', c='r', label='CS2SMOS', ls='')
ax.set_ylabel('SIT (m)')
ax.set_xticklabels('')

ax2.plot(chrono_a, [0]*len(chrono_a), '--', c ='grey', alpha=.7)
ax2.plot(chrono_e, Xem, 'k')
ax2.set_xlim(ax.get_xlim())
ax2.set_ylabel(f'Bias b-c (m)')


ax.legend()

modif_plot.resize(fig, s=18)

plt.show()

savefig = False
if savefig:
    filename = f'TOPAZ4b_c_bias.png'
    plt.savefig(f"{rootdir}{fig_dir}{filename}")
    print(f'Saved as {rootdir}{fig_dir}{filename}')