#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import numpy as np
import xarray as xr
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data_preparation import load_data
from src.data_preparation import mdl_dataset_prep
from src.feature_extraction import extract_pca
from src.visualization import visualize_pca
from src.visualization.visualize_local_area import plot_land_mask
from src.utils import load_config
from src.utils import tardisml_utils

rootdir = tardisml_utils.get_rootdir()

# Date to plot in example
idate = 100
# Path to config file
# file_config = '../config/config_default_2023.yaml'
file_config = '../config/for_paper/config_forcings_2.yaml'

# Path to template file
file_template = '../config/template_name.yaml'

# config
template = yaml.load(open(file_template),Loader=yaml.FullLoader)
load_config.update_config(file_config, verbose=True)
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

config = yaml.load(open(file_config), Loader=yaml.FullLoader) # just to get forcing_mean_days
suffix = '' # on whole dataset

# ---------- load forcings ---------- 
print('Loading forcings...')
data_kind = "forcing"
n_components = load_config.get_n_components(data_kind, file_config)
# forcings, Nf = load_data.load_forcing(forcing_fields, forcing_bdir)
forcings, Nf, chrono = load_data.load_forcing(forcing_fields, forcing_bdir, return_chrono=True)
# suffix = '_2011_2019'
suffix = '_2014_2022'
# suffix = '_2000_2011'  # NEED TO APPLY pca from 2011-2019 on 2000-2011
# import pdb; pdb.set_trace()


# load mask
maskok = load_data.load_land_mask(lim_idm, lim_jdm, rootdir, pca_dir)
mskok1d = maskok.stack(z=('y','x'))
# maskok3d = maskok.expand_dims({'time':forcings[forcing_fields[0]].shape[0]})

# for each forcing:
# inverse latitude
for forcing in forcing_fields:
     forcings[forcing][:] = forcings[forcing][:][::-1]

# ---------- selection ntrain ---------- 
ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(forcings[forcing_fields[0]].shape[0], train_p=1, val_p=0)
# suffix = '_train'
chrono = chrono[ntest+nval:]
for forcing in forcing_fields:
    forcings[forcing] = forcings[forcing][ntest+nval:]        
        
forcings2d = forcings.copy()
# stack lat and lon dimensions
# apply mask to exclude values not over sea-ice
# ---------- apply mask ---------- 

print('Apply land/ocean mask...')
for forcing in forcing_fields:
    tmp2D = xr.DataArray(forcings2d[forcing].reshape(forcings2d[forcing].shape[0], -1), dims=('time', 'z'))
    tmp2D_nonan = tmp2D.where(mskok1d, drop=True)
    forcings2d[forcing] = tmp2D_nonan.to_numpy()
    
# for forcing in forcing_fields:
#     forcings[forcing][:] = forcings[forcing][:][::-1]
#     tmp = xr.DataArray(forcings[forcing], dims=('time', 'y', 'x'))
#     forcings[forcing] = tmp.where(maskok, drop=True).to_numpy()



    

# ---------- compute pca ----------    
print('Compute PCA...')
odir = f'{rootdir}{pca_dir}/'
# mu, pca, PCs, EOFs = feature_pca.compute_pca_forcing(n_components, forcing_fields, forcings, saveraw=True, odir=odir)
mu, pca, PCs, EOFs = extract_pca.compute_pca_forcing(n_components, forcing_fields, forcings2d, saveraw=False, odir='')


# save PCA to files for futher use
short_forcing_fields = [item.split('_')[0] for item in forcing_fields]
field_str = '-'.join(sorted(short_forcing_fields))
filename = os.path.join(rootdir,pca_dir,f"pca_{field_str}_{n_components}N_{config['forcing_mean_days']}d{suffix}.pkl")

extract_pca.save_pca(filename, pca)
# print(f'PCA Forcings saved: {filename}')
# pkl.dump(pca, open(filename,"wb"))

exit()

# ----------- compute EOF -----------
EOF2D = {}
for forcing in forcing_fields:
    # should be correct bcs ntest+nval already removed 
    X = xr.DataArray(forcings[forcing], dims=('time', 'y', 'x')).where(maskok, drop=False)
#     X = xr.DataArray(forcings[forcing][ntest+nval:, :], dims=('time', 'y', 'x')).where(maskok, drop=False)
    _, EOF2D[forcing] = extract_pca.compute_eof(n_components, X, pca[forcing], maskok)



## Define chronology
chrono = pd.to_datetime(chrono)

for i, field in enumerate(forcing_fields):
    filename = f'{rootdir}{fig_dir}forcing_{field}_PCA{suffix}.png'
    visualize_pca.plot_pca_variance(n_components, pca[field], field, False, filename)


for field in forcing_fields:
    filename = f'{rootdir}{fig_dir}forcing_{field}_EOF{suffix}.png'
    visualize_pca.plot_eof2D(chrono, EOF2D[field], PCs[field], field, ofile=filename)
    
    
    
filename = f"{rootdir}{fig_dir}forcings_Ncomp_variance{suffix}.png"
visualize_pca.plot_ncomp_var(pca, forcing_fields, False, filename)
