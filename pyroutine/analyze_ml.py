#!/usr/bin/env python

"""
This script uses the PCs prediction (in file 'ypred_*LSTM*_2011_2022.nc') to reconstruct the SIT bias and obtain the SIT.
The final SIT is saved as netcdf file ('sit_*LSTM*_2011_2022.nc').
Several plots are saved as .png.

All saved products are saved in the same folder as the prediction.
"""

import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import datetime

from src.data_preparation import mdl_dataset_prep as dataset_prep

import src.utils.load_config as load_config
import src.data_preparation.load_data as load_data
from src.feature_extraction import extract_pca

from src.utils import modif_plot
import src.visualization.visualize_pca as visualize_pca
from src.visualization import visualize_sit_corr as vsc
from src.feature_extraction import baseline
from src.modelling import sit_corrected
from src.visualization import mdl_ypred_PCA


from src.utils import tardisml_utils
rootdir = tardisml_utils.get_rootdir()


# ------------------------------------------------
## Retrieving results from ML
# ------------------------------------------------


# AK model
#irootdir= '/cluster/work/users/leoede/'
#ipath = 'Leo/results/ak_221214-142044/'
#ml_name ='AK'

# CNN model
#irootdir= '/cluster/work/users/leoede/'
irootdir = rootdir
# ipath = 'Leo/results/cnn_221214-131355/'
ipath = 'Leo/results/cnn_230315-003644/'
ipath = 'Leo/results/cnn_230315-130931/'
ipath = 'Leo/results/cnn_230315-140503/'
ml_name = 'CNN'

## LSTM model
irootdir = rootdir
#ipath = 'Leo/results/lstm_221214-133340/'
# ipath = 'Leo/results/lstm_230201-112355/'
# ipath = 'Leo/results/lstm_230315-010148/'
# ipath = 'Leo/results/lstm_230601-141152/'

# TOPAZ4b 2023
ipath = 'Leo/results/lstm_230904-164505/'
ipath = 'Leo/results/lstm_230904-170405/'
ipath = 'Leo/results/lstm_230904-171446/'

ipath = 'Leo/results/lstm_230904-170733/'

ipath = 'Leo/results/lstm_230919-151834/'

ipath = 'Leo/results/lstm_230919-170402/'


# recent
# ipath = 'Leo/results/lstm_231004-175026/'
ipath = 'Leo/results/lstm_231005-151847/'
ipath = 'Leo/results/lstm_231006-185645/'

# -------------------------------------------
# for paper
ipath = 'Leo/results/lstm_231211-103258/'

# in table_comparison_build_LSTM.odt

## number of epochs
# ipath = 'Leo/results/lstm_231211-121441/'
# ipath = 'Leo/results/lstm_231211-152530/'
# ipath = 'Leo/results/lstm_231211-152533/'
# ipath = 'Leo/results/lstm_231211-152537/'
ipath = 'Leo/results/lstm_231211-152539/'



## number of variables
ipath = 'Leo/results/lstm_231211-160634/'



## history



## number of PC


ipath = 'Leo/results/lstm_231212-144512/'

# H4_EPO
ipath = 'Leo/results/lstm_240130-173040/'


# for now in merge_TOPAZ.py
ipath = 'Leo/results/lstm_231212-183758/'

# ipath = 'Leo/results/lstm_240404-151813/'  # diff inputs for each PC
# ipath = 'Leo/results/lstm_240404-171920/'  # diff inputs for each PC + adjSIC

# ipath = 'Leo/results/lstm_240405-180331/'  # adjSIC full opti1
# # ipath = 'Leo/results/lstm_240405-180337/'  # adjSIC full opti1 unclear
# ipath = 'Leo/results/lstm_240503-150136/'  # adjSIC full opti1, history [-30,-7,0,7,30]

## >>>>>>>>>> 
# ipath = 'Leo/results/lstm_240507-160336/'  # best of the best (?) (iteration 2)
# ipath = 'Leo/results/lstm_240507-160328/'  # best of the best - iteration 1
ipath = 'Leo/results/lstm_240507-160345/'  # best of the best - iteration 3
ipath = 'Leo/results/lstm_240510-221626/'  # best of the best - Fixed history + dataset

ipath = 'Leo/results/lstm_240521-120105/'  # for_paper_3 opti v2 - iteration 1
ipath = 'Leo/results/lstm_240521-120126/'  # for_paper_3 opti v2 - iteration 2
ipath = 'Leo/results/lstm_240521-120134/'  # for_paper_3 opti v2 - iteration 3


ipath = 'Leo/results/lstm_240523-133556/'  # for_paper_3 opti v2 - batch size = 4

ipath = 'Leo/results/lstm_240523-170100/'  # for_paper_3 opti v2 - batch size = 32 + SIA. N8
# ipath = 'Leo/results/lstm_240524-154750/'  # for_paper_3 opti v2 - batch size = 32 + SIA. N16
# ipath = 'Leo/results/lstm_240524-173523/'  # for_paper_3 opti v2 - batch size = 32 + SIA. N24

# ipath = 'Leo/results/lstm_240603-115720/'  # for_paper_3 opti v2 - batch size = 32 N8 History x2
# ipath = 'Leo/results/lstm_240604-155317/'  # LSTM units = 64
# ipath = 'Leo/results/lstm_240604-171111/'   # LSTM at
# ipath = 'Leo/results/lstm_240604-180526/'   # LSTM at TiemLagx2


# ipath = 'Leo/results/lstm_240605-185435/'   # LSTM bi 2

# ipath = 'Leo/results/lstm_240605-151145/'   # 


# ipath = 'Leo/results/lstm_240611-123849/'  # 240524-173523 iteration (N24)


# ipath = 'Leo/results/lstm_240611-153704/'  #  240524-170100 iteration
# ipath = 'Leo/results/lstm_240611-153812/'  # 240524-170100 iteration BUT low number of epochs


# ipath = 'Leo/results/lstm_240614-004737/'

# ipath = 'Leo/results/lstm_240614-162104/'

# ipath = 'Leo/results/lstm_240614-130712/'

# ipath = 'Leo/results/lstm_240614-182729/'


# ipath = 'Leo/results/lstm_240618-155530/'


# -------------------------------------------


# ipath = 'Leo/results/lstm_231009-114353/'


# ipath = 'Leo/results/lstm_230523-163358/'
# ipath = 'Leo/results/lstm_230524-114319/'
ml_name ='LSTM3_bk'

# RF model
#irootdir= '/cluster/work/users/leoede/'
# ipath = 'Leo/results/rf_221220-133644/'
#ipath = 'Leo/results/rf_230315-010207/'
# ml_name = 'RF'

# XGB model
# irootdir= '/cluster/work/users/leoede/'
# irootdir = rootdir
# ipath = 'Leo/results/xgb_221221-023342/'
#ipath = 'Leo/results/xgb_230315-010201/'
# ml_name ='XGB'

# ConvLSTM2D_H
# ncpath = 'Leo/results/C2D_230308-185753/'
# nc_name = 'ConvLSTM2D'
# ncfile = 'ypred_ConvLSTM2D_H_epochs10_6feat.nc'

# import netCDF4 as nc4
#nc = nc4.Dataset(f'{irootdir}{ncpath}ml/{ncfile}', mode='r')
#c2d_sit = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sit_pred']
#chrono = pd.DataFrame({'date':pd.to_datetime(c2d_sit['time'].to_numpy())})


# ------------------------------------------------

# file_config = f'{irootdir}{ipath}data_proc_full.yaml'

# get ml_dir and target_field
# nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config, verbose=True)
# timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

ifile = f"{irootdir}{ipath}"

# {iname}'
# ------------------------------------------------

# create new directory to save new results
# sit_rec_dir = f'{fig_dir}sit_reconstruct/'

# if not os.path.exists(f'{irootdir}{sit_rec_dir}'):
#     os.mkdir(f'{irootdir}{sit_rec_dir}') 
    
# ------------------------------------------------
## Reconstruct SIT values 
# ------------------------------------------------


s1 = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1) # , objective='apply')

print('\nReconstructing SIT...')
s1.reconstruct_sit()

s1.save_sit()

# exit()

# import pdb; pdb.set_trace()
# put sea ice from .nc instead of self.sit
#s1.sit = c2d_sit
#s1.chrono = chrono

s1.compute_rmse()
s1.compute_bias()
s1.compute_mean(sic_min=0.15)


s1.compute_corr()   # add correlation 

# compute new skill score: % of improvement
s1.compute_improvement()

# ------------------------------------------------
# Show reconstructed sea-ice thickness
# ------------------------------------------------
print('\nPlotting...')


# rmse
vsc.draw_rmse(s1, rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, savefig=True, showfig=False)

# bias
day = datetime.datetime(2013,2,15)
vsc.draw_bias_diff(s1, day=day, target_field='sithick', rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, savefig=True, showfig=False)
    
# RMSE and bias on time series
# vsc.draw_rmse_bias(s1, rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, savefig=True, showfig=False)

# same but + correlation
vsc.draw_rmse_bias_corr(s1, rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, savefig=True, showfig=False)


# SIT temporal reconstruction
vsc.draw_sit(s1, rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, showfig=False, savefig=True)

# add clean plot for remaining bias
vsc.draw_mean_remaining_bias(s1, rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, showfig=False, savefig=True)

# plot improvement
vsc.draw_improvement(s1, rootdir=s1.config.rootdir, fig_dir=s1.config.fig_dir, showfig=False, savefig=True)

# plot error of bias
s1.compare_bias_ml_da()
vsc.draw_error_bias(s1.bias_diff_mean, s1.bias_diff_std, s1.bins, s1.n_pixels, rootdir=s1.config.rootdir, fig_dir = s1.config.fig_dir, savefig=True)



# ------------------
    
# SIT spatial reconstruction - 1 day
# day = datetime.datetime(2012,1,1)
# vsc.spatial_reconstruct(s1, day, showfig=True, savefig=False, rootdir=rootdir, fig_dir=sit_rec_dir)
    
# SIT spatial reconstruction - multiple days
#d1 = datetime.datetime(2011,10,1)
#d2 = datetime.datetime(2019,12,31)
#all_days = np.array([d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)])

#vsc.spat_reco_save_all(s1, all_days, irootdir, fig_dir=s1.sit_rec_dir)
    
    
    
# ------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
