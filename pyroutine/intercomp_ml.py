'''Global approach
Comparison between all machine learning algorithms over 2011-2019
'''

import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import yaml
import pickle as pkl
import datetime
import netCDF4 as nc4

import src.utils.load_config as load_config

import src.visualization.visualize_pca as visualize_pca
from src.visualization import visualize_sit_corr as vsc
from src.utils import modif_plot
from src.feature_extraction import baseline
from src.modelling import sit_corrected

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# load CS2SMOS dataset
idir = f'{rootdir}Leo/SIT_observations/CS2SMOS/results/'
ifile = 'CS2SMOS_SIT_SIC_20101115_20210415.nc'

nc = nc4.Dataset(f'{idir}{ifile}', mode='r')
cs2smos = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['analysis_sea_ice_thickness']
# chrono = pd.DataFrame({'date':pd.to_datetime(X['time'].to_numpy())})


models = {}

# ### AK model

# # irootdir= f'{rootdir}'  # /cluster/work/users/leoede/'
# # ipath = 'Leo/results/ak_221214-142044/'
# # ml_name ='AK'

# # ifile = f'{irootdir}{ipath}' # '{iname}'
# # print('ifile:  ', ifile)
# # models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)

# ### LSTM model

# irootdir= f'{rootdir}'
# # ipath = 'Leo/results/lstm_221214-133340/'
# ipath = 'Leo/results/lstm_230217-153310/'
# ml_name ='LSTM3_bk'

# ifile = f'{irootdir}{ipath}' # '{iname}'
# print('ifile:  ', ifile)
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)


# ### CNN model

# irootdir= f'{rootdir}'
# ipath = 'Leo/results/cnn_221214-131355/'
# ml_name ='CNN'

# ifile = f'{irootdir}{ipath}'
# print('ifile:  ', ifile)
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)
# #import pdb; pdb.set_trace()
# ### RF model

# # irootdir= f'{rootdir}'
# # ipath = 'Leo/results/rf_221216-141433/'  # non recursive
# # ml_name = 'RF'

# # ifile = f'{irootdir}{ipath}' # '{iname}'
# # print('ifile:  ', ifile)
# # models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)

# ### XGB model

# irootdir= f'{rootdir}'
# # ipath = 'Leo/results/xgb_Npred22_7F_rw29d_N8844_H1333_Hn0000_sithick_artc_221102-144128/'  # non recursive
# # ipath = 'Leo/results/xgb_221216-100623/'  # non recursive
# ipath = 'Leo/results/xgb_221221-023342/'  # non recursive
# ml_name ='XGB'

# ifile = f'{irootdir}{ipath}' # '{iname}'
# print('ifile:  ', ifile)
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)


# ## Conv2D
# irootdir= f'{rootdir}'
# # ipath = 'Leo/results/C2D/' 
# ipath = 'Leo/results/C2D_230228-110411/' 

# ml_name ='Conv2D_sia'
# # ifile = f'ypred_Conv2Dencode_epochs10_scalefeat_5feat.nc'
# ifile = f'ypred_Conv2D_epochs10_scalefeat_6feat.nc'

# nc = nc4.Dataset(f'{irootdir}{ipath}ml/{ifile}', mode='r')
# c2d_sit = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sit_pred']

# ## ConvLSTM2D
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/C2D_230308-185753/' 
# ml_name ='ConvLSTM2d'
# ifile = f'ypred_ConvLSTM2D_H_epochs10_6feat.nc'

# nc = nc4.Dataset(f'{irootdir}{ipath}ml/{ifile}', mode='r')
# c2d_sit = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sit_pred']

# -----------------------------------------------------
#                   Number of PC
# -----------------------------------------------------
models_str = 'models_nPC'
models = {}
pathes = []
names = []
    
## LSTM model
ml_name ='LSTM3_bk'
irootdir= f'{rootdir}'
    
# Build1
pathes += ['Leo/results/lstm_231215-192237/']
names += ['N8']

# Build2
pathes += ['Leo/results/lstm_231215-185008/']
names += ['N16']

# # Build3
pathes += ['Leo/results/lstm_231215-185006/']
names += ['N24']


# -----------------------------------------------------
# -----------------------------------------------------

for ipath, name in zip(pathes, names):
    ifile = f'{irootdir}{ipath}'
    print('ifile:  ', ifile)
    models[f'{name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)

# ------------------------------------------------
## Reconstruct SIT values 
# ------------------------------------------------

# models

# print('\nReconstructing SIT...')

for mdl in models:
    models[mdl].reconstruct_sit()
    models[mdl].compute_mean(sic_min=0.15)
    
# s1.compute_rmse()
# s1.compute_bias()

# Cheating to use Conv2D
# models[ml_name] = models['XGB'] 
# models[ml_name].sit = c2d_sit
# models[ml_name].compute_mean()

# del models['XGB']


# ------------------------------------------------
##          new plot // comparison
# ------------------------------------------------

import pdb; pdb.set_trace()


## violon plot ?
## time diff ?
## RMSE/bias/correlation ?











# ------------------------------------------------
##          Plot    line average
# ------------------------------------------------
 
from src.visualization import annual_report_plot_intercomp_ml as arp

# arp.draw_test(models)
fig_dir = '/scratch/project_465000269/edelleo1/Leo/results/rapport_annuel/'
day = datetime.datetime(2011, 12, 15)
day = datetime.datetime(2012, 4, 15)
day = datetime.datetime(2012, 7, 15)
day = datetime.datetime(2012, 10, 15)



arp.draw_SIT(models, day=day, fig_dir=fig_dir)
    
    
from src.utils import modif_plot
from src.utils import save_name

# colors from plasma or inferno for all ML algo
# cmap = plt.cm.get_cmap('plasma')
cmap = plt.cm.get_cmap('rainbow')
color_arr = np.linspace(0, 1, len(models.keys()))
list_colors = [cmap(fl) for fl in color_arr]


fig, ax = plt.subplots(figsize=(16,9))

# first model loaded is the default one
model_default = models[list(models.keys())[0]] 

model_default.sit_am.plot(c='k', lw=2, label='TP4b', zorder=10)
model_default.sit_nam.plot(label='TP4 Free Run', c='#1f77b4')
model_default.sit_blm.plot(label='baseline', c='#2ca02c')

for mdl, cl in zip(models, list_colors):
    models[mdl].sit_m.plot(label=mdl, c=cl)

mini, maxi = ax.get_ylim()
ax.plot([model_default.chrono.iloc[model_default.ntest]]*2, 
        [mini+.1, maxi-.1],ls=':', c='k', label='test limit', zorder=-10)

    
ax.set_ylabel('SIT (m)')
ax.set_xlabel('')

modif_plot.resize(fig, s=18, rx=20)

plt.legend(ncol=3, loc='lower center')

savefig=True
fig_dir = 'Leo/results/intercomp_C2D_sia/'
if savefig:
    filename = f'intercomp_sit.png'
    plt.savefig(f"{model_default.config.rootdir}{fig_dir}{filename}", dpi=124)
    print(f'Saved as: {model_default.config.rootdir}{fig_dir}{filename}')


import pdb; pdb.set_trace()


# ------------------------------------------------
##          Plot    2d maps
# ------------------------------------------------

from src.visualization import intercomp_sit

# import pdb; pdb.set_trace()

# day = datetime.datetime(2012, 1, 1)
# intercomp_sit.draw_spatial_reconstruct(models, day, showfig=False, savefig=True,
#                                             rootdir='/cluster/work/users/leoede/', fig_dir='Leo/results/intercomp/',
#                                             cs2smos=cs2smos)

# exit()

# plot over period
# first day of time serie and last day
first_day = model_default.chrono.iloc[0].date.date()
last_day = model_default.chrono.iloc[-1].date.date()

d1 = datetime.datetime(2011,10,1)
d2 = datetime.datetime(2018,10,1)  #   2013,5,1)
# d1 = datetime.datetime.combine(first_day, datetime.time())
# d2 = datetime.datetime.combine(last_day, datetime.time())

all_days = np.array([d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)])


intercomp_sit.spat_reco_save_all(models, all_days, 
                                 rootdir=f'{rootdir}', fig_dir='Leo/results/intercomp_C2D_sia/',
                                cs2smos=cs2smos)




exit()

