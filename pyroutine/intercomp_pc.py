# Global approach
# Comparison between different machine learning algorithms over test period: 2011-2013


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

from src.feature_extraction import baseline
from src.modelling import sit_corrected
from src.visualization import visualize_pca
from src.visualization import visualize_sit_corr as vsc
from src.visualization import visu_intercomp_pc as vip
from src.utils import modif_plot
from src.utils import tardisml_utils
from src.utils import load_config


rootdir = tardisml_utils.get_rootdir()


# ------------------------------------------------
##              IMPORT 
# ------------------------------------------------

models = {}
pathes = []
names = []

## LSTM model
ml_name ='LSTM3_bk'
irootdir= f'{rootdir}'

# # -----------------------------------------------------
#                  Variables
# # -----------------------------------------------------
# models_str = 'model_vars'
 
# # Build1
# pathes += ['Leo/results/lstm_231212-183723/']
# names += ['var2']

# # Build2
# pathes += ['Leo/results/lstm_231212-183743/']
# names += ['var3']

# # Build3
# pathes += ['Leo/results/lstm_231212-183758/']
# names += ['var4']

# # Build4
# pathes += ['Leo/results/lstm_231212-183810/']
# names += ['var5']

# # Build5
# pathes += ['Leo/results/lstm_231212-183826/']
# names += ['var6']

# # Build6
# pathes += ['Leo/results/lstm_231212-183840/']
# names += ['var7']
# # -----------------------------------------------------
# # -----------------------------------------------------

# -----------------------------------------------------
#                   History
# -----------------------------------------------------
# models_str = 'models_history'
 
# # Build1
# pathes += ['Leo/results/lstm_231215-160924/']
# names += ['H2']

# # Build2
# pathes += ['Leo/results/lstm_231215-161010/']
# names += ['H3']

# # Build3
# pathes += ['Leo/results/lstm_231215-161416/']
# names += ['H4']

# # Build4
# pathes += ['Leo/results/lstm_231215-161445/']
# names += ['H5']

# -----------------------------------------------------
# -----------------------------------------------------

# -----------------------------------------------------
#                   History
# -----------------------------------------------------
# models_str = 'models_npc3'
 
# # Build1
# pathes += ['Leo/results/lstm_231215-192237/']
# names += ['N8']

# pathes += ['Leo/results/lstm_231215-185008/']
# names += ['N16']

# # Build2
# pathes += ['Leo/results/lstm_231215-185006/']
# names += ['N24']

# # -----------------------------------------------------
#                  Variables on H4 and adjSIC
# # -----------------------------------------------------

# models_str = 'model_vars-1'

# # # Build1
# pathes += ['Leo/results/lstm_240404-210031/']
# names += ['all']

# pathes += ['Leo/results/lstm_240404-205946/']
# names += ['all-1']

# pathes += ['Leo/results/lstm_240404-205948/']
# names += ['all-2']

# pathes += ['Leo/results/lstm_240404-205952_240404-205952/']
# names += ['all-3']

# pathes += ['Leo/results/lstm_240404-205952/']
# names += ['all-4']

# pathes += ['Leo/results/lstm_240404-205959/']
# names += ['all-5']

# pathes += ['Leo/results/lstm_240404-210000/']
# names += ['all-6']

# pathes += ['Leo/results/lstm_240404-210009/']
# names += ['all-7']

# pathes += ['Leo/results/lstm_240404-210008/']
# names += ['all-8']

# pathes += ['Leo/results/lstm_240404-210015/']
# names += ['all-9']

# pathes += ['Leo/results/lstm_240404-210014/']
# names += ['all-10']

# pathes += ['Leo/results/lstm_240404-210024/']
# names += ['all-11']

# pathes += ['Leo/results/lstm_240404-210029/']
# names += ['all-12']

# pathes += ['Leo/results/lstm_240404-210031_240404-210031/']
# names += ['all-13']


# # -----------------------------------------------------
#                  Variables on H3 and adjSIC
#                with formatting of dataset fixed                
# # -----------------------------------------------------

# models_str = 'model_3_vars-1'

# # # Build1
# pathes += ['Leo/results/lstm_240514-160421/']
# names += ['all']

# pathes += ['Leo/results/lstm_240514-164041/']
# names += ['all-1']

# pathes += ['Leo/results/lstm_240514-164042_240514-164042/']
# names += ['all-2']

# pathes += ['Leo/results/lstm_240514-164042/']
# names += ['all-3']

# pathes += ['Leo/results/lstm_240514-164040_240514-164040/']
# names += ['all-4']

# pathes += ['Leo/results/lstm_240514-164043/']
# names += ['all-5']

# pathes += ['Leo/results/lstm_240514-164953_240514-164953/']
# names += ['all-6']

# pathes += ['Leo/results/lstm_240514-164953/']
# names += ['all-7']

# pathes += ['Leo/results/lstm_240514-182022/']
# names += ['all-8']

# pathes += ['Leo/results/lstm_240514-164952/']
# names += ['all-9']

# pathes += ['Leo/results/lstm_240514-170305_240514-170305/']
# names += ['all-10']

# pathes += ['Leo/results/lstm_240514-165406/']
# names += ['all-11']

# pathes += ['Leo/results/lstm_240514-182750/']
# names += ['all-12']

# pathes += ['Leo/results/lstm_240514-183532/']
# names += ['all-13']


# # -----------------------------------------------------
#                  Variables on H3 and adjSIC
#                with formatting of dataset fixed                
# # -----------------------------------------------------

models_str = 'model_3_batch'

# # Build1
pathes += ['Leo/results/lstm_240514-170305/']
names += ['b4']

pathes += ['Leo/results/lstm_240514-170306_240514-170306/']
names += ['b32']

pathes += ['Leo/results/lstm_240514-170306/']
names += ['b64']

pathes += ['Leo/results/lstm_240514-164040/']
names += ['b128']

pathes += ['Leo/results/lstm_240514-163102/']
names += ['b256']

# -----------------------------------------------------
# -----------------------------------------------------



for ipath, name in zip(pathes, names):
    ifile = f'{irootdir}{ipath}'
    print('ifile:  ', ifile)
    models[f'{name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1)


# Parameters    
n_mdl = len(list(models.keys()))
n_pc = models[names[0]].ytrue.shape[1]  # get number of component from first model
    
# Time axis             
chrono = models[names[0]].ypred.isel(time=slice(None,1096)).time 
              
# ------------------------------------------------
##            Bias for violin plots
# ------------------------------------------------

bias_pc = np.zeros((1096, n_pc*n_mdl))

ii = 0  
for i_pc in range(n_pc):
    for nm in range(n_mdl):
        bias_pc[:, ii] = (models[names[nm]].ypred.isel(time=slice(None,1096)) -
                          models[names[nm]].ytrue.isel(time=slice(None,1096)))[:,i_pc]
        ii += 1

# ------------------------------------------------
##                 Bias
# ------------------------------------------------

biases = []
biases_pc = np.zeros((n_pc,n_mdl))

for name in names:
    biases += [models[name].ypred.isel(time=slice(None,1096)) - models[name].ytrue.isel(time=slice(None,1096))]

for n in range(n_mdl):
    biases_pc[:, n] = biases[n].mean('time')
    
# ------------------------------------------------
##                    RMSE
# ------------------------------------------------

rmse = []
rmse_pc = np.zeros((n_pc, n_mdl))  # reshape to have (PC1 all models), (PC2 all models), PC3...

for name in names:
    rmse += [np.array(np.sqrt(models[name].ypred.isel(time=slice(None,1096)) - 
                              models[name].ytrue.isel(time=slice(None,1096))).mean(dim='time'))]

for n in range(n_mdl):
    rmse_pc[:, n] = rmse[n]

# ------------------------------------------------
##                 Correlation
# ------------------------------------------------

correlations = []
corr_pc = np.zeros((n_pc, n_mdl))

for name in names:
    correlations += [xr.corr(models[name].ypred.isel(time=slice(None,1096)), models[name].ytrue.isel(time=slice(None,1096)), dim=('time'))] 

for n in range(n_mdl):
    corr_pc[:, n] = correlations[n]

# ------------------------------------------------
##                 Plots
# ------------------------------------------------

odir = '/scratch/project_465000269/edelleo1/Leo/results/intercomp_pc/'

vip.draw_errors_intercomp(rmse_pc, biases_pc, corr_pc, names=names, 
                              models_str=models_str, odir=odir, savefig=True)

vip.draw_bias_violin(bias_pc, names=names, n_pc=n_pc, n_mdl=n_mdl, 
                         models_str=models_str, odir=odir, savefig=True)

vip.draw_bias_time(bias_pc, chrono, names=names, n_pc=n_pc, n_mdl=n_mdl, 
                       models_str=models_str, odir=odir, savefig=True)

vip.draw_ypred_mulitple(models, n_pc, n_mdl, 
                        models_str=models_str, odir=odir, savefig=True)

    
    
    
    
    
    
    
    
    
    