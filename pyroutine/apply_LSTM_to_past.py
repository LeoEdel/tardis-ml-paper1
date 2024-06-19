#!/usr/bin/env python
# coding: utf-8

'''Apply GLOBAL trained ML algorithm (on 2011-2019) to the 2000-2010 period

Save PCs prediction
'''

from src.modelling import sit_corrected
from src.utils import reload_config
from src.data_preparation import mdl_dataset

from src.modelling import super_model_ml
from src.modelling import super_model_dl
from src.modelling import model_cnn
from src.modelling import model_lstm

from src.utils import tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# ---------------------------
# Parameters for application:
# ---------------------------

irootdir= f'{rootdir}'
#ipath = 'Leo/results/cnn_221214-131355/'
ipath = 'Leo/results/rf_221229-171734/'
ml_name ='RF'

irootdir= f'{rootdir}'
# ipath = 'Leo/results/xgb_221221-023342/'
ipath = 'Leo/results/xgb_221230-121451/'  # config updated
ml_name ='XGB'

irootdir= f'{rootdir}'
# ipath = 'Leo/results/lstm_230201-112355/'
# ipath = 'Leo/results/lstm_230215-010148/'
ipath = 'Leo/results/lstm_230523-163358/'
ipath = 'Leo/results/lstm_230919-170402/'

ipath = 'Leo/results/lstm_231005-151847/'  # default (=with SIT bias)
ipath = 'Leo/results/lstm_231006-185645/'  # 24 PCA

ipath = 'Leo/results/lstm_231212-183758/'  # var4


ipath = 'Leo/results/lstm_240404-151813/'  # diff inputs for each PC
ipath = 'Leo/results/lstm_240404-171920/'  # diff inputs for each PC + adjSIC

ipath = 'Leo/results/lstm_240405-180331/'  # adjSIC full opti1
# ipath = 'Leo/results/lstm_240405-180337/'  # adjSIC full opti1 unclear

ipath = 'Leo/results/lstm_240507-160336/'  # best of the best (?)

ipath = 'Leo/results/lstm_240523-170100/'  # for_paper_3 opti v2 - batch size = 32 + SIA


ipath = 'Leo/results/lstm_240524-173523/' # for_paper_3 opti v2 - batch size = 32 + SIA. N24

ipath = 'Leo/results/lstm_240614-004737/'  # LSTM with residual and reLU activation. 8PCs

ml_name ='LSTM'

# ---------------------------
#        Load config file
# ---------------------------

file_config = f'{irootdir}{ipath}'

conf = reload_config.Config(file_config, verbose=1)

# ----------------------------
#    Additional parameters 
# ----------------------------

retrained = True

if retrained:
    dir_weights = f'{irootdir}{ipath}ml/retrained/'
else:
    dir_weights = f'{irootdir}{ipath}ml/'
    

# ---------------------------
# Import dataset 2000-2010
# ---------------------------

# the prediction is split into 2 periods for easily save the outputs that are quite heavy (several GB).
# The period 1992-1998 and 1999-2010
# To predict 1992-1998, the parameter <objective> = 'apply91'
# To predict 1999-2010, the parameter <objective> = 'apply'

ds = mdl_dataset.Dataset(conf, setup=conf.setup, objective='apply', non_assimilated=conf.non_ass)
    

# ---------------------------
#      Import model ML
# ---------------------------
    
# to put in clean code
ds.ntrain = ds.config.ntrain
ds.nval = ds.config.nval
ds.ntest = ds.config.ntest
# ds.dataset.non_assimilated = ds.charac['non_ass']
   

#            m1 = model_lstm.ModelLSTM(ds, timesteps=ds.dataset['X'].shape[1], features=ds.dataset['X'].shape[2])
m1 = model_lstm.ModelLSTM(ds, timesteps=ds.dataset['X'].shape[1], features=ds.nfeatures)

# compile same architecture as during training
m1.compile_models(npca=conf.n_comp['tp'])
ifolder_pattern = f"model_weights_{conf.ml_name}_{conf.n_comp['tp']}N"  # 8 from config file

# load weights
m1.load_model_weights(ipath=f'{dir_weights}', ifolder_pattern=ifolder_pattern)

    
# apply the algorithm
m1.predict_apply()  # dataset=ds.dataset

m1.rootdir = conf.rootdir
m1.ml_dir = conf.ml_dir
m1.save_prediction()
    

# ---------------------------
# ---------------------------

# plot PCA prediction
# from src.visualization import mdl_ypred_PCA
# mdl_ypred_PCA.draw_apply(m1)  # or m2
    
    
    
    
# Reconstruct SIT from predictin (.nc)
# ifile = '/cluster/work/users/leoede/Leo/results/cnn_230131-175505/'
# rootdir = conf.rootdir
# pca_dir = conf.pca_dir

# s0 = sit_corrected.SITCorrected(ifile, name='CNN', verbose=1, objective='apply')
# s0.reconstruct_sit()
# s0.compute_mean()
# s0.sit[365].plot(vmin=0, vmax=5)
