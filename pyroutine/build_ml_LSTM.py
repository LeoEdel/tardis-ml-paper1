"""
Train LSTM for SIT bias prediction

Load dataset, train 1 ML model for each PC, save the weigths after training, save the prediction for the full training period
Plot (and save .png) the comparison between true bias and predicted bias for all PCs
"""


import os
# Filter out logs (additional) : 0 - all logs, 1 - INFO, 2 - WARNING, 3 - ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import yaml
import pickle as pkl
import subprocess
import tensorflow as tf
from tensorflow.keras.regularizers import L1L2


from src.data_preparation import load_data
from src.data_preparation.blocking_time_series_split import BlockingTimeSeriesSplit 
from src.data_preparation import mdl_dataset

from src.modelling import mdl_input_dico  # input variables class
from src.modelling import mdl_params  # parameters class
from src.modelling import mdl_history

# from src.modelling import model_cnn
from src.modelling import model_lstm
from src.modelling import super_model_dl

from src.visualization import visualize_pca
from src.visualization import mdl_introspect
from src.visualization import mdl_ypred_PCA

from src.utils import reload_config
from src.utils import tardisml_utils


# add argument to pass config file
import sys
narg = len(sys.argv)  # number of arguments passed
if narg > 1:
    file_config = sys.argv[1]
else:
    # file_config = '../config/config_to_jobs/config_LSTM_no_bias_0wk.yaml'
    # file_config = '../config/config_to_jobs/config_LSTM_no_bias_debug.yaml'
    file_config = '../config/config_default_2023.yaml'
    file_config = '../config/config_default_2023_inputs_PC.yaml'
    

rootdir = tardisml_utils.get_rootdir()
# file_config = '../config/data_proc_full.yaml'
conf = reload_config.Config(file_config, rootdir=rootdir, verbose=1)

# ---------------------------------------------------
#               Activate gpu if possible
# ---------------------------------------------------
ngpu = len(tf.config.list_physical_devices('GPU'))
print("\nNum GPUs Available: ", ngpu)


# print(tf.__version__)
# print(tf.config.list_physical_devices())
print(tf.config.list_logical_devices())
print(f'GPU device name:{tf.test.gpu_device_name()}')

# activation GPU
if ngpu > 0:
    tf.debugging.set_log_device_placement(True)  # debug: print placement (CPU or GPU) of operations
    print('Activation GPU')



# ---------------------------------------------------
#                 Loading data
# ---------------------------------------------------

#ds = mdl_dataset.Dataset(conf, setup='no_bias', history=new_hist, var_to_keep=var_to_keep)
# ds = mdl_dataset.Dataset(conf, setup=conf.setup, history=new_hist, var_to_keep=var_to_keep, non_assimilated=conf.non_ass)  #'adjSIC')

ds = mdl_dataset.Dataset(conf, setup=conf.setup, non_assimilated=conf.non_ass)


# -----------------------------------------------------------

regularizers = [L1L2(l1=0, l2=0.001)]  # [L1L2(l1=0, l2=0), L1L2(l1=0.01, l2=0), L1L2(l1=0, l2=0.01), L1L2(l1=0.1, l2=0.1)]

n_components = ds.config.n_comp['tp']


ireg = 0
reg = regularizers[ireg]
suffixe = f'_reg{ireg}'

            
# n_output = ds[0]['ytrain'].shape[0]
# m4 = model_lstm.ModelLSTM(ds, ds.dataset['Xtrain'].shape[1], ds.dataset['Xtrain'].shape[2], reg=reg, rootdir=ds.config.rootdir, ml_dir=ds.config.ml_dir, fig_dir=ds.config.fig_dir)
m4 = model_lstm.ModelLSTM(ds, ds.dataset['Xtrain'].shape[1], ds.nfeatures, reg=reg, rootdir=ds.config.rootdir, ml_dir=ds.config.ml_dir, fig_dir=ds.config.fig_dir)

m4.compile_models(name=conf.ml_name, npca=n_components)  # >>> check that
# m4.fit_multiple(ds.dataset, suffix=suffixe)
m4.fit_multiple(suffix=suffixe)

m4.print_histories()

# once fit_multiple has find the optimal parameters,
# we retrain ML algorithm by including the validation period
m4.retrain_wval()  # ds.dataset)


if conf.setup == 'default':
    m4.predict_multiple_wbias()  # ds.dataset)
else:
    m4.predict_multiple()  # ds.dataset)
m4.save_prediction()

# m4.save_model()
m4.save_model_weights()


# retrained with test period. For application prior to 2011
m4.retrain_wtest()  # ds.dataset)
m4.save_model_weights(retrained=True)



ofile = f'{m4.type}_ypred.png'
mdl_ypred_PCA.draw(m4, max_plot=n_components, odir=m4.rootdir+m4.fig_dir, savefig=True, showfig=False, ofile=ofile)















