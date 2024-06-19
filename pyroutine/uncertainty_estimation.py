import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
import numpy as np
import xarray as xr
from random import gauss
from random import seed

from src.data_preparation import load_data
from src.data_preparation import scaling
from src.data_preparation import mdl_dataset
from src.feature_extraction import extract_pca
from src.modelling import sit_corrected
from src.modelling import super_model_ml
from src.modelling import super_model_dl
from src.modelling import model_cnn
from src.modelling import model_lstm
from src.utils import reload_config
from src.utils import tardisml_utils

rootdir = tardisml_utils.get_rootdir()



# ---------------------------
# parameters for application:
# ---------------------------

irootdir= f'{rootdir}'
# ipath = 'Leo/results/lstm_230201-112355/'
# ipath = 'Leo/results/lstm_230215-010148/'
ipath = 'Leo/results/lstm_230523-163358/'
# ipath = 'Leo/results/lstm_230919-170402/'

# ipath = 'Leo/results/lstm_231005-151847/'  # default (=with SIT bias)
ipath = 'Leo/results/lstm_231006-185645/'  # 24 PCA

ml_name ='LSTM'


# ---------------------------
# Load config file
# ---------------------------

file_config = f'{irootdir}{ipath}'
conf = reload_config.Config(file_config, verbose=1)

# ---------------------------
# Import dataset 2000-2010
# ---------------------------

# for LSTM (only)
ds = mdl_dataset.Dataset(conf, setup=conf.setup, objective='apply', non_assimilated=conf.non_ass, do_scaling=True)

# ---------------------------
#      Import model ML
# ---------------------------
    
# to put in clean code
ds.ntrain = ds.config.ntrain
ds.nval = ds.config.nval
ds.ntest = ds.config.ntest
# ds.dataset.non_assimilated = ds.charac['non_ass']


m1 = model_lstm.ModelLSTM(ds, timesteps=ds.dataset['X'].shape[1], features=ds.dataset['X'].shape[2])

# compile same architecture as during training
m1.compile_models(npca=conf.n_comp['tp'])
ifolder_pattern = f"model_weights_{conf.ml_name}_{conf.n_comp['tp']}N"

# load weights
m1.load_model_weights(ipath=f'{irootdir}{ipath}ml/', ifolder_pattern=ifolder_pattern)


# ---------------------------
#      Perturb Prediction
# ---------------------------

from src.modelling import uncert_perturb_input

pp = uncert_perturb_input.PertPred(config=conf, model=m1, dataset=ds.dataset, objective=ds.objective)

pp.perturbe_inputs(n_pert=10, max_pert_array=.5) # .10 / .5 / 1


pp.predict()
pp.reconstruct_sit()

pp.compute_std()
pp.compute_mean()

pp.print_stats()

odir = '/scratch/project_465000269/edelleo1/Leo/results/uncert_inputs/'

pp.save_uncert(odir=odir)

# Plot 
from src.visualization import visu_mdl_uncert as vmu

vmu.draw_situ_t(pp.mean_sit_t, pp.ori_mean_sit_t, pp.std_t, pp.dataset_ori['chrono'], odir=odir, savefig=True)
vmu.draw_sit_pert(pp.means_pert, pp.mean_sit_t, pp.ori_mean_sit_t, pp.dataset_ori['chrono'], odir=odir, savefig=True)
vmu.draw_situ_xy(pp.ori_mean_sit_xy, pp.mean_xy, pp.std_xy, pp.sic_na, odir=odir, savefig=True)
vmu.draw_ypreds(pp.ypred_pert, pp.ori_ypred_pert, pp.dataset_ori['chrono'], max_plot=7, odir=odir, savefig=True)
































