'''From PCA prediction (.nc) to SIT (.nc)
'''


import datetime
import numpy as np

# from src.utils import modif_plot
from src.utils import save_name
from src.modelling import sit_corrected
from src.visualization import intercomp_sit

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# --------------------------------------------------
#      Get prediction to convert from PCA to SIT
# --------------------------------------------------

# could add an argument from sys to get folder straight from the command line

### AK model
# irootdir= '/cluster/work/users/leoede/'  # on fram
# irootdir= f'{rootdir}'  # on lumi
# ipath = 'Leo/results/ak_230215-144514/'
# ml_name = 'AK'

### CNN model
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/cnn_230131-175505/'
# ml_name ='CNN'

### LSTM model
irootdir= f'{rootdir}'
# ipath = 'Leo/results/lstm_230201-112355/'
# ipath = 'Leo/results/lstm_230217-153340/'
# ipath = 'Leo/results/lstm_230523-163358/'
ipath = 'Leo/results/lstm_230601-141152/'

ipath = 'Leo/results/lstm_231006-185645/'  # 24 PCA
ipath = 'Leo/results/lstm_240614-004737/'  # 24 PCA

ml_name ='LSTM3_bk'

### RF model
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/rf_221229-171734/'  # non recursive
# ml_name = 'RF'

### XGB model
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/xgb_221230-121451/'  # non recursive
# ml_name ='XGB'


# Initialise class
ifile = f'{irootdir}{ipath}'
# objective = 'train'  for period 2011-2019
#             'apply'  for period 2000-2011
#             'apply91'  for period 1991-2000

model = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1, objective='apply91')


# ------------------------------------------------
##            Reconstruct SIT values 
# ------------------------------------------------

print('\nReconstructing SIT...')

model.reconstruct_sit()
model.save_sit()
