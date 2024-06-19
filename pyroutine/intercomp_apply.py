
import matplotlib.pyplot as plt
import datetime
import numpy as np

from src.utils import modif_plot
from src.utils import save_name
from src.modelling import sit_corrected
from src.visualization import intercomp_sit

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

# ---------------------------------
#       Import all predictions
# ---------------------------------




models = {}

### AK model
# irootdir= '/cluster/work/users/leoede/'  # on fram
# irootdir= f'{rootdir}'  # on lumi
# ipath = 'Leo/results/ak_230215-144514/'
# ml_name ='AK'

# ifile = f'{irootdir}{ipath}'
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1, objective='apply')


# ### CNN model
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/cnn_230131-175505/'
# ml_name ='CNN'

# ifile = f'{irootdir}{ipath}'
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1, objective='apply')


### LSTM model
irootdir= f'{rootdir}'
ipath = 'Leo/results/lstm_230201-112355/'
ml_name ='LSTM'

ifile = f'{irootdir}{ipath}'
models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1, objective='apply')


### RF model
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/rf_221229-171734/'  # non recursive
# ml_name = 'RF'

# ifile = f'{irootdir}{ipath}'
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1, objective='apply')


### XGB model
# irootdir= f'{rootdir}'
# ipath = 'Leo/results/xgb_221230-121451/'  # non recursive
# ml_name ='XGB'

# ifile = f'{irootdir}{ipath}'
# models[f'{ml_name}'] = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=1, objective='apply')



# ------------------------------------------------
##            Reconstruct SIT values 
# ------------------------------------------------

print('\nReconstructing SIT...')

for mdl in models:
    models[mdl].reconstruct_sit()
    # models[mdl].save_sit()
    models[mdl].compute_mean(sit_min=None)    
    



# ------------------------------------------------
##       Plot  line average
# ------------------------------------------------
 

# colors from plasma or inferno for all ML algo
# cmap = plt.cm.get_cmap('plasma')
cmap = plt.cm.get_cmap('rainbow')
color_arr = np.linspace(0, 1, len(models.keys()))
list_colors = [cmap(fl) for fl in color_arr]


fig, ax = plt.subplots(figsize=(16,9))

# first model loaded is the default one
model_default = models[list(models.keys())[0]] 

model_default.sit_am.plot(c='k', lw=2, ls='--', label='TP4b', zorder=10)
model_default.sit_nam.plot(label='TP4 Free Run', c='#1f77b4')
model_default.sit_blm.plot(label='baseline', c='#2ca02c')

# for mdl, cl in zip(models, list_colors):  # to plot all models
#     models[mdl].sit_m.plot(label=mdl, c=cl)

# for LSTM only
model_default.sit_m.plot(label='LSTM', c='#ff7f0e')

# mini, maxi = ax.get_ylim()
# ax.plot([model_default.chrono.iloc[model_default.ntest]]*2, 
#         [mini+.1, maxi-.1],ls=':', c='k', label='test limit', zorder=-10)

    
ax.set_ylabel('SIT (m)')
ax.set_xlabel('')

modif_plot.resize(fig, s=18, rx=20)

plt.legend(ncol=2, loc='upper right')  # lower center')

savefig = True # False # True

if savefig:
    odir = f'{rootdir}Leo/results/application/'
    ofile = f'SIT_2000_2020_01.png'
    ofile = save_name.check(odir, ofile)
    plt.savefig(f"{odir}{ofile}", dpi=124)
    print(f'Saved as: {odir}{ofile}')

exit()
# ------------------------------------------------
##            Plot 2d maps
# ------------------------------------------------

# intercomp_sit.draw_spatial_reconstruct(models, datetime.datetime(2000, 8, 2), showfig=True, apply=True, savefig=True,
#                                       rootdir=f'{rootdir}', fig_dir='Leo/results/application/', vmax=5)


d1 = datetime.datetime(2008, 4, 1)  # 2000
d2 = datetime.datetime(2011, 10, 1)
#d2 = datetime.datetime(2011, 10, 1)


all_days = np.array([d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)])

intercomp_sit.spat_reco_save_all(models, all_days, 
                                 rootdir=f'{rootdir}', fig_dir='Leo/results/application/', apply=True, vmax=5)


exit()