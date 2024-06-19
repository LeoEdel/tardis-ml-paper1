'''Load SIT TOPAZ4-BL and save it as .nc
'''

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc4

from src.modelling import sit_corrected
from src.data_preparation import load_data
from src.utils import tardisml_utils


from src.data_preparation import merge_TOPAZ


# import the whole time series
# _,_, sit_bl = merge_TOPAZ.load(return_bl=True)


rootdir = tardisml_utils.get_rootdir()

# import just a portion of it
# ipath = 'Leo/results/lstm_231212-183758/'  # var4
ipath = 'Leo/results/lstm_240507-160336/'  # adjSIC

ml_name ='LSTM'

ifile = f'{rootdir}{ipath}'
m2000 = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=0, objective='apply')

print('\t1999-2010')
#  Reconstruct SIT values 
m2000.reconstruct_sit()
# m2000.compute_mean(sic_min=0.15)

odir = '/scratch/project_465000269/edelleo1/Leo/TP4_ML/'
# ofile = 'sit_bl_2000_2011.nc'
ofile = 'sit_bl_2000_2011_adjSIC.nc'

m2000.sit_bl.to_netcdf(f'{odir}{ofile}')

print(f'TOPAZ4-BL saved as: {odir}{ofile}')