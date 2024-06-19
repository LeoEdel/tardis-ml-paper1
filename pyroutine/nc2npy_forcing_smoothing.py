#!/usr/bin/env python
# coding: utf-8

# # ERA5 .nc colocated with TOPAZ to .npy

import os
import pandas as pd
from datetime import date
import xarray as xr
import numpy as np
from glob import glob
import datetime
import re
import yaml

from src.data_preparation.running_mean import grid_center_running_mean as gcrm
import src.utils.load_config as load_config
import src.utils.tardisml_utils as tardisml_utils


def sum_nval(arr, nval=4):
    '''Sum daily values
    Precipitation need to be cumulative over 1 day
    from m/6h to m/24h
    
    arr    : DataArray, dims= (time, y, x)
    nval   : int, nvalues to cumsum. Default is 4 (n timestep per day in ERA5)
    
    '''
    
    n_cumsum = int(arr.shape[0] / nval)
    new_arr = np.zeros((n_cumsum, arr.shape[1], arr.shape[2]))
    
    for t in range(n_cumsum):
#         print(t)
        new_arr[t] = np.sum(arr[t*nval:(t+1)*nval], axis=(0))
    
    return new_arr  # xr.DataArray(data=new_arr, dims=arr.dims)



rootdir = tardisml_utils.get_rootdir()
user = yaml.load(open('../config/data_proc_demo.yaml'), Loader=yaml.FullLoader)['user']

file_config = '../config/data_proc_full.yaml'

nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

datadir = os.path.join(rootdir,user+f'/forcings_full/ERA5_cTOPAZ/')


##  ---------  Smoothing parameters  ---------  
smooth = True
## timeofday = 0 # to save chrono at 00:00:00
ndays = 29 # 15 / 29
nval_per_day = 4  # for ERA5

##  ---------  timeline parameters  ---------  
#  years=[2000, 2011]
years=[2011, 2019]
# years=[1991,2020]


allyears = np.arange(years[0], years[-1]+1)

##  ---------  forcings parameters  ---------  
# var = '2T'
# list_var = ['2T', 'MSL', '10V', '10U', 'TP', 'SSR', 'STR'] # , 'SKT']
list_var = [fo.split('_')[0] for fo in forcing_fields]
list_var = ['TP']
print('\nFor the following forcings: ', list_var)

# new area for the new version of TOPAZ
# lim_jdm = (300, 629)
# lim_idm = (100, 550)

for var in list_var:
    print(f'\n--- {var} ---\n')
    # get files for forcing
    allfiles = sorted(glob(os.path.join(datadir,f'*{var}*.nc')))
    var_idx = len(var)+1
    listyear = [os.path.basename(name)[var_idx:var_idx+4] for name in allfiles]
    listfile=[]

    # selection corresponding years
    for idx, name in enumerate(listyear):
        if int(name)>=years[0] and int(name)<=years[-1]:
            listfile += [allfiles[idx]]

    print('Loading .nc files...')
    nc = xr.open_mfdataset(listfile, combine='nested', concat_dim='time')

    # # Smoothing
    dttm = pd.to_datetime(nc['time'].to_numpy())  # get datetime for variable
    raw_data = nc[f'{var}'].to_numpy()

    smooth = False
    if not smooth:
        print(f'Raw data at t = {timeofday}') 

        timeofday = 0
        # determine index to keep based on timeofday
        
        idx_time = (dttm.time==datetime.time(int(timeofday * 24),0))        
        forcing_data = raw_data[idx_time]
        chrono = nc['time'][idx_time].to_numpy()
        
        if var == 'TP':
#             import pdb; pdb.set_trace()            
            print('Summing daily values...')
            # check cumulative over one day for precipitation ERA5
            new_precip = sum_nval(raw_data[3:-1]) # forcing_data = 
            chrono = nc['time'][3:-1].to_numpy()

        
        savefile = os.path.join(rootdir, forcing_bdir, f'{var}_sum.npy')
        np.save(savefile, forcing_data)
        print(f'Forcing saved as: {savefile}')

        # save chrono
        savefile = os.path.join(rootdir, forcing_bdir, f'chrono_forcings_{years[0]}_{years[-1]}_sum.npy')
        np.save(savefile, chrono)

        import pdb; pdb.set_trace()
        exit()
        
    if smooth:
        print(f'Running windows of {ndays} days')    
        forcing_mean = np.empty((len(np.unique(dttm.date)), raw_data.shape[1], raw_data.shape[2]))
        
        print(f'Running mean over all gridpoints...')    
        smooth_data = gcrm(raw_data, ndays, npd=nval_per_day)

        savefile = os.path.join(rootdir, forcing_bdir, f'{var}_mean{ndays}d_{years[0]}_{years[-1]}.npy')
        np.save(savefile, smooth_data)
        print(f'Forcing saved as: {savefile}')
        del smooth_data 


