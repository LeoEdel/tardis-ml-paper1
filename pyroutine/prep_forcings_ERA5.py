#!/usr/bin/env python
# coding: utf-8

'''


------- Pseudo-code ------- 

check grid ERA5 (regular)
check grid TOPAZ (irregular)
determine closest point of ERA5 on TOPAZ grid
retrieve points for each time step (6h) on yearly .nc
save each year separatly otherwise it's too heavy
save to .nc
'''

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

import os
import pandas as pd

from datetime import date

import xarray as xr
import numpy as np
from glob import glob
import datetime
import re

import src.utils.load_config as load_config
from src.utils import long_trans

import yaml
user = yaml.load(open('../config/data_proc_demo.yaml'), Loader=yaml.FullLoader)['user']


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx



# ---------------------------

# file_config = '../config/data_proc_demo.yaml'
#file_config = '../config/data_proc_full.yaml'
file_config = '../config/config_default_2023.yaml'

# Path to template file
file_template = '../config/template_name.yaml'

# template = yaml.load(open(file_template),Loader=yaml.FullLoader)
# load_config.update_config(file_config, verbose=True)
nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config)
timeofday, target_field, forcing_fields, covar_fields, lim_idm, lim_jdm, n_comp = load_config.load_config_params(file_config)

# ## Get only interesting years and var
datadir = '/cluster/projects/nn9481k/ERA5_6h/' # os.path.join(rootdir,user+f'/Jiping/dailyforcing_2014_2017')
pkldir = os.path.join(rootdir,user+f'/forcings_full/ERA5_cTOPAZ/')  # f'Julien/forcings'

# PARAMETERS FOR SCRIPT 


# years=[2018, 2019]  # [2010, 2017]
years=[2000, 2009]  # [2010, 2017]

allyears = np.arange(years[0], years[-1]+1)
# years=[1991,2020]


# var = '2T'
# list_var = ['2T', 'MSL', '10V', '10U', 'TP', 'SSR', 'STR', 'SKT', 'TCC']
# list_var = ['SSR', 'STR', 'SKT', 'TCC']
list_var = [fo.split('_')[0] for fo in forcing_fields]
print('\nFor the following forcings: ', list_var)

# new area for the new version of TOPAZ
# lim_jdm = (300, 629)
# lim_idm = (100, 550)

for var in list_var:
    allfiles = sorted(glob(os.path.join(datadir,f'6h.{var}_*.nc')))
    listyear = [os.path.basename(name)[-7:-3] for name in allfiles]
    listfile = [allfiles[idx] for idx, name in enumerate(listyear) if int(name)>=years[0] and int(name)<=years[-1]]
#     listfile=[]

#     # selection corresponding years
#     for idx, name in enumerate(listyear):
#         if int(name)>=years[0] and int(name)<=years[-1]:
#             listfile += [allfiles[idx]]

    print('List of forcings')
    print(listfile)

    # Create yearly file otherwise to heavy (process killed)

    for idx_yr, yrfl in enumerate(listfile):
        print(f'\nWorking on {yrfl}')

        ## Grid ERA5
        # nc = xr.open_mfdataset(listfile[0], combine='nested', concat_dim='time')
        nc = xr.open_mfdataset(yrfl, combine='nested', concat_dim='time')

        ## Grid TOPAZ
        template = {}
        template['dailync_full'] = '{year:04d}{month:02d}{day:02d}_dm-12km-NERSC-MODEL-TOPAZ4B-ARC-RAN.fv2.0.nc'
#         withsit_dir = '/nird/projects/nird/NS2993K/Leo/Jiping_full/TP4b_with'
        withsit_dir = '/cluster/work/users/leoede/Leo/Jiping_full/TP4b_with'
        nc1 = xr.open_dataset(os.path.join(rootdir, withsit_dir, template['dailync_full'].format(year=2011, month=10, day=1)))

        # just to check area to consider
        nc_sel = nc1['sithick'].isel(y=slice(*lim_jdm), x=slice(*lim_idm))
        
        tlat = nc1['latitude'].isel(y=slice(*lim_jdm), x=slice(*lim_idm))
        tlon = nc1['longitude'].isel(y=slice(*lim_jdm), x=slice(*lim_idm))

        # -----
        ## Determine closest point of ERA5 on TOPAZ grid

        tlat1d = tlat.stack(z=('y','x'))
        tlon1d = tlon.stack(z=('y','x'))
        tlon1d_180 = long_trans.l180_to_360(tlon1d.data)
        close_lat = [find_nearest(nc['latitude'].data, lat) for lat in tlat1d.data]
        close_lon = [find_nearest(nc['longitude'].data, lon) for lon in tlon1d_180.data]
        lat_idx = [item[1] for item in close_lat]
        lat_nc = [item[0] for item in close_lat]
        lon_idx = [item[1] for item in close_lon]
        lon_nc = [item[0] for item in close_lon]


        # # retrieve points for each time step (6h)
        # converting to numpy and get the indexes is the most efficient and fast
        print('Dataset to numpy...')
        ds = nc[var].to_numpy()
        ds_sel = ds[:, lat_idx, lon_idx]
        ds3 = ds_sel.reshape(ds_sel.shape[0], 329, 450)

        # -----
        # save yearly era5 selection to .nc
        dttm = pd.to_datetime(nc['time'].to_numpy())
        print('Saving .nc...')
        to_save = xr.Dataset(data_vars={f'{var}':(['time','y','x'], ds3, 
                                                       {'name':nc[f'{var}'].name, 
                                                        'long_name':nc[f'{var}'].long_name,
                                                        'units':nc[f'{var}'].units,
                                                        'jdm':lim_jdm,
                                                        'idm':lim_idm}),}, 
                              coords=dict(time=nc['time'],
                                        longitude=tlon,
                                        latitude=tlat),
                              attrs=dict(
                                  description=f'Truncated ERA5 forcing fields for {allyears[idx_yr]} over the Arctic',
                                  author='Leo Edel, Nersc',
                                  project='TARDIS',
                                  date=f'{date.today()}')
                              )

        filename_nc = os.path.join(pkldir,f'{var}_{allyears[idx_yr]}_ERA5_coloc_TOPAZ.nc')

        to_save.to_netcdf(filename_nc)
        print(f'.nc saved as:  {filename_nc}')

        # ----------

        # Demo Load .nc
        # filename_nc = os.path.join(pkldir,f'{var}_{years[0]}_{years[-1]}.nc')
        # ncl = xr.open_mfdataset(filename_nc, combine='nested', concat_dim='time')
        # ncl[var][0].plot()

