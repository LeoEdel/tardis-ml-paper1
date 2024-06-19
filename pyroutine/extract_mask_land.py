# extract land_mask

import os
import numpy as np
import netCDF4 as nc4
import xarray as xr

import src.utils.load_config as load_config
import src.utils.tardisml_utils as tardisml_utils

# Import config

rootdir = tardisml_utils.get_rootdir()
# Path to config file
file_config = '../config/config_default_2023.yaml'

_, _, _, _, pca_dir, _, _, _, _ = load_config.load_filename(file_config)
_, _, _, _, lim_idm, lim_jdm, _ = load_config.load_config_params(file_config)

str_xy = f"i{lim_idm[0]}-{lim_idm[1]}_j{lim_jdm[0]}-{lim_jdm[1]}"


# import TOPAZ
idir = f'/scratch/project_465000269/edelleo1/Leo/results/pca_{str_xy}/'
ifile = f'sithick_TOPAZ4b23_2010_2022.nc'
filename = f'{idir}{ifile}'

nc = nc4.Dataset(filename, mode='r')
sit = xr.DataArray(nc['sithick'][:,:,:],
                  dims=['time','y','x'],
                  coords=dict(
                  x=(['x'], nc['x'][:].data),
                  y=(['y'], nc['y'][:].data),
                  time=(['time'], nc['time'][:].data)
                  ),     
              )

# compute mask
maskok = (np.isfinite(sit)).all(dim='time')
maskok.name = 'sithick'

# save mask in pca_dir from config
filename = os.path.join(rootdir,pca_dir,f"land_mask_{str_xy}.nc")
maskok.to_netcdf(filename)
