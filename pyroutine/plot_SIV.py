'''Draw SIV time series with trends
'''

import datetime
import numpy as np
import xarray as xr
import netCDF4 as nc4
import matplotlib.pyplot as plt


import pyproj
from src.modules.grid.grid import Grid
from src.modules.topaz.v4.confmap import ConformalMapping

from src.data_preparation import load_data
from src.data_preparation import merge_TOPAZ
from src.utils import modif_plot
from src.utils import save_name
from src.utils import tardisml_utils
from src.visualization import visu_volume as vv
from src.feature_extraction import linear_regression as lr

rootdir = tardisml_utils.get_rootdir()

# ------------------------------
# choose which SIV to plot

## DEFAULT is TOPAZ4-ML
do_FR = False
do_BL = False




# ------------------------------
# Import TOPAZ grid

mtp_proj = pyproj.Proj('+proj=stere +lon_0=-45 +lat_0=90 +k=1 +R=6378273 +no_defs')
ifile = '/scratch/project_465000269/edelleo1/Leo/Jiping_2023/TP4b/20221231_dm-12km-NERSC-MODEL-TOPAZ4B-ARC-RAN.fv2.0.nc'
nc = nc4.Dataset(f'{ifile}', mode='r')
sit = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sithick']
xt, yt = np.meshgrid(sit.x.to_numpy(), sit.y.to_numpy())
t_grid = Grid(mtp_proj, xt*100000, yt*100000)

# ------------------------------
# Get area of grid
# ------------------------------

area = (t_grid.dx / t_grid.mfx) * (t_grid.dy / t_grid.mfx)

# reshape
area = area[150:629, 100:550]
area = area[::-1]  # flip right side up

# Import SIT
# ------------------------------

# sit_ml, chrono_dt, sit_fr, sit_bl = merge_TOPAZ.load(return_na=True, return_bl=True, return_mean=False)
sit_ml, chrono_dt = merge_TOPAZ.load_nc()


# sit_nan = (sit_ml.where((0<=sit_ml))).where(np.isfinite(sit_ml))  # exclude open ocean (SIT=0m)
sit_nan = (sit_ml.where((0<sit_ml))).where(np.isfinite(sit_ml))  # exclude open ocean (SIT=0m)
sit_name = 'TOPAZ4-ML'

# for comparison:
# TOPAZ4-FR
if do_FR:
    sit_nan = (sit_fr.where((0<sit_fr))).where(np.isfinite(sit_fr))  # exclude open ocean (SIT=0m)
    sit_name = 'TOPAZ4-FR'
    
# TOPAZ4-BL
elif do_BL:
    sit_nan = (sit_bl.where((0<sit_bl))).where(np.isfinite(sit_bl))  # exclude open ocean (SIT=0m)
    sit_name = 'TOPAZ4-BL'


mask_ocean = sit_ml.isel(time=0).where(np.isnan(sit_ml.isel(time=0)), 1)  # ocean = 1, land = 0


# Compute sea ice volume (SIT x area x concentration)
# ------------------------------
sic_fr, chrono_sic = merge_TOPAZ.load_sic_fr()


volume = np.nansum(sit_nan * area * sic_fr, axis=(1,2)) / 1e9


xr_vol = xr.DataArray(
    data=volume,
    dims=["time"],
    coords=dict(
        time=(["time"], sit_nan.time.data)    #   <---- or sit_ml
    ),
    attrs=dict(
        name="Sea ice volume",
        description="Sea Ice Thickness * area of original grid",
        units="km³",
        standard_name='Volume',
    ),
)

# compute monthly volume for trends
vol_monthly = xr_vol.resample(time="1MS").mean(dim='time')
n_months = len(vol_monthly)

# ------------------------------
# Compute Trends
# ------------------------------

## Over the whole time period: daily trends
## Create dictionnary for several trends

dates_trends = [datetime.datetime(1992,1,1), datetime.datetime(2022,11,1)]

# dates_trends = [datetime.datetime(1991,1,1), datetime.datetime(2022,11,1), 
#                 datetime.datetime(1991,1,1), datetime.datetime(2002,1,1), 
#                 datetime.datetime(2002,1,1), datetime.datetime(2013,1,1), 
#                 datetime.datetime(2013,1,1), datetime.datetime(2022,11,1)]

trends = {}

for tr in range(len(dates_trends)//2):
    new_t, y_pp, sslope, ppval, nt =  lr.linreg(xr_vol, dates_trends[tr*2], dates_trends[tr*2+1])
    if new_t is not None:
        trends[tr] = (new_t, y_pp, sslope, ppval, dates_trends[tr*2], dates_trends[tr*2+1], nt)



# ------------------------------
## Trends for min 
## Create dictionnary for several trends

dates_trends = [datetime.datetime(1992,10,1), datetime.datetime(2022,10,1)]

# dates_trends = [datetime.datetime(1991,10,1), datetime.datetime(2022,10,1), 
#                 datetime.datetime(1991,10,1), datetime.datetime(2003,10,1), 
#                 datetime.datetime(2002,10,1), datetime.datetime(2013,10,1), 
#                 datetime.datetime(2013,10,1), datetime.datetime(2022,10,1)]

trends_min = {}

for tr in range(len(dates_trends)//2):
    new_t, y_pp, sslope, ppval, nt =  lr.linreg(vol_monthly.isel(time=range(9,n_months,12)), dates_trends[tr*2], dates_trends[tr*2+1])
    if new_t is not None:
        trends_min[tr] = (new_t, y_pp, sslope, ppval, dates_trends[tr*2], dates_trends[tr*2+1], nt)


# ------------------------------
## Trends for max
        
dates_trends = [datetime.datetime(1992,5,1), datetime.datetime(2022,5,1)]

# dates_trends = [datetime.datetime(1991,5,1), datetime.datetime(2022,5,1), 
#                 datetime.datetime(1991,5,1), datetime.datetime(2002,5,1), 
#                 datetime.datetime(2002,5,1), datetime.datetime(2013,5,1), 
#                 datetime.datetime(2013,5,1), datetime.datetime(2022,5,1)]
# 
trends_max = {}

for tr in range(len(dates_trends)//2):
    new_t, y_pp, sslope, ppval, nt =  lr.linreg(vol_monthly.isel(time=range(4,n_months,12)), dates_trends[tr*2], dates_trends[tr*2+1])
    if new_t is not None:
        trends_max[tr] = (new_t, y_pp, sslope, ppval, dates_trends[tr*2], dates_trends[tr*2+1], nt)
        

# ------------------------------
#          Plot 
# ------------------------------


odir = f'{rootdir}Leo/results/bin_fig/'
ofile = f'SIV_240523-170100_{sit_name}.png'
vv.draw_vol(xr_vol, vol_monthly, trends=trends, trends_min=trends_min, trends_max=trends_max, 
            odir=odir, ofile=ofile, savefig=True)






# ------------------------------
# Print Trends



print('Monthly trends')

print('Minimum: October')
print(trends_min[0][2]*10, 'km3 per decade')

print('Maximum: May')
print(trends_max[0][2]*10, ' km3 per decade')

print('Average over the full year:')
print(trends[0][2]*365*10, ' km3 per decade')




















