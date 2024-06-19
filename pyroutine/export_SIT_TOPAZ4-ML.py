'''
Save TOPAZ4-ML (final dataset corrected) as daily .nc files

Import .nc
Post process: - no sit < 0
Inform metadata
Save daily file as .nc
'''
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
from datetime import date
import datetime

from src.data_preparation import merge_TOPAZ


def save_daily_nc(idata, all_days, year):
    '''
    
    Parameters:
    -----------
    
        idata    :  xarray.Dataset. Dataset to save
        all_days :  np.array of datetime.date(). Save the days as .nc
        year     :  int, use for path
    
    '''
    
    odir = f'/scratch/project_465000269/edelleo1/Leo/export_SIT_TOPAZ4-ML/{year}/' 
    if not os.path.exists(odir):
        subprocess.run(['mkdir', f'{odir}'])
    
    for day in all_days:
        
        # identify index to plot
        chrono = pd.DataFrame({'date':pd.to_datetime(idata.sit.time.to_numpy())})
        chrono_dt = np.array([dt.date() for dt in chrono.date])
        t_idx = np.where(chrono_dt==day)[0][0]
        
        sday = day.strftime('%Y%m%d')
        
        oname = f'SIT_TOPAZ4-ML_{sday}.nc'
        idata.isel(time=t_idx).to_netcdf(f'{odir}{oname}')
        print(f'Saved as: {odir}{oname}')    

# ------------------------------   
##      get year to plot
# ------------------------------   
        
narg = len(sys.argv)  # number of arguments passed
if narg > 1:
    year = int(sys.argv[1])
else:
    year = 1992
    
        
# ------------------------------   
##           Import
# ------------------------------

sit_ml, chrono_ml = merge_TOPAZ.load_nc()
sic_fr, chrono_fr = merge_TOPAZ.load_sic_fr()


# ------------------------------
##         Post process
# ------------------------------

# Negative sea ice thickness is set to 0m
sit_pp = sit_ml.where((sit_ml>0) | (sit_ml.isnull()), 0)


# ------------------------------
##           Metadata
# ------------------------------

metadata = dict(
    title='Sea Ice Thickness (SIT)',
    description='Daily Sea Ice Thickness reconstructed by combining machine learning and data assimilation',
    project='TARDIS',
    comment='Dataset for Edel et al. (2024), Reconstruction of Arctic sea ice thickness (1992-2010) based on a '
    'hybrid machine learning and data assimilation approach, The Cryosphere, submitted.',
    summary='Using variables from TOPAZ4, CS2SMOS, ERA5, sea ice age. References can be found in Edel et al. (2024)',
    institution='Nansen Environmental and Remote Sensing Center, Jahnebakken 3, N-5007 Bergen, Norway',
    publisher_name='NERSC',
    publisher_email='post@nersc.no',
    publisher_url='https://nersc.no/',
    creator='Leo Edel',
    contact_email_primary='leo.edel@nersc.no',
    contact_email_secondary='laurent.bertino@nersc.no',
    product_version='0.0.1-submitted',
    conventions='CF-1.4',
    license='CC BY 4.0',
    production_date=f'{date.today()}',
    time_coverage_start='1992-01-01',
    time_coverage_end='2022-11-31',
    projection_grid_name='polar_stereographic',
    projection_proj4='+proj=stere +lon_0=-45 +lat_0=90 +k=1 +R=6378273 +no_defs',
    projection_latitude_of_projection_origin='90.',
    projection_longitude_of_projection_origin='-45.',
    projection_scale_factor_at_projection_origin='1.',
    projection_straight_vertical_longitude_from_pole='-45.',
    projection_earth_radius='6378273.',
    projection_false_easting='0.',
    projection_false_northing='0.',
   )

# ------------------------------
##           Attributes
# ------------------------------

del sic_fr.attrs['grid_mapping']
sic_fr.attrs['grid_mapping_name'] = 'polar_stereographic'

sit_pp.attrs['units'] = 'm'
sit_pp.attrs['standard_name'] = 'sea_ice_thickness'
sit_pp.attrs['grid_mapping_name'] = 'polar_stereographic'

# ------------------------------
##           Dataset
# ------------------------------


odata = xr.Dataset(data_vars={'sit':sit_pp,
                              'sic':sic_fr},
                   attrs=metadata
                  )
# ------------------------------
##             Save
# ------------------------------

# year = 1994
d1 = datetime.date(year,1,1)
d2 = datetime.date(year,12,31)
# d2 = datetime.date(year,1,3)
    
days = np.array([d1 + datetime.timedelta(days=x) for x in range((d2-d1).days + 1)])
    
save_daily_nc(odata, all_days=days, year=year)