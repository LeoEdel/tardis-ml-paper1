'''Reconstruct the full time serie for TOPAZ
Merge prediction period (approx. before 2011) and training period (approx. after 2011)
to get one unique time serie (easier to evaluate and plot)

Aslo returns TOPAZ4 non assimilated time series and/or baseline
'''

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc4
import datetime

from src.modelling import sit_corrected
from src.data_preparation import load_data
from src.utils import tardisml_utils



def cut_sit(sit_class, d1, d2):
    '''Remove extra days of a sit_corrected class
    
    Parameters:
    -----------
        sit_class     : class from sit_corrected.SITCorrected()
        d1           :   datetime.date() object, day 1 to identify
        d2           :   datetime.date() object, day 2 to identify
    '''

    start_idx, end_idx = return_idx_dates(sit_class.sit.time, d1, d2)
    start_bl, end_bl = return_idx_dates(sit_class.sit_bl.time, d1, d2)
    start_na, end_na = return_idx_dates(sit_class.sit_na.time, d1, d2)
    
    sit_class.sit = sit_class.sit.isel(time=slice(start_idx, end_idx+1))
    sit_class.sit_bl = sit_class.sit_bl.isel(time=slice(start_bl, end_bl+1))
    sit_class.sit_na = sit_class.sit_na.isel(time=slice(start_na, end_na+1))
    

def load(return_na=False, return_bl=False, return_mean=False, 
         chr_as_dt=True, verbose=1):
    '''

    Parameters:
    -----------

        return_na    :  bool, return non assimilated time series
        return_bl    :  bool, return baseline
        return_mean  :  bool, return average over the whole Arctic (with SIC>0.15) by default
        chr_as_dt    :  bool. if True, return chrono (chr, time variable) as an array of datetime.date() object (<chrono_dt>). if False, return chrono as panda.DataFrame object (<chrono>).
        verbose      :  int,  script will be talking through its execution


    '''

    rootdir = tardisml_utils.get_rootdir()
    
    if verbose == 1: print('Import...')
    
    # ---------------------------------
    #       Import 1991-1998
    # ---------------------------------

#     ipath = 'Leo/results/lstm_231212-183758/'  # var4
#     ipath = 'Leo/results/lstm_240405-180331/'  # adjSIC full opti1
#     ipath = 'Leo/results/lstm_240507-160336/'  # adjSIC full opti1
    ipath = 'Leo/results/lstm_240523-170100/'  # for_paper_3 opti v2 - batch size = 32 + SIA
#     ipath = 'Leo/results/lstm_240524-173523/'  # for_paper_3 opti v2 - batch size = 32 + SIA. 24 PCs
#     ipath = 'Leo/results/lstm_240614-004737/'  # for_paper_3 LSTM relu residual 8PCs
  
    
    ml_name ='LSTM'
#     ml_name ='LSTM3_bk'

    ifile = f'{rootdir}{ipath}'
    m91 = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=0, objective='apply91')
    
    if verbose == 1: print('\t1991-1998')
    ##  Reconstruct SIT values 
    m91.reconstruct_sit()
    
    
    # remove days before 1992-01-01 and after 1998-12-31
    cut_sit(m91, datetime.date(1992,1,1), datetime.date(1998,12,31))
    
    if return_mean:
        m91.compute_mean(sic_min=0.15)
    
    # ---------------------------------
    #       Import 1999-2010
    # ---------------------------------

    ifile = f'{rootdir}{ipath}'
    m2000 = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=0, objective='apply')

    if verbose == 1: print('\t1999-2010')
    #  Reconstruct SIT values 
    m2000.reconstruct_sit()
    
    cut_sit(m2000, datetime.date(1999,1,1), datetime.date(2011,1,30))
    
    if return_mean:
        m2000.compute_mean(sic_min=0.15)

    # ---------------------------------
    #       Import 2011-2022
    # ---------------------------------

    ifile = f'{rootdir}{ipath}'
    m2011 = sit_corrected.SITCorrected(ifile, name=ml_name, verbose=0)

    if verbose == 1: print('\t2011-2022')
    #  Reconstruct SIT values 
    m2011.reconstruct_sit()
    
    cut_sit(m2011, datetime.date(2011,1,31), datetime.date(2022,11,30))
    
    
    if return_mean:
        m2011.compute_mean(sic_min=0.15)
    
    # ---------------------------------------------
    #       Merge prediction and training
    # ---------------------------------------------
    if verbose == 1: print('Merging...')
    
    # join first/second part
#     if not return_mean:
    sit_ml = xr.concat([m91.sit, m2000.sit, m2011.sit], dim='time')  # , compat='override', coords='minimal')
    sit_na = xr.concat([m91.sit_na, m2000.sit_na, m2011.sit_na], dim='time')  # , compat='override', coords='minimal')
    sit_bl = xr.concat([m91.sit_bl, m2000.sit_bl, m2011.sit_bl], dim='time')  # , compat='override', coords='minimal')
    
    if return_mean:
        sit_mlm = xr.concat([m91.sit_m, m2000.sit_m, m2011.sit_m], dim='time')  # , compat='override', coords='minimal')
        sit_nam = xr.concat([m91.sit_nam, m2000.sit_nam, m2011.sit_nam], dim='time')  # , compat='override', coords='minimal')
        sit_blm = xr.concat([m91.sit_blm, m2000.sit_blm, m2011.sit_blm], dim='time')  # , compat='override', coords='minimal')

    # datetime
    chrono = pd.DataFrame({'date':pd.to_datetime(sit_ml.time)})
    
    # Add attributes
    # + which ML model is used for this merge product? = LSTM
    # + date of ML training = 231212-183758
    
    
    
    if chr_as_dt:
        chrono_dt = np.array([dt.date() for dt in chrono.date])
    else:
        chrono_dt = chrono
        
    if return_na and return_bl and return_mean:
        return sit_ml, chrono_dt, sit_na, sit_bl, sit_mlm, sit_nam, sit_blm
    
    if return_na and return_bl:
        return sit_ml, chrono_dt, sit_na, sit_bl
    elif return_na:
        return sit_ml, chrono_dt, sit_na
    elif return_bl:
        return sit_ml, chrono_dt, sit_bl
    else:
        return sit_ml, chrono_dt


def load_nc(chr_as_dt=True, verbose=1):
    '''Load full time series from .nc files
    '''
    
    
    # to avoid creating large chunks of data when importing dataset
    import dask
    dask.config.set({"array.slicing.split_large_chunks": True}) # to avoid creating the large chunk in the first place
    # dask.config.set({"array.slicing.split_large_chunks": False}) to allow the large chunk and silence the warning
    
    
    if verbose == 1: print('Loading ML-SIT...')
    
    #       Import 1991-1998
    # ---------------------------------
    if verbose == 1: print('\t1992-1998')
    
#     ifolder = '/scratch/project_465000269/edelleo1/Leo/results/lstm_231212-183758/ml/'
#     ifolder = '/scratch/project_465000269/edelleo1/Leo/results/lstm_240405-180331/ml/'
#     ifolder = '/scratch/project_465000269/edelleo1/Leo/results/lstm_240507-160336/ml/'
    ifolder = '/scratch/project_465000269/edelleo1/Leo/results/lstm_240523-170100/ml/'  # for_paper_3 opti v2 - batch size = 32 + SIA
#     ifolder = '/scratch/project_465000269/edelleo1/Leo/results/lstm_240524-173523/ml/'  # for_paper_3 opti v2 - batch size = 32 + SIA. 24 PCs
#     ifolder = '/scratch/project_465000269/edelleo1/Leo/results/lstm_240614-004737/ml/'  # for_paper_3 LSTM relu residual 8PCs

    ifile = 'sit_gLSTM3_bk_1991_1999_01.nc'
    f1 = f'{ifolder}{ifile}'
    
    sit_1 = xr.open_mfdataset([f1], combine='nested', concat_dim='time')['sit_ml']
    # get index of start (1992,1,1) and end (1998,12,31)
    start_1, end_1 = return_idx_dates(sit_1.time, datetime.date(1992,1,1), datetime.date(1998,12,31))
    
    #       Import 1999-2010
    # ---------------------------------
    if verbose == 1: print('\t1999-2010')
    
    ifile = 'sit_gLSTM3_bk_1998_2011_01.nc'
    f2 = f'{ifolder}{ifile}'
    
    sit_2 = xr.open_mfdataset([f2], combine='nested', concat_dim='time')['sit_ml']    
    # get index of start (1999,1,1) and end (2011,1,30)
    start_2, end_2 = return_idx_dates(sit_2.time, datetime.date(1999,1,1), datetime.date(2011,1,30))
    
    #       Import 2011-2022
    # ---------------------------------
    if verbose == 1: print('\t2011-2022')
    
    ifile = 'sit_gLSTM3_bk_2011_2022_01.nc'
    f3 = f'{ifolder}{ifile}'
    
    sit_3 = xr.open_mfdataset([f3], combine='nested', concat_dim='time')['sit_ml']
    # get index of start (1992,1,1) and end (1998,12,31)
    start_3, end_3 = return_idx_dates(sit_3.time, datetime.date(2011,1,31), datetime.date(2022,11,30))
    
    #       Merge prediction and training
    # ---------------------------------------------
    
    
    if verbose == 1: print('Merging...')
    
    sit_ml = xr.concat([sit_1.isel(time=slice(start_1, end_1+1)), 
                        sit_2.isel(time=slice(start_2, end_2+1)), 
                        sit_3.isel(time=slice(start_3, end_3+1))], 
                       dim="time")
    
    
    
    chrono_ml = pd.DataFrame({'date':pd.to_datetime(sit_ml.time)})
    
    if chr_as_dt:
        chrono_dt = np.array([dt.date() for dt in chrono_ml.date])
        return sit_ml, chrono_dt
    
    return sit_ml, chrono_ml


def return_idx_dates(arr, d1, d2):
    '''
    Return indexes (location) of day 1 and day 2 in time array
    
    
    Parametres:
    -----------
    
        arr          :   xarray.DataArray 'time'
        d1           :   datetime.date() object, day 1 to identify
        d2           :   datetime.date() object, day 2 to identify
    '''
    
    chrono = pd.DataFrame({'date':pd.to_datetime(arr)})
    chrono_dt = np.array([dt.date() for dt in chrono.date])
    
    idx1 = np.where(chrono_dt==d1)[0][0]
    idx2 = np.where(chrono_dt==d2)[0][0]
    
    return idx1, idx2


def load_sic_fr(adjSIC=False, verbose=1):
    '''Load SIC full time series from .nc files
    '''
    
    
    # to avoid creating large chunks of data when importing dataset
    import dask
    dask.config.set({"array.slicing.split_large_chunks": True}) # to avoid creating the large chunk in the first place
    # dask.config.set({"array.slicing.split_large_chunks": False}) to allow the large chunk and silence the warning
    
    if adjSIC:
        suffixe = '_adjSIC'
    else:
        suffixe = ''
    
    if verbose == 1: print('Loading SIC freerun...')
    
    #       Import 1991-1998
    # ---------------------------------
    if verbose == 1: print('\t1991-1998')
    
    ifolder = '/scratch/project_465000269/edelleo1/Leo/results/pca_i100-550_j150-629/'
    ifile = f'siconc_TOPAZ4b23_1991_1998_FreeRun{suffixe}.nc'
    f1 = f'{ifolder}{ifile}'
    
    sic_1 = xr.open_mfdataset([f1], combine='nested', concat_dim='time')['siconc']
    # get index of start (1992,1,1) and end (1998,12,31)
    start_1, end_1 = return_idx_dates(sic_1.time, datetime.date(1992,1,1), datetime.date(1998,12,31))
    
    #       Import 1999-2010
    # ---------------------------------
    if verbose == 1: print('\t1999-2010')
    
    ifile = f'siconc_TOPAZ4b23_1999_2010_FreeRun{suffixe}.nc'
    f2 = f'{ifolder}{ifile}'
    
    sic_2 = xr.open_mfdataset([f2], combine='nested', concat_dim='time')['siconc']    
    # get index of start (1999,1,1) and end (2011,1,30)
    start_2, end_2 = return_idx_dates(sic_2.time, datetime.date(1999,1,1), datetime.date(2011,1,30))
    
    #       Import 2011-2022
    # ---------------------------------
    if verbose == 1: print('\t2011-2022')
    
    ifile = f'siconc_TOPAZ4b23_2011_2022_FreeRun{suffixe}.nc'
    f3 = f'{ifolder}{ifile}'
    
    sic_3 = xr.open_mfdataset([f3], combine='nested', concat_dim='time')['siconc']
    # get index of start (1992,1,1) and end (1998,12,31)
    start_3, end_3 = return_idx_dates(sic_3.time, datetime.date(2011,1,31), datetime.date(2022,11,30))
    
    #       Merge prediction and training
    # ---------------------------------------------
    
    if verbose == 1: print('Merging...')
    
    sic = xr.concat([sic_1.isel(time=slice(start_1, end_1+1)), 
                     sic_2.isel(time=slice(start_2, end_2+1)), 
                     sic_3.isel(time=slice(start_3, end_3+1))], 
                     dim="time")
    
    
    
    chrono = pd.DataFrame({'date':pd.to_datetime(sic.time)})
    
    return sic, chrono
    
    
    
def load_freerun(verbose=1):
    '''Recompose freerun from 1991 to 2020 with the following 3 parts:
                - 1991 - 2000 :   noSIT or freerun are the same 
                - 2000 - 2011 :   freerun (!different from noSIT!)
                - 2011 - 2020 :   freerun (noSIT does not exist)
                
                
                
    "noSIT" means TOPAZ4 with assimilation of all variables except SIT (from CS2SMOS)
    "freerun" is truely TOPAZ4 running without assimilation
    '''

    idir = '/scratch/project_465000269/edelleo1/Leo/results/pca_i100-550_j150-629/'
    
#     ifile1 = 'sithick_TOPAZ4b23_1991_2000_noSIT.nc'
#     ifile2 = 'sithick_TOPAZ4b23_2000_2011_FreeRun.nc'
#     ifile3 = 'sithick_TOPAZ4b23_2011_2019_noSIT.nc'
    
    ifile1 = 'sithick_TOPAZ4b23_1991_1998_FreeRun.nc'
    ifile2 = 'sithick_TOPAZ4b23_1999_2010_FreeRun.nc'
    ifile3 = 'sithick_TOPAZ4b23_2011_2022_FreeRun.nc'
    
    
    # Load
    if verbose == 1: print('Loading...')
    nc = nc4.Dataset(f'{idir}{ifile1}', mode='r')
    sit1 = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sithick'][:,:,:]
    
    nc = nc4.Dataset(f'{idir}{ifile2}', mode='r')
    sit2 = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sithick'][:,:,:]

    nc = nc4.Dataset(f'{idir}{ifile3}', mode='r')
    sit3 = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sithick'][:,:,:]
    
    # Merge
    if verbose == 1: print('Merging...')
    
    sit_fr = xr.concat([sit1, sit2, sit3], dim='time') 
  
    # drop duplicates. keep 2011-2020 if available
    sit_fr = sit_fr.drop_duplicates(dim='time', keep='last')
    
    return sit_fr

    
def load_ass(return_mean=False, verbose=1, adjSIC=False):
    '''Load SIT from TOPAZ4b with CS2SMOS assimilation
    '''
    
    import os
    from src.data_preparation import load_data
    
    if verbose == 1: print('Loading TOPAZ4b with CS2SMOS assimilation...')
    
    idir = '/scratch/project_465000269/edelleo1/Leo/results/pca_i100-550_j150-629/'
#     ifile1 = 'sithick_TOPAZ4b23_2011_2022.nc'
    ifile1 = 'sithick_TOPAZ4b23_2011_2022_adjSIC.nc'

    
    nc = nc4.Dataset(f'{idir}{ifile1}', mode='r')
    sit_a = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sithick'][:,:,:]
    
    
    if return_mean:
        filename = os.path.join(idir, f"siconc_TOPAZ4b23_2011_2022_FreeRun.nc")
        if adjSIC:
            filename = os.path.join(idir, f"siconc_TOPAZ4b23_2011_2022_FreeRun_adjSIC.nc")
            
        
        siconc_fr, chronosc_fr = load_data.load_nc(filename, f'siconc', True)
    
        sit_am = sit_a.where(siconc_fr>.15).mean(dim=('y','x')).compute()
        
        return sit_a, sit_am
    else:
        return sit_a
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 
