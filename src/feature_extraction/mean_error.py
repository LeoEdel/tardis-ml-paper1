import xarray as xr
import pandas as pd
import numpy as np


def seasonal2(chrono, X):
    '''Return 2 seasonnall mean averaged over many years: Freezing (oct-April) and melting (Mai-Sept) season
    to get simplest mean correction: use only training period
    
    In:
        chrono, panda serie containing time dtype: datetime64
        X, variable to average (ntimes, lat, lon)
    
    Out:
        X_mm, variable averaged monthly (2, lat, lon)
    '''
    
    X_mm = xr.DataArray(data=np.zeros((2, X.shape[1], X.shape[2])))
    
#     seasons = [[12,1,2], [3,4,5], [6,7,8], [9,10,11]]
    seasons = [[10,11,12,1,2,3,4], [5,6,7,8,9]]
    
    
    for ni, n_month in enumerate(seasons):
        # get indexes for corresponding month
        month_idx = []
        month_idx += [get_idx_month(chrono, nm) for nm in n_month]
        season_idx = [item for sublist in month_idx for item in sublist]
        # do average        
        X_mm.values[ni] = X.isel(time=season_idx).mean(axis=(0))
    
    return X_mm 


def seasonally(chrono, X):
    '''Return seasonnaly mean averaged over many years
    to get simplest mean correction: use only training period
    
    In:
        chrono, panda serie containing time dtype: datetime64
        X, variable to average (ntimes, lat, lon)
    
    Out:
        X_mm, variable averaged monthly (4, lat, lon)
    '''
    
    X_mm = xr.DataArray(data=np.zeros((4, X.shape[1], X.shape[2])))
    
    seasons = [[12,1,2], [3,4,5], [6,7,8], [9,10,11]]
    
    for ni, n_month in enumerate(seasons):
        # get indexes for corresponding month
        month_idx = []
        month_idx += [get_idx_month(chrono, nm) for nm in n_month]
#         import pdb; pdb.set_trace()
        season_idx = [item for sublist in month_idx for item in sublist]
        # do average
        X_mm.values[ni] = X.isel(time=season_idx).mean(axis=(0))
    
    return X_mm 


def monthly(chrono, X):
    '''Return monthly mean averaged over many years
    to get simplest mean correction: use only training period
    
    In:
        chrono, panda serie containing time dtype: datetime64
        X, variable to average (ntimes, lat, lon)
    
    Out:
        X_mm, variable averaged monthly (12, lat, lon)
    '''
    
    X_mm = xr.DataArray(data=np.zeros((12, X.shape[1], X.shape[2])))
    
    for n_month in range(1,13):
        # get indexes for corresponding month
        month_idx = get_idx_month(chrono, n_month)
        # do average
        X_mm.values[n_month-1] = X.isel(time=month_idx.values).mean(axis=(0))
    
    return X_mm 


def get_idx_month(chrono, nm):
    '''return all indexes for the corresponding month
    
    chrono, panda serie containing time dtype: datetime64
    nm, number of month, int
    '''
    
    # check shape of chrono
    if chrono.ndim>=2:
        if chrono.shape[0]>chrono.shape[1]:
            chrono_val = chrono.values.reshape(chrono.shape[0])
        else:
            chrono_val = chrono.values.reshape(chrono.shape[1])
    else:
        chrono_val = chrono.values
        
        
    # we retrieve months
    tp = pd.DataFrame([chrono_val]).transpose()
    tp['month'] = tp[0].dt.month
    tp.drop(0, axis=1, inplace=True)

    # get indexes for corresponding month
    month_idx = tp.index.where((tp['month']==nm))
    return month_idx.dropna().astype(int)  # integer without nan
    # month_idx.values
    
    
    
def apply_mean_correction(chrono, Xf, Xe_mm):
    '''Apply the simplest correction to any time serie of the SIT
    
    Simplest correction: monthly mean on the training period, obtained with func mean_error.monthly()
    !! To be fair: correction must be extracted from training period only !!
    
    
    In:
        chrono, panda serie containing time dtype: datetime64
        Xf, time serie of the forecast (SIT without CS2 assimilation) (ntimes, lat, lon)
        Xe_mm, monthly mean error (12, lat, lon), uses for correction
        
    Out:
        Xf_mc, time serie of the forecast monthly corrected (ntimes, lat, lon)
    
    '''
    
    Xf_mc = Xf.copy()
    
    for n_month in range(1, 13):
        # get indexes for corresponding month
        month_idx = get_idx_month(chrono, n_month)
        Xf_mc.values[month_idx] = Xf_mc.values[month_idx] + Xe_mm.values[n_month-1]       
        
    return Xf_mc
    
    
    