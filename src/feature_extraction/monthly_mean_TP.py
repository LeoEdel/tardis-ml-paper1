'''Monthly mean for TOPAZ data
From format (days, y, x) to (month, y, x)
'''

import numpy as np
import xarray as xr
import pandas as pd

def compute_mm(sit):  # , chrono):
    '''Average for each month
    
    Parameters:
    -----------
    
        sit         :     xarray.DataArray. format (time, y, x) with time in days
        ## chrono      :     pandas.DataFrame. format (time) with time in days. Contains year-month-day
    
    
    Returns:
    --------
    
        mon_mean    :     monthly average value of sit
        mon_chrono  :     first day of each month
    
    '''
    
    chrono = pd.DataFrame({'date':pd.to_datetime(sit['time'].to_numpy())})
    
    ## Return a different index for each month

    mon_idx = 0  # init first month as index 0
    indexes = [mon_idx]  # first day of the first month

    last_month = chrono.date[0].month  # month [1-12] at t-1

    for i in range(1, len(chrono)):  # loop over days

        if last_month != chrono.date[i].month:  # if month is different
            mon_idx += 1                        # increments month index

        indexes += [mon_idx]                    # save month index
        last_month = chrono.date[i].month       # update month [1-12]

    mon_indexes = np.array(indexes)
    
    ## Average all days that have the same index = monthly mean

    n_months = np.unique(mon_indexes).size
    mon_mean = np.zeros((n_months, 479, 450), dtype=np.float16)
    mon_chrono = []

    for n, idx in enumerate(np.unique(mon_indexes)):          # loop over unique month index
        tmp_idx = np.where(mon_indexes==idx)[0]               # get locations of a given month index
        mon_mean[n] = sit.isel(time=tmp_idx).mean('time')       # average over all days of the month
        mon_chrono += [sit.isel(time=slice(tmp_idx[0],tmp_idx[0]+1)).time.to_numpy()[0]]      # save date (1st day of the month)

    mon_chrono = np.array(mon_chrono)
    
    
    # convert numpy to xarray
    mm = xr.DataArray(
                data=mon_mean,
                dims=["time", "y", "x"],
                coords=dict(
                    time=(["time"], mon_chrono),
                    x=(["x"], sit.x.data),
                    y=(["y"], sit.y.data),
                    longitude=(["y", "x"], sit.longitude.data),
                    latitude=(["y", "x"], sit.latitude.data)
                ),
            )

    mm.name = sit.name
    
    
    
    return mm # mon_mean, mon_chrono