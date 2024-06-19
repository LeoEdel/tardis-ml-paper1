'''Function to extract linear tendency from dataset
'''

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats



def linreg(xr_vol, date_1, date_2):
    '''
    Computes the indexes of 2 days
    Returns the slope of the linear regression, with p value
    
    Parameters:
    -----------
    
        xr_vol    :     xarray.DataArray, sea ice volume, average over the Arctic. dimension: 'time'
        
        
    # Demo for 1 period:
    day_1 = datetime.datetime(2014,1,1)
    day_2 = datetime.datetime(2016,1,1)
    new_t, y_pp, sslope, ppval = linreg(xr_vol, day_1, day_2)
    plt.plot(new_t, y_pp)
    '''
    # identify index to plot
    chrono_tp = pd.DataFrame({'date':pd.to_datetime(xr_vol.time.to_numpy())})
    chrono_dt = np.array([dt.date() for dt in chrono_tp.date])
    idx_1 = np.where(chrono_dt==date_1.date())[0]
    idx_2 = np.where(chrono_dt==date_2.date())[0]
    
    ## if date not found:
    if len(idx_1) > 0:  ## date found in time axis
        idx_1 = idx_1[0]
    else:
        # import pdb; pdb.set_trace()
        return None, None, None, None, None
    if len(idx_2) > 0:  ## date found in time axis
        idx_2 = idx_2[0]
    else:
        return None, None, None, None, None
    
    ## Inputs of linear regression
    nt = np.arange(len(xr_vol.time))
    lr_vol = xr_vol.isel(time=slice(idx_1, idx_2)).to_numpy()
    lr_time = nt[idx_1:idx_2]

    m1 = LinearRegression().fit(lr_time.reshape(-1,1), lr_vol)  ## fit

    ## Prediction
    new_x = nt[idx_1:idx_2].reshape(-1,1)  ## for prediction
    new_time = xr_vol.time[idx_1:idx_2]  ## for plot
    y_pred = m1.predict(new_x)

    ## just to get p-value
    slope, intercept, r_val, p_val, std_err = stats.linregress(lr_time, lr_vol)

    # return line to plot X, Y, slope coefficient, p value, nt (number of time steps)
    return new_time, y_pred, slope, p_val, nt