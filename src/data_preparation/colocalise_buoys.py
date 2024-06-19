'''Colocalise IMB and TOPAZ
also applicable for ULS and TOPAZ
and any other array of points that need to be colocalise with TOPAZ grid


Call:

# arrays of datetime objects, to check easily dates
chrono_dt_imb = np.array([dt.date() for dt in chrono_imb.date])

sit_ml_loc, lats, lons, valid_ll = get_closest_arr(nlat, nlon, chrono_dt_imb, sit_ml, chrono_dt_ml)


'''


import numpy as np
import pandas as pd



def get_tindex(chrono_dt_ml, chrono_dt_imb):
    '''
    Get the indexes of IMB in TOPAZ ML-corrected
    
    Parameters:
    -----------
        chrono_dt_ml     :   ref time, from Machine Learning
        chrono_dt_imb    :   from buoy
    '''

    t_idx = []

    for dt in chrono_dt_imb:
        idx = np.where(chrono_dt_ml==dt)[0]
        if idx.size == 1:
            t_idx.append(int(idx[0]))
        elif idx.size == 0:
            t_idx.append(int(-999))
            
    return np.array(t_idx, dtype=int)



def get_closest_arr(lat, lon, chrono_dt_imb, sit_tp, chrono_dt_tp):
    '''
    Returns TOPAZ SIT (mean and std) for closest point in space and time
    for one given (lat, lon) point (from IMB buoys)
    
    Parameters:
    -----------
        lat                 :    array of latitude of given points
        lon                 :    array of longitude of given points
        chrono_dt_imb       :    array of datetime object, date of one given point
        sit_tp              :    xarray.DataArray, Sea Ice Thickness ML-corrected from TOPAZ4
        chrono_dt_tp        :    array of datetime object, dates associated to sit_tp
    
    Returns:
    --------
    
        sit                 : array of SIT from Topaz localisation in time and space with buoy
        lats                : array of valid lats
        lons                : array of valid lons
        valid               : valid lat lon points. may be non valid if time index is not found
    
    '''

    # for one given (lat,lon) point
    # get time index tm in TOPAZ chrono
    t_idx = get_tindex(chrono_dt_tp, chrono_dt_imb)
    
    tloc = []
    yloc = []
    xloc = []
    
    lats = []
    lons = []
    valid = []
    
    for tm, lt, ln in zip(t_idx, lat, lon):
        if tm == - 999:
            valid.append(False)
            continue
        
        valid.append(True)
        # localise the closest TOPAZ points for the (lat, lon) point
        # from https://stackoverflow.com/questions/58758480/xarray-select-nearest-lat-lon-with-multi-dimension-coordinates
        # First, find the index of the grid point nearest a specific lat/lon.   
        abslat = np.abs(sit_tp.latitude-lt)
        abslon = np.abs(sit_tp.longitude-ln)
        c = np.maximum(abslon, abslat)
        closest = np.where(c == np.min(c))

        if closest[0].size > 1:  # rare case where one (lat,lon) point is at equal distance of 2 grid points
            # then we just pick the first one
            yloc.append(closest[0][0])
            xloc.append(closest[1][0])
        else:
            yloc.append(closest[0][0])
            xloc.append(closest[1][0])
            
        tloc.append(tm)
        # save valid lat/lon
        lats.append(lt)
        lons.append(ln)
    
    sit = sit_tp.to_numpy()[tloc, yloc, xloc]
    
    
    # print(f'TOPAZ4 global corrected with RF, SIT: {point_ds.mean().data}')
    # print(f'ICESAT , SIT: {dico_ice["Avg_thkns"][index][pi]}')
    
    return sit, np.array(lats), np.array(lons), np.array(valid)
