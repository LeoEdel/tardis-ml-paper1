#!/usr/bin/env python
# coding: utf-8

import numpy as np

def center_running_mean(arr, ndays, npd=1):
    '''
    Centered Running Mean
    Running mean of a 1d array (arr) over a windows of ndays (odd)
    the windows is averaging from day t=0 until day t=0+ndays
    
    Parameters:
    -----------
    
        npd    :    integer, is the Number of values Per Day (default:1)
    
    '''

    window_size = ndays*npd + 1 # always odd: 5 for npd=1
    
    i = 0
    # Initialize an empty list to store moving averages
    # no mean for the 1st element of array
    moving_averages = []

    # Loop through the array to consider every window
    while i < window_size//2:    # for the first points of the running mean
        window_average = np.nanmean(arr[0:i+window_size//2])
        moving_averages.append(window_average)
        i += npd
    
    while i < len(arr) - window_size//2:
        # central window
        window_average = np.nanmean(arr[i-window_size//2:i+window_size//2])
        # Store the average of current window in moving average list
        moving_averages.append(window_average)
        # Shift window to right by 4 position = 1 day
        i += npd
        
    while i < len(arr) - npd +1:  # for the last points of the running mean
        idx = len(arr) - i

        window_average = np.nanmean(arr[-idx:-1])
        moving_averages.append(window_average)
        i += npd
        
    return moving_averages



def grid_center_running_mean(grid, ndays, npd=1):
    '''
    Centered Running Mean for 3D array
    Running mean of a 3d array over a windows of ndays (odd)
    the windows is averaging from day t=0 until day t=0+ndays
    
    Parameters:
    -----------
    
        grid     : 3D array, format (time, lat, lon)
        ndays    : integer, size on the running window in days
        npd      : integer, Number of values Per Day (default:1, ERA5 = 4)
    '''

    window_size = ndays*npd + 1 # always odd: 5 for ndp=1
    
    i = 0
    # Initialize an empty array to store moving averages
    nt = int(grid.shape[0] / npd)
    moving_averages = np.zeros(shape=(nt, grid.shape[1], grid.shape[2]))

    # Loop through the array to consider every window
    while i < window_size//2:    # for the first points of the running mean
        moving_averages[i//npd] = np.nanmean(grid[0:i+window_size//2], axis=(0))
        i += npd
    
    while i < grid.shape[0] - window_size//2:  # central windows
        # Store the average of current window in moving average list
        moving_averages[i//npd] = np.nanmean(grid[i-window_size//2:i+window_size//2], axis=(0))
        # Shift window to right by 4 position = 1 day
        i += npd
        
    while i < grid.shape[0] - npd +1:  # for the last points of the running mean
        idx = grid.shape[0] - i
        moving_averages[i//npd] = np.nanmean(grid[-idx:-1], axis=(0))
        i += npd
        
    return moving_averages