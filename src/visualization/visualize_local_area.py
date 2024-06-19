import matplotlib.pyplot as plt
import os
import numpy as np
import netCDF4 as nc4
import xarray as xr
from src.data_preparation.load_data import get_land_mask 


### ----
#   This may contain all the functions that do some plots of land mask
#   and local areas
### ----



def plot_land_mask(lim_idm, lim_jdm, rootdir, pca_dir, drwzne=True, showfig=True, savefig=False):
    '''Plot current local zone on the arctic map
    based on the mask between (200, 600) (600, 881)
    
    full resolution of TOPAZ: 880 800
    
    '''
    # get reference land mask  
    pca_dir_ref = 'Leo/results/pca_8N_i200-600_j600-881/'
    ref_idm = (200, 600)
    ref_jdm = (600, 881)
    _, data = get_land_mask(ref_idm, ref_jdm, rootdir, pca_dir_ref)
    
    lonmin = lim_idm[0] - ref_idm[0]
    lonmax = lim_idm[1] - ref_idm[0]
    latmin = lim_jdm[0] - ref_jdm[0]
    latmax = lim_jdm[1] - ref_jdm[0]

    data.plot(cmap=plt.get_cmap('gray'))  # plot full arctic
    
    if drwzne:  # draw local zone
        draw_zone(latmin, latmax, lonmin, lonmax, lw=None)

    if showfig:
        plt.show()
        
    if savefig:
        filename = f'land_mask_local_i{lim_idm[0]}-{lim_idm[1]}_j{lim_jdm[0]}-{lim_jdm[1]}.png'
        plt.savefig(f"{rootdir}{pca_dir}/{filename}")
        print(f'Saved as: {rootdir}{pca_dir}/{filename}')

    plt.close()
    
    

def draw_zone(latmin, latmax, lonmin, lonmax, lw=None):
    """trace les contours de la zone voulue 
    latz et lonz contiennet les 4 coins de la zone

    latz = [latmin,latmax,latmax,latmin,latmin]
    lonz = [lonmin,lonmin,lonmax,lonmax,lonmin]
    
    lw, float, line width of the zone
    """
    if lw==None:
        lw=1.5
    x = np.empty([0], float)
    y = np.empty([0], float)
    latz = [latmin, latmax, latmax, latmin, latmin]
    lonz = [lonmin, lonmin, lonmax, lonmax, lonmin]
    # np.linspace between different points to have a smooth curve
    for i in range(len(lonz)-1): 
        x = np.append(x, np.linspace(lonz[i], lonz[i + 1], 50))
        y = np.append(y, np.linspace(latz[i], latz[i + 1], 50))

    plt.plot(x, y, color='r', lw=lw, zorder=70)  # draw rectangle