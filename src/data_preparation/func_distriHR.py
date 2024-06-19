'''Functions that convert/transform colocalised data between ICESat-2 and TOPAZ4
'''

import numpy as np
import netCDF4 as nc4
import xarray as xr
import pyproj
from scipy.interpolate import griddata
import xarray as xr
import matplotlib.pyplot as plt


def convert_y_to_proba(sp_distri, bins=None):
    '''Convert distribution of X samples to distribution on SIT TOPAZ5 categories

    Convert y from array of values (raw distributions)
                 to  probabilities of bins (distributions projected on bins)

    Parameters:
    -----------

        bins     :    if None, bins of TOPAZ5 sea ice categories will be used


    Normalization:
    https://stackoverflow.com/questions/5498008/pylab-histdata-normed-1-normalization-seems-to-work-incorrect

    '''

    if bins is None:
        bins = np.array([0, 0.64, 1.39, 2.47, 4.57, 30])


    # y_tmp = self.sp_distri.copy()

    n = sp_distri.shape[0]
    nbins = len(bins)-1
    y_prob = np.zeros((n, nbins))

    for idx in range(n):
        # instead of : density=True (shows sum >1 in some cases), using weights 
        y_prob[idx] = np.histogram(sp_distri[idx], bins=bins, 
                                   weights=np.ones_like(sp_distri[idx])/float(len(sp_distri[idx]))
                                  )[0]

    #self.sp_distri = y_prob
    return y_prob




def get_gridTP_xy():
    '''Returns grid for x and y in TOPAZ4, and SIT to access dimensions
    '''

    # Get TOPAZ4 grid
    ifile = '/scratch/project_465000269/edelleo1/Leo/Jiping_2023/TP4b/20221231_dm-12km-NERSC-MODEL-TOPAZ4B-ARC-RAN.fv2.0.nc'

    nc_tp = nc4.Dataset(f'{ifile}', mode='r')
    sit = xr.open_dataset(xr.backends.NetCDF4DataStore(nc_tp))['sithick']

    xt, yt = np.meshgrid(sit.x.to_numpy(), sit.y.to_numpy())
    
    return xt, yt, sit


def project_ll_to_tp(lon, lat):
    '''Project IS2 to TOPAZ projected coordinates
    
    Convert ICESat2 (lat, lon) to TOPAZ x/y
    '''

    # Define the input and output coordinate systems
    in_proj = pyproj.CRS('EPSG:4326')  # WGS 84 - lat/lon

    # define TOPAZ projection: a bit offset compared to North Polar Stereographic
    proj_str = ('+proj=stere +a=6378273 +b=6356889.44891 +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=10901.11 +y_0=10901.447')
    out_proj = pyproj.CRS(proj_str)

    # Create a transformer to convert from lat/lon to North Polar Stereographic
    transformer = pyproj.Transformer.from_crs(in_proj, out_proj)

    # Convert the latitude and longitude to North Polar Stereographic
    x, y = transformer.transform(lat, lon)

    return x/1e5, y/1e5  # from m to 100 km (used in Topaz)




def interp_TPgrid(lats, lons, values, method='linear', return_latlon=False):
    '''Project unstructured dataset (lat, lon, value) onto TOPAZ4 grid 
    Returns 2D xarray and npoints in each cell
    
    Parameters:
    -----------
    
        - lats        :    1D array, latitude for each IS2 profile 
        - lons        :    1D array, longitude for each IS2 profile 
        - values      :    1D array, Z value each IS2 profile. Such as SIT, bias, error, KL/CRPS
        - method      :    string, Method of interpolation for scipy.griddata 'linear'. 
                                   'nearest', 'linear', 'cubic'
        - return_latlon:   bool, return 2 additionals parameters: latitude and longitude from TOPAZ grid
        
    '''
    # -----------------------------------------------
    # Convert ICESat2 (lat,lon) to TOPAZ grid
    lats_c, lons_c = project_ll_to_tp(lons, lats)
    
    # get TOPAZ grid x/y
    gx, gy, gsit = get_gridTP_xy()
    
    # Make 1 array for lat/lon
    points_c = np.hstack((lats_c[:,np.newaxis], lons_c[:,np.newaxis]))    
    
    # Interpolate on TOPAZ grid
    grid_c = griddata(points_c, values, (gx, gy), method=method, fill_value=np.nan)
    
    # -----------------------------------------------
    # Compute number of points in each cell
    binx = np.concatenate((gx[0],   np.array([gx[0,-1]+ gx[0, -1] - gx[0, -2]]) ))
    biny = np.concatenate((gy[:,0], np.array([gy[-1,0]+ gx[-1, 0] - gx[-2, 0]]) ))
    bins = [biny, binx]   
    
    # Count
#     grid_nbin, _, _, _ = plt.hist2d(lons_c, lats_c, bins=bins)
    grid_nbin, _, _ = np.histogram2d(lons_c, lats_c, bins=bins)
    
    # -----------------------------------------------
    # Put nan where land
    grid_c[np.isnan(gsit.isel(time=0))] = np.nan
    grid_nbin[np.isnan(gsit.isel(time=0))] = np.nan
    
    # Cut to the area we desire
    grid_interp = grid_c[150:629, 100:550]
    grid_nbin = grid_nbin[150:629, 100:550]
    
    # Transform in xarray
    grid_z = xr.DataArray(grid_interp[np.newaxis,:,:], 
                coords={'time': gsit.time.data, 'y': gsit.y.data[150:629], 'x': gsit.x.data[100:550]}, 
                dims=["time", "y", "x"])
    grid_n = xr.DataArray(grid_nbin[np.newaxis,:,:],
                coords={'time': gsit.time.data, 'y': gsit.y.data[150:629], 'x': gsit.x.data[100:550]}, 
                dims=["time", "y", "x"])
    
    if return_latlon:
        return grid_z, grid_n, gsit.latitude.to_numpy()[150:629, 100:550], gsit.longitude.to_numpy()[150:629, 100:550]
    
    return grid_z, grid_n

