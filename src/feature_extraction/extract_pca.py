#################################################################
#
# Functions to compute PCA and EOF
#
#################################################################

import yaml
import sys
import os
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.decomposition import PCA
import pickle as pkl
from datetime import datetime

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()


def define_pca_ass(target_field, n_components, nc_a_sel, maskok, pca_dir):
    """ Save SIT assimilated for intercomparison between TOPAZ if doesn't exist
    In:
        target_field : String
        n_components : int
        nc_a_sel     : xarray.core.dataarray.DataArray 3D
        maskok       : xarray.core.dataarray.DataArray 2D
        pca_dir      : String
    """
    # save SIT assimilated for intercomparison between TOPAZ

    filename = os.path.join(rootdir, pca_dir, f"{target_field}_forecast_SITass.nc")
    ofile_pca = os.path.join(rootdir, pca_dir, f"pca_{target_field}_{n_components}N_SITass.pkl")
    
    if not os.path.exists(filename) or not os.path.exists(ofile_pca):
        mu_a =  nc_a_sel.mean(dim='rdim').compute()
        Xa = nc_a_sel.compute()
        Xa.to_netcdf(filename)
        print(f'SIT ass saved in: {filename}')

        mskok1d= maskok.stack(z=('jdim','idim'))
        pca_a, _ = fit_pca(Xa, mskok1d, n_components)
        save_pca(ofile_pca, pca_a) 
        
        
def load_TOPAZ(list_files, target_field=None, lim_idm=None, lim_jdm=None):
    '''Load dataset for TOPAZ
    
    Parameters:
    -----------
    list_files       : list of string, daily .nc to load (from TOPAZ)
    target_field     : String, variable to select
    lim_idm          : tuple, longitude zone
    lim_jdm          : tuple, latitude zone
    
    Spatial selection (lim_idm, lim_jdm) is done only if a variable is selected
    
    '''
    
    print('Define chronology from .nc files...')
    listfields = sorted([os.path.basename(name) for name in list_files])
    dt = np.array([datetime.strptime(lf[:8], "%Y%m%d") for lf in listfields])
    chrono = pd.DataFrame({'date':dt})
    
    print('Loading .nc ...')
    nc = xr.open_mfdataset(list_files, combine='nested', concat_dim='time')
    
    if target_field is None and lim_idm is None and lim_jdm is None:
        return nc, chrono
    
    if target_field is not None:
        print('Variable selection...')
        nc_sel = nc[target_field]
    
    if lim_idm is not None and lim_jdm is not None and target_field is not None:
        print('Spatial selection...')
        nc_sel = nc_sel.isel(y=slice(*lim_jdm),x=slice(*lim_idm))
    
    return nc_sel, chrono
        
        
def compute_pca_TOPAZ(nc_sel, n_components, lim_idm, lim_jdm, ofile_pca='', ofile_X=''):
#(chrono, n_components, data_dir, target_field, lim_idm, lim_jdm, template, file_save='', nc_f=0, pca_dir=None, maskok=None, list_files=None):
    """ Compute PCA
    Parameters:
    -----------
        list_files      : list of string, daily .nc to load (from TOPAZ)
        n_components : int, number of PCA
        
        
        nc_sel       :  from load_TOPAZ()
        target_field : String
        lim_idm      : tuple, longitude zone
        lim_jdm      : tuple, latitude zone
        ofile_pca    : String, output filename to save pca (doesn't save if empty)
        ofile_X      : String, output filename to save raw values of variable (doesn't save if empty)
        
        
    Out:
        mu        : xarray.core.dataarray.DataArray
        X         : xarray.core.dataarray.DataArray
        X1d_nonan : xarray.core.dataarray.DataArray
        pca       : sklearn.decomposition._pca.PCA
        maskok    : xarray.core.dataarray.DataArray
        nc        : xarray.core.dataarray.DataArray
    """    
      
    mu =  nc_sel.mean(dim='time').compute()
    X = nc_sel.compute()
    
#     print("TODO : redo by combinbing forecast and analyis mask (even though they should be the same)")
#     if maskok is None:
    print('Compute ocean/land mask...')
    maskok = (np.isfinite(X)).all(dim='time')
    mskok1d = maskok.stack(z=('y','x'))
           
    X1d = X.stack(z=('y','x'))
    X1d_nonan = X1d.where(mskok1d, drop=True)
    
#     print('Todo: split between train/evaluation datasets')
    #ntest, nval, ntrain, nsplit = tardis_ml.compute_dataset_splits(X1d_nonan)
    #print("FOR NOW PCA ON ALL POINTS")
    #ntest = 0

    #X1d_nonan = X1d_nonan[ntest:]
    #print('TODO: PCA only on training when more years available')
    
    print(f'Compute PCA with ncomp = {n_components}...')
    pca = PCA(n_components=n_components).fit(X1d_nonan)

    
    if ofile_pca != "":
        save_pca(ofile_pca, pca)
#         print(f'PCA saved as {ofile_pca}')
    
    if ofile_X != '':
        X.to_netcdf(ofile_X)
        print(f'X saved as {ofile_X}')
        
    
    return mu, X, X1d_nonan, pca, maskok
    

    
    
def compute_pca_forcing(n_components, forcing_fields, forcings, saveraw=False, odir=''):
    """ Compute PCA for all forcing in forcing_fields
    In:
        X       : 3D array, ('rdim','jdim','idim')
        masknan : 1d array, (already stacked on 'jdim','idim')
        Ncomp   : integer   -- number of components for PCA
        saveraw : bool, save forcing dataset without PCA
    Out:
        mu   : Dict of numpy.ndarray
        pca  : Dict of sklearn.decomposition._pca.PCA
        PCs  : Dict of numpy.ndarray
        EOFs : Dict of numpy.ndarray
    """    
    mu = dict()
    pca = dict()
    PCs = dict()
    EOFs = dict()
    
#     maskok1d = maskok.stack(z=('y','x'))
    
    for field in forcing_fields:
        print(f'{field}')
        
        X = forcings[field].reshape(forcings[field].shape[0],-1)
        if saveraw:
            filename = f'{odir}{field}.pkl'
            pkl.dump(X, open(filename,"wb"))
#             print(f'Forcing {field.split('_')[0]} save as: {filename}')
        
        # exclude nan from mask
#         X1d_nonan = X
        mu[field] = np.mean(X, axis=0)
        pca[field] = PCA(n_components=n_components)
#         print("# TODO train/val")
#         import pdb; pdb.set_trace()
        pca[field].fit(X)
        PCs[field] = pca[field].transform(X)
        
        # compute EOF
        EOFs[field] = pca[field].components_
#         _, EOFs[field] = compute_eof(n_components, X, pca[field], maskok)
        
    return mu, pca, PCs, EOFs
    

    
def fit_pca(X, masknan, Ncomp):
    """ Compute PCA
    In:
        X       : 3D array, ('rdim','jdim','idim')
        masknan : 1d array, (already stacked on 'jdim','idim')
        Ncomp   : integer   -- number of components for PCA
        
    Out:
        pca : sklearn.decomposition._pca.PCA : X array stacked on 1d with only non nan values
    """
    
    X1d =  X.stack(z=('y','x'))

    X1d_nonan = X1d.where(masknan, drop=True)
#     X1d_nonan.shape

    return PCA(n_components=Ncomp).fit(X1d_nonan), X1d_nonan
#     pca_a.fit(Xa1d_nonan)
      
    
    
                
def compute_eof(n_components, X, pca, maskok):
    """ Compute EOF
    In:
        n_components : int
        X            : xarray.core.dataarray.DataArray
        pca          : sklearn.decomposition._pca.PCA
        maskok       : xarray.core.dataarray.DataArray
        
    Out:
        EOF1d        : xarray.core.dataarray.DataArray
        EOF2d        : xarray.core.dataarray.DataArray
    """
    X1d = X.stack(z=('y','x'))
    mskok1d = maskok.stack(z=('y','x'))
    
    EOF1d = xr.DataArray(np.nan*np.ones ((n_components,X1d.shape[1])),dims=['comp','z'])
    multi_index = pd.MultiIndex.from_tuples(X1d.coords['z'].data,names=['y','x'])
    EOF1d = EOF1d.assign_coords(z=('z',multi_index))
    EOF1d[{'z':mskok1d}] = pca.components_
    EOF2d = EOF1d.unstack('z')
    
    return EOF1d, EOF2d




def compute_EOF_2d(n_components, X, pca, maskok1d):
    """ Compute EOF
    In:
        n_components : int
        X            : xarray.core.dataarray.DataArray
        pca          : sklearn.decomposition._pca.PCA
        maskok       : xarray.core.dataarray.DataArray
        
    Out:
        EOF1d        : xarray.core.dataarray.DataArray
        EOF2d        : xarray.core.dataarray.DataArray
    """
#     X1d = X.stack(z=('y','x'))
#     mskok1d = maskok.stack(z=('y','x'))
    
    EOF1d = xr.DataArray(np.nan*np.ones ((n_components,X.shape[1])),dims=['comp','z'])
    multi_index = pd.MultiIndex.from_tuples(X.coords['z'].data,names=['y','x'])
    EOF1d = EOF1d.assign_coords(z=('z',multi_index))
    EOF1d[{'z':maskok1d}] = pca.components_
    EOF2d = EOF1d.unstack('z')
    
    return EOF1d, EOF2d



def save_pca(filename, pca):
    '''Save PCA in .pkl file
    '''
    print(f'PCA saved: {filename}')
    pkl.dump(pca, open(filename,"wb"))
    return




def pca_to_PC(pca, X, maskok1d):
    """Apply transform of PCA to X for valid pixels of maskok
    In:
        pca      : sklearn.decomposition._pca.PCA   
        X        : xarray.core.dataarray.DataArray
        maskok1d : xarray.core.dataarray.DataArray
    
    Out:
        PCs      : xarray.core.dataarray.DataArray
    """
    # and to put Xf in 1d and apply mask to only keep non nan values
    X1d = X.stack(z=('y','x'))
    X1d_nonan = X1d.where(maskok1d, drop=True)

#     print(X1d_nonan.shape)
    # retrieve PC
    PCs = xr.DataArray(pca.transform(X1d_nonan), dims=['time','comp'])
    
    return PCs


def pca_to_PC_isnan(pca, X):
    '''Same as pca_to_PC without using mask
    base on np.isnan already included in X
    '''
    X1d = X.stack(z=('y','x')).to_numpy()
    X1d_nonan = X1d[~np.isnan(X1d)]
    X1d_nonan = X1d_nonan.reshape(X1d.shape[0], X1d_nonan.shape[0]//X1d.shape[0])

    PCs = xr.DataArray(pca.transform(X1d_nonan), dims=['time','comp'])
    
    return PCs


def reshape_nonan(X):
    '''Reshape from 3D to 2D and remove nan values
    '''
    X1d = X.stack(z=('y','x')).to_numpy()
    X1d_nonan = X1d[~np.isnan(X1d)]
    X1d_nonan = X1d_nonan.reshape(X1d.shape[0], X1d_nonan.shape[0]//X1d.shape[0])

    return X1d_nonan


