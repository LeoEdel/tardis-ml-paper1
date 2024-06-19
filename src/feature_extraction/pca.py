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

import src.utils.tardisml_utils as tardisml_utils
rootdir = tardisml_utils.get_rootdir()

#sys.path.append(os.getcwd() + '/../..')
#import tardis_ml




def missing_file(chrono, data_dir, template):
    """
    In:
        chrono      : pandas.core.series.Series -- Chronology
        data_dir    : String                    -- Name of the file containing data
        template    : dict
        
    Out:
        list_files : String list -- Paths to data file
    """
    list_files = [os.path.join(rootdir, data_dir, template['dailync'].format(year=date.year, dayinyear = date.dayofyear-1)) for date in chrono]
    missing = [os.path.basename(file) for file in list_files if not os.path.isfile(file)]
    
    if len(missing)>0:
        print('hack for missing files')
        with open(os.path.join(rootdir, data_dir, 'missing.yaml'),'w') as f:
            yaml.dump(missing, f)
        for file in list_files:
            if os.path.isfile(file):
                last_ok_file = file
            else:
                os.symlink(last_ok_file, file)
                
    return list_files        
     
  

    
    
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
        
        
        
def compute_pca(chrono, n_components, data_dir, target_field, lim_idm, lim_jdm, template, file_save='', nc_f=0, pca_dir=None, maskok=None, list_files=None):
    """ Compute PCA
    In:
        chrono       : pandas.core.series.Series
        n_components : int
        data_dir     : String
        target_field : String
        lim_idm      : tuple
        lim_jdm      : tuple
        template     : dict
        nc_f         : xarray.core.dataarray.DataArray -- only for withsit data ! nc forecast
        pca_dir      : String                          -- usefull only for withsit data
        maskok       : xarray.core.dataarray.DataArray -- usefull only for withsit data
        file_save    : String                          -- name in which to save the file (don't save if empty)
        
    Out:
        mu        : xarray.core.dataarray.DataArray
        X         : xarray.core.dataarray.DataArray
        X1d_nonan : xarray.core.dataarray.DataArray
        pca       : sklearn.decomposition._pca.PCA
        maskok    : xarray.core.dataarray.DataArray
        nc        : xarray.core.dataarray.DataArray
    """    
    
    if list_files is None:
        list_files = missing_file(chrono, data_dir, template)
    
    nc = xr.open_mfdataset(list_files, combine='nested', concat_dim='rdim')
    nc_sel = nc[target_field].sel(jdim=slice(*lim_jdm),idim=slice(*lim_idm))
    
    if pca_dir is not None:
        # save PCA for target var with SIT assimilated
        define_pca_ass(target_field, n_components, nc_sel, maskok, pca_dir)
        # then go on with model bias (SIT assimilation - no assimilation)
        nc_f_sel = nc_f[target_field].sel(jdim=slice(*lim_jdm),idim=slice(*lim_idm))
        nc_sel = nc_sel - nc_f_sel
    
    mu =  nc_sel.mean(dim='rdim').compute()
    X = nc_sel.compute()
    
    print("TODO : redo by combinbing forecast and analyis mask (even though they should be the same)")
    if maskok is None:
        maskok = (np.isfinite(X)).all(dim='rdim')
    mskok1d= maskok.stack(z=('jdim','idim'))
           
    X1d =  X.stack(z=('jdim','idim'))
    X1d_nonan = X1d.where(mskok1d, drop=True)
    
    #ntest, nval, ntrain, nsplit = tardis_ml.compute_dataset_splits(X1d_nonan)
    #print("FOR NOW PCA ON ALL POINTS")
    #ntest = 0

    #X1d_nonan = X1d_nonan[ntest:]
    #print('TODO: PCA only on training when more years available')
    
    pca = PCA(n_components=n_components).fit(X1d_nonan)

    
    if file_save != "":
        save_pca(file_save, pca)
    
    return mu, X, X1d_nonan, pca, maskok, nc
    

    
    
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
    for field in forcing_fields:
        print(f'{field}')
        
        X = forcings[field].reshape(forcings[field].shape[0],-1)
        if saveraw:
            filename = f'{odir}{field}.pkl'
            pkl.dump(X, open(filename,"wb"))
            print(f'Forcing {field[:6]} save as: {filename}')
        
        mu[field] = np.mean(X, axis=0)
        pca[field] = PCA(n_components=n_components)
        print("# TODO train/val")
        pca[field].fit(X)
        PCs[field] = pca[field].transform(X)
        EOFs[field] = pca[field].components_
        
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
    
    X1d =  X.stack(z=('jdim','idim'))

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
    X1d =  X.stack(z=('jdim','idim'))
    mskok1d= maskok.stack(z=('jdim','idim'))
    
    EOF1d = xr.DataArray(np.nan*np.ones ((n_components,X1d.shape[1])),dims=['comp','z'])
    multi_index = pd.MultiIndex.from_tuples(X1d.coords['z'].data,names=['jdim','idim'])
    EOF1d = EOF1d.assign_coords(z=('z',multi_index))
    EOF1d[{'z':mskok1d}] = pca.components_
    EOF2d= EOF1d.unstack('z')
    
    return EOF1d, EOF2d


def save_pca(filename, pca):
    '''Save PCA in .pkl file
    '''
    print(f'PCA saved: \n{filename}')
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
    X1d =  X.stack(z=('jdim','idim'))
    X1d_nonan = X1d.where(maskok1d, drop=True)

    # retrieve PC
    PCs = xr.DataArray(pca.transform(X1d_nonan), dims=['rdim','comp'])
    
    return PCs














