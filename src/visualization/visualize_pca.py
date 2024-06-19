#################################################################
#
# Functions for ploting graph abput PCA & EOF informations
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt

import src.utils.tardisml_utils as tardisml_utils
from src.utils import modif_plot

rootdir = tardisml_utils.get_rootdir()




def plot_pca_variance(n_components, pca, target_field, showfig=False, filename=''):
    """ Plot Cumulative explained variance of the first axes and save it in the given filename.
    In:
        n_components : int
        pca_f        : sklearn.decomposition._pca.PCA
        target_field : String
        filename     : String -- name in which to save the file (don't save if empty)     
    """
  
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9,9))
    
    plt.bar(x=np.arange(n_components)+1,height=pca.explained_variance_ratio_.cumsum());
#     plt.title(f'Cumulative explained variance of the first axes ({target_field})')
    plt.title(f'{target_field}')
#     plt.title('SIT bias')
    plt.xlabel('PCA index')
    plt.ylabel('Explained variance');
    
    plt.ylim([0,1])
    ax.set_xticks(np.arange(0, len(pca.explained_variance_ratio_)+1, step=1))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True) 
    
    modif_plot.resize(fig, s=18)
    
    if filename != "":
        plt.savefig(filename)
        print(f'Fig saved as: {filename}')

    if showfig:
        plt.show()
        
    plt.close()

        
def plot_ncomp_var(PCAs, var_fields, showfig=False, filename=''):
    '''Plot cumulative variance as a function of number of components for PCA
    for all forcings in forcing_fields 
    '''
    
    
    max_plot = len(var_fields)
    fig, ax = plt.subplots(ncols=1, nrows=max_plot, figsize=(8,max_plot*4))
    
    for idx, (pca, var) in enumerate(zip(PCAs, var_fields)):
        stp = 1  # step for xi
        xi = np.arange(1, len(PCAs[pca].explained_variance_ratio_)+1, step=stp)
        y = np.cumsum(PCAs[pca].explained_variance_ratio_)

        ax[idx].set_ylim(0.0,1.1)
        ax[idx].plot(xi, y, marker='o', linestyle='--', color='b')

        ax[idx].set_xlabel('Number of Components')
        if len(PCAs[pca].explained_variance_ratio_) +1 > 20:
            stp = 5  # step for labels and grid

        ax[idx].set_xticks(np.arange(0, len(PCAs[pca].explained_variance_ratio_)+1, step=stp))
        ax[idx].set_ylabel('Cumulative variance (%)')
        ax[idx].set_title(f'{var}')

        ax[idx].axhline(y=0.95, color='r', linestyle='-')
        ax[idx].text(0.5, 0.85, '95% threshold', color = 'red', fontsize=10)

        ax[idx].grid(axis='x')
        
    plt.tight_layout()     
        
    if showfig:
        plt.show()
        
    if filename != '':
        plt.savefig(filename)
        print(f'Fig saved as: {filename}')
    
    plt.close()

def plot_save_eof_annotation(chrono, EOF2d, PCs, var_cum, max_plot=8, n_components=8, ofile='', savefig=True, showfig=False):
    """ Plot EOF and PC and save it in the given filename.
    In:
        chrono       : pandas.core.series.Series
        max_plot     : int
        n_components : int
        EOF2d        : xarray.core.dataarray.DataArray
        PCs_f        : xarray.core.dataarray.DataArray
        var_cum      : cumulated variance
        filename     : String  -- name in which to save the file (don't save if empty)    
    """
    
    if n_components < max_plot:
        max_plot = n_components
    
    # same colorbar
    mini, maxi = np.nanmin(EOF2d.isel(comp=0)), np.nanmax(EOF2d.isel(comp=0))
    mini, maxi = -np.max([np.abs(mini), maxi]), np.max([np.abs(mini), maxi])
    
    my_cmap = plt.get_cmap('bwr')
    
    fig, axes = plt.subplots(ncols=2, nrows=max_plot, figsize=(12,max_plot*3), constrained_layout=True)
    
    for i in range(max_plot):
        im = axes[i, 0].imshow(EOF2d.isel(comp=i),origin='lower', vmin=mini, vmax=maxi, cmap=my_cmap)
        axes[i, 1].plot(chrono, PCs[:,i])
        axes[i, 0].set_ylabel(f'EOF #{i+1}')
        
        plt.colorbar(im, ax=axes[i, 0], format='%.0e')
        
        # decoration
        axes[i,0].set_facecolor('grey')
        axes[i,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)     
        
        # annotation
        axes[i,0].annotate(f'{var_cum[i]:.02f}', xy=(.95,.85), xycoords='axes fraction', color='w', ha='right', fontsize=24)

    
    
    axes[0,0].set_title('EOF')
    axes[0,1].set_title('PC');
    
    modif_plot.resize(fig, s=14, rx=-25)
    
    if ofile != "":
        plt.savefig(ofile, bbox_inches='tight')
        print(f'Saved as: {ofile}')
        
    if showfig:
        plt.show()
        
    plt.close()        

def plot_save_eof(chrono, max_plot, n_components, EOF2d, PCs, target_field, ntrain, ofile="", showfig=False):
    """ Plot EOF and PC and save it in the given filename.
    In:
        chrono       : pandas.core.series.Series
        max_plot     : int
        n_components : int
        EOF2d        : xarray.core.dataarray.DataArray
        PCs_f        : xarray.core.dataarray.DataArray
        target_field : String
        ntrain       : int
        filename     : String  -- name in which to save the file (don't save if empty)    
    """
    
    if n_components < max_plot:
        max_plot = n_components
    
    # same colorbar
    mini, maxi = np.nanmin(EOF2d.isel(comp=0)), np.nanmax(EOF2d.isel(comp=0))
    mini, maxi = -np.max([np.abs(mini), maxi]), np.max([np.abs(mini), maxi])
    
    my_cmap = plt.get_cmap('bwr')
    
    fig, axes = plt.subplots(ncols=2, nrows=max_plot, figsize=(12,max_plot*3), constrained_layout=True)
    
    for i in range(max_plot):
        im = axes[i, 0].imshow(EOF2d.isel(comp=i),origin='lower', vmin=mini, vmax=maxi, cmap=my_cmap)
        axes[i, 1].plot(chrono[ntrain:], PCs[:,i])
        axes[i, 0].set_ylabel(f'EOF #{i+1}')
        
        plt.colorbar(im, ax=axes[i, 0], format='%.0e')
        
        # decoration
        axes[i,0].set_facecolor('grey')
        axes[i,0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)     
    
    
    axes[0,0].set_title('EOF')
    axes[0,1].set_title('PC');
    
    modif_plot.resize(fig, s=14, rx=-25)
    
    if ofile != "":
        plt.savefig(ofile)
        print(f'Saved as: {ofile}')
        
    if showfig:
        plt.show()
        
    plt.close()

def plot_eof2D(chrono, EOF, PC, target_field, ofile="", showfig=False, max_plot=None):
    """ Plot EOF and PC for numpy.ndarray EOF and PC, and save it in the given filename.
    for new TOPAZ dataset
    
    In:
        chrono       : pandas.core.series.Series
        EOF          : DataArray already format (npca, y, x)
        PC           : numpy.ndarray 
        target_field : String
        filename     : String -- name in which to save the file (don't save if empty)    
        max_plot     : int
    """
    
    n_pca = PC.shape[1]
    if max_plot is None:
        max_plot = n_pca
    
    fig, ax = plt.subplots(ncols=2, nrows=max_plot,sharex='col', figsize=(4*max_plot, 9))
    for i in range(max_plot):
        ax[i, 0].imshow(EOF[i][::-1])
        ax[i, 1].plot(chrono, PC[:,i])
        ax[i, 0].set_ylabel(f'EOF #{i+1}')

    ax[0,0].set_title('EOF')
    ax[0,1].set_title('PC');
    fig.suptitle(target_field)
    plt.tight_layout()
    
    if ofile != "":
#         filename = f'forcing_{field}_EOF.png'
        plt.savefig(f"{ofile}")
        print(f'Saved as {ofile}')        
        
    if showfig:
        plt.show()
        
    plt.close()         
        
        
def plot_save_eof_np(chrono, max_plot, n_components, EOF, PC, xdim, ydim, target_field, ntrain, ofile="", showfig=True):
    """ Plot EOF and PC for numpy.ndarray EOF and PC, and save it in the given filename.
    In:
        chrono       : pandas.core.series.Series
        max_plot     : int
        n_components : int
        EOF          : numpy.ndarray
        PC           : numpy.ndarray 
        xdim         : int
        ydim         : int
        target_field : String
        ntrain       : int
        filename     : String -- name in which to save the file (don't save if empty)    
    """
      
    if n_components < max_plot:
        max_plot = n_components
        
    fig, ax = plt.subplots(ncols=2, nrows=max_plot,sharex='col', figsize=(4*max_plot, 9))
    for i in range(max_plot):
        ax[i, 0].imshow(EOF[i].reshape(xdim, ydim))
        ax[i, 1].plot(chrono, PC[:,i])
        ax[i, 0].set_ylabel(f'EOF #{i+1}')

    ax[0,0].set_title('EOF')
    ax[0,1].set_title('PC');
    fig.suptitle(target_field)
    
    if ofile != "":
        #plt.savefig(filename)
#         filename = f'forcing_{field}_EOF.png'
        plt.savefig(f"{ofile}")
        print(f'Saved as {ofile}')
        
        
    if showfig:
        plt.show()
        
    plt.close()   
        
        
        
        
def plot_save_reconstruction_eof(chrono, idate, Xf, Xf_rec, ntrain, filename):
    """ Plot EOF rec, TOPAZ field' and REC - TOPAZ
    In:
        chrono       : pandas.core.series.Series
        idate        : int     -- date to plot in example
        Xf           : xarray.core.dataarray.DataArray
        Xf_rec       : xarray.core.dataarray.DataArray
        ntrain       : int
        filename     : String  -- name in which to save the file (don't save if empty)    
    """
    
    fig, ax = plt.subplots(ncols=3, figsize=(21,6))
    fig.suptitle(f'Sample{chrono.iloc[idate]}')
    Xf_rec.isel(rdim=idate).plot(ax=ax[0],vmin=0, vmax=4)
    Xf.isel(rdim=idate).plot(ax=ax[1],vmin=0, vmax=4)
    (Xf_rec-Xf[ntrain:]).isel(rdim=idate).plot(ax=ax[2], vmin=-1, vmax=1, cmap=plt.get_cmap('coolwarm'))
    ax[0].set_title('EOF rec.')
    ax[1].set_title('TOPAZ field')
    ax[2].set_title('REC - TOPAZ')
    
    if filename != "":
        plt.savefig(filename)
        print(f'Saved as {filename}')
        
        
    plt.close()