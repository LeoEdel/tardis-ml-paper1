import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pickle as pkl
import netCDF4 as nc4
import matplotlib.colors as colors

# import src.utils.load_config as load_config
from src.utils import reload_config
from src.utils import save_name
from src.utils import modif_plot

import src.data_preparation.load_data as load_data
from src.feature_extraction import extract_pca
from src.feature_extraction import mean_error
from src.data_preparation import mdl_dataset_prep # as target_history

import src.utils.tardisml_utils as tardisml_utils
from src.utils import modif_plot

def _load_SIT_na(rootdir, pca_dir, chrono_e, adjSIC=False):
    '''Load SIT non assimilated (from TOPAZ4b FreeRun or TOPAZ4c)
    to obtain the SIT corrected from ML
    '''
    filename = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4b23_2011_2022_FreeRun.nc")
    if adjSIC:
        filename = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4b23_2011_2022_FreeRun_adjSIC.nc")
    
    sit_na, chronof = load_data.load_nc(filename, 'sithick', X_only=True)
    _, [sit_na] = load_data.trunc_da_to_chrono(chrono_e, chronof, [sit_na])
    return sit_na, chronof

def _load_SIT_err(rootdir, pca_dir, adjSIC=False):
    '''Load SIT assimilated (from TOPAZ4b)
    for comparison
    '''
    filename = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4err23_2011_2022.nc")
    if adjSIC:
        filename = os.path.join(rootdir, pca_dir, f"sithick_TOPAZ4err23_2011_2022_adjSIC.nc")
        
    Xe, chronoe = load_data.load_nc(filename, 'sithick', X_only=True)
    return Xe, chronoe

def load_nc(filename, target_field='sithick', X_only=False):
    """load .netcdf   
    """
    nc = nc4.Dataset(filename, mode='r')
    X = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))[target_field]
    chrono = pd.DataFrame({'date':pd.to_datetime(X['time'].to_numpy())})
    
    if X_only:
        return X, chrono
    
    RMSE = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['rmse']
    bias = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['bias']
    
       
    return X, RMSE, bias, chrono


def draw_corr_mm(Xe_mm, idx_month, rootdir=None, fig_dir=None, savefig=True):
    '''Draw monthly mean correction
    '''
    
    fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(8,6))
    Xe_mm[idx_month].plot(ax=ax, cbar_kwargs={'label':'SIT bias (m)'})
    ax.set_title(f'Month {idx_month+1}');
    modif_plot.resize(fig, s=18)

    if savefig:
        filename = f'Baseline_SITbias23_month{idx_month+1}.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f'Saved as {rootdir}{fig_dir}{filename}')

def save_as_xrdataset(Xe_mm_xr, config, adjSIC=False):
    '''
    '''
#     Xe_mm_xr = xr.DataArray(Xe_mm,
#                         coords={'month':np.arange(12), 'y':ydim, 'x':xdim}, 
#                         dims=["month","y","x"])
    to_save = xr.Dataset(data_vars = {'Xe_mm':Xe_mm_xr},
                         attrs=dict(
                           description='Monthly mean error between TOPAZ4b23 - 4b FreeRun over 2013-2022. Used as baseline correction',
                           author='Leo Edel, Nersc',
                           project='TARDIS')
                        )
    
    
    filename = f'Baseline_monthly_error_2014_2022.nc'
    if adjSIC:
        filename = f'Baseline_monthly_error_2014_2022_adjSIC.nc'
        
    ofile = save_name.check(f"{config.rootdir}{config.pca_dir}", filename)

    to_save.to_netcdf(f"{config.rootdir}{config.pca_dir}/{ofile}")
    print(f'Saved as: {config.rootdir}{config.pca_dir}/{ofile}')


def draw_error_mm(Xe_mm, odir='', showfig=True, savefig=False):
    '''Plot 3x4 subplots with monthly error
    
    
        Parameters:
        -----------
         Xe_mm       : monthly bias between TOPAZ 4b - 4b FreeRun
    '''
    
    fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(28,28), constrained_layout=True)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # diverent colormap [black - white - black]
    colors_under = plt.get_cmap('Greys_r')(np.linspace(0, 1, 256))
    colors_over = plt.get_cmap('Greys')(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_under, colors_over))
    mymap = colors.LinearSegmentedColormap.from_list('mymap', all_colors)
    
    # pcolormesh
    cmap_extend = 'both'
    cmap = plt.get_cmap('coolwarm_r')
    imC = Xe_mm.isel(month=0).plot(ax=axes[0][0], vmin=-2, vmax=2, add_colorbar=False, cmap=cmap) # , center=0, robust=True)
    
    # contour
    levels = np.arange(-2, 2.5, 0.5)
    cl = Xe_mm.isel(month=0).plot.contour(ax=axes[0][0], levels=levels, vmin=-3, vmax=3, add_colorbar=False, cmap=mymap, center=0)
    plt.clabel(cl, fontsize=18, inline=1)
    
    axes[0][0].set_title(f'{months[0]}')
    
    ax_fl = axes.flatten()
    for idx in range(1, Xe_mm.shape[0]):
        ax = ax_fl[idx]
        Xe_mm.isel(month=idx).plot(ax=ax, add_colorbar=False, cmap=cmap, vmin=-2, vmax=2) # center=0, robust=True)
        
        levels = np.arange(-2, 2.5, 0.5)
        cl = Xe_mm.isel(month=idx).plot.contour(ax=ax, levels=levels, vmin=-3, vmax=3, add_colorbar=False, cmap=mymap, center=0)
        ax.clabel(cl, fontsize=18, inline=1)
        
        ax.set_title(f'{months[idx]}')

    for ax in ax_fl:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        
    
    fig.colorbar(imC, ax=axes, label='SIT TOPAZ 4b-4b FreeRun (m)', extend=cmap_extend, shrink=0.5, location="bottom")
    
   # fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')

    modif_plot.resize(fig, s=24, rx=0)

    # plt.tight_layout()
    
    if savefig:
        ofile = f'{odir}Baseline23_2014_2022.png'    
        plt.savefig(f"{ofile}", dpi=124, facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()        

    
     
         

    
     
    
    
     
    
    


if __name__ == "__main__":
    
#     file_config = f'../../config/config_default_2023.yaml'
    file_config = f'../../config/for_paper_3/config_default_2023_adjSIC_full_pm1M_batch32-opti1-v2-sia.yaml'
    rootdir = tardisml_utils.get_rootdir()
    # nosit_dir, withsit_dir, _, forcing_bdir, pca_dir, res_dir, fig_dir, ml_dir, _ = load_config.load_filename(file_config, verbose=True)    
    conf = reload_config.Config(file_config, rootdir=rootdir, verbose=1)
    
    if 'adjSIC' in conf.non_ass:  # check if the SIT with adjustement with SIC > 15% must be used
        adjSIC = True
    else:
        adjSIC = False
        
    # load TP4b and TP4b Free Run
    Xe, chrono_e = _load_SIT_err(rootdir, conf.pca_dir, adjSIC=adjSIC)
    sit_na, chrono_na = _load_SIT_na(rootdir, conf.pca_dir, chrono_e, adjSIC=adjSIC)

    # remove test and validation parts of the data
    ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(Xe.shape[0], 1, 0)
    
    Xe_train = Xe.isel(time=slice(ntest,ntest+nval+ntrain))  # mean on training period
    chrono_train = chrono_e[-ntrain:]

    # get monthly error
    Xe_mm = mean_error.monthly(chrono_train, Xe_train)

    # save monthly error to be applied to other time period
    Xe_mm_xr = xr.DataArray(Xe_mm,
                    coords={'month':np.arange(12), 'y':Xe.y, 'x':Xe.x}, 
                    dims=["month","y","x"])
    
    save_as_xrdataset(Xe_mm_xr, config=conf, adjSIC=adjSIC)
    # + save figure to visually check out the errors
    draw_error_mm(Xe_mm_xr, savefig=True, odir=conf.rootdir+conf.fig_dir)
#     import pdb; pdb.set_trace()
    
    
    # apply correction to mean
    sit_nac = mean_error.apply_mean_correction(chrono_e, sit_na, Xe_mm)

    # compute rmse on test dataset
    RMSE_c = np.sqrt((np.square(sit_nac.isel(time=slice(None,ntest))-sit_na.isel(time=slice(None,ntest)))))
    
    bl = sit_nac.to_dataset()  # baseline dataset
    bl = bl.assign({'rmse':RMSE_c})
    
    # save baseline SIT .nc
    # ofile = f'{rootdir}{conf.pca_dir}/SIT_baseline_2011_2019_FreeRun.nc'
    ofile = f'{rootdir}{conf.pca_dir}/SIT_baseline_2011_2022_FreeRun.nc'
    if adjSIC:
        ofile = f'{rootdir}{conf.pca_dir}/SIT_baseline_2011_2022_FreeRun_adjSIC.nc'
    
    bl.to_netcdf(ofile)
    print(f'SIT baseline saved as: {ofile}')
    
    
    # nice plot ?

    # nice loading function

