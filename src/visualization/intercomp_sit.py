import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import save_name
from src.utils import modif_plot

def spat_reco_save_all(models, days, rootdir, fig_dir, apply=False, cs2smos=None, vmax=4):
    '''Loop draw_spatial_reconstruct() on all dates in days
    
    will call draw_spatial_reconstruct() with argument:
        - bias = False 
        - bias = True
    
    or only draw_spatial_reconstruct(bias=False, apply=True) if apply is True.
    
    '''
    for day in days:
        draw_spatial_reconstruct(models, day, rootdir, fig_dir, savefig=True, apply=apply, cs2smos=cs2smos, vmax=vmax)
        if not apply:  # bias only possible for training period
            draw_spatial_reconstruct(models, day, rootdir, fig_dir, savefig=True, bias=True, cs2smos=cs2smos, vmax=vmax)
        
        


def draw_spatial_reconstruct(models, day, rootdir='', fig_dir='', showfig=False, savefig=False, bias=False, apply=False, cs2smos=None, vmax=4):
    '''Plot spatial reconstruction
    
    Parameters:
    -----------
        models     :     array containing multiple class SITCorrected
        day        :     datetime.datetime object, day to plot
        
        bias       :     bool, if False (default) will plot SIT. if True, will plot SIT (TOPAZ4b - model) for model in models
        apply      :     bool, if True: will hide TOPAZ4b (do not exist) nor Baseline (not done yet)
        
        cs2smos    :     xarray DataArray, sea ice thickness merged product from CS2SMOS
                            will be plotted if not None
        vmax       :     int, threshold maximum for SIT. 4 is default and optimal for 2010-2020, 5/6 may be optimal for 2000-2010
    '''
    
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(28,28), constrained_layout=True)

    # first model in models will be usd to plot default variables
    model_default = models[list(models.keys())[0]] 

    # identify index to plot
    chrono_dt = np.array([dt.date() for dt in model_default.chrono.date])
    idx = np.where(chrono_dt==day.date())[0]
    
    if len(idx) < 1:
        print('Day not found')
        return
    
    # contour plot
    levels = np.arange(1, vmax+1, 1)
    
    # plot TOPAZ4b if exists
    if not apply:
        model_default.sit_a.isel(time=idx).plot(ax=axes[0][0], vmin=0, vmax=vmax, add_colorbar=False)  # TOPAZ4b    
        # contour need an int
        cl = model_default.sit_a.isel(time=idx[0]).plot.contour(ax=axes[0][0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
        
    
    if not bias:  # print SIT for each model
        cmap_extend = 'max'
        imC = model_default.sit_na.isel(time=idx).plot(ax=axes[0][1], vmin=0, vmax=vmax, add_colorbar=False)  # label='TP4 Free Run', c='#1f77b4')
        model_default.sit_bl.isel(time=idx).plot(ax=axes[0][2], vmin=0, vmax=vmax, add_colorbar=False)  # label='baseline', c='#2ca02c')

        cl = model_default.sit_na.isel(time=idx[0]).plot.contour(ax=axes[0][1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
        # axes[0][1].clabel(cl, fontsize=18, inline=1)
        
        cl = model_default.sit_bl.isel(time=idx[0]).plot.contour(ax=axes[0][2], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
        # axes[0][2].clabel(cl, fontsize=18, inline=1)
        
        
        ax_fl = axes.flatten()
        for i, mdl in enumerate(models):
            ax = ax_fl[i+4]
            models[mdl].sit.isel(time=idx).plot(ax=ax, vmin=0, vmax=vmax, add_colorbar=False)
            
            # contour plot
            # levels = np.arange(1, 5, 1)
            # contour need an int
            cl = models[mdl].sit.isel(time=idx[0]).plot.contour(ax=ax, levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
            # ax.clabel(cl, fontsize=18, inline=1)

            ax.set_title(f'{mdl}')
            
        
    elif bias:  # print TOPAZ4B - model
        cmap = plt.get_cmap('coolwarm_r')
        cmap_extend = 'both'
        imC = (model_default.sit_a.isel(time=idx) - model_default.sit_na.isel(time=idx)).plot(ax=axes[0][1], vmin=-2, vmax=2, add_colorbar=False, cmap=cmap)
        (model_default.sit_a.isel(time=idx) - model_default.sit_bl.isel(time=idx)).plot(ax=axes[0][2], vmin=-2, vmax=2, add_colorbar=False, cmap=cmap)

        ax_fl = axes.flatten()
        for i, mdl in enumerate(models):
            ax = ax_fl[i+4]
            (model_default.sit_a.isel(time=idx) - models[mdl].sit.isel(time=idx)).plot(ax=ax, vmin=-2, vmax=2, add_colorbar=False, cmap=cmap)
            ax.set_title(f'{mdl}')
        

    

    axes[0][0].set_title(f'TOPAZ4b')  # ass')
    axes[0][1].set_title(f'TOPAZ4b FreeRun')  # no ass')
    axes[0][2].set_title(f'Baseline')
    
    
    # if we got observation data from CS2SMOS:
    # we plot
    # weekly mean on each day (average including +- 3 days)
    if cs2smos is not None:
        # identify index to plot
        cs_chrono = pd.DataFrame({'date':pd.to_datetime(cs2smos['time'].to_numpy())})
        cs_chrono_dt = np.array([dt.date() for dt in cs_chrono.date])
        cs_idx = np.where(cs_chrono_dt==day.date())[0]
        # plot if day exists (no data in summer)
        if len(cs_idx) == 1:
            cs2smos.isel(time=cs_idx, yc=slice(90,310), xc=slice(100,332)).plot(ax=axes[1][0], vmin=0, vmax=vmax, add_colorbar=False)
            # contour plot
            levels = np.arange(1, 5, 1)
            # contour need an int
            cl = cs2smos.isel(time=cs_idx[0]).plot.contour(ax=axes[1][0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
            axes[1][0].set_title(f'CS2SMOS')
            # axes[1][0].clabel(cl, fontsize=18, inline=1)
        else:
            axes[1][0].set_visible(False)
    else:
        axes[1][0].set_visible(False)
    
    
    if apply:
        axes[0][0].set_visible(False)  # TOPAZ4b assimilation does not exist
        

    for ax in ax_fl:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        
    
    fig.colorbar(imC, ax=axes, label='SIT (m)', extend=cmap_extend, shrink=0.5, location="bottom")
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')

    modif_plot.resize(fig, s=24, rx=0)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
        ofile = f'{rootdir}{fig_dir}SIT_intercomp_{sdate}.png'
        if bias:
            ofile = f'{rootdir}{fig_dir}SIT_intercomp_bias_{sdate}.png'
            
        plt.savefig(f"{ofile}", dpi=124, facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()        
