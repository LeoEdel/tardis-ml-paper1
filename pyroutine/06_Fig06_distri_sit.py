'''Load full time period of TOPAZ ML-adjusted
and plot 
'''

import numpy as np
import pandas as pd
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from matplotlib import dates

from src.data_preparation import load_data
from src.data_preparation import merge_TOPAZ
from src.utils import tardisml_utils
from src.utils import save_name
from src.utils import modif_plot


def compute_distri_time(sit, bins=None, bin_width=None):
    '''
    Returns distribution (pdf) of SIT over time
    Daily or monthly time step
    
    Parameters:
    -----------
    
        sit           :      xarray of dimension (time, y, x)
        bins          :      array of bins. if None, will select values for thickness categories from TOPAZ5
        bin_width     :      float. If given, bins between 0 and 8 meters will be created spaced by bin_width
    '''
    
    if bins is None:
        bins = np.array([0, 0.64, 1.39, 2.47, 4.57, 20]) # from icecat TOPAZ5
    if bin_width is not None:
        bins = np.arange(0,8+bin_width,bin_width)
    
    distri = np.zeros((sit.shape[0], bins.size-1))
    
    ntime = sit.shape[0]
    
    for nt in range(ntime)[:]:
        # remove nan: from xarray.plot.hist()
        no_nan = np.ravel(sit.isel(time=nt).to_numpy())
        no_nan = no_nan[pd.notnull(no_nan)]
        
        hist, _ = np.histogram(no_nan, bins=bins) # , density=True)
        distri[nt] = hist/sum(hist)
#         distri[nt] = hist/sum(hist)
    
    
    dist = xr.DataArray(distri.T,
                        coords={'bins': bins[:-1], 'time': sit.time}, 
                        dims=["bins", "time"])
    
    return dist


def draw_dist_sit_uneven(dist, sit_am, sit_nam, sit_m, sit_blm, tp_model='', savefig=False, showfig=True, rootdir=None, fig_dir=None):
    ''' 
    Draw time series of SIT for:
            - ML reconstruction
            - TOPAZ4b FreeRun
            - baseline
            - TOPAZ 4b (our 'truth')
    
    Parameters:
    -----------
    tp_model     :     string, TOPAZ version used. Ex: 4b, 4bFR, or ML-corrected: LSTM, RF, CNN
    sc           :  SITCorrected instance
    '''
    from matplotlib.colors import LogNorm
    from matplotlib import gridspec  # uneven subplot
    
    # GridSpec tuto:
    # http://127.0.0.1:6868/?token=5ce1e1a313df890cbc23e9240932c9523b747b308604c37a
    fig = plt.figure(figsize=(28,12), constrained_layout=True)
    gs = gridspec.GridSpec(20, 20)
    ax0 = fig.add_subplot(gs[0:8, :])
    ax1 = fig.add_subplot(gs[9:, :])

    mini = 0.001
    maxi = 0.1
       
    # cut at 5m of SIT as very little to see above
    imC = dist.isel(bins=slice(None,58)).plot(ax=ax0, vmax=maxi, add_colorbar=False,
                                              cmap=plt.get_cmap('cubehelix'),
                                              norm=LogNorm(vmin=mini, vmax=maxi))
    
    
    ax0.set_ylabel('SIT (m)')
    ax0.set_xlabel('')
    
    ax0.grid(axis='y',color='white', alpha=0.6, ls =':')
    ax0.tick_params(labelbottom=False, labelleft=True, bottom=True, left=True)     

    
   
    cb = fig.colorbar(imC, ax=ax0, label='Frequency', extend='max', shrink=0.3, location="top")
    cb.ax.minorticks_off()
    

    
    # colors GOOD for daltonism
    sit_am.plot(ax=ax1, label='TOPAZ4-RA', ls='--', c='k', lw=2, zorder=6)
    sit_nam.plot(ax=ax1,label='TOPAZ4-FR', c='#1295B2', zorder=0)
    sit_m.plot(ax=ax1, label=f'TOPAZ4-ML', c='#FB6949', zorder=8)
    sit_blm.plot(ax=ax1, label=f'TOPAZ4-BL', c='#7E8A8A', zorder=5)  # #5cad47

    
    mini = np.nanmin(np.concatenate((sit_nam.data, sit_blm.data, sit_m.data, sit_am.data)))
    maxi = np.nanmax(np.concatenate((sit_nam.data, sit_blm.data, sit_m.data, sit_am.data)))
    ax1.plot([datetime.datetime(2011,1,1)]*2,[mini-0.01,maxi+0.01], ':k')

    # vertical line on the 1st Sept 2007
    ax1.plot([datetime.datetime(2007,9,1)]*2,[mini-0.01,maxi+0.01], 'k', ls='-.')
    
    
    ax1.set_ylabel(f'SIT (m)')
    ax1.set_xlabel(f'')    
    ax1.legend(fontsize=18, loc='lower left', ncol=4)
    ax1.set_ylim([0.5, ax1.get_ylim()[-1]])
    
    ax1.set_xlim([ax0.get_xlim()[0], ax0.get_xlim()[-1]])
    
    
    ax1.xaxis.grid(alpha=0.6)
    ax1.spines[['right', 'top']].set_visible(False)
    fig.align_ylabels([ax0, ax1])

    
    # grey alternative background (winter-summer)
    for yr in range(1992, 2023):
        ax1.axvspan(dates.date2num(datetime.datetime(yr,10,1)),
                        dates.date2num(datetime.datetime(yr+1,4,30)),
                        facecolor='grey', alpha=0.2)
    
    modif_plot.resize(fig, s=24)


    if savefig:
        filename = f'fig6.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}", facecolor='white', dpi=200, bbox_inches='tight')
        print(f'Figure saved as: {rootdir}{fig_dir}{filename}')
        
    if showfig:
        plt.show()
        
    plt.close()


def draw_dist_sit(dist, sit_am, sit_nam, sit_m, sit_blm, tp_model='', savefig=False, showfig=True, rootdir=None, fig_dir=None):
    ''' 
    Draw time series of SIT for:
            - ML reconstruction
            - TOPAZ4b FreeRun
            - baseline
            - TOPAZ 4b (our 'truth')
    
    Parameters:
    -----------
    tp_model     :     string, TOPAZ version used. Ex: 4b, 4bFR, or ML-corrected: LSTM, RF, CNN
    sc           :  SITCorrected instance
    '''
    from matplotlib.colors import LogNorm
    
    fig, axes = plt.subplots(nrows=2, figsize=(14,6), constrained_layout=True)

    mini = 0.001
    maxi = 0.1
       
    # cut at 5m of SIT as very little to see above
    imC = dist.isel(bins=slice(None,58)).plot(ax=axes[0], vmax=maxi, add_colorbar=False,
                                              cmap=plt.get_cmap('cubehelix'),
                                              norm=LogNorm(vmin=mini, vmax=maxi))
    
    
    axes[0].set_ylabel('SIT (m)')
    axes[0].set_xlabel('')
    
    axes[0].grid(axis='y',color='white', alpha=0.6, ls =':')
    axes[0].tick_params(labelbottom=False, labelleft=True, bottom=True, left=True)     

    
   
    cb = fig.colorbar(imC, ax=axes[0], label='Frequency', extend='max', shrink=0.3, location="top")
    cb.ax.minorticks_off()
    

    
    # colors GOOD for daltonism
    sit_am.plot(ax=axes[1], label='TOPAZ', ls='--', c='k', lw=2, zorder=10)
    sit_nam.plot(ax=axes[1],label='TOPAZ Freerun', c='#1295B2', zorder=0)
    sit_m.plot(ax=axes[1], label=f'ML-adjusted', c='#FB6949', zorder=8)
    sit_blm.plot(ax=axes[1], label=f'Baseline', c='#7E8A8A', zorder=5)  # #5cad47

    
    mini = np.nanmin(np.concatenate((sit_nam.data, sit_blm.data, sit_m.data, sit_am.data)))
    maxi = np.nanmax(np.concatenate((sit_nam.data, sit_blm.data, sit_m.data, sit_am.data)))
    axes[1].plot([datetime.datetime(2014,1,1)]*2,[mini-0.01,maxi+0.01], ':k')

    
    
    axes[1].set_ylabel(f'SIT (m)')
    axes[1].set_xlabel(f'')    
    axes[1].legend(fontsize=18, loc='lower center', ncol=4)
    axes[1].set_ylim([0, axes[1].get_ylim()[-1]])
    
    axes[1].set_xlim([axes[0].get_xlim()[0], axes[0].get_xlim()[-1]])
    
    
    axes[1].xaxis.grid(alpha=0.6)
    axes[1].spines[['right', 'top']].set_visible(False)
    
    fig.align_ylabels(axes)

    
    modif_plot.resize(fig, s=24)


    if savefig:
        filename = f'fig6.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}", facecolor='white', dpi=300)
        print(f'Figure saved as: {rootdir}{fig_dir}{filename}')
        
    if showfig:
        plt.show()
        
    plt.close()
    
    
def draw_2dist(dist_1, dist_2, name_1='TOPAZ4-ML', name_2='TOPAZ4-FR',
               savefig=False, showfig=True, rootdir=None, fig_dir=None):
    ''' 
    Draw 2 time series of SIT distribution:
            - ML reconstruction
            - TOPAZ4b FreeRun
    
    Parameters:
    -----------
        dist_1           :     output from compute_distri_time(). Distribution of SIT over times, daily histogram
        dist_2           :     output from compute_distri_time().
    '''
    from matplotlib.colors import LogNorm
    
    fig, axes = plt.subplots(nrows=2, figsize=(28,12), constrained_layout=True)

    mini = 0.001
    maxi = 0.1
       
    # cut at 5m of SIT as very little to see above
    imC = dist_1.isel(bins=slice(None,58)).plot(ax=axes[0], vmax=maxi, add_colorbar=False,
                                              cmap=plt.get_cmap('cubehelix'),
                                              norm=LogNorm(vmin=mini, vmax=maxi))
    
    
    axes[0].set_ylabel(f'SIT {name_1}(m)')
    axes[0].set_xlabel('')
    
    axes[0].grid(axis='y',color='white', alpha=0.6, ls =':')
    axes[0].tick_params(labelbottom=False, labelleft=True, bottom=True, left=True)     

    # -------------------------------------------------
    imC = dist_2.isel(bins=slice(None,58)).plot(ax=axes[1], vmax=maxi, add_colorbar=False,
                                              cmap=plt.get_cmap('cubehelix'),
                                              norm=LogNorm(vmin=mini, vmax=maxi))
    
    
    axes[1].set_ylabel(f'SIT {name_2}(m)')
    axes[1].set_xlabel('')
    
    axes[1].grid(axis='y',color='white', alpha=0.6, ls =':')
    axes[1].tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)     

    
   
    cb = fig.colorbar(imC, ax=axes[0], label='Frequency', extend='max', shrink=0.3, location="top")
    cb.ax.minorticks_off()
    
  
    
    modif_plot.resize(fig, s=24)


    if savefig:
        filename = f'SITdistri_{name_1}_{name_2}.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}", facecolor='white', dpi=150)
        print(f'Figure saved as: {rootdir}{fig_dir}{filename}')
        
    if showfig:
        plt.show()
        
    plt.close()

    
# ---------------------------------------------------
# ---------------------------------------------------




rootdir = tardisml_utils.get_rootdir()
fig_dir = 'Leo/results/bin_fig/'

# LOAD THICKNESS
sit_ml, chrono_dt, sit_fr, sit_bl, sit_mlm, sit_frm, sit_blm = merge_TOPAZ.load(return_na=True, return_bl=True, return_mean=True)
# this function is too slow and not updated with the latest results


# sit_ml, chrono_dt = merge_TOPAZ.load_nc()
# sit_mlm = sit_ml.mean(('x','y'))

# sit_fr = merge_TOPAZ.load_freerun()
# sit_frm = sit_fr.mean(('x','y'))

# sit_bl, _ = load_data.load_nc('/scratch/project_465000269/edelleo1/Leo/TP4_ML/sit_bl_2000_2011_adjSIC.nc', 'sithick', X_only=True)
# sit_blm = sit_bl.mean(('x','y'))

sit_a, sit_am = merge_TOPAZ.load_ass(return_mean=True, adjSIC=True)  # False

# ---------------------------------------------------
#         DATA TREATMENT: compute distribution
# ---------------------------------------------------

# Cap negative value at 0 m (while keeping nan)
sit0 = (sit_ml.where((0<sit_ml), -999)).where(np.isfinite(sit_ml))

# histogram
bin_width = 0.1
bins = np.arange(0,8+bin_width,bin_width)

sit_dist0 = compute_distri_time(sit0, bin_width=0.1)
name_ml = 'TOPAZ4-ML'

# ---------------------------------------------------
# Cap negative value at 0 m (while keeping nan)
sit0_fr = (sit_fr.where((0<sit_fr), -999)).where(np.isfinite(sit_fr))

sit_fr_dist0 = compute_distri_time(sit0_fr, bin_width=0.1)
name_fr = 'TOPAZ4-FR'

# ---------------------------------------------------
# Cap negative value at 0 m (while keeping nan)
sit0_bl = (sit_bl.where((0<sit_bl), -999)).where(np.isfinite(sit_bl))

sit_bl_dist0 = compute_distri_time(sit0_bl, bin_width=0.1)
name_bl = 'TOPAZ4-BL'


# ---------------------------------------------------
# Plot
# draw_dist_sit(sit_dist0, sit_am, sit_nam, sit_m, sit_blm)

draw_dist_sit_uneven(sit_dist0, sit_am=sit_am, sit_nam=sit_frm, sit_m=sit_mlm, sit_blm=sit_blm,
              savefig=True, showfig=False, rootdir=rootdir, fig_dir=fig_dir)


draw_2dist(dist_1=sit_dist0, dist_2=sit_fr_dist0, name_1=name_ml, name_2=name_fr,
          savefig=True, showfig=False, rootdir=rootdir, fig_dir=fig_dir)

draw_2dist(dist_1=sit_dist0, dist_2=sit_bl_dist0, name_1=name_ml, name_2=name_bl,
          savefig=True, showfig=False, rootdir=rootdir, fig_dir=fig_dir)

