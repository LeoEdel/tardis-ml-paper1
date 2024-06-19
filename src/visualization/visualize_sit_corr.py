#################################################################
#
# Functions ploting graph about errors
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec  # uneven subplot
from matplotlib import dates
from datetime import datetime

from src.utils import save_name
from src.utils import modif_plot



def draw_error_bias(mean, std, bins, n_pixels, rootdir=None, fig_dir=None, savefig=False, showfig=True):
    '''
    '''
    
    fig = plt.figure(constrained_layout=True, figsize=(16,9))

    gs = gridspec.GridSpec(20, 20)
    ax0 = fig.add_subplot(gs[0:15, :])
    ax1 = fig.add_subplot(gs[16:, :])

    ax0.plot(bins[1:], mean[1:], lw=2)
    ax0.fill_between(bins[1:], mean[1:]-std[1:], mean[1:]+std[1:], color='grey', alpha=0.4)
    ax0.axhline(0, c='grey', ls='--', zorder=-5)
#     ax0.axhline(0.5, c='gray', ls=':', zorder=-5)
#     ax0.axhline(-0.5, c='gray', ls=':', zorder=-5)

    ax0.plot(bins[1:], bins[1:] * 20 /100, c='gray', ls=':', zorder=-10)
    ax0.plot(bins[1:], -bins[1:] * 20 /100, c='gray', ls=':', zorder=-10)
    
    

    ax0.set_ylabel('SIT bias difference:\n[TOPAZ4-ML - TOPAZ4-RA] (m)')
#     ax0.set_ylabel('SIT bias difference:\n[ML - ass]/SIT_freerun (%)')
    # ax0.set_xlabel('Sea ice thickness Freerun (m)')

    ax0.set_ylim([-2,2]) # absolute values
#     ax0.set_ylim([-50,50]) # in %

    ax0.set_xlim([-.2,9])
    ax0.xaxis.grid(zorder=-10, ls=':')


    ax1.plot(bins[1:], n_pixels[1:], 'r*')
    
    ind_less20 = np.where(n_pixels<50)[0]
    ax1.plot(bins[ind_less20], n_pixels[ind_less20], color='grey', marker='*', lw=0)
    

    ax1.set_yscale('log')
    ax1.xaxis.grid(zorder=-10, ls=':')
    ax1.yaxis.grid(ls=':')
    ax1.set_ylabel('Number\nof pixels')
    ax1.set_xlabel('SIT TOPAZ4-FR (m)');
    ax1.set_xlim([-.2,9])

    ax0.set_title('Error of SIT bias on test period')

    fig.align_ylabels([ax0, ax1])

    modif_plot.resize(fig, s=18)


    if savefig:
        filename = f'sit_bias_diff_01.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f'Saved as: {rootdir}{fig_dir}{filename}')
    
    if showfig:
        plt.show()
    
    plt.close()
    


def draw_improvement(model, rootdir=None, fig_dir=None, savefig=False, showfig=True):
    '''Draw skill score: % of improvement
    '''

    fig, ax = plt.subplots(ncols=1, figsize=(14,6))
    
    model.ss_clim_t.plot(ax=ax)
    ax.set_ylabel('Skill score')
    ax.xaxis.grid(alpha=0.6)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('')
    
    ax.set_ylim([0.0, 1.1])


    modif_plot.resize(fig, s=18)
    
    if savefig:
        filename = f'sithick_improvement_time_01.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f'Saved as: {rootdir}{fig_dir}{filename}')
    
    if showfig:
        plt.show()
    
    plt.close()
    
    
    
    fig, ax = plt.subplots(ncols=1, figsize=(10,8))

    levels = np.array([0, 0.5])

# ax.set_facecolor('grey')
    model.ss_clim_xy.plot(ax=ax, vmin=0, vmax=1, center=0) # cmap=plt.get_cmap('bwr'))
    cl = model.ss_clim_xy.plot.contour(ax=ax, levels=levels, add_colorbar=False, 
                                 cmap=plt.get_cmap('Greys'))
    
    
#     for ax in axes.flatten():
#         ax.set_facecolor('grey')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    ax.set_title('Skill score')
    
    modif_plot.resize(fig, s=18)
    
    if savefig:
        filename = f'sithick_improvement_xy_01.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f'Saved as: {rootdir}{fig_dir}{filename}')
    
    if showfig:
        plt.show()
    
    plt.close()
    
    
    
    
    # monthly average of skill score

    
    
#     landmask = landmask.where(~np.isnan(nosit),1)
  
#     # monthly average
#     skm = sk.resample(time='1M').mean(dim='time')

#     # cap negative value to 0
#     skm_ = skm.where(skm>=0, 0)

#     # put nan when there is not sea ice
#     skm_ = skm.where(landmask!=1)

#     # invalid skill score (-inf)
#     skm_ = skm.where(skm!=-np.Inf, -1)

#     skm_.where(skm_>0.001).mean(dim=('x','y')).plot()
#     plt.ylabel('improvement compared to no assimilation');
    
    
    
    

def draw_rmse(model, rootdir=None, fig_dir=None, savefig=False, showfig=True):
    """ Draw 3 subplots: RMSE of :
            - ML reconstructed (the result from our ML)
            - error reconstruction: due to EOF/PCA transformation (lower bound = best we can obtain)
            - baseline reconstruction (higher bound = worst that we should beat)
    
    Parameters:
    -----------
        model     : class SITCorrected  
    """
    
    fig, axes = plt.subplots(ncols=3, figsize=(8*3,8), constrained_layout=True)
    
    #fig.suptitle('RMSE of')
    model.RMSE_corr.plot(vmax=2, ax=axes[0], cbar_kwargs={'orientation':'horizontal', 'ticks':[0,0.5,1,1.5,2],
                                                                       'label':'RMSE (m)', 'aspect':25})
    model.RMSE_rece.plot(vmax=2, ax=axes[1], add_colorbar=False)  #  cbar_kwargs={'orientation':'horizontal','ticks':[0,0.5,1,1.5,2],
#                                                                        'label':'RMSE (m)', 'aspect':25})
    model.RMSE_bl.plot(vmax=2, ax=axes[2], add_colorbar=False)  #cbar_kwargs={'orientation':'horizontal','ticks':[0,0.5,1,1.5,2],
#                                                                        'label':'RMSE (m)', 'aspect':25})

    axes[0].set_title('TOPAZ4-ML error');  # ML-adjusted error
    axes[1].set_title('EOF error')  # EOF error # 'error reconstruction (lower bound)'
    axes[2].set_title('TOPAZ4-BL error')  #  Baseline (higher bound)
    
    
    
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)     
        ax.set_facecolor('grey')


    modif_plot.resize(fig, s=24)
    
    if savefig:
        filename = f'sithick_rmse_ML_HL_bounds.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}", dpi=400, facecolor='white')
        print(f'Saved as: {rootdir}{fig_dir}{filename}')
    
    if showfig:
        plt.show()
    
    plt.close()

def draw_mean_remaining_bias(model, rootdir=None, fig_dir=None, savefig=True, showfig=False):
    '''Plot TOPAZ4b - ML
    average over test period
    '''
    import matplotlib.colors as colors

     # diverent colormap [black - white - black]
    colors_under = plt.get_cmap('Greys_r')(np.linspace(0, 1, 256))
    colors_over = plt.get_cmap('Greys')(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_under, colors_over))
    mymap = colors.LinearSegmentedColormap.from_list('mymap', all_colors)
    
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,8), constrained_layout=True)

    mean_bias = (model.Xc.isel(time=slice(None, model.ntest))-model.Xe.isel(time=slice(None, model.ntest))).mean(dim='time')
    imC = mean_bias.plot(ax=ax, vmin=-1, vmax=1, cmap=plt.get_cmap('coolwarm_r'), add_colorbar=False)
    
    levels = np.arange(-1, 1.5, 0.5)
    cl = mean_bias.plot.contour(ax=ax, levels=levels, vmin=-3, vmax=3, add_colorbar=False, cmap=mymap, center=0)
        
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    fig.colorbar(imC, ax=ax, label='SIT bias (m)', extend='both', shrink=0.9, location="bottom")
#     fig.colorbar(imC, ax=ax, label=f'SIT TOPAZ 4b-{model.name} (m)', extend='both', shrink=0.9, location="bottom")
    
    
    modif_plot.resize(fig, s=24)
    
    if savefig:
        filename = f'sithick_remaining_error_{model.name}_TOPAZ4b.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}", dpi=300)
        print(f'Saved as: {rootdir}{fig_dir}{filename}')
        
    if showfig:
        plt.show()
    
    plt.close()    
    
    
    
def draw_bias_diff(model, day, target_field='sithick', rootdir=None, fig_dir=None, savefig=True, showfig=False):
    '''Draw 3 subplots:
           - bias between SIT from ML and from TOPAZ4b FreeRun
           - bias between SIT from TOPAZ4b and TOPAZ4b FreeRun
           - bias between SIT from ML and from TOPAZ4b
           
    Parameters:
    -----------
        model     : class SITCorrected
           
    '''
    
        # identify index to plot
    chrono_dt = np.array([dt.date() for dt in model.chronoe.date])
    # chrono_dt = np.array([dt.date() for dt in chrono])
    idate = np.where(chrono_dt==day.date())[0]
    
    if len(idate) < 1:
        print('Day not found')
        return
    
    fig, axes = plt.subplots(ncols=3, figsize=(21,6))
    fig.suptitle(f'{model.chronoe.iloc[idate[0]].date.strftime("%Y %m %d")} - {target_field}')
    
    model.Xc.isel(time=idate).plot(ax=axes[0],vmin=-2, vmax=2)
    axes[0].set_title(f'bias estimated by ML');
    
    model.Xe.isel(time=idate).plot(ax=axes[1],vmin=-2, vmax=2)
    axes[1].set_title(f'bias TOPAZ ass');
    
#     import pdb; pdb.set_trace()
    
    (model.Xc.isel(time=idate[0])-model.Xe.isel(time=idate[0])).plot(ax=axes[2], vmin=-1, vmax=1, cmap=plt.get_cmap('coolwarm'));
    axes[2].set_title('ML - TOPAZ');
    
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    modif_plot.resize(fig, s=18)
    
    if savefig:
        filename = f'{target_field}_error_ML_TOPAZ_diff.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f'Saved as: {rootdir}{fig_dir}{filename}')
        
    if showfig:
        plt.show()
    
    plt.close()

def draw_rmse_bias(model, rootdir=None, fig_dir=None, savefig=True, showfig=False):
    '''Draw time series of RMSE and bias for the following data:
                - TOPAZ 4b Free Run
                - baseline
                - ML reconstructed
                
    Parameters:
    -----------
        model     : class SITCorrected
                
    '''

    fig, ax = plt.subplots(ncols=1, nrows=2,figsize=(16,12))

   # chrono = model.chronoe

    ax[0].plot(model.chronoe, model.RMSE_na_t, label='TOPAZ4b FR')
    ax[0].plot(model.chronoe, model.RMSE_bl_t, '#5cad47', label='baseline') # alpha=0.5
    ax[0].plot(model.chrono, model.RMSE_ml_t, label=f'{model.name}')
    ax[0].plot([model.chronoe.iloc[model.ntest]]*2,[0.0, 0.98], ':k')  # label='train limit'
    ax[0].legend()
    ax[0].set_xticklabels('')
    ax[0].set_ylabel(f'RMSE (m)')
    ax[0].xaxis.grid()

    ax[1].plot([model.chronoe.iloc[0], model.chronoe.iloc[-1]], [0]*2, '--', c='grey', alpha=.7)
    ax[1].plot(model.chronoe, model.bias_na, label='TOPAZ4c')
    ax[1].plot(model.chronoe, model.bias_bl, '#5cad47', label='baseline') # , alpha=0.5)
    ax[1].plot(model.chrono, model.bias_ml, label=f'{model.name}')
    ax[1].plot([model.chronoe.iloc[model.ntest]]*2,[-0.28, 0.43], ':k')

    ax[1].set_ylim([-.3,.45])
    ax[1].set_ylabel(f'Bias (m)')
    ax[1].xaxis.grid()
    
    plt.xticks(rotation= -25)

    modif_plot.resize(fig, s=18)
    fig.tight_layout()

    if savefig:
        filename = f'ML_RMSE_bias_01.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f"{rootdir}{fig_dir}{filename}")
    
    if showfig:
        plt.show()
        
    plt.close()

    
    
def draw_rmse_bias_corr(model, rootdir=None, fig_dir=None, savefig=True, showfig=False):
    '''Draw time series of RMSE and bias for the following data:
                - TOPAZ 4b Free Run
                - baseline
                - ML reconstructed
                
    Parameters:
    -----------
        model     : class SITCorrected
                
    '''

    fig, ax = plt.subplots(ncols=1, nrows=3,figsize=(16,12))

   # chrono = model.chronoe

    ax[0].plot(model.chronoe, model.RMSE_na_t, label='TOPAZ4b FR')
    ax[0].plot(model.chronoe, model.RMSE_bl_t, '#5cad47', label='baseline') # alpha=0.5
    ax[0].plot(model.chrono, model.RMSE_ml_t, label=f'{model.name}')
    ax[0].plot([model.chronoe.iloc[model.ntest]]*2,[0.0, 0.98], ':k')  # label='train limit'
    ax[0].legend()
    ax[0].set_xticklabels('')
    ax[0].set_ylabel(f'RMSE (m)')
    ax[0].xaxis.grid()

    ax[1].plot([model.chronoe.iloc[0], model.chronoe.iloc[-1]], [0]*2, '--', c='grey', alpha=.7)
    ax[1].plot(model.chronoe, model.bias_na, label='TOPAZ4c')
    ax[1].plot(model.chronoe, model.bias_bl, '#5cad47', label='baseline') # , alpha=0.5)
    ax[1].plot(model.chrono, model.bias_ml, label=f'{model.name}')
    
    
    min_bias = np.nanmin([np.nanmin(model.bias_na),
                          np.nanmin(model.bias_bl), 
                          np.nanmin(model.bias_ml)])
    max_bias = np.nanmax([np.nanmax(model.bias_na),
                           np.nanmax(model.bias_bl), 
                           np.nanmax(model.bias_ml)])
    
    ax[1].plot([model.chronoe.iloc[model.ntest]]*2,[min_bias+0.02, max_bias-0.02], ':k')
    ax[1].set_ylim([min_bias, max_bias])
    ax[1].set_ylabel(f'Bias (m)')
    ax[1].xaxis.grid()
    ax[1].set_xticklabels('')
    
    ax[2].plot(model.chronoe, model.corr_na_t)
    ax[2].plot(model.chronoe, model.corr_bl_t, '#5cad47')
    ax[2].plot(model.chrono, model.corr_ml_t)
    
    min_corr = np.nanmin([np.nanmin(model.corr_na_t),
                          np.nanmin(model.corr_bl_t), 
                          np.nanmin(model.corr_ml_t)])
#     max_corr = np.nanmax([np.nanmax(model.corr_na_t),
#                           np.nanmax(model.corr_bl_t), 
#                           np.nanmax(model.corr_ml_t)])
    ax[2].plot([model.chronoe.iloc[model.ntest]]*2,[min_corr+0.01, .99], ':k')
    
    ax[2].set_ylim([min_corr, 1])
    ax[2].set_ylabel(f'Correlation')
    ax[2].xaxis.grid()
    
    
    plt.xticks(rotation= -25)

    modif_plot.resize(fig, s=18)
    fig.tight_layout()

    if savefig:
        filename = f'ML_RMSE_bias_corr_01.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f"{rootdir}{fig_dir}{filename}")
    
    if showfig:
        plt.show()
        
    plt.close()

def draw_rmse_bias_corr_poster(model, rootdir=None, fig_dir=None, savefig=True, showfig=False):
    '''Draw time series of RMSE and bias for the following data:
                - TOPAZ 4b Free Run
                - baseline
                - ML reconstructed
                
    Parameters:
    -----------
        model     : class SITCorrected
                
    '''

    fig, ax = plt.subplots(ncols=1, nrows=3,figsize=(16,12))

   # chrono = model.chronoe

    ax[0].plot(model.chronoe, model.RMSE_na_t, label='TOPAZ4b FR')
    ax[0].plot(model.chronoe, model.RMSE_bl_t, '#5cad47', label='baseline') # alpha=0.5
    ax[0].plot(model.chrono, model.RMSE_ml_t, label=f'LSTM')
    ax[0].plot([model.chronoe.iloc[model.ntest]]*2,[0.0, 0.98], ':k')  # label='train limit'
    ax[0].legend(fontsize=24, ncol=3, loc='upper center')
    ax[0].set_xticklabels('')
    ax[0].set_ylabel(f'RMSE (m)')
    ax[0].xaxis.grid()

    ax[1].plot([model.chronoe.iloc[0], model.chronoe.iloc[-1]], [0]*2, '--', c='grey', alpha=.7)
    ax[1].plot(model.chronoe, model.bias_na, label='TOPAZ4c')
    ax[1].plot(model.chronoe, model.bias_bl, '#5cad47', label='baseline') # , alpha=0.5)
    ax[1].plot(model.chrono, model.bias_ml, label=f'{model.name}')
    
    
    min_bias = np.nanmin([np.nanmin(model.bias_na),
                          np.nanmin(model.bias_bl), 
                          np.nanmin(model.bias_ml)])
    max_bias = np.nanmax([np.nanmax(model.bias_na),
                           np.nanmax(model.bias_bl), 
                           np.nanmax(model.bias_ml)])
    
    ax[1].plot([model.chronoe.iloc[model.ntest]]*2,[min_bias+0.02, max_bias-0.02], ':k')
    ax[1].set_ylim([min_bias, max_bias])
    ax[1].set_ylabel(f'Bias (m)')
    ax[1].xaxis.grid()
    ax[1].set_xticklabels('')
    
    ax[2].plot(model.chronoe, model.corr_na_t)
    ax[2].plot(model.chronoe, model.corr_bl_t, '#5cad47')
    ax[2].plot(model.chrono, model.corr_ml_t)
    
    min_corr = np.nanmin([np.nanmin(model.corr_na_t),
                          np.nanmin(model.corr_bl_t), 
                          np.nanmin(model.corr_ml_t)])
#     max_corr = np.nanmax([np.nanmax(model.corr_na_t),
#                           np.nanmax(model.corr_bl_t), 
#                           np.nanmax(model.corr_ml_t)])
    ax[2].plot([model.chronoe.iloc[model.ntest]]*2,[min_corr+0.01, .99], ':k')
    
    ax[2].set_ylim([min_corr, 1])
    ax[2].set_ylabel(f'Correlation')
    ax[2].xaxis.grid()
    
    
    plt.xticks(rotation= -25)

    modif_plot.resize(fig, s=32)
    fig.tight_layout()

    if savefig:
        filename = f'ML_RMSE_bias_corr_poster_01.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}")
        print(f"{rootdir}{fig_dir}{filename}")
    
    if showfig:
        plt.show()
        
    plt.close()

    
# plot sea ice thickness
# idealement ajouter les donnees in situ
def draw_sit(sc, rootdir=None, fig_dir=None, savefig=False, showfig=False):
    '''
    Draw time series of SIT for:
            - ML reconstruction
            - TOPAZ4b FreeRun
            - baseline
            - TOPAZ 4b (our 'truth')
    
    Parameters:
    -----------
    sc           :  SITCorrected instance
    '''
    
    fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(16,6), constrained_layout=True)

    sc.sit_am.plot(label='TOPAZ4-RA', ls='--', c='k', lw=2, zorder=10)
    sc.sit_nam.plot(label='TOPAZ4-FR', c='#1295B2', zorder=0)
    sc.sit_m.plot(label=f'TOPAZ4-ML', c='#FB6949', zorder=8)
    sc.sit_blm.plot(label=f'TOPAZ4-BL', c='#7E8A8A', zorder=5)

    
    mini = np.nanmin(np.concatenate((sc.sit_nam.data, sc.sit_blm.data, sc.sit_m.data, sc.sit_am.data)))
    maxi = np.nanmax(np.concatenate((sc.sit_nam.data, sc.sit_blm.data, sc.sit_m.data, sc.sit_am.data)))
#     ax.plot([sc.chronoe.iloc[sc.ntest]]*2,[mini-0.01,maxi+0.01], ':k')
    ax.plot([sc.chronoe.iloc[sc.ntest]]*2,[0.3, ax.get_ylim()[-1]], ':k')


    # grey alternative background (winter-summer)
    for yr in range(2010, 2023):
        ax.axvspan(dates.date2num(datetime(yr,10,1)),
                        dates.date2num(datetime(yr+1,4,30)),
                        facecolor='grey', alpha=0.2)
    
    ax.set_ylabel(f'SIT (m)')
    ax.set_xlabel(f'')    
    ax.legend(fontsize=18, loc='lower center', ncol=4)
    ax.set_ylim([0, ax.get_ylim()[-1]])
    ax.set_xlim([datetime(2011,1,1).date(), datetime(2022,12,31).date()])
    

    ax.xaxis.grid(alpha=0.6)
    ax.spines[['right', 'top']].set_visible(False)

    
    modif_plot.resize(fig, s=24)


    if savefig:
        filename = f'sithick_ts_reconstruct.png'
        plt.savefig(f"{rootdir}{fig_dir}{filename}", facecolor='white', dpi=300)
        print(f'Figure saved as: {rootdir}{fig_dir}{filename}')
        
    if showfig:
        plt.show()
        
    plt.close()
    
    
def spat_reco_save_all(sc, days, rootdir, fig_dir):
    '''Loop spatial_reconstruct() on all dates: 2014-2017
    '''
    
#     for idate in range(0, 365):
    for day in days:
        spatial_reconstruct(sc, day, rootdir, fig_dir, savefig=True)
        
#     print('Finish')
        
def spatial_reconstruct(sc, day, rootdir='', fig_dir='', showfig=False, savefig=False):
    '''Plot spatial reconstruction
    
     old: Xf, Xf_mc, Xc, Xa, chrono
    
    Parameters:
    -----------
        sc : class SITCorrected
        Xf: SIT forecast
        Xf_mc: monthly mean simplest correction
        Xc    : SIT error reconstructed by ML
        Xa    : SIT from TOPAZ with CS2 assimilated
        chrono: time series
        idate : i-th time to plot
    '''
    
    fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(28,12))

    # identify index to plot
    chrono_dt = np.array([dt.date() for dt in sc.chronoe.date])
    # chrono_dt = np.array([dt.date() for dt in chrono])
    idx = np.where(chrono_dt==day.date())[0]
    
    if len(idx) < 1:
        print('Day not found')
        return

    # h ice  
    sc.sit_a.isel(time=idx).plot(ax=ax[0][0], vmax=4)
    sc.sit_na.isel(time=idx).plot(ax=ax[0][1], vmax=4)
    
    sc.sit_bl.isel(time=idx).plot(ax=ax[0][2], vmin=0, vmax=4)
    sc.sit.isel(time=idx).plot(ax=ax[0][3], vmin=0, vmax=4)    
    
    # h ice differences
    (sc.sit_a.isel(time=idx) - sc.sit_na.isel(time=idx)).plot(ax=ax[1][1], vmin=-2, vmax=2, cmap=plt.get_cmap('coolwarm'))
    (sc.sit_a.isel(time=idx) - sc.sit_bl.isel(time=idx)).plot(ax=ax[1][2], vmin=-2, vmax=2, cmap=plt.get_cmap('coolwarm'))   
    (sc.sit_a.isel(time=idx) - sc.sit.isel(time=idx)).plot(ax=ax[1][3], vmin=-2, vmax=2, cmap=plt.get_cmap('coolwarm'))
    

    ax[0][0].set_title(f'TOPAZ4b')  # ass')
    ax[0][1].set_title(f'TOPAZ4b FreeRun')  # no ass')
    ax[0][2].set_title(f'Baseline')
    ax[0][3].set_title(f'{sc.name}')
    
    ax[1][0].set_visible(False)
    ax[1][1].set_title(f'TOPAZ 4b - 4b FR')  #ass - no ass')
    ax[1][2].set_title(f'TOPAZ 4b - baseline')
    ax[1][3].set_title(f'TOPAZ 4b - {sc.name}')

    for axx in ax.flatten():
        axx.set_xlabel('')
        axx.set_ylabel('')
        
    
    fig.suptitle(f'{sc.chronoe.iloc[idx[0]].date.strftime("%Y %m %d")}')

    modif_plot.resize(fig, s=18, rx=0)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
        ofile = f'{rootdir}{fig_dir}SIT_RF_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()        
    
def spatial_reconstruct_v0(Xf, Xf_mc, Xc, Xa, chrono, idate, rootdir='', fig_dir='', showfig=False, savefig=False):
    '''Plot spatial reconstruction
    
    In:
        Xf: SIT forecast
        Xf_mc: monthly mean simplest correction
        Xc    : SIT error reconstructed by ML
        Xa    : SIT from TOPAZ with CS2 assimilated
        chrono: time series
        idate : i-th time to plot
    '''
    
    fig, ax = plt.subplots(ncols=3, nrows = 2, figsize=(21,12))

    Xf.isel(time=idate).plot(ax=ax[0][0], vmax=4)
    Xf_mc.isel(time=idate).plot(ax=ax[0][1], vmin=0, vmax=4)
    (Xa.isel(time=idate) - Xf_mc.isel(time=idate)).plot(ax=ax[0][2], vmin=-2, vmax=2, cmap=plt.get_cmap('coolwarm'))
    
    Xa.isel(time=idate).plot(ax=ax[1][0], vmax=4)
    (Xf.isel(time=idate) + Xc.isel(time=idate)).plot(ax=ax[1][1], vmin=0, vmax=4)
    (Xa.isel(time=idate) - (Xf.isel(time=idate) + Xc.isel(time=idate))).plot(ax=ax[1][2], vmin=-2, vmax=2, cmap=plt.get_cmap('coolwarm'))
    

    ax[0][0].set_title(f'TOPAZ no ass')
    ax[0][1].set_title(f'Monthly mean corrected')
    ax[0][2].set_title(f'TOPAZb - monthly mean corr')
    
    ax[1][0].set_title(f'TOPAZ ass')
    ax[1][1].set_title(f'ML corrected')
    ax[1][2].set_title(f'TOPAZb - ML corr')

    fig.suptitle(f'{chrono[idate]}')

    if showfig:
        plt.show()
    
    if savefig:
        ofile = f'{rootdir}{fig_dir}reconstruct_t{idate:04}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    plt.close()
    
    

def get_hist_without_0(X, bins):
    '''return values to plot histogram for a X on bins
    while removing 0 values
    
    In:
        X        : 2d Array
        bins     : list of bin
    '''
    
    y = X.copy()
    y[y==0] = np.nan
    vals,_ = np.histogram(np.clip(y,bins[0],bins[-1]),bins=bins, density=True)
    return vals
    
    
    
def hist(Xf, Xf_mc, Xc, Xa, chrono, idate, odir='', showfig=False, savefig=False):
    '''Plot histogramme of error
    '''
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(21,12))
    
    
#     ax.hist(Xf.isel(time=idate))  # [0][0], vmax=4)
    
#     bins = [0,1,2,3,4,5,6,7,8,9]
    dist_btw_bins = .05
    bins = np.arange(0, 8, dist_btw_bins)
#     print(bins)
    
    space = 0.3
    nbin = len(bins)
    width = .5  # (1 - space) #  / (len(inst))
    
#     vals,_ = np.histogram(np.clip((Xa.isel(time=idate)-Xf.isel(time=idate)),bins[0],bins[-1]),bins=bins)  # calcul histogramme
#     pos = [j - (1 - space) / 2. + 1 * width for j in range(1,nbin+1)]  # calcul position des barres
#     ax.bar(pos[:-1], vals, width=width)  # , color = clrs[i], label=lbl[i]) 
#     ax.bar(bins[:-1], vals, width=width)  # , color = clrs[i], label=lbl[i]) 
    vals = get_hist_without_0((Xa.isel(time=idate).values-Xf.isel(time=idate).values), bins)
    ax.plot(bins[:-1], vals, label='TOPAZ no ass') # , width=width)  # , color = clrs[i], label=lbl[i]) 
    
    vals = get_hist_without_0(Xa.isel(time=idate).values, bins)
#     vals,_ = np.histogram(np.clip(Xa.isel(time=idate),bins[0],bins[-1]),bins=bins)
    ax.plot(bins[:-1], vals, '-', label='TOPAZ ass')
    
    vals = get_hist_without_0((Xa.isel(time=idate).values-Xf_mc.isel(time=idate).values), bins)
    ax.plot(bins[:-1], vals, 'r', alpha=.7, label='Monthly mean corrected')
    
    vals = get_hist_without_0(Xc.isel(time=idate).values, bins)
    ax.plot(bins[:-1], vals, 'k', label='ML corrected')
    
    ax.set_ylabel('Density')
    ax.set_xlabel('hice error (m)')
    
    
    plt.legend()
#     ax.set_ylim([0, 20000])
    ax.set_xlim([0, 5])
    
#     ax.set_ya
#     plt.yscale('log')
    
    
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return

        filename = f'hist_error.png'
        plt.savefig(f"{odir}{filename}")
        print(f'Saved as: {odir}{filename}')


    if showfig:
        plt.show()


    plt.close()

def hist_mean(Xf, Xf_mc, Xc, Xa, chrono, odir='', showfig=False, savefig=False):
    '''Plot histogramme of error
    '''
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(21,12))
    
#     bins = [0,1,2,3,4,5,6,7,8,9]
    dist_btw_bins = .05
    bins = np.arange(0, 8, dist_btw_bins)
#     print(bins)
    
    space = 0.3
    nbin = len(bins)
    width = .5  # (1 - space) #  / (len(inst))
    
#     vals = get_hist_without_0((Xa.values-Xf.values), bins)
    vals = get_hist_without_0(Xf.values, bins)
    ax.plot(bins[:-1], vals, label='TOPAZ no ass')
    
    vals = get_hist_without_0(Xa.values, bins)
    ax.plot(bins[:-1], vals, '-', label='TOPAZ ass')
    
    vals = get_hist_without_0(Xf_mc.values, bins)
    ax.plot(bins[:-1], vals, 'r', alpha=.7, label='Monthly mean corrected')
    
    vals = get_hist_without_0(Xc.values, bins)
    ax.plot(bins[:-1], vals, 'k', label='ML corrected')
    
    ax.set_ylabel('Density')
    ax.set_xlabel('hice (m)')
    
    plt.legend()
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 2])
    

    
    
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return

        filename = f'hist_error.png'
        plt.savefig(f"{odir}{filename}")
        print(f'Saved as: {odir}{filename}')


    if showfig:
        plt.show()


    plt.close()
    
    
    
def hist_monthly(Xf, Xf_mc, Xc, Xa, chrono, odir='', showfig=False, savefig=False):
    '''Plot histogramme of error
    '''
    import matplotlib.colors as mcolors
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(21,12))
    
#     bins = [0,1,2,3,4,5,6,7,8,9]
    dist_btw_bins = .05
    bins = np.arange(0, 8, dist_btw_bins)
#     print(bins)
    
    space = 0.3
    nbin = len(bins)
    width = .5  # (1 - space) #  / (len(inst))
    
#     colors = plt.cm.viridis(np.linspace(0,1,12))
    colors = plt.cm.rainbow(np.linspace(0,1,12))
    
#     mycmap = plt.cm.get_cmap('twilight', 12)  #(np.linspace(0,1,nbin))
#     print(mycmap.N)
#     colors = [mcolors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
    
#     plt.style.use('classic')
    
#     vals = get_hist_without_0((Xa.values-Xf.values), bins)
    for n in range(12):
        vals = get_hist_without_0(Xf.values[n], bins)
        ax.plot(bins[:-1], vals, c=colors[n], label=f'Month {n+1}')
    
#     vals = get_hist_without_0(Xa.values, bins)
#     ax.plot(bins[:-1], vals, '-', label='TOPAZ ass')
    
#     vals = get_hist_without_0(Xf_mc.values, bins)
#     ax.plot(bins[:-1], vals, 'r', alpha=.7, label='Monthly mean corrected')
    
#     vals = get_hist_without_0(Xc.values, bins)
#     ax.plot(bins[:-1], vals, 'k', label='ML corrected')
    
    ax.set_ylabel('Density')
    ax.set_xlabel('hice (m)')
    
    plt.legend()
    plt.title('SIT forecast')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 2])
    

    
    
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return

        filename = f'hist_error.png'
        plt.savefig(f"{odir}{filename}")
        print(f'Saved as: {odir}{filename}')


    if showfig:
        plt.show()


    plt.close()
    
    
    
    
    
def hist_monthly_4(Xf, Xf_mc, Xc, Xa, chrono, odir='', showfig=False, savefig=False):
    '''Plot histogramme of error
    '''
    import matplotlib.colors as mcolors
    
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(21,12))
    
#     bins = [0,1,2,3,4,5,6,7,8,9]
    dist_btw_bins = .05
    bins = np.arange(0, 8, dist_btw_bins)
#     print(bins)
    
    space = 0.3
    nbin = len(bins)
    width = .5  # (1 - space) #  / (len(inst))
    
#     colors = plt.cm.viridis(np.linspace(0,1,12))
    colors = plt.cm.rainbow(np.linspace(0,1,12))

    
    datasets = [Xf, Xa, Xf_mc, Xc]
    labels = ['forecast', 'TOPAZ ass','montlhy mean corr', 'ML corr']
    
#     vals = get_hist_without_0((Xa.values-Xf.values), bins)
    for data, ax, lb in zip(datasets , ax.flatten(), labels):
        for nm in range(12):
            vals = get_hist_without_0(data.values[nm], bins)
            ax.plot(bins[:-1], vals, c=colors[nm], label=f'Month {nm+1}')

        ax.set_ylabel('Density')
        ax.set_xlabel('hice (m)')
    
        plt.legend()
        ax.set_title(f'SIT {lb}')
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 2])
    

    
    
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return

        filename = f'hist_error.png'
        plt.savefig(f"{odir}{filename}")
        print(f'Saved as: {odir}{filename}')


    if showfig:
        plt.show()
    
    plt.close()