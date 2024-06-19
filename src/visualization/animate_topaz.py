import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from src.utils import save_name
from src.utils import modif_plot


def multiple_draw_4b_fr_cs_ld_v2p(sit_4b, sit_fr, sit_cs2, sit_l, chrono_l, days, 
                              sitm_4b, sitm_fr, sitm_cs2, sitm_l,
                              rootdir, fig_dir, showfig=False, savefig=True):
    '''Run draw_4b_4c for multiple days
    
    Parameters:
    -----------
    days        :  array of datetime.datetime(yyyy,mm,dd)
    '''
    for day in days:
        draw_4b_fr_cs_ld_v2p(sit_4b, sit_fr, sit_cs2, sit_l, chrono_l, day,
                             sitm_4b, sitm_fr, sitm_cs2, sitm_l,
                             rootdir=rootdir, fig_dir=fig_dir, showfig=showfig, savefig=savefig)
        
        
def draw_4b_fr_cs_ld_v2p(Xa, Xf, Xc, Xl, chrono_l, day, 
                         Xam, Xfm, Xcm, Xlm,
                         rootdir='', fig_dir='', showfig=True, savefig=False):
    '''
    Draw Landy 22 reprojected on TOPAZ grid
    + add subplot for the mean
    
    Parameters:
    -----------
    
        Xa          : SIT assimilated
        Xf          : SIT free run
        Xc          : SIT from CS2SMOS
        Xl          : SIT from Landy product <<reprojected on TOPAZ grid>>
        chrono_l    : datetime array for landy product (because time axis is fucking shit)
        
        Xa and Xf must have the same chrono (time axis)
    '''
    from matplotlib import gridspec  # uneven subplot
    
    # Parameters for colormap
    vmin = 0
    vmax = 4
    my_cmap = plt.cm.get_cmap('viridis')
    my_cmap.set_under('white')       
    
    # identify index to plot
    chrono_tp = pd.DataFrame({'date':pd.to_datetime(Xa['time'].to_numpy())})
    chrono_dt = np.array([dt.date() for dt in chrono_tp.date])
    idx = np.where(chrono_dt==day.date())[0]

    # index for cs2smos
    chrono_cs2 = pd.DataFrame({'date':pd.to_datetime(Xc['time'].to_numpy())})
    chrono_dt_cs2 = np.array([dt.date() for dt in chrono_cs2.date])
    idx_c = np.where(chrono_dt_cs2==day.date())[0]
    idx_most_recent = (np.where(chrono_dt_cs2-day.date()<datetime.timedelta(days=0))[0]).argmax()
    
    # index for Landy
    # select the most recent (and past) observation
    chrono_dt_l = np.array([dt.date() for dt in chrono_l.date])
    idx_l = (np.where(chrono_dt_l-day.date()<datetime.timedelta(days=0))[0]).argmax()
    
    # get TOPAZ land mask to plot on top of Landy
    land_mask = Xa.isel(time=0).isnull()
    land_mask = land_mask.where(land_mask>0.5)
    land_mask2 = land_mask.assign_coords({'y':Xl.y[::-1], 'x':Xl.x})  # need same coordinates than Landy



    # put SIT < 0.002 m as SIT = -1
    # will be white areas on plot
    Xa_ = Xa.isel(time=idx).where((Xa.isel(time=idx)>0.002) | Xa.isel(time=idx).isnull(), -1)
    Xf_ = Xf.isel(time=idx).where((Xf.isel(time=idx)>0.002) | Xf.isel(time=idx).isnull(), -1)
    Xc_ = Xc.isel(time=idx_c).where((Xc.isel(time=idx_c)>0.002) | Xc.isel(time=idx_c).isnull(), -1)
    Xl_ = Xl.isel(time=idx_l).where((Xl.isel(time=idx_l)>0.002) | Xl.isel(time=idx_l).isnull(), -1)
    
    
    # ----- Define subplots -----
#     fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(22*2,22*2), constrained_layout=True)
    fig = plt.figure(figsize=(22*2,22*2), constrained_layout=True)
#     fig = plt.figure(figsize=(22,22), constrained_layout=True)
    gs = gridspec.GridSpec(44, 20, figure=fig)
    ax2 = fig.add_subplot(gs[0:9, :])

    
    ax00 = fig.add_subplot(gs[10:26, 0:10])
    ax01 = fig.add_subplot(gs[10:26, 10:20])

    ax10 = fig.add_subplot(gs[27:43, 0:10])
    ax11 = fig.add_subplot(gs[27:43, 10:20])
    ax_cb = fig.add_subplot(gs[43:44, 0:10])

    # -------------------------------
    # plot SIT mean
    
    Xfm.isel(time=slice(None, idx[0])).plot(ax=ax2, label='TOPAZ Freerun', color='#1295B2', lw=3, zorder=0)
    Xam.isel(time=slice(None, idx[0])).plot(ax=ax2, label='TOPAZ', lw=3, ls='--', color='k', zorder=10)
    Xlm.isel(time=slice(None, idx_l)).plot(ax=ax2, color='green', ls='--', lw=3, label='Landy')
    Xcm.isel(time=slice(None, idx_most_recent)).plot(ax=ax2, marker='.', c='#9f5ebd', label='CS2SMOS', ls='', markersize=15, zorder=5)
    
    ax2.set_ylabel(f'SIT (m)')
    ax2.xaxis.grid(alpha=0.6)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.legend(fontsize=35, ncol=4, markerscale=2)
    ax2.set_xlim(datetime.datetime(2011,1,1), datetime.datetime(2022,12,31))
    ax2.set_ylim([0.5,2.3])
    ax2.set_xlabel('')
    
    # --- Plot 2d maps ---
    Xf_.plot(ax=ax00, vmin=vmin, vmax=vmax, add_colorbar=False, cmap=my_cmap,
                                  extend='max')
    Xa_.plot(ax=ax01, vmin=vmin, vmax=vmax, add_colorbar=False, cmap=my_cmap,
                          extend='max')
    imC = Xl_.plot(ax=ax10, vmin=vmin, vmax=vmax, add_colorbar=False, cmap=my_cmap) #,
#                           extend='max', cbar_kwargs={'orientation':'horizontal',
#                                                                        'label':'SIT (m)',
#                                                                        'ticks':[0,1,2,3,4], 'aspect':25})
    land_mask2.where(land_mask2>0.5).plot(ax=ax10, add_colorbar=False, cmap=plt.get_cmap('Greys'))

    
    if idx_c.size > 0:
        Xc_.plot(ax=ax11, vmin=vmin, vmax=vmax, add_colorbar=False, cmap=my_cmap,
                              extend='max')       
        
    # contour plot
    levels = np.arange(1, vmax+1, 1)
    cl = Xa.isel(time=idx[0]).plot.contour(ax=ax01, levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    cl = Xf.isel(time=idx[0]).plot.contour(ax=ax00, levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    if idx_c.size > 0:
        cl = Xc.isel(time=idx_c[0]).plot.contour(ax=ax11, levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    cl = Xl.isel(time=idx_l).plot.contour(ax=ax10, levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    
    ax00.set_title('TOPAZ freerun')
    ax01.set_title('TOPAZ')
    ax10.set_title(f'Landy {chrono_l.iloc[idx_l].date.strftime("%Y %m %d")}')
    if idx_c.size > 0:
        ax11.set_title('CS2SMOS')
    else:
        ax11.set_visible(False)

   
    for ax in [ax00,ax01,ax10,ax11]:  #axes.flatten():
        ax.set_facecolor('grey')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    plt.colorbar(imC, cax=ax_cb, orientation='horizontal', extend='max', label='SIT (m)', ticks=[0,1,2,3,4])
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')
    modif_plot.resize(fig, s=45)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
        ofile = f'{rootdir}{fig_dir}TOPAZ4b23_FR_CS2SMOS_Landy_v2p_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white', dpi=200)
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()  

    
def multiple_draw_4b_fr_cs_ld(sit_4b, sit_fr, sit_cs2, sit_l, chrono_l, days, rootdir, fig_dir, showfig=False, savefig=True):
    '''Run draw_4b_4c for multiple days
    
    Parameters:
    -----------
    days        :  array of datetime.datetime(yyyy,mm,dd)
    '''
    for day in days:
#         draw_4b_fr_cs_ld(sit_4b, sit_fr, sit_cs2, sit_l, chrono_l, day, rootdir, fig_dir, showfig=showfig, savefig=savefig)
        draw_4b_fr_cs_ld_v2(sit_4b, sit_fr, sit_cs2, sit_l, chrono_l, day, rootdir, fig_dir, showfig=showfig, savefig=savefig)

    
        
def draw_4b_fr_cs_ld_v2(Xa, Xf, Xc, Xl, chrono_l, day, rootdir='', fig_dir='', showfig=True, savefig=False):
    '''
    Draw Landy 22 reprojected on TOPAZ grid
    + add subplot for the mean
    
    Parameters:
    -----------
    
        Xa          : SIT assimilated
        Xf          : SIT free run
        Xc          : SIT from CS2SMOS
        Xl          : SIT from Landy product <<reprojected on TOPAZ grid>>
        chrono_l    : datetime array for landy product (because time axis is fucking shit)
        
        Xa and Xf must have the same chrono (time axis)
    '''
    
    print('todo add the last subplot')
    
    # Parameters for colormap
    vmin = 0
    vmax = 4
    my_cmap = plt.cm.get_cmap('viridis')
    my_cmap.set_under('white')       
    
    # identify index to plot
    chrono_tp = pd.DataFrame({'date':pd.to_datetime(Xa['time'].to_numpy())})
    chrono_dt = np.array([dt.date() for dt in chrono_tp.date])
    idx = np.where(chrono_dt==day.date())[0]

    # index for cs2smos
    chrono_cs2 = pd.DataFrame({'date':pd.to_datetime(Xc['time'].to_numpy())})
    chrono_dt_cs2 = np.array([dt.date() for dt in chrono_cs2.date])
    idx_c = np.where(chrono_dt_cs2==day.date())[0]
    
    # index for Landy
    # select the most recent (and past) observation
    chrono_dt_l = np.array([dt.date() for dt in chrono_l.date])
    idx_l = (np.where(chrono_dt_l-day.date()<datetime.timedelta(days=0))[0]).argmax()
    
    # get TOPAZ land mask to plot on top of Landy
    land_mask = Xa.isel(time=0).isnull()
    land_mask = land_mask.where(land_mask>0.5)
    land_mask2 = land_mask.assign_coords({'y':Xl.y, 'x':Xl.x})  # need same coordinates than Landy



    # put SIT < 0.002 m as SIT = -1
    # will be white areas on plot
    Xa_ = Xa.isel(time=idx).where((Xa.isel(time=idx)>0.002) | Xa.isel(time=idx).isnull(), -1)
    Xf_ = Xf.isel(time=idx).where((Xf.isel(time=idx)>0.002) | Xf.isel(time=idx).isnull(), -1)
    Xc_ = Xc.isel(time=idx_c).where((Xc.isel(time=idx_c)>0.002) | Xc.isel(time=idx_c).isnull(), -1)
    Xl_ = Xl.isel(time=idx_l).where((Xl.isel(time=idx_l)>0.002) | Xl.isel(time=idx_l).isnull(), -1)
    
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(22*2,22*2), constrained_layout=True)
    
    

    imC = Xf_.plot(ax=axes[0][0], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                                  extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    Xa_.plot(ax=axes[0][1], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                          extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    Xl_.plot(ax=axes[1][0], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                          extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    land_mask2.where(land_mask2>0.5).plot(ax=axes[1][0], add_colorbar=False, cmap=plt.get_cmap('Greys'))
#     plt.show()
#     return
    
    if idx_c.size > 0:
        Xc_.plot(ax=axes[1][1], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                              extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})       
        
    # contour plot
    levels = np.arange(1, vmax+1, 1)
    cl = Xa.isel(time=idx[0]).plot.contour(ax=axes[0][1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    cl = Xf.isel(time=idx[0]).plot.contour(ax=axes[0][0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    if idx_c.size > 0:
        cl = Xc.isel(time=idx_c[0]).plot.contour(ax=axes[1][1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    cl = Xl.isel(time=idx_l).plot.contour(ax=axes[1][0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    
    axes[0][0].set_title('TOPAZ freerun')
    axes[0][1].set_title('TOPAZ')
    axes[1][0].set_title(f'Landy {chrono_l.iloc[idx_l].date.strftime("%Y %m %d")}')
    if idx_c.size > 0:
        axes[1][1].set_title('CS2SMOS')
    else:
        axes[1][1].set_visible(False)

   
    for ax in axes.flatten():
        ax.set_facecolor('grey')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')
    modif_plot.resize(fig, s=45)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
        ofile = f'{rootdir}{fig_dir}TOPAZ4b23_FR_CS2SMOS_Landy_v2_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()  

        


def draw_4b_fr_cs_ld(Xa, Xf, Xc, Xl, chrono_l, day, rootdir='', fig_dir='', showfig=True, savefig=False):
    '''
    Parameters:
    -----------
    
        Xa          : SIT assimilated
        Xf          : SIT free run
        Xc          : SIT from CS2SMOS
        Xl          : SIT from Landy product
        chrono_l    : datetime array for landy product (because time axis is fucking shit)
        
        Xa and Xf must have the same chrono (time axis)
    '''
    
    # Parameters for colormap
    vmin = 0
    vmax = 4
    my_cmap = plt.cm.get_cmap('viridis')
    my_cmap.set_under('white')       
    
    # identify index to plot
    chrono_tp = pd.DataFrame({'date':pd.to_datetime(Xa['time'].to_numpy())})
    chrono_dt = np.array([dt.date() for dt in chrono_tp.date])
    idx = np.where(chrono_dt==day.date())[0]

    # index for cs2smos
    chrono_cs2 = pd.DataFrame({'date':pd.to_datetime(Xc['time'].to_numpy())})
    chrono_dt_cs2 = np.array([dt.date() for dt in chrono_cs2.date])
    idx_c = np.where(chrono_dt_cs2==day.date())[0]
    
    # index for Landy
    # select the most recent (and past) observation
    idx_l = (np.where(chrono_l-day<datetime.timedelta(days=0))[0]).argmax()
    
    # put SIT < 0.002 m as SIT = -1
    # will be white areas on plot
    Xa_ = Xa.isel(time=idx).where((Xa.isel(time=idx)>0.002) | Xa.isel(time=idx).isnull(), -1)
    Xf_ = Xf.isel(time=idx).where((Xf.isel(time=idx)>0.002) | Xf.isel(time=idx).isnull(), -1)
    Xc_ = Xc.isel(time=idx_c).where((Xc.isel(time=idx_c)>0.002) | Xc.isel(time=idx_c).isnull(), -1)
    Xl_ = Xl.isel(t=idx_l).where((Xl.isel(t=idx_l)>0.002) | Xl.isel(t=idx_l).isnull(), -1)
    
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(22*2,22*2), constrained_layout=True)
    
    

    imC = Xf_.plot(ax=axes[0][0], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                                  extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    Xa_.plot(ax=axes[0][1], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                          extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    Xl_.plot(ax=axes[1][0], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                          extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    
    if idx_c.size > 0:
        Xc_.plot(ax=axes[1][1], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                              extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})       
        
    # contour plot
    levels = np.arange(1, vmax+1, 1)
    cl = Xa.isel(time=idx[0]).plot.contour(ax=axes[0][1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    cl = Xf.isel(time=idx[0]).plot.contour(ax=axes[0][0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    if idx_c.size > 0:
        cl = Xc.isel(time=idx_c[0]).plot.contour(ax=axes[1][1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    cl = Xl.isel(t=idx_l).plot.contour(ax=axes[1][0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    
    axes[0][0].set_title('TOPAZ freerun')
    axes[0][1].set_title('TOPAZ')
    axes[1][0].set_title(f'Landy {chrono_l[idx_l].date().strftime("%Y %m %d")}')
    if idx_c.size > 0:
        axes[1][1].set_title('CS2SMOS')
    else:
        axes[1][1].set_visible(False)
   
   
    for ax in axes.flatten():
        ax.set_facecolor('grey')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')
    modif_plot.resize(fig, s=45)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
        ofile = f'{rootdir}{fig_dir}TOPAZ4b23_FR_CS2SMOS_Landy_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()  




def multiple_draw_4b_fr_bias_23(sit_4b, sit_fr, chrono, days, rootdir, fig_dir, showfig=False, savefig=True):
    '''Run draw_4b_4c for multiple days
    
    Parameters:
    -----------
    days        :  array of datetime.datetime(yyyy,mm,dd)
    
    
    '''
    
    for day in days:
        draw_4b_fr_bias_23(sit_4b, sit_fr, chrono, day, rootdir, fig_dir, showfig=showfig, savefig=savefig)
    
    
    
def draw_4b_fr_bias_23(Xb, Xc, chrono, day, rootdir, fig_dir, showfig=True, savefig=False):
    
    
    # Parameters for colormap
    vmin = 0
    vmax = 4
    my_cmap = plt.cm.get_cmap('viridis')
    my_cmap.set_under('white')   
    bias_cmap = plt.cm.get_cmap('bwr')
    
     # diverent colormap [black - white - black]
    import matplotlib.colors as colors
#     colors_under = plt.get_cmap('Greys_r')(np.linspace(0, 1, 256-100))
    colors_over = plt.get_cmap('Greys')(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_over[::-1], colors_over)) # colors_over))
    mymap = colors.LinearSegmentedColormap.from_list('mymap', all_colors)
    
    
    # identify index to plot
    chrono_dt = np.array([dt.date() for dt in chrono.date])
    # chrono_dt = np.array([dt.date() for dt in chrono])
    idx = np.where(chrono_dt==day.date())[0]

        # put SIT < 0.002 m as SIT = -1
    # will be white areas on plot
    Xb_ = Xb.isel(time=idx).where((Xb.isel(time=idx)>0.002) | Xb.isel(time=idx).isnull(), -1)
    Xc_ = Xc.isel(time=idx).where((Xc.isel(time=idx)>0.002) | Xc.isel(time=idx).isnull(), -1)
    
    
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(22*3,22), constrained_layout=True)
    
    
    Xb_.plot(ax=axes[0], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                          extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    imC = Xc_.plot(ax=axes[1], vmin=vmin, vmax=vmax, add_colorbar=True, cmap=my_cmap,
                                  extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})

    mean_bias = (Xc.isel(time=idx)-Xb.isel(time=idx)).mean(dim='time')
    mean_bias.plot(ax=axes[2], vmin=-3, vmax=3, add_colorbar=True, cmap=bias_cmap,
                                               extend='both', cbar_kwargs={'orientation':'horizontal',
                                                                          'label':'Bias SIT (m)',
                                                                          'ticks':[-3,-2,-1,0,1,2,3], 'aspect':25})

        # contour plot
    levels = np.arange(1, vmax+1, 1)
    cl = Xb.isel(time=idx[0]).plot.contour(ax=axes[0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))

    cl = Xc.isel(time=idx[0]).plot.contour(ax=axes[1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    
    blevels = np.arange(-2, 2.5, 0.5)
    cl = mean_bias.plot.contour(ax=axes[2], levels=blevels, vmin=-3, vmax=3, add_colorbar=False, cmap=mymap, center=0)
    
#     cax1 = fig.add_axes([0.25, -0.05, 0.5, 0.04])   
#     fig.colorbar(imC,cax=cax1, label='SIT (m)', extend='max', shrink=0.5, orientation='horizontal', ticks=[0,1,2,3,4])
    
    
    axes[0].set_title('TOPAZ + CS2SMOS')
    axes[1].set_title('TOPAZ')
    axes[2].set_title('TOPAZ - [TOPAZ+CS2SMOS]')
    
   
    for ax in axes.flatten():
        ax.set_facecolor('grey')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')
    modif_plot.resize(fig, s=30)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
        ofile = f'{rootdir}{fig_dir}TOPAZ4b23_FR_bias_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()  



    

def draw_4b_fr_23(Xb, Xc, chrono, day, rootdir, fig_dir, showfig=True, savefig=False):
    
    vmax = 4

    # identify index to plot
    chrono_dt = np.array([dt.date() for dt in chrono.date])
    # chrono_dt = np.array([dt.date() for dt in chrono])
    idx = np.where(chrono_dt==day.date())[0]

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(22*2,22), constrained_layout=True)
    
    Xb.isel(time=idx).plot(ax=axes[0], vmax=vmax, add_colorbar=True, 
                          extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})
    imC = Xc.isel(time=idx).plot(ax=axes[1], vmax=vmax, add_colorbar=True,
                                  extend='max', cbar_kwargs={'orientation':'horizontal',
                                                                       'label':'SIT (m)',
                                                                       'ticks':[0,1,2,3,4], 'aspect':25})

    
        # contour plot
    levels = np.arange(1, vmax+1, 1)
    cl = Xb.isel(time=idx[0]).plot.contour(ax=axes[0], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))

    cl = Xc.isel(time=idx[0]).plot.contour(ax=axes[1], levels=levels, vmin=0, vmax=vmax, add_colorbar=False, cmap=plt.get_cmap('Greys'))
    

#     cax1 = fig.add_axes([0.25, -0.05, 0.5, 0.04])   
#     fig.colorbar(imC,cax=cax1, label='SIT (m)', extend='max', shrink=0.5, orientation='horizontal', ticks=[0,1,2,3,4])
    
    
    axes[0].set_title('TOPAZ + CS2SMOS')
    axes[1].set_title('TOPAZ')
   
    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')
    modif_plot.resize(fig, s=30)

    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
#         ofile = f'{rootdir}{fig_dir}TOPAZ4b_4c_{sdate}.png'
        ofile = f'{rootdir}{fig_dir}TOPAZ4b_4bFR_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()  










def multiple_draw_4b_4c(X4b, X4c, chrono, days, rootdir, fig_dir, showfig=False, savefig=True):
    '''Run draw_4b_4c for multiple days
    
    Parameters:
    -----------
    days        :  array of datetime.datetime(yyyy,mm,dd)
    
    
    '''
    
    for day in days:
        draw_4b_4c(X4b, X4c, chrono, day, rootdir, fig_dir, showfig=showfig, savefig=savefig)
    
    print('Finish')


def draw_4b_4c(Xb, Xc, chrono, day, rootdir, fig_dir, showfig=True, savefig=False):
    '''Plot 
    
    Parameters:
    -----------
    Xb        : Topaz4b  
    Xc        : Topaz4c
    chrono    : from Topaz4b, array of DatetimeIndex
    day       : day to plot, datetime.datetime(yyyy,mm,dd)
     
    '''
    
    vmax = 5

    # identify index to plot
    chrono_dt = np.array([dt.date() for dt in chrono.date])
    # chrono_dt = np.array([dt.date() for dt in chrono])
    idx = np.where(chrono_dt==day.date())[0]

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18*2,14))
    
    Xb.isel(time=idx).plot(ax=ax[0], vmax=vmax, add_colorbar=False) # , shading='nearest')
    imC = Xc.isel(time=idx).plot(ax=ax[1], vmax=vmax, add_colorbar=False)

#     imB = ax[0].imshow(Xb.isel(time=idate)[::-1], vmax = 4, aspect='auto')
#     fig.colorbar(imB, ax=ax[0], label='SIT (m)', extend='max', shrink=0.5)   
# #     imC = ax[1].imshow(Xc.isel(time=idate)[::-1], vmax = 4)
#     ax[1].pcolormesh(Xc.isel(time=idate)[::-1], Xc.y, Xc.x, vmax = 4)
    cax1 = fig.add_axes([0.45, 0.5, 0.01, 0.4])
    fig.colorbar(imC,cax=cax1, label='SIT (m)', extend='max', shrink=0.5)
    
    ax[0].set_title('Topaz 4b')
#     ax[1].set_title('Topaz 4c')
    ax[1].set_title('Topaz 4b Free Run')
    ax[1].set_ylabel('')
    ax[0].set_ylabel('')
    ax[1].set_xlabel('')
    ax[0].set_xlabel('')
    
    fig.suptitle(f'{chrono_dt[idx][0].strftime("%Y %m %d")}')
    modif_plot.resize(fig, s=24)
    plt.tight_layout()
    
    if savefig:
        sdate = chrono_dt[idx][0].strftime("%Y%m%d")
#         ofile = f'{rootdir}{fig_dir}TOPAZ4b_4c_{sdate}.png'
        ofile = f'{rootdir}{fig_dir}TOPAZ4b_4bFR_{sdate}.png'
        plt.savefig(f"{ofile}", facecolor='white')
        print(f'Saved as : {ofile}')
    
    if showfig:
        plt.show()
    
    plt.close()  