import numpy as np
import matplotlib.pyplot as plt
from src.utils import save_name
from src.utils import modif_plot
from mpl_toolkits.basemap import Basemap
import datetime

def multiple_4sf(lats, lons, sit_obs, lats2, lons2, sit2, 
                lats3, lons3, sit3, lats4, lons4, sit4,
                chrono_cl, chrono, days,
                odir='', savefig=True, showfig=False, **kwargs):
    '''Run draw_4sf for multiple days
    
    Parameters:
    -----------
    days        :  array of datetime.datetime(yyyy,mm,dd)
    
    
    '''
    
    for day in days:
        basemap_4sf(lats, lons, sit_obs, lats2, lons2, sit2, 
                lats3, lons3, sit3, lats4, lons4, sit4,
                chrono_cl, chrono, day,
                odir=odir, savefig=savefig, showfig=showfig, **kwargs)
    
    print('Finish')

    
    

def basemap_4sf(lats, lons, sit_obs, lats2, lons2, sit2, 
                lats3, lons3, sit3, lats4, lons4, sit4,
                chrono_cl, chrono, day,
                suptitle='', odir='', savefig=False, showfig=True, **kwargs):
    '''
    Parameters:
    -----------
            lats         :        grid of latitude, dimension from sit_obs
            lons         :        grid of longitude
            sit_obs      :        xr.DataArray, snowfall rates from CloudSat
            lats2        :        grid of latitude, dimension from sit_tp
            lons2        :        grid of longitude
            sit_tp       :        xr.DataArray, Sea Ice Thickness from ToPaz
    
    
    
            chrono_cl    :        pandas.DataFrame, date for monthly values from CloudSat
            chrono       :        pandas.DataFrame, date for daily values from ERA5/TOPAZ
            day          :        datetime object, date to plot
    
    '''
    
    
         # identify index to plot
    chrono_dt = np.array([dt.date() for dt in chrono.date])
    idx = np.where(chrono_dt==day.date())[0]#[0]
    if len(idx)>0: idx = idx[0]
    # print(idx)

    chrono_dt_cl = np.array([dt.date() for dt in chrono_cl.date])
    day_cl = datetime.date(day.year, day.month, 1)
    idx_cl = np.where(chrono_dt_cl==day_cl)[0] # [0]
    if len(idx_cl)>0: idx_cl = idx_cl[0]
    # print(idx_cl)

    vmax_monthly=30  # mm/month
        
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(9*3, 9*2), constrained_layout=True)
    
    m = Basemap(projection='npstere', boundinglat=68, lon_0=0, resolution='l', round=True, ax=axes[1][1])
    m.drawcoastlines(color='white', zorder=11)
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))
    
    mlons, mlats = m(lons, lats)
    
    pcm = m.pcolormesh(mlons, mlats, sit_obs.isel(time=idx_cl), zorder=10, vmax=vmax_monthly, **kwargs)
    # cbar = plt.colorbar(cs)
    # cbar.set_label('{}'.format(label_cb))
    axes[1][1].set_title('CloudSat monthly')
    cbar=fig.colorbar(pcm, ax=axes[1][1], shrink=0.7, location="bottom", extend='max',
                     label='(mm/m)') # , pad=-0.43)
    
    
    m2 = Basemap(projection='npstere', boundinglat=68, lon_0=0, resolution='l', round=True, ax=axes[0][0])
    m2.drawcoastlines()
    m2.drawparallels(np.arange(-80.,81.,20.))
    m2.drawmeridians(np.arange(-180.,181.,20.))
    
    mlons2, mlats2 = m2(lons2, lats2)
    
    pcm2 = m2.pcolormesh(mlons2, mlats2, sit2.isel(time=idx)*1000, zorder=10, vmax=1, **kwargs)
    axes[0][0].set_title('Snowfall ERA5 on SI')
    
    cbar=fig.colorbar(pcm2, ax=axes[0][0], shrink=0.7, location="bottom", extend='max',
                     label='(mm/d)') # , pad=-0.43)
    
    
    
    m3 = Basemap(projection='npstere', boundinglat=68, lon_0=0, resolution='l', round=True, ax=axes[0][1])
    m3.drawcoastlines()
    m3.drawparallels(np.arange(-80.,81.,20.))
    m3.drawmeridians(np.arange(-180.,181.,20.))
    
    mlons3, mlats3 = m3(lons3, lats3)
    
    pcm3 = m3.pcolormesh(mlons3, mlats3, sit3.isel(time=idx)*1000, zorder=10,  vmax=vmax_monthly, **kwargs)
    axes[0][1].set_title('Sf ERA5 cumul over last month')
    
    cbar=fig.colorbar(pcm3, ax=axes[0][1], shrink=0.7, location="bottom", extend='max',
                     label='(mm/m)') # , pad=-0.43)
    
    
    m4 = Basemap(projection='npstere', boundinglat=68, lon_0=0, resolution='l', round=True, ax=axes[0][2])
    m4.drawcoastlines()
    m4.drawparallels(np.arange(-80.,81.,20.))
    m4.drawmeridians(np.arange(-180.,181.,20.))
    
    mlons4, mlats4 = m4(lons4, lats4)
    
    pcm4 = m4.pcolormesh(mlons4, mlats4, sit4.isel(time=idx), zorder=10, vmax=0.4, **kwargs)
    axes[0][2].set_title('Snow thickness on SI\nTOPAZ4b FR')
    
    cbar=fig.colorbar(pcm4, ax=axes[0][2], shrink=0.7, location="bottom", extend='max',
                     label='(m)') # , pad=-0.43)
    cbar.ax.tick_params(rotation=45)
        
    axes[1][0].set_visible(False)
    axes[1][2].set_visible(False)

    fig.suptitle(f'{chrono_dt[idx].strftime("%Y %m %d")}') # , y=1.04)   
    
    modif_plot.resize(fig, s=32, rx=20)
#     fig.tight_layout(pad=3)
    
    
    if savefig:
        sdate = chrono_dt[idx].strftime("%Y%m%d")
        ofile = f'sf_ERA5_TP4b_CLDST_{sdate}.png'

        # ofile = save_name.check(f"{odir}", ofile)
        plt.savefig(f"{odir}{ofile}", dpi=124, facecolor='white')
        print(f'Figure saved as : {odir}{ofile}')
        
    if showfig:
        plt.show()
        
    plt.close()


