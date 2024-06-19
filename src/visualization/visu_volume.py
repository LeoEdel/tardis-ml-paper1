'''Draw SIV
'''
import datetime
import matplotlib.pyplot as plt

from src.utils import modif_plot
from src.utils import save_name


def draw_vol(vol, volm, trends={}, trends_min={}, trends_max={}, odir='', ofile='', showfig=True, savefig=False):
    '''
    Parameters:
    -----------
    
        vol    :    xarray.DataArray, daily volume 
        volm   :    xarray.DataArray, volume average each month
        trends :    dictionnary, contains trends to plot
    '''
    
    n_months = len(volm)
    
    fig, ax = plt.subplots(figsize=(16,9), constrained_layout=True)

    ## Draw volume
    (volm/1000).isel(time=range(4,n_months,12)).plot(ms=15, lw=2, marker='.', color='b', label='May')
    (volm/1000).isel(time=range(9,n_months,12)).plot(ms=15, lw=2, marker='.', color='r', label='October')
    (vol/1000).plot(lw=2, color='k')

    ## Draw trends
    list_trends = list(trends.keys())
    if len(list_trends)>0:
        for n in list_trends:
            plt.plot(trends[n][0], trends[n][1]/1000, c='k', ls='--')
        
    list_trends_min = list(trends_min.keys())
    if len(list_trends_min)>0:
        for n in list_trends_min:
            plt.plot(trends_min[n][0], trends_min[n][1]/1000, c='r', ls='--', marker='+', ms=10)
        
    list_trends_max = list(trends_max.keys())
    if len(list_trends_max)>0:
        for n in list_trends_max:
            plt.plot(trends_max[n][0], trends_max[n][1]/1000, c='b', ls='--', marker='+', ms=10)
        
      
    ax.yaxis.grid(alpha=0.6, ls='-.')
    ax.xaxis.grid(alpha=0.6)
    ax.set_ylabel('Sea Ice Volume (1000 km³)')
    ax.set_xlabel('')
    ax.spines[['right', 'top']].set_visible(False)

    plt.legend(fontsize=18)

    ax.set_xlim([datetime.datetime(1991,12,1), datetime.datetime(2022,12,31)])
    
    modif_plot.resize(fig, s=24)

    if savefig:
        if ofile == '':
            ofile = f'SIV_01.png'
        ofile = save_name.check(odir, ofile)
        plt.savefig(f"{odir}{ofile}", dpi=120, facecolor='white')
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()    
