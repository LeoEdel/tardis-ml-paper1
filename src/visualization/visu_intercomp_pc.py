import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

from src.utils import modif_plot
from src.utils import save_name

def draw_ypred_mulitple(models,
                        n_pc:int, n_mdl:int, max_plot=8,
                        models_str='models',
                        odir='', savefig=False, showfig=True):
    '''model, data classe ModelML or SModelDL
    '''

    # Parameters:
    ntest = models[list(models.keys())[0]].config.ntest
    colors = ['#7250B0', '#64CFBF', '#8ab17d', '#e9c46a', '#e76f51', '#ae2d68']*5
    linestyle = ['-']*6 + ['--']*6 + ['-.']*6 + [':']*6 + [(0, (3, 1, 1, 1, 1, 1))]*6
    
    if max_plot > n_pc: max_plot = n_pc
    
    fig, ax = plt.subplots(ncols=1, nrows=max_plot, figsize=(12,max_plot*3), constrained_layout=True)

    for i in range(max_plot):
        for nm, name in enumerate(models):
            model = models[name]
            if nm == 0:
                ax[i].plot(model.chrono, model.ytrue[:,i], ls='--', c='k')  # label='TOPAZ', 

            ax[i].plot(model.chrono, model.ypred[:,i], ls=linestyle[nm], label=f'{name}', c=colors[nm])
    
            ax[i].set_ylabel(f'PC #{i+1}')
            ax[i].xaxis.grid(alpha=0.6)

            mini, maxi = model.ytrue[:,i].min(), model.ytrue[:,i].max()
            ax[i].plot([model.chrono.iloc[model.config.ntest]]*2, [mini, maxi],ls=':', c='k')
            
        if i < max_plot-1:
            ax[i].spines[['right', 'top', 'bottom']].set_visible(False)
            ax[i].tick_params(labelbottom=False, bottom=False)     

        elif i == max_plot-1:
            ax[i].spines[['right', 'top']].set_visible(False)


    ax[1].legend(fontsize=18, ncol=3, loc='upper right')
    modif_plot.resize(fig, s=22)

   
    if savefig:
        ofile = f'ML_ypred_intercomp_{models_str}.png'
        ofile = save_name.check(f"{odir}", ofile)
        plt.savefig(f"{odir}{ofile}", facecolor='white', dpi=150, bbox_inches='tight')
        print(f'Saved as: {odir}{ofile}')


    if showfig:
        plt.show()


    plt.close()


def draw_bias_time(bias_pc, chrono, names, 
                     n_pc:int, n_mdl:int,
                     models_str='models', 
                     odir='', savefig=False, showfig=True):
    '''
    Parameters:
    -----------
        bias_pc       :    np.array, format (time, number of pc*number of model), bias for each PC
                                     in this order:
                                         PC 1, model 1,2,3,4,5 ...
                                         PC 2, model 1,2,3,4,5 ...
                                         PC 3...
        names         :    array of string, names of each model
        n_pc          :    int, number of PC
        n_mdl         :    int, number of model
    '''
    
    
    ## check min/max date along models
    
    
    fig, ax = plt.subplots(nrows=1, constrained_layout=True, figsize=(16,9))

    plt.imshow(bias_pc.T, aspect='auto', cmap=plt.get_cmap('RdBu_r'), interpolation='None', vmin=-80, vmax=80)
    plt.colorbar(extend='both', shrink=0.6, label='Bias [Predicted-Truth]')


    
    # Time TickLabels
    time_ticks = np.arange(0,365*3,365//2)  # test period spans 3 years. 1 date every 6 months
    time_labels = chrono[time_ticks]
    time_labels_str = [pd.to_datetime(x.data).strftime('%Y-%m-%d') for x in time_labels]
    
    ytl = [f'PC {x}' for x in range(1,9)]

    ax.set_yticks(range(0, n_pc*n_mdl,n_mdl))
    ax.set_yticklabels(ytl)

    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels_str)

    plt.hlines(y=np.arange(n_mdl-.5, n_pc*n_mdl,n_mdl), xmin=0, xmax=1096, color='k')

    ax.set_xlim([0,1096])

    # Top axis for variables
    ax2 = ax.secondary_yaxis('right')
    yticks = [n+n*n_mdl for n in range(n_mdl)]
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(names, fontsize=20)

    modif_plot.resize(fig, s=20, do_annotation=False)



    if savefig:
        ofile = f'intercomp_PC_time_{models_str}_01.png'
        ofile = save_name.check(odir, ofile)
        plt.savefig(f"{odir}{ofile}", dpi=150, bbox_inches='tight')
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()  

    
    
    

def draw_bias_violin(bias_pc, names, 
                     n_pc:int, n_mdl:int,
                     models_str='models', 
                     odir='', savefig=False, showfig=True):
    '''
    Parameters:
    -----------
    
        bias_pc       :    np.array, format (time, number of pc*number of model), bias for each PC
                                     in this order:
                                         PC 1, model 1,2,3,4,5 ...
                                         PC 2, model 1,2,3,4,5 ...
                                         PC 3...
        names         :    array of string, names of each model
        n_pc          :    int, number of PC
        n_mdl         :    int, number of model
    '''

   
    pos = np.arange(0, int(n_pc*n_mdl))

    fig, ax = plt.subplots(nrows=1, constrained_layout=True, figsize=(16,9))

    violons_mdl = {}  # dico of bias to plot different colors
    for nm in range(n_mdl):
        idx_mdl = np.arange(nm, n_pc*n_mdl, n_mdl)
        violons_mdl[nm] = ax.violinplot(bias_pc[:,idx_mdl], pos[idx_mdl], showmeans=True, widths=1)

    # color each model differently
    grey = '#4D4D4D'
    # colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']  # 3 "blue"
    colors = ['#7250B0', '#64CFBF', '#8ab17d', '#e9c46a', '#e76f51', '#ae2d68']*5


    for idx, kk in enumerate(list(violons_mdl.keys())):
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = violons_mdl[idx][partname]
            vp.set_edgecolor(grey)
            vp.set_linewidth(2)
            vp.set_alpha(.5)

        for vp in violons_mdl[idx]['bodies']:
            vp.set_facecolor(colors[idx])
            vp.set_edgecolor(grey)
            vp.set_linewidth(0)
            vp.set_alpha(.9)


    plt.vlines(x=np.arange(n_mdl-0.5, (n_pc-1)*n_mdl,n_mdl), ymin=-100, ymax=100, color='grey',ls='--')
    plt.hlines(y=0, xmin=-10, xmax=100, color='grey',ls='-.', zorder=-10)


    ax.set_ylabel('Bias [Predicted-Truth]')

    ax.set_xlim([-1,n_pc*n_mdl])
    ax.set_ylim([-80,80])

    xtl = [f'PC {x}' for x in range(1,9)]
    ax.set_xticks(np.arange(n_mdl//2, n_pc*n_mdl,n_mdl)-.5)
    ax.set_xticklabels(xtl)

    lgd = []  # create fake legend
    for n in range(n_mdl):
        lgd.append(mpatches.Patch(color=colors[n], label=names[n]))

        
    if n_mdl <= 6:
        ncols = n_mdl
    else:
        ncols = n_mdl // 3

    plt.legend(handles=lgd, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncols=ncols, fancybox=True, fontsize=18)

    modif_plot.resize(fig, s=20, do_annotation=False)


    if savefig:
        ofile = f'intercomp_PC_violin_{models_str}_01.png'
        ofile = save_name.check(odir, ofile)
        plt.savefig(f"{odir}{ofile}", dpi=150, bbox_inches='tight')
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()  

def draw_errors_intercomp(rmse, bias, corr, names, models_str='models', odir='', savefig=False, showfig=True):
    '''
    Parameters:
    -----------
    
        rmse           : np.array, format(number of PC, number of model),  RMSE
        bias           : np.array, format(number of PC, number of model),  bias 
        corr           : np.array, format(number of PC, number of model),  correlation
        names          : array of string, contains the names of each model
        models_str     : string, used in output filename .png
    
    '''
    
    fig, axes = plt.subplots(ncols=3, constrained_layout=True, figsize=(24,6))

    ax = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    
    n_pc = rmse.shape[0]
    n_mdl = rmse.shape[1]
    
    if n_mdl <= 8:
        size_edges = 12
        score_fontsize = 16
        cb_pad = -0.05
    else:  # if n_mdl <= 15:
        size_edges = 10
        score_fontsize = 12
        cb_pad = 0
    
    
    
    # --------------       RMSE      ----------------------------------------------------
    
    cax = ax.imshow(rmse.T, cmap=plt.get_cmap('Reds'), interpolation='None', aspect=0.8)
    cbar = plt.colorbar(cax,
    #                     ticks=[-1,0,1],
    #                     format=mticker.FixedFormatter(['-1', '0', '1']),
                        extend=None, shrink=0.5, label='RMSE', pad=cb_pad)


    # get the middle value of the cbar
    cbar_ticks = []
    for t in cbar.ax.get_yticklabels():
        cbar_ticks += [t._y]

    mid_cbar = cbar_ticks[len(cbar_ticks)//2]  # so we can turn the font color to white when cmap is dark


    for (i, j), z in np.ndenumerate(rmse.T):
        if abs(z) > mid_cbar: tc = 'w'
        else: tc = 'k'
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=score_fontsize, color=tc)


    xtl = [f'PC {x}' if x==1 else f'{x}' for x in range(1,9)]  # set x tick labels
    ax.set_xticks(range(n_pc))
    ax.set_xticklabels(xtl)

    ax.set_yticks(range(n_mdl))
    ax.set_yticklabels(names)

    ## Add edges
    # Minor ticks
    ax.set_xticks(np.arange(-.5, n_pc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_mdl, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=size_edges)

    # Remove minor ticks
    ax.tick_params(which='both', bottom=False, left=False)

    ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax.set_xlim([-.5, n_pc])
    
    
    # --------------       Bias      ----------------------------------------------------
    tmp_maxi = np.nanmax(abs(bias))

    mini = -tmp_maxi
    maxi = tmp_maxi

    cax1 = ax1.imshow(bias.T, cmap=plt.get_cmap('RdBu_r'), interpolation='None', aspect=0.8, vmin=mini, vmax=maxi)
    cbar1 = plt.colorbar(cax1,
    #                     ticks=[-1,0,1],
    #                     format=mticker.FixedFormatter(['-1', '0', '1']),
                        extend=None, shrink=0.5, label='Bias', pad=cb_pad)


    # get the middle value of the cbar
    cbar_ticks = []
    for t in cbar1.ax.get_yticklabels():
        cbar_ticks += [t._y]

    mid_cbar = cbar_ticks[len(cbar_ticks)//4]  # so we can turn the font color to white when cmap is dark


    for (i, j), z in np.ndenumerate(bias.T):
        if abs(z) > abs(mid_cbar): tc = 'w'
        else: tc = 'k'
        ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize=score_fontsize, color=tc)


    ax1.set_xticks(range(n_pc))
    ax1.set_xticklabels(xtl)

    ax1.set_yticks(range(n_mdl))
    ax1.set_yticklabels(names)

    ## Add edges
    # Minor ticks
    ax1.set_xticks(np.arange(-.5, n_pc, 1), minor=True)
    ax1.set_yticks(np.arange(-.5, n_mdl, 1), minor=True)

    # Gridlines based on minor ticks
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=size_edges)

    # Remove minor ticks
    ax1.tick_params(which='both', bottom=False, left=False)

    ax1.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax1.set_xlim([-.5, n_pc])
    
    
    # --------------       Correlation      ----------------------------------------------------
    
    
    cax2 = ax2.imshow(corr.T, cmap=plt.get_cmap('RdBu_r'), interpolation='None', vmin=-1, vmax=1, aspect=0.8)
    cbar2 = plt.colorbar(cax2,
                        ticks=[-1,0,1],
                        format=mticker.FixedFormatter(['-1', '0', '1']),
                        extend=None, shrink=0.5, label='Correlation', pad=cb_pad)


    for (i, j), z in np.ndenumerate(corr.T):
        if abs(z) > 0.75: tc = 'w'
        else: tc = 'k'
        ax2.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=score_fontsize, color=tc)


    ax2.set_xticks(range(n_pc))
    ax2.set_xticklabels(xtl)

    ax2.set_yticks(range(n_mdl))
    ax2.set_yticklabels(names)

    ## Add edges
    # Minor ticks
    ax2.set_xticks(np.arange(-.5, n_pc, 1), minor=True)
    ax2.set_yticks(np.arange(-.5, n_mdl, 1), minor=True)

    # Gridlines based on minor ticks
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=size_edges)

    # Remove minor ticks
    ax2.tick_params(which='both', bottom=False, left=False)

    ax2.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax2.set_xlim([-.5, n_pc])

    
    # --------------------  Finition   ---------------------------------------------------------

    # lines between each subplot
    lines = [plt.Line2D([ax.get_position().x0+ax.get_position().width]*2, [0.2,.8], transform=fig.transFigure, color="grey", lw=2),
             plt.Line2D([ax2.get_position().x0]*2, [.2,.8], transform=fig.transFigure, color="grey", lw=2)]
    
    for line in lines:
        fig.add_artist(line)

    
    
    modif_plot.resize(fig, s=20, do_annotation=False)    
    
    if savefig:
#         models_str = 'models_vars'
        ofile = f'intercomp_PC_errors_{models_str}_01.png'
        ofile = save_name.check(odir, ofile)
        plt.savefig(f"{odir}{ofile}", dpi=150, bbox_inches='tight')
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()  


