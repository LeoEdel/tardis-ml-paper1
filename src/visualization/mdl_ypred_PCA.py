# plot figures for PCA prediction of model of machine learning

import matplotlib.pyplot as plt
from src.utils import modif_plot
from src.utils import save_name


def draw(model, max_plot=8, odir='',  ofile='', savefig=False, showfig=True):
    '''model, data classe ModelML or SModelDL
    '''

    if max_plot > model.npca: max_plot = model.npca
    
    fig, ax = plt.subplots(ncols=1, nrows=max_plot, figsize=(12,max_plot*3), constrained_layout=True)

    for i in range(max_plot):
        ax[i].plot(model.chrono[:], model.ytrue[:,i], label='TOPAZ4-RA', ls='--', c='k')
        ax[i].plot(model.chrono[:], model.ypred[:,i], label='TOPAZ4-ML', c='#FB6949')
        
        mini, maxi = model.ytrue[:,i].min(), model.ytrue[:,i].max()
        ax[i].plot([model.chrono.iloc[model.ntest]]*2, [mini, maxi],ls=':', c='k')  # , label='test limit')
#         ax[i].plot([model.chrono.iloc[model.ntest+model.nval]]*2, [mini, maxi], ls=':', c='grey', label='val limit')
        ax[i].set_ylabel(f'PC #{i+1}')
        
        ax[i].xaxis.grid(alpha=0.6)

        if i < max_plot-1:
            ax[i].spines[['right', 'top', 'bottom']].set_visible(False)
            ax[i].tick_params(labelbottom=False, bottom=False)     

        elif i == max_plot-1:
            ax[i].spines[['right', 'top']].set_visible(False)


    ax[1].legend(fontsize=18, ncol=2, loc='upper right')

    modif_plot.resize(fig, s=22)

   
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return
        if ofile == '':
            ofile = f'ML_ypred_{model.type}.png'
            
        ofile = save_name.check(f"{odir}", ofile)
        plt.savefig(f"{odir}{ofile}", facecolor='white', dpi=150)
        print(f'Saved as: {odir}{ofile}')


    if showfig:
        plt.show()


    plt.close()


    
def draw_apply(model, max_plot=8, odir='',  ofile='', savefig=False, showfig=True):
    '''model, data classe ModelML or SModelDL
    '''

    if max_plot > model.npca: max_plot = model.npca
    
    fig, ax = plt.subplots(ncols=1, nrows=max_plot,sharex='col', figsize=(12,max_plot*3))

    for i in range(max_plot):
        ax[i].plot(model.chrono[:], model.ypred[:,i], label='predict')
        
        # mini, maxi = model.ytrue[:,i].min(), model.ytrue[:,i].max()
        #ax[i].plot([model.chrono.iloc[model.ntest]]*2, [mini, maxi],ls=':', c='k', label='test limit')
        #ax[i].plot([model.chrono.iloc[model.ntest+model.nval]]*2, [mini, maxi], ls=':', c='grey', label='val limit')
        ax[i].set_ylabel(f'PC{i}')

    ax[0].legend()
#     plt.suptitle(f'{model.type}', y=1.005)
    plt.tight_layout()
   
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return
        if ofile == '':
            ofile = f'ML_ypred_{model.type}.png'
            
        ofile = save_name.check(f"{odir}", ofile)
        plt.savefig(f"{odir}{ofile}")
        print(f'Saved as: {odir}{ofile}')


    if showfig:
        plt.show()


    plt.close()

def draw_local(model, odir='',  ofile='', savefig=False, showfig=True):
    '''model, data classe ModelML or SModelDL
    '''

     
    fig, ax = plt.subplots(sharex='col', figsize=(12,7))


    ax.plot(model.chrono, model.ytrue, label='target')
    ax.plot(model.chrono, model.ypred, label='predict')

    mini, maxi = model.ytrue.min(), model.ytrue.max()
    ax.plot([model.chrono.iloc[model.ntest]]*2, [mini, maxi],ls=':', c='k', label='test limit')
    ax.plot([model.chrono.iloc[model.ntest+model.nval]]*2, [mini, maxi], ls=':', c='grey', label='val limit')
#     ax[i].set_ylabel(f'PC{i}')

    ax.legend(ncol=2)
    plt.suptitle(f'{model.type}:  {model.point_train} applied to {model.point_target}', y=1.005)
    plt.tight_layout()
   
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return
        if ofile == '':
            ofile = f'ML_ypred_{model.type}.png'
            
        ofile = save_name.check(f"{odir}", ofile)
        plt.savefig(f"{odir}{ofile}")
        print(f'Saved as: {odir}{ofile}')


    if showfig:
        plt.show()


    plt.close()    
    