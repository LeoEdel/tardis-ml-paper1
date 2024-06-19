import matplotlib.pyplot as plt
import numpy as np

from src.utils import modif_plot


def draw_ypreds(predictions, ori_ypred, chrono, max_plot=8, odir='', savefig=False, showfig=False):
    '''Draw ensemble of predictions for perturbated inputs (PCA)
    '''

    fig, ax = plt.subplots(ncols=1, nrows=max_plot, sharex='col', figsize=(12,max_plot*3), constrained_layout=True)

    for pred in predictions[::-1]:
        for i in range(max_plot):
            ax[i].plot(chrono, pred[:,i], color='orange', alpha=0.4)
        
    for i in range(max_plot):
        ax[i].plot(chrono, ori_ypred[:,i], 'k--', lw=2)
        ax[i].set_ylabel(f'PC {i+1}')
        
        
    ax[0].set_title(f'{len(predictions)} members')
    
    if savefig:
        ofile = f'ypreds_PCA_ensemble_pert_01.png'
        plt.savefig(f"{odir}{ofile}", facecolor='white', dpi=120)
        print(f'Saved as: {odir}{ofile}')
    
    if showfig:
        plt.show()

    plt.close()




def draw_situ_xy(ori_mean_sit, mean_sit, std, sic_na, odir='', showfig=False, savefig=False):
    '''
    
        sic_na     :   used to plot invalid data (SIC < 15%)
    
    '''

    fig, axes = plt.subplots(ncols=3, figsize=(8*3,8), constrained_layout=True)

    # compute labels for colorbar of [ensemble mean SIT - original SIT]
    diff = (mean_sit-ori_mean_sit)
    maxi = np.max([abs(diff.min()), diff.max()])
    maxi_rnd = np.round(maxi,2)
#     if maxi_rnd < 0.1: 
#         maxi_rnd = 0.1
    tcks = np.linspace(-maxi_rnd, maxi_rnd, 5)
#     print(tcks)
    
    # Add 0 where SIC is < 15% (just to make the plot looks nicer)
    mask_ocean = sic_na.isel(time=0).where(np.isnan(sic_na.isel(time=0)), 1)  # ocean = 1, land = 0
    mean_sit = mean_sit.where((mask_ocean!=1 & np.isnan(mean_sit)), 0)
    diff = diff.where((mask_ocean!=1 & np.isnan(diff)), 0)
    std = std.where((mask_ocean!=1 & np.isnan(std)), 0)
    


    mean_sit.plot(ax=axes[0], vmin=0, vmax=4, cbar_kwargs={'orientation':'horizontal', 'extend':'max',
                                                                       'label':'SIT (m)', 'aspect':25})   
        
    diff.plot(ax=axes[1], cbar_kwargs={'orientation':'horizontal', 'extend':'both',
                                                                       'label':'SIT (m)', 'aspect':25,
                                                         'ticks':tcks})

    std.plot(ax=axes[2], cbar_kwargs={'orientation':'horizontal', 'extend':'max',
                                            'label':'Uncertainty SIT (m)', 'aspect':25})

#     axes[0].set_title('Original inputs')
    axes[0].set_title('Ensemble mean')
    axes[1].set_title('Ensemble mean - Original inputs')
    axes[2].set_title('Ensemble std')
    
    for ax in axes.flatten():
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)     
        ax.set_facecolor('grey')
    
    modif_plot.resize(fig, s=24, rx=30)
    
    if savefig:
        ofile = f'uncert_2d_01.png'
        plt.savefig(f"{odir}{ofile}", facecolor='white', dpi=120)
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()


def draw_sit_pert(means_pert, mean_sit, ori_mean_sit, chrono,  odir='', showfig=False, savefig=False):
    '''Quick Plot to visualize SIT of each member of perturbed ensemble
    
    Parameters:
    -----------
    
        means_pert    :    list of array, SIT average time series for each member of the ensemble of perturbation
        mean_sit      :    array, SIT average time series of all members of the ensemble
        ori_mean_sit  :    array, SIT of the initial dataset (without perturbation)
        chrono        :    pd.Dataframe (?), time axis
    
    '''
    fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)

    # each perturbed member of the ensemble
    for pred in means_pert:
        ax.plot(chrono, pred, color='orange', alpha=0.3, zorder=-10)

    ax.plot(chrono, mean_sit, color='orangered', lw=2, label='Ensemble mean')  # SIT mean of ensemble
    ax.plot(chrono, ori_mean_sit, 'k--', lw=2, label='Original inputs')  # SIT without perturbation

    ax.set_ylabel('SIT (m)')
    ax.spines[['right', 'top']].set_visible(False)

    ax.xaxis.grid(alpha=0.6)
    plt.legend(ncol=4, loc='lower left', fontsize=16, ncols=2)
    
    modif_plot.resize(fig, s=24)

    if savefig:
        ofile = f'ypred_pert_all_member_01.png'
        plt.savefig(f"{odir}{ofile}", facecolor='white', dpi=120)
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()


def draw_situ_t(mean_sit, ori_mean_sit, std_t, chrono, odir='', showfig=False, savefig=False):
    '''Quick Plot to visualize mean and std of ensemble
    
       Parameters:
       -----------
       
          mean_sit      :    array, SIT average time series of all members of the ensemble
          ori_mean_sit  :    array, SIT of the initial dataset (without perturbation)
          std_t         :    array, SIT uncertainty (computed from the ensemble of perturbation)
          chrono        :    pd.Dataframe (?), time axis
    '''
    fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)

    ax.plot(chrono, mean_sit, color='orangered', lw=2, label='Ensemble mean', zorder=0)  # SIT mean of ensemble
    
    # Uncertainty of the ensemble
    ax.fill_between(chrono.to_numpy()[:,0], mean_sit - std_t, mean_sit + std_t,
                    color='orange', alpha=0.4, label='Standard deviation', zorder=-10)  # grey


    ax.plot(chrono, ori_mean_sit, 'k--', lw=2, label='Original inputs', zorder=5)  # SIT without perturbation

    ax.set_ylabel('SIT (m)')
    ax.spines[['right', 'top']].set_visible(False)

    ax.xaxis.grid(alpha=0.6)
    plt.legend(ncol=4, loc='lower left', fontsize=16, ncols=2)
    
    modif_plot.resize(fig, s=24)
    
    if savefig:
        ofile = f'ypred_situ_mean_01.png'
        plt.savefig(f"{odir}{ofile}", facecolor='white', dpi=120)
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()