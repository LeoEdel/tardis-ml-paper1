

# differents plots that aim to deeply understand what's going on inside ML prediction (GradientBoostRegressor)

import matplotlib.pyplot as plt
from src.utils import save_name
from skopt.plots import plot_objective
import numpy as np

def plot_cvscore_split(model, odir='', savefig=False, showfig=False):
    '''Plot score (r2) for different TimeSeriesSplit, different PCA
    '''
    nrow = model.nsplit
    ncol = model.npca
    fig, axes = plt.subplots(ncols=ncol, nrows=nrow, figsize=(16, 9),sharex='col') # , sharey='row')

    for tt in range(nrow):
    #     print(tt)
        axes[tt][0].set_ylabel(f'TS{tt}')
        for ipca in range(ncol):
            n_search = len(model.dict_grid[f'pc{ipca}'].cv_results_[f'split{tt}_test_score'])
            
            axes[tt][ipca].plot([0,n_search],[0,0],'k--')
            axes[0][ipca].set_title(f'PC{ipca}')           
            axes[-1][ipca].set_xlabel(f'n_iter')

            axes[tt][ipca].plot(model.dict_grid[f'pc{ipca}'].cv_results_[f'split{tt}_train_score'],'.', label='train')
            axes[tt][ipca].plot(model.dict_grid[f'pc{ipca}'].cv_results_[f'split{tt}_test_score'],'.', label='test')
            
#             xmin = np.floor(np.min(model.dict_grid[f'pc{ipca}'].cv_results_[f'split{tt}_train_score']))
#             if xmin < -10: xmin = -10
#             xmin = -10

            if tt == 0 and ipca == 0:
                axes[tt][ipca].legend()

            axes[tt][ipca].set_ylim([-1.5,1.5])


    fig.suptitle('r2 - Bayesian search')
    

    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return
        filename = f'ML_TSsplits_score.png'
        ofile = save_name.check(f"{odir}", filename)
        plt.savefig(f"{odir}{ofile}")
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()

    plt.close()

    
    
    
def plot_cvscore_nboosting_iter(model, odir='', savefig=False, showfig=False):
    '''Plot score (r2) for best_estimator_ as function of the boosting iterations (number tree used in GradientBoostingRegressor)
    one plot per PCA
    '''
    
    for ipca in range(model.npca):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 9))

        ax.plot(model.r2_train[f'pc{ipca}'][:], 'b')
        ax.set_ylabel('r2 train')
        ax.set_xlabel('boosting iterations')

        ax.spines['left'].set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.tick_params(axis='y', colors='blue')

        ax2 = ax.twinx()
        ax2.plot(model.r2_test[f'pc{ipca}'][:], 'red')
        ax2.set_ylabel('r2 test')
        ax2.spines['right'].set_color('red')
        ax2.yaxis.label.set_color('red')
        ax2.tick_params(axis='y', colors='red')

        if savefig:
            if odir == '':
                print('Please specify folder for save .png')
                return
            filename = f'ML_nBoostingOp_score_pc{ipca}.png'
            ofile = save_name.check(f"{odir}", filename)
            plt.savefig(f"{odir}{ofile}")
            print(f'Saved as: {odir}{ofile}')

        if showfig:
            plt.show()

        plt.close()
        
        
        
        
        
        
def plot_optimizer_full(model, sample_source='random', odir='', savefig=False, showfig=False):
    '''Plot in depth results of the Bayesian search
    
    In:
        sample_source    :   ‘random’ - n_samples random samples will used, a partial dependence plot will be generated
                             ‘result’ - Use only the best observed parameters
    '''
    
    for ipca in range(model.npca):
        dim = [f'{item.name[27:]}' for item in model.dict_grid[f'pc{ipca}'].optimizer_results_[0].space.dimensions]
        _ = plot_objective(model.dict_grid[f'pc{ipca}'].optimizer_results_[0],
                          dimensions=dim,
                          sample_source=sample_source,
                          size = 4  
                          )

        plt.tight_layout()
        
        
        if savefig:
            if odir == '':
                print('Please specify folder for save .png')
                return
            filename = f'BayesianSearchCV_dependence_pc{ipca}.png'
            ofile = save_name.check(f"{odir}", filename)
            plt.savefig(f"{odir}{ofile}")
            print(f'Saved as: {odir}{ofile}')

        if showfig:
            plt.show()

        plt.close()
        
        
        
#         odir = f'{m1.rootdir}{m1.fig_dir}'
#         filename = f'BayesianSearchCV_dependence_pc{ipca}.png'


    
    