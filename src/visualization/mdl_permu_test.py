# draw permutation test

import matplotlib.pyplot as plt
import numpy as np
from src.utils import save_name


def draw_permu_test(model, max_plot=4, savefig=False, showfig=True):
    '''draw permutation importance test for all PC (for the same PC for other variables)
    computed from: model.permu_imp_test()
    
    In:
        model    :  ModelML class
    
    
    '''

    if max_plot > model.ytrain.shape[1]:
        max_plot = model.ytrain.shape[1]

    fig, ax = plt.subplots(ncols=1, nrows=max_plot, figsize=(8,6*max_plot))

    for Nf in range(max_plot):
        result = model.dict_imp[f'pc{Nf}']
        sorted_idx = result.importances_mean[Nf::model.nfeat].argsort()

        ax[Nf].plot([0,0], [1,len(sorted_idx)], '--k', alpha=.6)

        ax[Nf].boxplot(result.importances[Nf::model.nfeat][sorted_idx].T,
            vert=False,
            labels=np.array(model.totlabels[::model.nfeat])[sorted_idx])
#                 labels=np.array(self.totlabels[::self.nfeat])[sorted_idx])


        ax[Nf].set_xlabel('Permutation Importance')
        ax[Nf].set_title(f"PC{Nf}")

    fig.tight_layout()

    if savefig:
        filename = f'Permutation_importance_testset_{model.type}.png'
        ofile = save_name.check(f"{model.rootdir}{model.fig_dir}", filename)
        plt.savefig(f"{model.rootdir}{model.fig_dir}{ofile}")
        print(f'Saved as: {model.rootdir}{model.fig_dir}{ofile}')
        

    if showfig:
        plt.show()

    plt.close()
    
    
    
    
def draw_permu_1pc(model, i_pca=0, train=False, savefig=True, showfig=False):
    '''draw permutation importance test for one PC (all the PC for other variables)
    computed from: model.permu_imp_test()
    
    In:
        model    :  ModelML class
        i_pca    :  target var PCA to use  [0 : n_components]
    
    
    '''


#     full_label = [f'{item} PC{n%model.nfeat}' for n,item in enumerate(model.totlabels)] 
    full_label = [f'{item}' for n,item in enumerate(model.input_label[f'pc{i_pca}'])] 
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,60))

    result = model.dict_imp[f'pc{i_pca}']
    if train:
        result = model.dict_imp_train[f'pc{i_pca}']
        
    sorted_idx = result.importances_mean.argsort()

    ax.plot([0,0], [1,len(sorted_idx)], '--k', alpha=.6)

    ax.boxplot(result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(full_label)[sorted_idx])
#                 labels=np.array(self.totlabels[::self.nfeat])[sorted_idx])


    ax.set_xlabel('Permutation Importance')
    ax.set_title(f"PC{i_pca}")

    fig.tight_layout()

    if savefig:
        filename = f'PermuImp_testset_{model.type}_PC{i_pca}.png'
        if train:
            filename = f'PermuImp_trainset_{model.type}_PC{i_pca}.png'
            
        ofile = save_name.check(f"{model.rootdir}{model.fig_dir}", filename)
        plt.savefig(f"{model.rootdir}{model.fig_dir}{ofile}")
        print(f'Saved as: {model.rootdir}{model.fig_dir}{ofile}')

    if showfig:
        plt.show()

    plt.close()