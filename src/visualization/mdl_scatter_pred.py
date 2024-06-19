# plot scatter prediction value as function of true value

import matplotlib.pyplot as plt
import numpy as np
from src.utils import save_name

def draw_non_recursive(model, i_pca=0, savefig=False, showfig=True):
    '''Draw prediction value as a function of true value for ypredict (simple prediction)
    
    model, class ModelML
    '''
    
    mini = np.floor(np.min([model.ytrue[:,i_pca], model.ypredict[:,i_pca]]))
    maxi = np.ceil(np.max([model.ytrue[:,i_pca], model.ypredict[:,i_pca]]))
    
    plt.plot([mini, maxi], [mini, maxi], '--', c='grey')
    plt.plot(model.ytrue[:model.ntrain,i_pca],model.ypredict[:model.ntrain,i_pca],'.b', label='train')
    plt.plot(model.ytrue[model.ntrain:,i_pca],model.ypredict[model.ntrain:,i_pca],'.r', label='val')
    plt.legend()
    plt.xlabel('True value')
    plt.ylabel('Pred. value');
    plt.title('one step prediction scatter plot')
    
    if showfig:
        plt.show()
    if savefig:
        print('todo')
    plt.close()
    
    
def draw_recursive(model, i_pca=0, savefig=False, showfig=True):
    '''Draw recurisve prediction value as a function of true value for ypred (recursive prediction)
    
    model, class ModelML
    '''
    
    if not 'rec' in model.pred_type:
        print('Recursive prediction missing. Use ModelML.recursive_prediction()')
        return
    
    mini = np.floor(np.min([model.ytrue[:,i_pca], model.ypred[:,i_pca]]))
    maxi = np.ceil(np.max([model.ytrue[:,i_pca], model.ypred[:,i_pca]]))
    
    plt.plot([mini, maxi], [mini, maxi], '--', c='grey')
    plt.plot(model.ytrue[:model.ntrain,i_pca], model.ypred[:model.ntrain,i_pca], '.b', label='train')
    plt.plot(model.ytrue[model.ntrain:,i_pca], model.ypred[model.ntrain:,i_pca], '.r', label='val')
    plt.legend()
    plt.xlabel('True value')
    plt.ylabel('Pred. value');
    plt.title('one step prediction recursive scatter plot')
    
    if showfig:
        plt.show()
    if savefig:
        print('todo')
    plt.close()
    
    
    
def draw_recursive_allfeat(model, savefig=True, showfig=False):
    '''Draw recurisve prediction value as a function of true value for ypred (recursive prediction)
    for all the features
    
    model, class ModelML
    '''
    
#     if not 'rec' in model.pred_type:
#         print('Recursive prediction missing. Use ModelML.recursive_prediction()')
#         return
    
    colors_red = plt.cm.Reds.reversed()(np.linspace(0,1,model.npca))
    colors_blue = plt.cm.Blues.reversed()(np.linspace(0,1,model.npca))
#     colors_blue = ['b']*model.npca
#     colors_red = ['r']*model.npca
    
    
    mini = np.floor(np.min([model.ytrue[:,:], model.ypred[:,:]]))
    maxi = np.ceil(np.max([model.ytrue[:,:], model.ypred[:,:]]))
    plt.axes().set_facecolor('whitesmoke')
    
    plt.plot([mini, maxi], [mini, maxi], '--', c='grey')
    
    for i_pca in range(model.npca)[::-1]:
        if i_pca == 0:
            plt.plot(model.ytrue[:model.ntrain,i_pca], model.ypred[:model.ntrain,i_pca], '.', c=colors_blue[i_pca], label='train')
            plt.plot(model.ytrue[model.ntrain:,i_pca], model.ypred[model.ntrain:,i_pca], '.', c=colors_red[i_pca], label='val')

        plt.plot(model.ytrue[:model.ntrain,i_pca], model.ypred[:model.ntrain,i_pca], '.', c=colors_blue[i_pca])  #, alpha=1-0.1*i_pca)
        plt.plot(model.ytrue[model.ntrain:,i_pca], model.ypred[model.ntrain:,i_pca], '.', c=colors_red[i_pca])  # , alpha=1-0.1*i_pca)
        
    
    
    
    plt.legend()
    plt.xlabel('True value')
    plt.ylabel('Pred. value');
    plt.title('one step prediction recursive scatter plot')
    

    if savefig:
        filename = f'scatter_recursive_allfeat.png'
        ofile = save_name.check(f"{model.rootdir}{model.fig_dir}", filename)
        plt.savefig(f"{model.rootdir}{model.fig_dir}{ofile}")
        print(f'Saved as: {model.rootdir}{model.fig_dir}{ofile}')
        
    if showfig:
        plt.show()
        
    plt.close()
    
