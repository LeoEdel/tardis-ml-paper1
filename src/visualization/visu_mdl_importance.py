import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.colors as mcolors
from src.utils import save_name


    
    
    
def plot_importance_var_allpca(model, showfig=False, savefig=True):
    '''2D plot fo importance for all target var PCA at the same time
    
    '''
#     from matplotlib import colors  # cbar lognormal (=ugly)
    n_comp = model.ytrain.shape[1]
    
    imp = np.zeros(shape=(len(model.totlabels), n_comp)) * np.nan
#     full_label = [f'{item} PC{n%model.npca}'      for n,item in enumerate(model.totlabels)] 
    pc_label = [f'PC{n}' for n in range(n_comp)]

    fig, ax = plt.subplots(figsize=(16,60))

    # same that the part commented above but prettier. remove above if working well
    for i_pca in range(n_comp):  # loop on pca
        if model.type == 'xgb':
            temp_imp = model.dict_grid[f'pc{i_pca}'].best_estimator_['xgbregressor'].feature_importances_
        elif model.type == 'grdbst':
            temp_imp = model.dict_grid[f'pc{i_pca}'].best_estimator_['gradientboostingregressor'].feature_importances_
        elif model.type == 'rf':
            temp_imp = model.dict_grid[f'pc{i_pca}'].best_estimator_['randomforestregressor'].feature_importances_
        elif model.type == 'ridge':
            temp_imp = model.dict_grid[f'pc{i_pca}'].best_estimator_['ridge'].coef_[:]
            
        if model.input_dico == '':  # if all variables are included in PCA
            idx_valid = range(len(model.totlabels))
        else:  # differentes variables as function of the PCA
            idx_valid = np.where(model.input_dico[f'pc{i_pca}'][:,1]=='True')
            
        imp[idx_valid, i_pca] = temp_imp
            
    c = ax.imshow(imp, cmap=plt.cm.hot.reversed(), interpolation='None', aspect='auto')  # , norm=colors.LogNorm())
    # plt.cm.hot.reversed()
    
    # annotate importance
    for (j,i),label in np.ndenumerate(imp):
        if j%n_comp == 0:  # draw line to help reading the table
            ax.plot([0-.5,n_comp-.5],[j-.5]*2,'grey', lw=.2)
        if label >= 0.001:  # if value > threshold, annotate
            cl = 'k'
            if label > .5 and model.type != 'ridge':
                cl = 'white'
            elif label > 50 and model.type == 'ridge':
                cl = 'white'
            ax.text(i,j,f'{label:0.3f}',ha='center',va='center', color=cl)
        elif label < 0.001 and not np.isnan(label):
            ax.text(i,j,f'.',ha='center',va='center')
#         if np.isnan(label):  # annotate nan
#             ax.text(i,j,f'NaN',ha='center',va='center')
            
    
    
    cbar_ax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 * .90, ax.get_position().width, 0.005])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', ticks=[0,.5,1])
    cbar.set_label('Importance')
    
    ax.set_yticks(range(len(model.full_label)), model.full_label)
    ax.set_xticks(range(n_comp), pc_label, rotation= -25)
    
    # get colorbar
    
    if savefig:
        filename = f'importance_allPC.png'
        ofile = save_name.check(f"{model.rootdir}{model.fig_dir}", filename)
        plt.savefig(f"{model.rootdir}{model.fig_dir}{ofile}", facecolor='white')
        print(f'Saved as: {model.rootdir}{model.fig_dir}{ofile}')
        
    if showfig:
        plt.show()
        
    plt.close()
    
    
#    ----------------------------------------
#                    to delete 
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    
def plot_importance_var(model, i_pca, showfig=True, savefig=False):
    '''Plot variables importances
    I_pca, int, index PCA to draw
    '''
    print('Function not up to date - do not use')
    return
    
    fig, ax = plt.subplots(figsize=(12,6))
#     feat_imp = get_bg_feat_importances(model.dict_grid, i_pca, model.Xtrain.shape[1], model.npca)
    
    if model.type == 'grdbst':
        feat_imp = model.dict_grid[f'pc{i_pca}'].best_estimator_['gradientboostingregressor'].feature_importances_
    elif model.type == 'rf':
        feat_imp = model.grid.best_estimator_['randomforestregressor'].feature_importances_
    elif model.type == 'ridge':
#         feat_imp = model.grid.best_estimator_['ridge'].coef_[i_pca,:]  # old
         feat_imp = model.dict_grid[f'pc{i_pca}'].best_estimator_['ridge'].coef_[:]
    
    plt.plot(feat_imp)
    plt.xticks(range(0,len(model.totlabels),model.npca),model.totlabels[::model.npca], rotation= -25)
    plt.ylabel('Importance')

    if savefig:
        filename = f'importance_PC{i_pca}.png'
        plt.savefig(f"{model.rootdir}{model.fig_dir}{filename}", facecolor='white')
        print(f'Saved as: {model.rootdir}{model.fig_dir}{filename}')
        
    if showfig:
        plt.show()
        
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    