# plot figures for recursive prediction of model of machine learning

import matplotlib.pyplot as plt
from src.utils import save_name

def draw(model, max_plot=8, savefig=False, showfig=True, force=False):
    '''model, data classe ModelML
    '''
    
    if model.pred_type != 'recursive':
        print(f'Values predicted with following method: {model.pred_type}.\n\
                Use corresponding plotting function to visualize results')
        if not force:
            return
    
    if max_plot > model.npca: max_plot = model.npca
    
    fig, ax = plt.subplots(ncols=1, nrows=max_plot,sharex='col', figsize=(12,max_plot*3))

    for i in range(max_plot):
        ax[i].plot(model.chrono[:], model.ytrue[:,i], label='true')
        ax[i].plot(model.chrono[:], model.ypred[:,i], label='pred recursive')  # chrono[:-1]
        mini, maxi = model.ytrue[:,i].min(), model.ytrue[:,i].max()
        ax[i].plot([model.chrono.iloc[model.ntest], model.chrono.iloc[model.ntest]], [mini, maxi],':k',label='train limit')
#        ax[i].plot([model.chrono[model.ntest], model.chrono[model.ntest]], [mini, maxi],':k',label='train limit')
        ax[i].set_ylabel(f'PC{i}')
#         ax[i].set_ylim([-100,100])
        
    ax[0].legend()
#     plt.suptitle(f'forcing mean_{config["forcing_mean_days"]}d')
   
    if savefig:
        if odir == '':
            odir = model.rootdir + model.fig_dir
#             print('Please specify folder for save .png')
#             return
        filename = f'ML_prediction_recursive.png'
        ofile = save_name.check(f"{odir}", filename)
        plt.savefig(f"{odir}{ofile}")
        print(f'Saved as: {odir}{ofile}')

    if showfig:
        plt.show()
        
        
    plt.close()
