# plot figures for recursive SEQUENCE  prediction of model of machine learning

import matplotlib.pyplot as plt
import numpy as np
from src.utils import save_name

def draw_seq(model, max_plot=4, odir='', savefig=True, showfig=False, force=False):
    '''plot several sequences over the total number of sequence
    
    model, data classe ModelML
    odir, output folder, string path containing location for save .png
    '''

    if model.pred_type != 'sequence recursive':
        print(f'Values predicted with following method: {model.pred_type}.\n\
            Use corresponding plotting function to visualize results.')
        if not force:
            return
        
    
    if max_plot > model.npca: max_plot = model.npca
    
    fig, ax = plt.subplots(ncols=1, nrows=max_plot,sharex='col', figsize=(12,max_plot*3))

    for i in range(max_plot):
        ax[i].plot(model.chrono[:-1], model.y[::-1,i], label='target')
        for it in range(model.nseq, model.nsample-model.nseq, 50):
            for nt in range(model.nseq):
                ax[i].plot(np.array(model.chrono.iloc[:][it-model.nseq:it]), model.yseq[::-1,:,::-1][it,i,:], 'orange')
    
                
        mini, maxi = model.y[:,i].min(), model.y[:,i].max()
        ax[i].plot([model.chrono.iloc[model.ntest], model.chrono.iloc[model.ntest]], [mini, maxi],':k',label='train limit')

    ax[0].legend()
#     plt.suptitle(f'forcing mean_{config["forcing_mean_days"]}d')
    
    if showfig:
        plt.show()
    
    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return
        filename = f'ML_multicurves_recursive_sequence.png'
        ofile = save_name.check(f"{odir}", filename)
        plt.savefig(f"{odir}{ofile}")
        print(f'Saved as: {odir}{ofile}')
        
    plt.close()

    
    
def draw(model, max_plot=4, odir='', savefig=True, showfig=False, force=False):
    '''plot prediction averaged over all sequences
    
    model, data classe ModelML
    odir, output folder, string path containing location for save .png
    '''
    if model.pred_type != 'sequence recursive':
        print(f'Values predicted with following method: {model.pred_type}.\n\
            Use corresponding plotting function to visualize results.')
        if not force:
            return
    
    
    if max_plot > model.npca: max_plot = model.npca
    
    nseq = model.nseq  # pour plus facile a lire
    
    fig, ax = plt.subplots(ncols=1, nrows=max_plot,sharex='col', figsize=(12,max_plot*3))

    for i in range(max_plot):
        ax[i].plot(model.chrono[:-1], model.y[::-1,i], label='target')
    
        ax[i].plot(model.chrono[:-1], model.yms[::-1,i], label=f'pred mean seq{nseq}')
        ax[i].plot(model.chrono[:-1], model.yms[::-1,i]+model.yms_ds[::-1,i], c='grey', alpha = .3, label=f'Std deviation')
        ax[i].plot(model.chrono[:-1], model.yms[::-1,i]-model.yms_ds[::-1,i], c='grey', alpha = .3)
        

        mini, maxi = model.y[:,i].min(), model.y[:,i].max()
        ax[i].plot([model.chrono.iloc[model.ntest], model.chrono.iloc[model.ntest]], [mini, maxi],':k',label='train limit')
        ax[i].set_ylabel(f'PC{i}')
        
    ax[0].legend()
#     plt.suptitle(f'forcing mean_{config["forcing_mean_days"]}d')

    if savefig:
        if odir == '':
            print('Please specify folder for save .png')
            return
        filename = f'ML_prediction_recursive_sequence.png'
        ofile = save_name.check(f"{odir}", filename)
        plt.savefig(f"{odir}{ofile}")
        print(f'Saved as: {odir}{ofile}')
        
    if showfig:
        plt.show()
    
    plt.close()    
    