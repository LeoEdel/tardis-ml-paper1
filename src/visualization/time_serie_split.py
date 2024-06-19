# plot for TimeSerieSplit object

import matplotlib.pyplot as plt
import numpy as np


def PlotTimeSerieSplit(ts, X, chrono, ofolder, savefig=False, showfig=False):
    '''
    Plot nSplit used by the machine learning for evaluation
    
    
    
    ts, TimeSeriesSplit object from sklearn.model_selection._split.TimeSeriesSplit
    X, np.ndarray (time, features)
    chrono, pandas.core.series.Series (time,)
    
    ofolder, output folder, string containing path to save .png
    
    '''

    max_plot = ts.n_splits
    fig, ax = plt.subplots(ncols=1, nrows=max_plot,sharex='col', figsize=(6,max_plot*3))

    ipca = 0  # PCA to show
    gmin = np.nanmin(X[:,ipca])
    gmax = np.nanmax(X[:,ipca])
    
    i = 0
    for itr, itt in ts.split(X):
        ax[i].plot(chrono[itr[0]:itr[-1]], X[itr[0]:itr[-1], ipca], label='train')
        ax[i].plot(chrono[itt[0]:itt[-1]], X[itt[0]:itt[-1], ipca], label='test')
        ax[i].set_ylabel(f'CV {i+1}')
        i += 1
        
    for axx in ax:
        axx.set_ylim([gmin, gmax])
        
    ax[0].legend()
    plt.xticks(rotation= -25)
    
    if showfig:
        plt.show()
        
    if savefig:
        filename = f'ML_TimeSerieSplit.png'
        plt.savefig(f"{ofolder}{filename}")
        print(f'Png saved as : {ofolder}{filename}')

    plt.close()