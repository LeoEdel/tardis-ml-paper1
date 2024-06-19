import numpy as np
import copy
import pickle

from sklearn.preprocessing import MinMaxScaler

def scaleuh_3D(X):
    '''To remove
    '''
    X_scaled = copy.deepcopy(X)
    
    scalers = {}
    for n in range(X.shape[0]):
        scalers[n] = MinMaxScaler(feature_range=(-1,1))
        X_scaled[n,:,:] = scalers[n].fit_transform(X_scaled[n,:,:])

    return X_scaled


def scale_fit_transform(X, scalers_file, verbose=1):
    '''
    Fit scalers (one for each variable) and transform the inputs <X> 
    So each variable is between (min, max) [0,1]
    Save a dictionnary containing all scalers in <scalers_file>
    
    Parameters:
    -----------
            X     :     np.array 2D or 3D, inputs, last dimension must be 'variables'
            scalers_file : string, path+file name to the scalers (fitted on training dataset)
    '''
    X_scaled = copy.deepcopy(X)
    
    scalers = {}
    for n in range(X.shape[-1]):
        scalers[n] = MinMaxScaler()
        if X.ndim == 2:
            X_scaled[...,n:n+1] = scalers[n].fit_transform(X_scaled[...,n:n+1]) ## scale inputs
        elif X.ndim == 3:
            X_scaled[:,:,n] = scalers[n].fit_transform(X_scaled[:,:,n])
        else:
            print(f'Inputs should be dimensions 2 or 3. Found {X.ndim}')

#     [:,:,n]  >>> for 3D if does not work
        
        
    with open(scalers_file, 'wb') as handle: ## and save fitted scalers
        pickle.dump(scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if verbose == 1: print(f'\tScalers saved as {scalers_file}')
    
    return X_scaled



def scale_transform(X, scalers_file, verbose=1):
    '''
    Transform the inputs <X> using the fitted Scalers <scalers>
    
    
    Parameters:
    -----------
            X     :     np.array 2D (should work on 3D?), inputs, last dimension must be 'variables'
            scalers_file : string, path+file name to the scalers (fitted on training dataset)    
    '''

    ## Transform from saved MinMaxScaler (fit on training dataset)
    with open(scalers_file, 'rb') as handle:
        scalers = pickle.load(handle)

    if verbose == 1: print(f'\tScalers loaded from {scalers_file}')
    X_scaled = np.empty(X.shape)
    
    for n in range(X.shape[-1]):  ## Transform new values with scalers
        if X.ndim == 2:
            X_scaled[..., n:n+1] = scalers[n].transform(X[..., n:n+1])
        elif X.ndim == 3:
            X_scaled[:,:,n] = scalers[n].transform(X[:,:, n])
            
    
    return X_scaled




def scale_3D(data):
    '''scale each variable btw 0 and 1
    Scale along axis==0
    
    Parameters:
    -----------
        data       :  format numpy array 3D
    '''
    assert len(data.shape) == 3, "Data should be 3D"
    
    data_scaled = data.copy()
    
    for ni in range(data.shape[1]):
        for nj in range(data.shape[2]):
            max_val = data[:, ni, nj].max()
            min_val = data[:, ni, nj].min()
            data_scaled[:, ni, nj] = (data[:, ni, nj] - min_val) / (max_val - min_val)            

        
    return data_scaled


def scale_2D(data):
    '''scale each variable btw 0 and 1 (independently of each other)
    Scale along axis==0
    
    Parameters:
    -----------
        data       :  format numpy array 2D, such as (time, features)
    '''
    assert len(data.shape) == 2, "Data should be a 2D array"
    
    data_scaled = data.copy()
        
    for var in range(data.shape[1]):
        max_val = data[:,var].max()
        min_val = data[:,var].min()
        data_scaled[:,var] = (data[:,var] - min_val) / (max_val - min_val)
        
    return data_scaled


def scale_1D(data):
    '''Scale data btw 0 and 1
    Parameters:
    -----------
        data       :  format numpy array 1D
    '''
    assert len(data.shape) == 1, "Data should be a 1D array"
    
    data_scaled = data.copy()
        
    max_val = data.max()
    min_val = data.min()
    data_scaled = (data - min_val) / (max_val - min_val)
        
    return data_scaled