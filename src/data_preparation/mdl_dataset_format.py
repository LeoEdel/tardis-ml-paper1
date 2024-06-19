# everthing related to dataset reshaping / fomratting to fit for Deep Learning algorithm such as LSTM, CNN, etc.

import numpy as np
import copy

from src.data_preparation import mdl_dataset_prep

def format_ConvLSTM(ds, H):
    '''   
    from -> (alltimes=time serie, features) 
    to   ->> (nsamples=n points, time=time serie, rows=time lag, channels=features)
    '''
    
    
def format_2D_to_3D(arr2D, points):
    '''
       from     (timesteps, features)
            to  (nsample, timesteps, features)
            
    Parameters:
    -----------
    
        arr2D        : xarray.DataArray with shape (time, y, x)
    '''
    
    n_samples = len(points)  # ds.points ?
    ntimes = arr2D.shape[0]
    
    idx_y = [pt[0] for pt in points]
    idx_x = [pt[1] for pt in points]
    
    new_data = arr2D.data[:, idx_y, idx_x]  # fucking shit, dim y and x: -> nb_points **2
        
    # add one dimension for number of feature
    # (n_samples, timesteps, 1)
    return new_data.T[..., None]



def format_CNN_LSTM_fixed(ds, H, train_p=0.8, val_p=0.2):
    """ Formating data to CNN input shape
    from -> (alltimes=samples, features) 
    to   ->> (alltimes, timeslagsgiven, features)
    
    all features have to have the same timesteps: for the dataset to be a 3D array with regular dimensions
    
    returns the true number of timestep used in test dataset
    
    Parameters:
    ------------
    dataset               : class object from mdl_dataset
    H                     : list of integer, unique history for all features
    
    """
    
    new_ds = copy.deepcopy(ds.dataset)  # need deepcopy() to remove links between dictionary and its copy
    old_shape = ds.X.shape
    X = ds.X  # shall remained unchanged for futher formatting to work
#     y = ds.y  # shall remained unchanged
#     ------------------------------------------

    nfeat = X.shape[1]
            
    # If need different timesteps (History)
    needfutur, needpast = 0, 0
    if max(H)>0:
        needfutur = max(H)
    if min(H)<0:
        needpast = abs(min(H))
        
#     import pdb; pdb.set_trace()
                
    # save needfutur and needpast in config
    ds.config.modify_in_yaml('needfutur', needfutur)
    ds.config.modify_in_yaml('needpast', needpast)
    
            
    # Number of data
    n = X.shape[0] # - needpast - needfutur # needpast needfutur already removed from dataset
    
    if 'train' in ds.objective:  #  == 'train':
#         ntrain, nval, ntest = mdl_dataset_prep.dataset_split(n, train_p=train_p, val_p=val_p)
#         print('\n\n\nOLD FUNCTION FOR DATASET_SPLIT\n\n\n')
        ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(n, train_p=train_p, val_p=val_p)

        true_ntest = ntest # - needpast - needfutur
                
    # add history (negative and positive)        
    if len(H) > 1:
        # 3D dataset for LSTM (alltimes,timeslagsgiven, allfeatures) (add several time step in input)
        X2 = np.empty([n, len(H), int(nfeat/len(H))])
        

        # --------------------------------------------------------------------------
        # split time lag and input features into two different dimensions (instead of being concatenated)
    
        lb_pos = np.zeros((len(ds.inputs)))

        for n_xh, xh in enumerate(H):
            for n_lb, lb in enumerate(ds.inputs):
                if f't{xh:+} ' in lb:  # all labels of input feature with 't+0 ' are saved with the same index
                    lb_pos[n_lb] = n_xh
        
        # indexes are attributed to the same row in the reshaped dataset
        for t_idx in range(len(H)):
            X2[:, t_idx, :] = X[:, lb_pos==t_idx]

            
        # --------------------------------------------------------------------------
            
        new_ds['X'] = X2
        
        # already done for training
#         new_ds['y'] = new_ds['y'][needpast:-needfutur]
#         new_ds['chrono'] = new_ds['chrono'][needpast:-needfutur]
      
#     if needfutur == 0:
#         new_ds['y'] = new_ds['y'][needpast:]
#         new_ds['chrono'] = new_ds['chrono'][needpast:]
        
        
#     else:  # something to do ?
#         pass
#         dataset["ytrain"] = Y[ntest+nval+needpast:]
#         dataset['chronotrain'] = chrono[ntest+nval+needpast:-1]
        

        
    # split
    if 'train' in ds.objective:
        new_ds['Xtrain'] = new_ds['X'][ntest+nval:]
        new_ds['Xval'] = new_ds['X'][ntest:ntest+nval]
        new_ds['Xtest'] = new_ds['X'][:ntest]

        new_ds['ytrain'] = new_ds['y'][ntest+nval:]
        new_ds['yval'] = new_ds['y'][ntest:ntest+nval]
        new_ds['ytest'] = new_ds['y'][:ntest]
        
        new_ds['chrono_train'] = new_ds['chrono'][ntest+nval:]
        new_ds['chrono_val'] = new_ds['chrono'][ntest:ntest+nval]
        new_ds['chrono_test'] = new_ds['chrono'][:ntest]
    
        new_ds['ntrain'] = ntrain
        new_ds['nval'] = nval
        new_ds['ntest'] = ntest
    
    # attribute new variables to dataset  
    ds.dataset = new_ds
    
    if ds.config.verbose == 1:
        print(f"Dataset formatted from {old_shape} to {new_ds['X'].shape} for H={H}.")

    if 'train' in ds.objective:
        return true_ntest

def format_CNN_LSTM(ds, H, train_p=0.8, val_p=0.2):
    """ Formating data to CNN input shape
    from -> (alltimes=samples, features) 
    to   ->> (alltimes, timeslagsgiven, features)
    
    all features have to have the same timesteps: for the dataset to be a 3D array with regular dimensions
    
    returns the true number of timestep used in test dataset
    
    Parameters:
    ------------
    dataset               : class object from mdl_dataset
    H                     : list of integer, unique history for all features
    
    """
    
    new_ds = copy.deepcopy(ds.dataset)  # need deepcopy() to remove links between dictionary and its copy
    old_shape = ds.X.shape
    X = ds.X  # shall remained unchanged for futher formatting to work
#     y = ds.y  # shall remained unchanged
#     ------------------------------------------

    nfeat = X.shape[1]
            
    # If need different timesteps (History)
    needfutur, needpast = 0, 0
    if max(H)>0:
        needfutur = max(H)
    if min(H)<0:
        needpast = abs(min(H))
        
    import pdb; pdb.set_trace()
                
    # save needfutur and needpast in config
    ds.config.modify_in_yaml('needfutur', needfutur)
    ds.config.modify_in_yaml('needpast', needpast)
    
            
    # Number of data
    n = X.shape[0] - needpast - needfutur
    
    if 'train' in ds.objective:  #  == 'train':
#         ntrain, nval, ntest = mdl_dataset_prep.dataset_split(n, train_p=train_p, val_p=val_p)
#         print('\n\n\nOLD FUNCTION FOR DATASET_SPLIT\n\n\n')
        ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(n, train_p=train_p, val_p=val_p)

        true_ntest = ntest - needpast - needfutur
        
    # add history (negative and positive)        
    if len(H) > 1:
        # 3D dataset for LSTM (alltimes,timeslagsgiven, allfeatures) (add several time step in input)
        X2 = np.empty([n, len(H), nfeat])
        for time in range(n):
            for i, ts in enumerate(H):
                X2[time, i] = X[needpast+time+ts]

                        
        new_ds['X'] = X2  # np.reshape(X2, [n, len(H), nfeat])
        new_ds['y'] = new_ds['y'][needpast:-needfutur]
        new_ds['chrono'] = new_ds['chrono'][needpast:-needfutur]
      
    if needfutur == 0:
        new_ds['y'] = new_ds['y'][needpast:]
        new_ds['chrono'] = new_ds['chrono'][needpast:]
        
        
#     else:  # something to do ?
#         pass
#         dataset["ytrain"] = Y[ntest+nval+needpast:]
#         dataset['chronotrain'] = chrono[ntest+nval+needpast:-1]
        

        
    # split
    if 'train' in ds.objective:
        new_ds['Xtrain'] = new_ds['X'][ntest+nval:]
        new_ds['Xval'] = new_ds['X'][ntest:ntest+nval]
        new_ds['Xtest'] = new_ds['X'][:ntest]

        new_ds['ytrain'] = new_ds['y'][ntest+nval:]
        new_ds['yval'] = new_ds['y'][ntest:ntest+nval]
        new_ds['ytest'] = new_ds['y'][:ntest]
        
        new_ds['chrono_train'] = new_ds['chrono'][ntest+nval:]
        new_ds['chrono_val'] = new_ds['chrono'][ntest:ntest+nval]
        new_ds['chrono_test'] = new_ds['chrono'][:ntest]
    
        new_ds['ntrain'] = ntrain
        new_ds['nval'] = nval
        new_ds['ntest'] = ntest
    
    # attribute new variables to dataset  
    ds.dataset = new_ds
    
    if ds.config.verbose == 1:
        print(f"Dataset formatted from {old_shape} to {new_ds['X'].shape} for H={H}.")

    if 'train' in ds.objective:
        return true_ntest

def build_dataset_1pt(X, Y, chrono, train_p=.65, val_p=0.15, times=[0]):
    """ Construct the CNN dataset:
        - split train, val and test
        - formating data to CNN input shape
    
    
    Parameters:
    ------------
    
    
        ntvt            :  tuple containing Number of timestep for Train, Validation, Test period
    
    In: 
        point       : int list           --  (i, j) coordinates of the point for which we want to calculate the dataset
        cov_sel     : numpy.ndarray dict -- dict containing (t, y, x) data for each choosen covariables. Need to keep ilim/jlim values for x:y!
        fgs         : numpy.ndarray dict -- dict containing (t, y, x) data for each forcings (supposed to be already scale)
        Y           : numpy.ndarray      -- (t), error time serie for the chosen point
        chrono      : 
        times       : int list           -- allows to include different time steps in an input data (for LSTM). Each element corresponds to the time lag to be given
                                            - if times==[0] (DNN mode) : shape X data (n, nb_param)
                                            - else          (LSTM mode): shape X data (n, len(times), nb_param) -> the number of data is reduced here because the extreme data do not have the necessary lag data and are therefore removed
        
    Out:
        dataset            : dict            -- NN data (Y: wanted output / X: given input)
                                                Xtest, ytest : test data 
                                                Xtrain, ytrain : train data
                                            
        (n, ntrain, ntest) : (int, int, int) -- size of the differents dataset (n=ntrain+ntest)
        rchrono            :                 -- len(rchrono) = n, times associated to the dataset (usefull for LSTM as it's cut)

    """
    
    
    
#     keep_cov = [c for c in cov_sel.keys() if c not in exclude_cov]
    
#     t = cov_sel[keep_cov[0]].shape[0]
#     nb_pts = pow((size_around)*2 + 1, 2)
    
#     if size_around == 0:
#         nb_params = len(keep_cov) + len(keep_fg)
#     else:
#         nb_params = (len(keep_cov) + len(keep_fg))*nb_pts
    
    nb_params = X.shape[1]
            
    # If need differents times
    needfutur, needpast = 0, 0
    if max(times)>0:
        needfutur = max(times)
    if min(times)<0:
        needpast = abs(min(times))
        
        
#     print(needfutur, needpast)
    # Number of data
    n = X.shape[0] - needpast - needfutur
#     print('n', n)

        
        
    if len(times) > 1:
        # 3D dataset for LSTM (alltimes,timeslagsgiven, allfeatures) (add several time step in input)
        X2 = np.empty([n, len(times), nb_params])
        for time in range(n):
            for i, ts in enumerate(times):
                X2[time, i] = X[needpast+time+ts]
        X = X2

   
    ntrain, nval, ntest = mdl_dataset_prep.dataset_split(n)
#     print(ntrain, nval, ntest)

    # Split dataset, train with older years
    # X = (t1_pt1,  ..., tn_pt1, t1_pt2, ..., tn_ptn)
    dataset = dict()
    Xtrain = X[ntest+nval:]
    Xval = X[ntest:ntest+nval]
    Xtest = X[:ntest]

    dataset['yval'] = Y[ntest+needpast:ntest+nval+needpast]
    dataset["ytest"] = Y[needpast:ntest+needpast]
    
    if len(times) > 1:
        Xtrain = np.reshape(Xtrain, [ntrain, len(times), nb_params])
        Xval = np.reshape(Xval, [nval, len(times), nb_params])
        Xtest = np.reshape(Xtest, [ntest, len(times), nb_params])
#         import pdb; pdb.set_trace()
        dataset["ytrain"] = Y[ntest+nval+needpast:-needfutur-1]  # Y[ntest+nval+needpast:-needfutur]
#         dataset["yval"] = Y[ntest+needpast:ntest+nval+needpast]
        dataset['chronotrain'] = chrono[ntest+nval+needpast:-needfutur-1]

        

    else:
        dataset["ytrain"] = Y[ntest+nval+needpast:]
        dataset['chronotrain'] = chrono[ntest+nval+needpast:-1]
        
#         dataset["yval"] = Y[ntest:ntest+nval]

    
    dataset['Xtrain'] = Xtrain 
    dataset['Xval'] = Xval
    dataset['Xtest'] = Xtest
    
    
    # Add noise to avoid overfitting
    # Xtrain = Xtrain + np.random.normal(0, np.std(Xtrain) / 1000, Xtrain.shape)
    # dataset["Xtrain"], dataset["Xtest"] = scale_data(Xtrain, Xtest)
#    dataset["Xtrain"], dataset["Xtest"] = Xtrain, Xtest

# scaling done before
#     dataset["Xtrain"], dataset["Xtest"] = scale_data_var(Xtrain, Xtest)
          
    rchrono = chrono[needpast:-needfutur] if needfutur > 0 else chrono[needpast:]
    
    dataset['chrono'] = rchrono
#     dataset['chronotrain'] = chrono[ntest+nval+needpast:-needfutur-1]
    dataset['chronoval'] = chrono[ntest+needpast:ntest+nval+needpast]
    dataset['chronotest'] = chrono[needpast:ntest+needpast]
    
    dataset['ntrain'] = ntrain
    dataset['nval'] = nval
    dataset['ntest'] = ntest
    
    
    
    return dataset, (n, ntrain, ntest), rchrono

    # in near future:
    # return dataset
