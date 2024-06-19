# machine learning database preparation
import numpy as np
import copy




def scale_data_var(data):
    '''scale each variable btw 0 and 1
    
    for data: format (time,  features)
    '''
    data_scaled = data.copy()
        
    for var in range(data.shape[1]):
        max_val = data[:,var].max()
        min_val = data[:,var].min()
        data_scaled[:,var] = (data[:,var] - min_val) / (max_val - min_val)
        
    return data_scaled




def data_to_dataset(data, history, invert_t=False):
    '''TO BE REMOVED SOON
    
    from PCA to (X, y)
    
    duplicate and shift variables depending on history parameters
    '''
    
    # -------- data --------
    
    Xf = data[0] 
    PCs_f = data[1]
    Xe = data[2]
    PCs_e = data[3]
    PCs_co = data[4]
    PCs_fo = data[5]
    dsCo = data[6]
    dsFo = data[7]
    chrono = data[8]
    
    
    # -------- history --------

    H = history['bias']['H']
    H_neg = history['bias']['H_neg']
    
    Xtp, nhypfeat, hyplabels = target_history(PCs_e.values, H, H_neg, f'Xe', False)
    
    Hnc = history['noass']['H']
    Hnc_neg = history['noass']['H_neg']
    Xnc, _, nclabels = target_history(PCs_f.values, Hnc, Hnc_neg, 'SITf')

    
    # if 'H' not in history['forcing'] and 'H_neg' not in history['forcing']:
    fo_hist = make_fo_history(history['forcing']['names'], 
                              history['forcing']['to_explore'], 
                              must_be=history['forcing']['must_be'],
                              interval=history['forcing']['interval'],
                              nexplo=history['forcing']['nexplo'])
    Xfo, folabels, fohyperfeat = forcing_history(history['forcing']['names'], PCs_fo, fo_hist)
    
    print(fo_hist)
    
    
    co_hist = make_fo_history(history['covar']['names'], history['covar']['to_explore'], must_be=history['covar']['must_be'],interval=history['covar']['interval'], nexplo=history['covar']['nexplo'])
    Xco, colabels, cohyperfeat = forcing_history(history['covar']['names'], PCs_co, co_hist)
    
    #  resize bcz same size for concatenate
#     import pdb; pdb.set_trace()
#     nresize = 2920
#     Xtp = Xtp[-nresize:]
#     Xnc = Xnc[-nresize:]
#     Xfo = Xfo[-nresize:]
#     Xco = Xco[-nresize:]
    
    # add noise on Xtp
    Xtp, _ = add_noise(Xtp, Xtp.shape[1], pert=1)
    
    # -------- assemble final dataset --------
    
    # without covariables
    # X = np.concatenate((Xtp, Xnc, Xfo),axis=1)
    # totlabels = hyplabels + nclabels + folabels

    # with covariables
    X = np.concatenate((Xtp, Xnc, Xfo, Xco),axis=1)
    totlabels = hyplabels + nclabels + folabels + colabels

    # without Xtp
    # X = np.concatenate((Xnc, Xfo, Xco),axis=1)
    # totlabels = nclabels + folabels + colabels

    # without forcing
    # X = np.concatenate((Xtp, Xnc),axis=1)
    # totlabels = hyplabels + nclabels

    # everything + forcing without PCA
    # X = np.concatenate((Xtp, Xnc, Xfo, Xco, DSco, DSfo),axis=1)
    # pca_labels = hyplabels + nclabels + folabels + colabels 
    # totlabels = hyplabels + nclabels + folabels + colabels + clabels + flabels



    y = PCs_e.values[:-1, :]  # target var t
    chrono = chrono[:-1]
    
    # resize (fait a la main pour l'instant)
    # y = y[-nresize:]
    # chrono = chrono[-nresize:]
    
    if invert_t:
        return X[::-1], y[::-1], chrono[::-1], totlabels
    
    return X, y, chrono, totlabels
    
    
def ds_split_random(n, train_p=.65, val_p=0.15):
    '''Return 3 array of indexes for train, evaluation and test
    '''

    ntrain = int(n*train_p)
    nval = int(n*val_p)
    ntest = int(n - ntrain - nval)
    
    # array of bool
    atra = np.zeros((n))
    aval = np.zeros((n))
    ates = np.zeros((n))
    
    all_idx = [i for i in range(n)]  # list containing all indexes

    # random selection
    idx_tra = np.array([all_idx.pop(np.random.randint(0, n-i)) for i in range(ntrain)])
    idx_val = np.array([all_idx.pop(np.random.randint(0, n-ntrain-i)) for i in range(nval)])
    idx_tes = np.array([all_idx.pop(np.random.randint(0, n-ntrain-nval-i)) for i in range(ntest)])
    
    return idx_tra, idx_val, idx_tes
    
    

def dataset_split(n, train_p=.65, val_p=0.15):
    '''return index ofr dataset splitting into train, valuation and test
    
    for true time (forward):
    dataset_ntest = dataset[:ntest]
    dataset_nval = dataset[ntest:ntest+nval]
    dataset_train = dataset[ntest+nval:]
    '''
    
    ntrain = int(n*train_p)
    nval = int(n*val_p)
    ntest = int(n - ntrain - nval)
    
    return ntrain, nval, ntest
    
    
def dataset_split_3yrs(n, train_p=0.8, val_p=0.2, needpast=None):
    '''Define the test period as 3 years
    The rest is split as 80/20
    
    Same as dataset_split()
    
    if needpast is given: the amount of days will be substracted from the test period
    
    '''
    ndays = int((365*3) + 1)  # +1 because 2012 is a leap year
    if needpast is not None:
        ndays = ndays - needpast
    
    
    n_ = int(n - ndays)  # number of days left once test period is substracted
    ntrain = int(n_*train_p)
    nval = int(n_*val_p)
    ntest = ndays  
    
    return ntrain, nval, ntest
    
    
def add_noise(Xtrain, nhyperfeat, pert=1):
    '''Addition of noise to avoid overfitting
    use for the Xtp feature
    
    Xtrain, ndarray 2d (time, nfeat)
    nhyperfeat, number of features in Xtp to add noise to
    
    pert, integer, perturbation 
    
    '''
    
    sigma = np.std(Xtrain[:,:nhyperfeat],keepdims=True)
    noise = pert * np.random.randn(Xtrain.shape[0], nhyperfeat)
    Xtrain[:,:nhyperfeat] += noise
    return Xtrain, noise

def add_history(array, H, needpast, needfutur, label='var', label_pca=True): # , npca=8):
    '''target_history but better
    
    
    array      : 1D/2D array
    H          : array of integer, history to add 
    needpast   : int, number of indices to remove from beginning of array
    needfutur  : int, same but end of array
    
    label_pca  : bool, if True, labels will be done for N pca
    '''
    # only for H positive for NOW !


    # --------   Initialisation ----------------------
    try:  # number of PCA contain in array is shape[1]
        array.shape[1]
    except:
        array = array.reshape((array.shape[0], -1))

    npca = array.shape[1]  # minimum 1
    length = array.shape[0] - needpast - needfutur
    
    H_neg = [x for x in H if x < 0]
    H_pos = [x for x in H if x >= 0]
    
    
#     new_arr = np.empty([length, len(H)*npca])
    new_arr = np.empty([length, (len(H_neg)+len(H_pos))*npca]) #*
    
    labels = []
    # -----------------------------------------------

    # * [x:0] or [0:x] - returns undesired indexes selection. case needs to be excluded with 'if'
    
    # if H_neg empty: we skip this loop
    for nidx, ti in enumerate(H_neg):  # negative 'ti': time_index
        if needpast + ti == 0:  # if time lag is the absolute highest
            new_arr[:, nidx*npca:(nidx+1)*npca] = array[:-needfutur+ti]
        elif needpast > 0:
            new_arr[:, nidx*npca:(nidx+1)*npca] = array[needpast+ti:-needfutur+ti]
            
            
        if label_pca:  # add name of variable to list of labels
            labels += [f'{label} t-{abs(ti)} PC{n}' for n in range(npca)]
        else:
            labels += [f'{label} t-{abs(ti)}']
    
        last_idx = nidx +1 # ugly way to keep index correct between the 2 for-loops
    
    
    if len(H_neg)>0:
        nidx = last_idx*npca  # next index to use for H
    else:
        nidx = 0
    # print(f'nidx={nidx} negative history elements added')
    
    for idx, ti in enumerate(H_pos):  # positive 'ti': time_index
        if needfutur == 0 or (needfutur - ti) == 0:  # if time lag is the absolute highest
            new_arr[:, nidx + idx*npca: nidx + (idx+1)*npca] = array[ti+needpast:]
        elif needfutur > 0:
            new_arr[:,  nidx + idx*npca: nidx + (idx+1)*npca] = array[ti+needpast:-needfutur+ti]
        
        if label_pca:  # add name of variable to list of labels
            labels += [f'{label} t+{ti} PC{n}' for n in range(npca)]
        else:
            labels += [f'{label} t+{ti}']

            
    return new_arr, len(H), labels
    
    


def add_history_dict(forcing_fields, PCf, history_dic, needpast, needfutur):
    '''
    for forcing and covar
    
    Inverse time axis, add the future time steps (in history_dic) to use as input of ML algo


    Parameters:
    -----------
    forcing_fields      : array of string containing names of forcings
    PCf                 : dictionary of PCA for forcings
    history_dic         : dictionary containing, for each forcing name, the list of indexes of the history to be returned
    
    Returns:
    --------
    X                   : PCA at the corresponding history times
    labels              : labels type ['forcing_name PCX t+H', ...]
    hyperfeature        : number of features for each forcing field
    
    '''
    # for each forcing: returns History (future dataset IRL reference)
    
    Hfo_neg = 0
    folabels = []
    fohyperfeat = []
    
    for idx, field in enumerate(forcing_fields):
        Hfo = history_dic[field.split('_')[0]]
        fotmp, fofeat, folab = add_history(PCf[field], Hfo, needpast, needfutur,
                                           field.split('_')[0], label_pca=True)
        
        folabels += folab
        fohyperfeat += [fofeat]
        if idx == 0:  # field == forcing_fields[0]:
            Xfo = fotmp
            continue
        Xfo = np.hstack([Xfo, fotmp])
    
    return Xfo, folabels, fohyperfeat    

# ------------------------------------------------------------------------------------
# -------------------------------- OLD VERSION  -------------------------------------
# TO REMOVE when all dependencies have been changed to use: add_history() and add_history_dict()
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv


def target_history(array, H, Hneg=0, label='', opt_t0=True, label_pca=True):
    '''Add H past values (history) of target field at t+1, t+2, t+3
    to predictors for n features
    
    input:
    array, Xtp predictors for H0 (without history) of dimensions (time, features)
    H, int, number of past values to add, starts at 1
    label, string for H (xx t+1 or xx t+0)
    Hneg, int, number of future values to add, starts at 0
    opt_t0, bool, if false, slice the t=0 value for the array to begin at t=t+1 (for Xtp)
                  if true (default), begin at t=0 (same as prediction), but remove last value (to keep same dimension)
    
    label_pca, bool, if true add ' PCX' to label: 'Xe t+1 PC0'
                     if false, only var and history in label: 'Xe t+1'
    
    
    return 3D array of dim (time, features, history)
    '''
    
    if type(H) == int:
        if H == 0:
            raise ValueError('History index is == 0')
        indH = range(1, H+1)
        nH = H
#         nfeat = H
    elif type(H) == list:
        if len(H) == 0:
            raise ValueError('History List is empty')
        indH = H
        nH = len(H)
#         nfeat = len(H)
    else:
        raise TypeError('History type not supported: should be int or list.')
    
    nfeat = array.shape[1]
    
    if not opt_t0:  # pour Xtp
        array = array[:-1]  # X t+1  [:-1]
        hlabel = 1
        
    if opt_t0:  # pour Forcing and Xnc
        array = array[1:]  # X t0  [:] ?
        # pour commencer au meme temps que la premiere prevision a faire
        hlabel = 0 
    
    new_array = np.copy(array)
    if label_pca:
        hyperlabels = [f'{label} t+{hlabel} PC{n}' for n in range(nfeat)]  #*nfeat
    else:
        hyperlabels = [f'{label} t+{hlabel}']
    
#     if H == 0:
#         print('H == 0')
#         return new_array, nfeat, hyperlabels
     
    nhyperfeat = nfeat * nH + nfeat
    
    # instead of H = range(x)
    # list_history = [1,2,3,4, 8, 12, 16...]
    
    for h in indH:  # range(H):
        tmp = np.vstack([array[:h], array[:-(h)]])  # we put serveral time the same value at the beginning
        # to keep the same length
        if label_pca:
            hyperlabels += [f'{label} t+{(hlabel+h)} PC{n}' for n in range(nfeat)]
        else:
            hyperlabels += [f'{label} t+{(hlabel+h)}']
            
        new_array = np.hstack([new_array, tmp])
        
    # (time, n hyper features) to dim (time, features, history)
    arr3d = np.reshape(new_array, (array.shape[0], nfeat, nH+1))
    
    # add Hneg (reverse) future value: t-1, t-2, t-3
    if Hneg > 0:
        nhyperfeat += nfeat * Hneg
        for hn in range(Hneg):
            print(hn)
            tmp = np.vstack([array[hn+1:], array[-(hn+1):]])
            hyperlabels += [f'{label} t-{(hn+1)}']*nfeat
            new_array = np.hstack([new_array, tmp])
    
        arr3d2 = np.reshape(new_array, (array.shape[0], nfeat, H+Hneg+1))
    
    return new_array, nhyperfeat, hyperlabels


def forcing_history(forcing_fields, PCf, history_dic):
    '''
    
    Inverse time axis, add the future time steps (in history_dic) to use as input of ML algo


    Parameters:
    -----------
    forcing_fields      : array of string containing names of forcings
    PCf                 : dictionary of PCA for forcings
    history_dic         : dictionary containing, for each forcing name, the list of indexes of the history to be returned
    
    Returns:
    --------
    X                   : PCA at the corresponding history times
    labels              : labels type ['forcing_name PCX t+H', ...]
    hyperfeature        : number of features for each forcing field
    
    '''
    # for each forcing: returns History (future dataset IRL reference)
    
    # Hfo = 1  # Xfo
    # Hfo = params['History'][params['type_var_short'].index('fo')]
#     Hfo_neg = params['History_neg'][params['type_var_short'].index('fo')]
    Hfo_neg = 0
    folabels = []
    fohyperfeat = []
    # Xfo  = np.concatenate([PCs[field] for field in forcing_fields],axis=1)[::-1][:-1]

    for idx, field in enumerate(forcing_fields):
        Hfo = history_dic[field]
        fotmp, fofeat, folab = target_history(PCf[field], Hfo, Hfo_neg, field.split('_')[0])
        folabels += folab
        fohyperfeat += [fofeat]
        if idx == 0:  # field == forcing_fields[0]:
            Xfo = fotmp
            continue
        Xfo = np.hstack([Xfo, fotmp])
    
    return Xfo, folabels, fohyperfeat    
    
    
    
def make_fo_history(forcing_fields, fo_to_explore=['2T', 'MSL'], must_be=[1,2,3], interval=3, nexplo=5):
    '''Create dictionary containing future time steps (H) of History
    to use as input in ML
    for forcings
    
    '''
    dic_hist = {}
    
    exploration = [must_be[-1]+i*interval for i in range(1,nexplo+1)]  # indexes for long timescale exploration
    thistory = must_be + exploration

    # fo_to_explore = 'airtmp'
    fo_to_explore = fo_to_explore  # ['airtmp', 'mslprs']

    # var in fo_to_explore will have History on long timescale
    for field in forcing_fields:
        if field.split('_')[0] in fo_to_explore:
            dic_hist[f'{field}'] = thistory
        else:
            dic_hist[f'{field}'] = must_be
            
    return dic_hist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

