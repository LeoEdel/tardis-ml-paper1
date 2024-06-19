import autokeras as ak
import tensorflow_addons as tfa
import tensorflow as tf

from src.modelling import super_model_dl
from src.data_preparation  import mdl_dataset_prep

def scale_data_var_2D(train_data, test_data):
    '''scale each variable btw 0 and 1
    
    for train_data, test_data: format (sample, timestep, features)
    '''
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()
    
    
    for var in range(train_data.shape[1]):  # on each feature
        max_val = train_data[:,var].max()
        min_val = train_data[:,var].min()
        train_scaled[:,var] = (train_data[:,var] - min_val) / (max_val - min_val)
        test_scaled[:,var] = (test_data[:,var] - min_val) / (max_val - min_val)
    return train_scaled, test_scaled

def build_dataset_autokeras(X, y, chrono, ntvt, times=[0]):
    '''Construct the NN dataset for autokeras.StructureDataRegressor
    
    Shape:
        Xtrain (samples=time, features)
        ntvt    . tuple containing (ntrain, nval, ntest)
    '''
    
    
    # test to see if use the good year for validation
#     train_p = 0.65
#     val_p = 0.15
    
    # Number of data
    # n = X.shape[0] # - needpast - needfutur
    
#     ntrain, nval, ntest = mdl_dataset_prep.dataset_split(X.shape[0], train_p=train_p, val_p=val_p)
    ntrain, nval, ntest = ntvt
    
    
     # Split dataset, train with older years
    dataset = dict()

    Xtrain = X[ntest+nval:]
    Xval = X[ntest:ntest+nval]
    Xtest = X[:ntest]
    
    dataset["ytrain"] = y[ntest+nval:]
    dataset["yval"] = y[ntest:ntest+nval]
    dataset["ytest"] = y[:ntest]
    
    dataset["Xtrain"], dataset["Xval"], dataset["Xtest"] = Xtrain, Xval, Xtest  #= scale_data_var_2D(Xtrain, Xtest)

    dataset['chrono'] = chrono
    dataset['chrono_train'] = chrono[ntest+nval:]
    dataset['chrono_val'] = chrono[ntest:ntest+nval]
    dataset['chrono_test'] = chrono[:ntest]

    dataset['ntrain'] = ntrain
    dataset['nval'] = nval
    dataset['ntest'] = ntest
    
    return dataset, 'toutcajustepour avoir le bon format--achanger'


class ModelAK(super_model_dl.SModelDL):
    
    def __init__(self, ds, rootdir=None, ml_dir=None, fig_dir=None):  #, ntvt=(None, None, None)):
        '''
        
        Parameters:
        -----------
        
            ntvt         :     array of int, Number of timesteps for Train Validation and Test dataset
                                                                  ex:   (1958, 451, 604)
        
        
        '''

       # self.max_trials = max_trials
        self.rootdir = rootdir
        self.ml_dir = ml_dir
        self.fig_dir = fig_dir
        self.type = 'AK'
        

        # new vvvvvvvv
        self.name = ds.config.ml_name
        self.epochs = ds.config.epochs
        self.batch_size = ds.config.batch_size
        self.max_trials = ds.config.max_trials
        
        self.ntrain = ds.ntrain  # ntvt[0]
        self.nval = ds.nval  # ntvt[1]
        self.ntest = ds.ntest  # ntvt[2]
        
        self.ds_objective = ds.objective
        self.verbose = ds.config.verbose
        
        
        self.models = {}
        self.regs = {}
        self.histories = {}
        
        super().__init__()
        

    def compile_models(self, name=None, npca=1):
        '''Compile for npca
        '''
        
        if name is not None:
            self.name = name
            self.type = name
        
        self.npca = npca
        
        for ipca in range(self.npca):
#             self.regs[f'pc{ipca}'] = self.mdl_ak_base()
            self.regs[f'pc{ipca}'] = ak.StructuredDataRegressor(overwrite=True, max_trials=self.max_trials,
                                        objective='val_loss', 
                                        metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

            
    def compile_model(self, name='AK'):
                
        if name == 'AK':
            self.reg = self.mdl_ak_base()
        else:
            print(f'No model for this name {name}')
            return
        
        self.type = name
    
#     def explore(self, dataset, epochs=40):
#         '''depreciated -- to delete
#         '''
        
#         self.epochs = epochs        
#         ipca = 0
#         ytrain = dataset['ytrain'][:-1,ipca].reshape((dataset['ytrain'].shape[0]-1,-1))
        
#         self.history = self.reg.fit(dataset['Xtrain'], ytrain, epochs=self.epochs)

    
    def explore_multiple(self, dataset, epochs=None, batch_size=None):
        '''Fit max_trials and select the best model 
        
        Parameters:
        -----------
        
            epochs      : int, (old default=40)
            batch_size  : int, (old default=None)
        
        '''
        if epochs is not None:
            self.epochs = epochs
            
        
        self.batch_size = batch_size
        
        for ipca in range(self.npca):
            ytrain = dataset['ytrain'][:,ipca].reshape((dataset['ytrain'].shape[0],-1))  # [:-1, pca]
            yval = dataset['yval'][:,ipca].reshape((dataset['yval'].shape[0],-1))
            # fit and get the best model
            if batch_size is None:
                self.regs[f'pc{ipca}'].fit(dataset['Xtrain'], ytrain, epochs=self.epochs,
                                                               validation_data=(dataset['Xval'], yval))
            else:
                self.regs[f'pc{ipca}'].fit(dataset['Xtrain'], ytrain, epochs=self.epochs,
                                                                     batch_size=self.batch_size, 
                                                               validation_data=(dataset['Xval'], yval))
                                       #validation_split=0.2)
            
            # export direct, otherwise would overwrite model at next PC fit
            self.models[f'pc{ipca}'] = self.regs[f'pc{ipca}'].export_model()
            # fit one more the model to get validation metrics: 'val_loss'
            if batch_size is None:
                self.histories[f'pc{ipca}'] = self.models[f'pc{ipca}'].fit(dataset['Xtrain'], ytrain, epochs=self.epochs,
                                                               validation_data=(dataset['Xval'], yval))
            else:
                self.histories[f'pc{ipca}'] = self.models[f'pc{ipca}'].fit(dataset['Xtrain'], ytrain, epochs=self.epochs,
                                                                     batch_size=self.batch_size, 
                                                               validation_data=(dataset['Xval'], yval))
                                                                       
                                                                       #validation_split=0.2)

            
            
#     def export_models(self):
#         for ipca in range(self.npca):
#             self.models[f'pc{ipca}'] = self.regs[f'pc{ipca}'].export_model()
        
    def export_model(self):
        self.model = self.reg.export_model()
        
    def mdl_ak_base(self):
        reg = ak.StructuredDataRegressor(overwrite=True, max_trials=self.max_trials, objective='val_loss',
                                        metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        
        return reg
