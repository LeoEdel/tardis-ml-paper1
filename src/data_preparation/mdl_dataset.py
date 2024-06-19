"""Class preparing the dataset of the machine learning model

Manage:
 - input features
 - time steps
 - scaling
 - formatting

"""

import os
import copy
import pickle
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
        
from src.data_preparation import load_data
from src.data_preparation import mdl_dataset_prep
from src.data_preparation import mdl_dataset_format
from src.data_preparation import scaling
from src.feature_extraction import extract_pca
from src.modelling import mdl_history


models_ml = ['xgb', 'rf', 'grdbst', 'ridge']
models_dl = ['cnn', 'lstm', 'ak']

class Dataset:
    def __init__(self, config, setup='default', non_assimilated:str='freerun', 
                     objective:str='global_train', invert_t=False, return_raw=False, do_scaling=True,
                     history=None, var_to_keep=None, point=None,
                     gfile=None, 
                ):
        '''Class for dataset
        Prepare the dataset to be used
        
        to change input variables: self.update_var_selection()
        to change history: self.set_history(new_history_class)
        
        !! after changing history: self.data_to_dataset() needs to be called again !!
        
        
        Parameters:
        -----------
        
           config                 :   configuration class from reload_config.Config() 
           setup                  :   string, for assemble setup:
                                        'default'    : SIT assimilated, non assimilated, forcings and covariables
                                        'sit_only'   : default but no forcings, no covariables
                                        'no_bias'    : default but no bias SIT (ass-non ass)
                                        'no_forcing' : default but no forcings
                                        'no_covar'   : default but no covariables
                                        'all_raw'    : default setup + raw forcings + raw covariables      
                                        'no_bias_sia': default but no bias SIT (ass-non ass) + sea ice age
                                        'default_sia': default + sea ice age
                    
           non_assimilated       :   string, reference for non assimilated TOPAZ data to use:
                                               - 'freerun'   : data from TOPAZ4, 2000-2011 without any assimilation
                                               - 'noSIT'     : data from TOPAZ4, 2000-2011 with assimilation of all variables except sea-ice Thickness from CS2SMOS
                                               - 'adjSIC'    : data from TOPAZ4, 2000-2011 without any assimilation, but only with SIC > 15%
           
           
           
           
           objective              :   string, usage of the dataset 'train'/'apply'
                                      updated to:      'global_train'      'local_train'
                                                       'global_apply'      'local_apply'
                                                       'global_apply91'      'local_apply91'
                                                       
           
           return_raw             :   bool, if true, will load covariables and forcings without PCA
           history                :   class from mdl_history.History(), will replace the current history
           var_to_keep            :   array of string, variables to keep from all loaded variables
           point                  :   tuple (y, x), indexes of (lat, lon) position inside 2D dataset
           gfile                  :   absolute path to .nc file containing SIT corrected by GLOBAL method
                                              only used if objective == 'local' for now
                                              only used if objective == 'local_train' in the future (?- once local_apply is done)
        '''
        
        if config.verbose == 1:
            print('\nInitialisation dataset...')
        
        self.freerun = True
        self.train = True
        
        self.config = config

        self.setup = setup
        self.non_assimilated = non_assimilated
        self.objective = objective
        self.invert_t = invert_t
        self.return_raw = return_raw
        self.do_scaling = do_scaling
        
        self.var_to_keep = var_to_keep
        
        self.point = point
        self.gfile = gfile
        
        if non_assimilated == '4c':
            self.freerun = False
        if objective == 'apply':
            self.train = False
        

        
        self.charac = {}  # store characteristics of the dataset (to be saved in config)
        self.charac['setup'] = self.setup
        self.charac['non_ass'] = self.non_assimilated
        self.charac['invert_time'] = self.invert_t
        self.charac['return_raw'] = return_raw
    
        self.dataset = {}
    
        # init history        
        if history is not None:
            self.history = history
        else:
            self.history = mdl_history.History(self.config, self.config.verbose)
        
        # determine which names of files to load from the config file
        self.attribute_files_to_load()
        
        # load data
        self.load_big()
        
        # create dataset with variables selected
        if not 'local' in self.objective:
            self.data_to_dataset()
        elif 'local' in self.objective:
            self.local_data_to_dataset()
        
        # transform dataset by adding History selection (t+x, t-x)
        # transform dataset by scaling, 
                            #  addition of noise, if required
        
        if self.var_to_keep is not None:
            self.update_var_selection(self.var_to_keep)
        
        # splitting dataset between train/eval/test if required
        if 'train' in self.objective: # ok for 'global_train' and 'local_train' #  == 'train' and self.objective == 'local':
            self.dataset_split(save_nsplit=self.train)  # train_p, val_p)
        
        if self.invert_t:  # inverse data for prediction to go backward
            self.inverse_time()
        
        # formating to a specific ML/DL algorithm
        if not self.config.ml_model in ['xgb', 'rf', 'grdbst', 'ridge']: # if in models_dl
            self.format_for_dl()
   

        self.get_nfeatures()  # count number of variables for each PC> will be used to compile models

#         self.save_charac_to_config()
    def create_dataset_PC(self, var_to_keep):
        '''After loading data
        Is used to remove undesired variables
        
        -- is the last part of __init__ (?)
        -- for global dataset only
        '''
    
        self.data_to_dataset()
        
        self.update_var_selection(var_to_keep)
        
        # splitting dataset between train/eval/test if required
        if 'train' in self.objective: # ok for 'global_train' and 'local_train' #  == 'train' and self.objective == 'local':
            self.dataset_split()  # train_p, val_p)
        
        if self.invert_t:  # inverse data for prediction to go backward
            self.inverse_time()
        
        # formating to a specific ML/DL algorithm
        if not self.config.ml_model in ['xgb', 'rf', 'grdbst', 'ridge']: # if in models_dl
            self.format_for_dl()
   
    
    
    def redefine_point(self, new_point):
        '''Change point used to get data
        Only works for 
                objective=='local_train' and 'local_apply'
        '''
        
        self.point = new_point        
        self.local_data_to_dataset()
        
        if self.var_to_keep is not None:
            self.update_var_selection(self.var_to_keep)
        
        # splitting dataset between train/eval/test if required
        if 'train' in self.objective: # ok for 'global_train' and 'local_train' #  == 'train' and self.objective == 'local':
            self.dataset_split()  # train_p, val_p)
        
        if self.invert_t:  # inverse data for prediction to go backward
            self.inverse_time()
        
        # formating to a specific ML/DL algorithm
        if not self.config.ml_model in ['xgb', 'rf', 'grdbst', 'ridge']: # if in models_dl
            self.format_for_dl()
    
    def print_charac(self):
        '''characteristics of the dataset (that need to go in config file ? - for application to new dataset from the saved model)
        '''
        print('\nDataset characteristics:')
        print('setup:                         ', self.charac['setup'])
        print('TOPAZ4 non asssimiliated from: ', self.charac['non_ass'])
        print('Time reversed:                 ', self.charac['invert_time'])
        print('Forcing/covar raw included:    ', self.charac['return_raw'])
        
        
        print('Dataset used for:              ', self.objective)
        
    
    def save_charac_to_config(self):
        '''Save useful characteristics of the dataset to config file
        for the charac necessary for reload dataset (or application to 2000-2010
        '''
        
        self.config.modify_to_yaml('dataset_charac', self.charac)
        
        
    def attribute_files_to_load(self):
        '''Set options for the dataset (todo: will need 'set_options_from_config' for application)
        
        Define file names as function of the options set for the dataset
        '''
        
        self.nc_filenames = {}
        self.pkl_filenames = {}
        
        # ------------- FreeRun on 2011-2022 -------------
        if 'train' in self.objective and self.non_assimilated == 'freerun':
            self.nc_filenames = {'nosit': f"{self.config.target_field}_TOPAZ4b23_2011_2022_FreeRun.nc",
                            'withsit': '',
                            'err': f"{self.config.target_field}_TOPAZ4err23_2011_2022.nc",
                            'covar': '',
                            'forcing': '',
                            'sia': 'Topaz_arctic25km_sea_ice_age_v2p1_20110101_20221231.nc'
                           } #  'Topaz_arctic25km_sea_ice_age_v2p0_20111001_20191231.nc'

            self.pkl_filenames = {'nosit': f"PC_{self.config.target_field}_{self.config.n_comp['nc']}N_2011_2022_FreeRun.pkl",  # noSITass
                             'withsit': '',
                             'err': f"PC_SITbias_{self.config.n_comp['tp']}N_2011_2022.pkl",
                             'covar': '',
                             'forcing': '',
                             'sia': f'PC_sia_8N_2011_2022.pkl'     
                            }
              #                'sia': f'pca_sia_{self.config.n_comp["co"]}N_2011_2019.pkl' 
        
        # ------------- adjSIC on 2011-2022 -------------
        elif 'train' in self.objective and self.non_assimilated == 'adjSIC':  # also freerun but with adjSIC (SIC > 15% only)
            self.nc_filenames = {'nosit': f"{self.config.target_field}_TOPAZ4b23_2011_2022_FreeRun_adjSIC.nc",
                            'withsit': '',
                            'err': f"{self.config.target_field}_TOPAZ4err23_2011_2022_adjSIC.nc",
                            'covar': '',
                            'forcing': '',
                            'sia': 'Topaz_arctic25km_sea_ice_age_v2p1_20110101_20221231.nc'
                           } #  'Topaz_arctic25km_sea_ice_age_v2p0_20111001_20191231.nc'

            self.pkl_filenames = {'nosit': f"PC_{self.config.target_field}_{self.config.n_comp['tp']}N_2011_2022_FreeRun_adjSIC.pkl",  # noSITass
                             'withsit': '',
                             'err': f"PC_SITbias_{self.config.n_comp['tp']}N_2011_2022_adjSIC.pkl",
                             'covar': '',
                             'forcing': '',
                             'sia': f'PC_sia_8N_2011_2022.pkl'
                            }
            # f'PC_sia_8N_2011_2022.pkl' # for SIA average
            # f'PC_fyi1_8N_2011_2022.pkl' # for SIA with age == 1
              #                'sia': f'pca_sia_{self.config.n_comp["co"]}N_2011_2019.pkl' 
        
        
         # ------------- adjSIC on 1999-2010 -------------
        elif self.non_assimilated == 'adjSIC' and self.objective == 'apply':    
            self.nc_filenames = {'nosit': f"{self.config.target_field}_TOPAZ4b23_1999_2010_FreeRun_adjSIC.nc",
                            'withsit': '',
                            'err': None,
                            'covar': '',
                            'forcing': '',
                            'sia': 'Topaz_arctic25km_sea_ice_age_v2p1_19981001_20110331.nc'
                           }  # sia file does not exist

            # err filenames do not exist for application in the past (neither does assimilation)
            self.pkl_filenames = {'nosit': f"PC_{self.config.target_field}_{self.config.n_comp['nc']}N_1999_2010_FreeRun_adjSIC.pkl",
                             'withsit': '',
                             'err': None,
                             'covar': '',
                             'forcing': '',
                             'sia': f'PC_sia_8N_1999_2010.pkl'
                            }  # sia file does not exist
            
        # ------------- FreeRun on 1999-2010 -------------
        elif self.non_assimilated == 'freerun' and self.objective == 'apply':    
            self.nc_filenames = {'nosit': f"{self.config.target_field}_TOPAZ4b23_1999_2010_FreeRun.nc",
                            'withsit': '',
                            'err': None,
                            'covar': '',
                            'forcing': '',
                            'sia': 'Topaz_arctic25km_sea_ice_age_v2p1_19981001_20110331.nc'
                           }

            # err filenames do not exist for application in the past (neither does assimilation)
            self.pkl_filenames = {'nosit': f"PC_{self.config.target_field}_{self.config.n_comp['nc']}N_1999_2010_FreeRun.pkl",
                             'withsit': '',
                             'err': None,
                             'covar': '',
                             'forcing': '',
                             'sia': f'PC_sia_8N_1999_2010.pkl'
                            }
        # ------------- adjSIC on 1991-1998 -------------
        elif self.non_assimilated == 'adjSIC' and self.objective == 'apply91':    
            self.nc_filenames = {'nosit': f"{self.config.target_field}_TOPAZ4b23_1991_1998_FreeRun_adjSIC.nc",
                            'withsit': '',
                            'err': None,
                            'covar': '',
                            'forcing': '',
                            'sia': 'Topaz_arctic25km_sea_ice_age_v2p1_19911001_19990331.nc'
                           }

            # err filenames do not exist for application in the past (neither does assimilation)
            self.pkl_filenames = {'nosit': f"PC_{self.config.target_field}_{self.config.n_comp['nc']}N_1991_1998_FreeRun_adjSIC.pkl",
                             'withsit': '',
                             'err': None,
                             'covar': '',
                             'forcing': '',
                             'sia': f'PC_sia_8N_1991_1998.pkl'
                            }
        # ------------- FreeRun on 1991-1998 -------------
        elif self.non_assimilated == 'freerun' and self.objective == 'apply91':    
            self.nc_filenames = {'nosit': f"{self.config.target_field}_TOPAZ4b23_1991_1998_FreeRun.nc",
                            'withsit': '',
                            'err': None,
                            'covar': '',
                            'forcing': '',
                            'sia': 'Topaz_arctic25km_sea_ice_age_v2p1_19911001_19990331.nc'
                           }

            # err filenames do not exist for application in the past (neither does assimilation)
            self.pkl_filenames = {'nosit': f"PC_{self.config.target_field}_{self.config.n_comp['nc']}N_1991_1998_FreeRun.pkl",
                             'withsit': '',
                             'err': None,
                             'covar': '',
                             'forcing': '',
                             'sia': f'PC_sia_8N_1991_1998.pkl'
                            }
    
    
        self.config.nc_filenames = self.nc_filenames
        self.config.pkl_filenames = self.pkl_filenames
    
    def load_big(self):
        '''Load the whole dataset
        '''
        
        # if 'global' in self.objective:
        if not 'local' in self.objective:
            self.outputs = load_data.REREload_dataset_PCA(self.config, return_raw=self.return_raw, 
                                                        freerun=self.freerun, objective=self.objective,
                                                         non_assimilated=self.non_assimilated)
            
        elif 'local' in self.objective:
            self.outputs = load_data.load_dataset_g2l(self.config, 
                                                      freerun=self.freerun,
                                                      gfile=self.gfile)
            
#         print('Adding sia')
            
            
#         self.Xf, self.PCs_f, self.Xe, self.PCs_e, 
#         self.PCs_co, self.PCs_fo, self.dsCo, 
#         self.dsFo, self.chrono = load_data.REload_dataset_PCA(self.config, return_raw=True, freerun=True)
    
    def get_nfeatures(self):
        '''Count number of variables for each PC
        '''
        
        nvar = []
        for pc_name in list(self.config.input_fields.keys()):
            
            var_to_keep = self.config.input_fields[pc_name]
            indexes = []
            for mi in var_to_keep:
                for idx, lb in enumerate(self.inputs):
                    if mi in lb:
                        indexes.append(idx)


            nvar += [len(indexes)]
        
        self.nfeatures = np.array(nvar)
        
        if self.config.ml_model in ['lstm', 'cnn']: # if in models_dl
            # number of timelag
            ntimelag = len(self.history.params['keras']['H'])        
            self.nfeatures = np.array(nvar)//ntimelag
            
        
        print('Number of variables for each PC:')
        print(self.nfeatures) # nvar)
    
    def update_var_selection(self, var_to_keep):
        '''Enable to only select some variables
        
        Variable name must match exactly or will be skipt
        No need to add the time difference 't+x'/'t-x' or the PC number 'PC0', 'PC1'...
        '''
        
        indexes = []
        for mi in var_to_keep:
            for idx, lb in enumerate(self.inputs):
                if mi in lb:
                    indexes.append(idx)

#         import pdb; pdb.set_trace()

        var_name = np.array(self.inputs)[indexes]
        
        if self.config.verbose == 1:
            print(f'New variable selection:\n  {var_name}')
        self.inputs = var_name  # uncomment if ok
#         import pdb; pdb.set_trace()
        
        
        # Assign modified variables to self
          
        # for 2D dataset (before formatting to CNN-LSTM)
        self.X = self.X[:, indexes]
        
#         if dataset need to be split and has been split already
        if hasattr(self, 'ntrain') and self.train:
            self.assemble_dataset()
        elif not self.train:  # if dataset does not need to be split
            self.assemble_dataset()
            
        
        
#         self.dataset['X'] = self.X
        
        # if dataset split
#         if self.objective == 'train':
#             self.dataset['Xtrain'] = self.dataset['Xtrain'][:, indexes]
#             self.dataset['Xval'] = self.dataset['Xval'][:, indexes]
#             self.dataset['Xtest'] = self.dataset['Xtest'][:, indexes]
        
        if self.config.verbose == 1:
            print('Variables updated: do not forget to execute self.format_for_dl() if Deep Learning')
        
    def set_history(self, new_history):
        '''set additional time step (t+x, t-x)
        '''
        
        if hasattr(self, 'history'):
            if new_history == self.history:
                print('New history equals current history: no change.')
                return
        
        self.history = new_history
        self.history.save_to_config()
        
        if self.config.verbose == 1:
            print('New history: do not forget to execute self.data_to_dataset()')
        
        
    
    def load(self):
        '''do nothing - just good recap of what load should do. - to be removed
        
        should be able to load for different sets of variables: 
                    - for training:
                        - SIT bias PCA and SIT bias only for TOPAZ4b
                        - TOPAZ4b FreeRun (or TOPAZ4c)
                        - list of covariables PCA
                        - list of forcings PCA
                        - covariables and forcings 'raw' (=no PCA)
                        
                    - for application on 2000-2010:
                        - same dataset configuration than for training
                        
            need to retrieve the dataset variables (and history) from config file (to application easily done for different
            kind of training)
                    
            * Preparing the dataset
            * Scaling
            * building dataset for Deep Learning (CNN, LSTM, AK) (different than for ML: RF, GB, XGB)
            
            * when splitting dataset for training: save train, validation and test period 
                    
        '''
        
    def local_data_to_dataset(self):    
        '''From outputs from load_dataset_g2l to array 2d (time, variables)
        for one point in the dataset
        '''
        
        
        Xeg = self.outputs[0]
        chrono = self.outputs[1]
        Xna = self.outputs[2] 
        dsCo = self.outputs[3]
        dsFo = self.outputs[4]
        self.maskok = self.outputs[5]
        sia = self.outputs[6]  # todo
        
        
        dim_y = Xeg.shape[1]
        dim_x = Xeg.shape[2]
        
        # ------------------------------------------------------------
        #         selection data from one point or group of points 
        # ------------------------------------------------------------
        
        
        if len(self.point)==1:  # one point
            point = self.point[0]
            # check point is good
            idx_x, idx_y = point[1], point[0]  # self.point[1], self.point[0]
            assert 0<=idx_x<dim_x, f'dimension x outside of limits [0-{dim_x}], got: {idx_x}'
            assert 0<=idx_y<dim_y, f'dimension y outside of limits [0-{dim_y}], got: {idx_y}'

            # select one point
            Xeg_p = Xeg[:, idx_y, idx_x].data[:, None]  # before: no .data[]
            Xna_p = Xna[:, idx_y, idx_x].data[:, None]  # before: no .data[]
            forc_p = self.dict_select_1p(dsFo, [idx_y], [idx_x])
            cova_p = self.dict_select_1p(dsCo, [idx_y], [idx_x])
        
        # average: one group of 9 points
        if 8>=len(self.point)>1:
            idx_x = [pt[1] for pt in self.point]
            idx_y = [pt[0] for pt in self.point]
            
            # select
            #import pdb; pdb.set_trace()
            Xeg_p = Xeg[:, idx_y, idx_x].mean(('y', 'x')).data[:, None]  # before: no .data[]
            Xna_p = Xna[:, idx_y, idx_x].mean(('y', 'x')).data[:, None]
           #  import pdb; pdb.set_trace()
            
            forc_p = self.dict_select_1p(dsFo, idx_y, idx_x)        
            cova_p = self.dict_select_1p(dsCo, idx_y, idx_x)
            
        # reshape to get all points in training dataset
        # (nsample, timesteps, features)
        # (nsample, timesteps, time lag, features) -> later: 4D, not the good order ?
        if len(self.point)>=9:
            # all points in first axis
            Xeg_p = mdl_dataset_format.format_2D_to_3D(Xeg, self.point)
            Xna_p = mdl_dataset_format.format_2D_to_3D(Xna, self.point)
#             sia_p = mdl_dataset_format.format_2D_to_3D(sia, self.point)
            
#             import pdb; pdb.set_trace()
            
            forc_p = self.dict_select_points(dsFo, self.point)
            cova_p = self.dict_select_points(dsCo, self.point)
            
            
        # all the cluster  # todo
        
        # ----------------------------------------------------------------
        

        # labels
        folabels = list(dsFo.keys())
        colabels = list(dsCo.keys())
        nclabels = ['SITf t+0']
        
        # history
        # >>> no history for now
        
        
        # noise if training
        #>>>no  noise
        
        # assemble dataset with setup
        # only one situation ?
#         import pdb; pdb.set_trace()
        X = np.concatenate((Xna_p.data, forc_p, cova_p),axis=-1)   # << issue
        totlabels = nclabels + colabels + folabels
        

        # attribute y and chrono
        y = Xeg_p  # .data[:, None]  # always shape (time, 1) - to be usable when looping on PCA (here no PCA like if PCA=1)
        
        # just to try LSTM '3d'
        X = np.swapaxes(X, 0, 1)
        y = np.swapaxes(y, 0, 1)
        
        self.X = X
        self.y = y
        self.chrono = chrono
        self.inputs = totlabels
        
        # scaling
        self.scale()
        
        
        # datset split + dataset assemble done in self.dataset_split() called by __init__()


    def dict_select_points(self, dico, points):
        '''From dictionnary containing multiple keys
        to 2d array
        select data for many points
        '''
 
        idx_x = [pt[1] for pt in points]
        idx_y = [pt[0] for pt in points]


        ntimes = dico[list(dico.keys())[0]].shape[0]
        nkeys = len(list(dico.keys()))
        npoints = len(points)
        arr_1p = np.zeros((npoints, ntimes, nkeys))
        
        for nf, field in enumerate(list(dico.keys())):
            arr_1p[:, :, nf] = dico[field].data[:, idx_y, idx_x].T
             
                            
        return arr_1p
        
                           
    def dict_select_1p(self, dico, idx_y, idx_x):
        '''From dictionnary containing multiple keys
        to 2d array
        select data for 1 point
        '''
 
        ntimes = dico[list(dico.keys())[0]].shape[0]
        nkeys = len(list(dico.keys()))
        arr_1p = np.zeros((ntimes, nkeys))
        for nf, field in enumerate(list(dico.keys())):
           # arr_1p[:, nf] = np.nanmean(dico[field][:, idx_y, idx_x], axis=(1))  # .mean(('y', 'x'))
            arr_1p[:, nf] = dico[field][:, idx_y, idx_x].mean(('y', 'x'))
             
                            
        return arr_1p
        
    def data_to_dataset(self, invert_t=False, do_scaling=None):
        '''from PCA to (X, y)

        duplicate and shift variables depending on history parameters
        
        Parameters:
        -----------
        
            invert_t       : bool, invert time axis if true. False (default) is time forward
                             depreciated: no use
            scaling        : bool, Scale input datasets or not (between 0-1)
        '''
        
        if do_scaling is not None:
            self.do_scaling = do_scaling
        
        
        # -------- data --------
        Xf = self.outputs[0] 
        PCs_f = self.outputs[1]
        Xe = self.outputs[2]
        PCs_e = self.outputs[3]
        PCs_co = self.outputs[4]
        PCs_fo = self.outputs[5]
        dsCo = self.outputs[6]
        dsFo = self.outputs[7]
        chrono = self.outputs[8]
        PCs_sia = self.outputs[9]
               
#         import pdb; pdb.set_trace()

        # -------- history --------
#         if self.config.ml_model in ['rf', 'ridge', 'xgb', 'grdbst']:

        needpast, needfutur = self.history.get_minmax()
   
        # non corrected (=non assimilated)
        # import pdb; pdb.set_trace()
        Xnc, _, nclabels = mdl_dataset_prep.add_history(PCs_f.values,
                                                           self.history.params['noass']['H'],
                                                           needpast, needfutur,
                                                           'SITf', label_pca=True)  

        if PCs_e is not None:
            Xtp, self.nhypfeat, hyplabels = mdl_dataset_prep.add_history(PCs_e.values, 
                                                               self.history.params['bias']['H'],
                                                               needpast, needfutur,
                                                               f'Xe', label_pca=True)
        elif PCs_e is None:
            Xtp = np.zeros((Xnc.shape[0], self.config.n_comp['tp']))
            self.nhypfeat = self.config.n_comp['tp']
            hyplabels = []
    

        Xfo, folabels, fohyperfeat = mdl_dataset_prep.add_history_dict(self.config.forcing_fields, 
                                                                      PCs_fo, 
                                                                      self.history.params['forcing']['H'],
                                                                      needpast, needfutur)

        
        Xco, colabels, cohyperfeat = mdl_dataset_prep.add_history_dict(self.config.covar_fields, 
                                                                      PCs_co,
                                                                      self.history.params['covar']['H'],
                                                                      needpast, needfutur)
     

        Xsa, sahyperfeat, salabels = mdl_dataset_prep.add_history(PCs_sia.values,
                                                           self.history.params['noass']['H'],
                                                           needpast, needfutur,
                                                           'SIA', label_pca=True)  

        
        if self.objective == 'train':
            # add noise on Xtp
            Xtp, _ = mdl_dataset_prep.add_noise(Xtp, Xtp.shape[1], pert=1)

        # -------- assemble final dataset --------
#         import pdb; pdb.set_trace()

        # with covariables
        if self.setup == 'default':  
            X = np.concatenate((Xtp, Xnc, Xfo, Xco),axis=1)
            totlabels = hyplabels + nclabels + folabels + colabels    
        
        elif self.setup == 'sit_only':
            X = np.concatenate((Xtp, Xnc),axis=1)
            totlabels = hyplabels + nclabels
            
        elif self.setup == 'no_bias':  
            X = np.concatenate((Xnc, Xfo, Xco),axis=1)
            totlabels = nclabels + folabels + colabels
            
        elif self.setup == 'no_forcing':  
            X = np.concatenate((Xnc, Xco),axis=1)
            totlabels = hyplabels + nclabels + folabels
            
        elif self.setup == 'no_covar':  
            X = np.concatenate((Xtp, Xnc, Xfo),axis=1)
            totlabels = hyplabels + nclabels + folabels
            
        elif self.setup == 'all_raw':  
            X = np.concatenate((Xtp, Xnc, Xfo, Xco, DSco, DSfo),axis=1)
            pca_labels = hyplabels + nclabels + folabels + colabels 
            totlabels = hyplabels + nclabels + folabels + colabels + clabels + flabels
            
        elif self.setup == 'no_bias_sia':  # no bias + sea ice age
            X = np.concatenate((Xnc, Xfo, Xco, Xsa),axis=1)
            totlabels = nclabels + folabels + colabels + salabels
            
        elif self.setup == 'default_sia':  # default + sea ice age
            X = np.concatenate((Xtp, Xnc, Xfo, Xco, Xsa),axis=1)
            totlabels = hyplabels + nclabels + folabels + colabels + salabels
            
        else:
            raise ValueError(f'This dataset setup is not defined: {self.setup}')

        if 'apply' in self.objective:
            y = np.zeros((PCs_f.shape))
        else:
            y = PCs_e.values

        # shorten ytrue and chrono according to History    
        if needfutur > 0:
            y = y[needpast:-needfutur, :]  # target var t #  [:-1]
            chrono = chrono[needpast:-needfutur]
        else:
            y = y[needpast:, :]
            chrono = chrono[needpast:]
            # y = PCs_e.values  # << nothing to do here. to remove if no bug
            



        self.X = X
        self.y = y
        self.chrono = chrono
        self.inputs = totlabels

#         import pdb; pdb.set_trace()
        
        if self.do_scaling:
            self.scale()
        
        if 'apply' in self.objective:  # else need to call dataset_split() before assembling dataset
            self.assemble_dataset()
        
    def inverse_time(self):
        '''inverse time axis
        needs to be done after dataset_split()
        '''
            
        self.X = self.X[::-1]
        self.y = self.y[::-1]
        self.chrono = self.chrono[::-1]
        
    
        
    def dataset_split(self, train_p=None, val_p=0, save_nsplit=True):
        '''Define size of training, validation and test dataset
        Save the number of days in the config files:
            - temporal order is always: test, validation, training (forward in time ->)
            
        since the dataset is now at the last version (sept 2023), the test dataset is defined as 3 years.
        
        Parameters:
        -----------
        
            save_nsplit        : bool. If True, save ntest, nval, ntrain to the config (.yaml) file
        '''
        
        n = self.y.shape[0]  # number of points
        
        # default for CNN, LSTM, AK
        if train_p is None:
            train_p = 0.8  # 65
            val_p = 0.2  # 15
        
        if self.config.ml_model in ['xgb', 'ridge', 'rf', 'grdbst']:  # default for ML // in models_ml
            train_p = 0.8
            val_p = 0
    
        # proportion of train period and validation period are defined as
        # proportion of the full dataset MINUS the test period
        ntrain, nval, ntest = mdl_dataset_prep.dataset_split_3yrs(n, train_p=train_p, val_p=val_p, needpast=self.config.needpast)
        
        
        if self.config.verbose == 1:
            print(f'Size of the training set:   {ntrain:>5} days')
            print(f'Size of the validation set: {nval:>5} days')
            print(f'Size of the test set:       {ntest:>5} days')
        
        self.ntrain = ntrain
        self.nval = nval
        self.ntest = ntest

        # save them in config:
        if save_nsplit:
            self.config.modify_in_yaml('ntrain', ntrain)
            self.config.modify_in_yaml('nval', nval)
            self.config.modify_in_yaml('ntest', ntest)
    
        self.assemble_dataset()
    
    def assemble_dataset(self):
        '''Attribute self data to the dict() self.dataset
        if dataset used for training:
                slpit the data in order to have chronologically (time -->-->-->)
                test period - validation period - train period
        '''
        
        self.dataset['X'] = self.X
        self.dataset['y'] = self.y
        self.dataset['chrono'] = self.chrono
        
        # if self.objective == 'train':
        if 'train' in self.objective:  # allow 'train' >>> now:'global_train' and 'local_train' to be True
            self.dataset['ntrain'] = self.ntrain
            self.dataset['nval'] = self.nval
            self.dataset['ntest'] = self.ntest

            self.dataset['chrono_train'] = self.chrono[self.ntest+self.nval:]
            self.dataset['chrono_val'] = self.chrono[self.ntest:self.ntest+self.nval]
            self.dataset['chrono_test'] = self.chrono[:self.ntest]

            self.dataset['Xtrain'] = self.X[self.ntest+self.nval:]
            self.dataset['Xval'] = self.X[self.ntest:self.ntest+self.nval]
            self.dataset['Xtest'] = self.X[:self.ntest]

            self.dataset['ytrain'] = self.y[self.ntest+self.nval:]
            self.dataset['yval'] = self.y[self.ntest:self.ntest+self.nval]
            self.dataset['ytest'] = self.y[:self.ntest]
                

    def scale(self):
        '''Scale the variables used inputs
        for now: scale betweeen [0-1].
        later: add scale between [-1,1]
        '''

        if self.config.verbose == 1: print('Scaling...')
        
        # old way
#         scale between 0 and 1
#         self.X = mdl_dataset_prep.scale_data_var(self.X)
        
        filename = f'{self.config.rootdir}{self.config.ml_dir}scalers.pkl'
        
        if 'train' in self.objective:  ## Fit and transform
            X_scaled = scaling.scale_fit_transform(X=self.X, scalers_file=filename)
                        
        else:  ## Transform from saved MinMaxScaler (fit on training dataset)
            X_scaled = scaling.scale_transform(X=self.X, scalers_file=filename)

        self.X = X_scaled
        
        
    def format_for_dl(self, ml_model=''):
        '''Change format of the dataset to fit CNN/LSTM/AK etc
        
        not necessary to have 'ml_model' as input for this function
        only to test
        
        '''
        
        if not ml_model == '':
            self.config.ml_model = ml_model
        
        print(f'Dataset Format for {self.config.ml_model}')
                
        if self.config.ml_model in ['cnn', 'lstm']:
#             true_ntest = mdl_dataset_format.format_CNN_LSTM(self, self.history.params['keras']['H'])
            true_ntest = mdl_dataset_format.format_CNN_LSTM_fixed(self, self.history.params['keras']['H'])
            
            if 'train' in self.objective:  ## write down number of days in test period (ntest) in config file only if training
                self.config.modify_in_yaml('ntest', true_ntest)
        elif self.config.ml_model == 'ak':
            print('No need for formatting')
        else:
            print(f'Dataset non-existant formatting for: {self.config.ml_model}')
        
            
        
        
        
