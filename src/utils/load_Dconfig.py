'''Class necessary to load config file used for Distribution High Resolution machine learning
'''

import os
import yaml
import glob
import subprocess
import datetime

from src.utils import tardisml_utils
from src.utils import save_name
from src.utils import reload_config

class DConfig(reload_config.Config):
    def __init__(self, ifile, rootdir=None, verbose:int=0):
        '''
        let rootdir=None if you do not want to create a new directory
        
        Parameters:
        -----------

            ifile            : string, absolute path to configuration file (.yaml)
                                if does not contain filename, will glob.glob folder

            rootdir         : string, path where to create the results directory.
                                None (default) if the folder directory already exists and is given as 'ifile'

            verbose          : int. 0 = not details, 1 = details

        '''

        self.ifile = ifile
        self.verbose = verbose
        self._check_filename()  # check config file
        
        # rootdir is before subfolder /Leo/ . select it from full path
        if rootdir is None:  # from an existant result folder
            self.rootdir = os.path.dirname(self.ifile)[:os.path.dirname(self.ifile).find('Leo')]
        else:  # to create the result folder
            self.rootdir = rootdir
            self.update_folders()  # create folders to store results
            # self.copy_config_to_results()  # copy config file to results
            
        # load different metrics from ifile
        self._load_directories()
#         self._load_variable_params()
        self.load_ml_params()  # for ml_model
#         self.load_dataset_params()  # for dataset


        # when init is finished:
        # if config is NEW
        if rootdir is not None:
            self.copy_config_to_results()
        # once the config has been copied, it becames the default config file 
        # (so any modification will further be done on the TRUE config file and not the one in /config
        # that may be used for/by other jobs/scripts as well)
            self.ifile = f'{self.rootdir+self.res_dir}/'
            self._check_filename()  # check config file
            if self.verbose == 1:
                print(f'Default config file is now the copied following one:')
                print(self.ifile)

        
    def _check_filename(self):
        '''check if filename exist
                 if path alone, will glob.glob folder and identify config.yaml file
        '''

        if os.path.basename(self.ifile) == '':  # path only
            yaml_file = glob.glob(f'{self.ifile}*.yaml', recursive=False)
            if len(yaml_file)==1:
                self.ifile = yaml_file[0]
                if self.verbose == 1:
                    print(f'Config file found: {yaml_file[0]}')
            else:
                raise ValueError(f'Too many (>1) or 0 .yaml file(s) found: {self.ifile}')

        else: # with filename.yaml
            if not os.path.exists(self.ifile): # if does not exist
                raise ValueError(f'File does not exist: {self.ifile}')
            else:
                if self.verbose == 1: print(f'Config file found: {self.ifile}')

        

    def _load_directories(self):
        """ Load directory names from config file

        Parameters:
        -----------
            filename : String -- Path to config file


        nosit_dir    : String -- Name of the file containing nosit data
        withsit_dir  : String -- Name of the file containing withsit data
        forcing_adir : String -- Name of the file containing forcing data
        forcing_bdir : String -- Name of the file containing forcing data
        pca_dir      : String -- Specific subfolder for PCA results
        res_dir      : String -- Specific folder to save results (figures and machine learning)
        fig_dir      : String --

        """

        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)


        self.fig_dir = config['user'] + config['results_dir'] + config['fig_dir']
        self.pca_dir = config['user'] + config['pca_dir']
        self.res_dir = config['user'] + config['results_dir']
        self.ml_dir = self.res_dir + config['ml_dir']

#     def _load_variable_params(self):
#         '''Load parameters for PCA
#         '''
#         config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)

# #         self.timeofday = config['timeofday']
# #         self.target_field = config['target_field']
#         self.covar_fields = config['covariable_fields']
# #         self.forcing_mean_days = config['forcing_mean_days']
        
#         if config["forcing_mean_days"]>0:
#             self.forcing_fields = [f'{item}_mean{config["forcing_mean_days"]}d' for item in config['forcing_fields']]
#         else:  # for forcings at 12h
#             self.forcing_fields = config['forcing_fields']
            
#         self.forcing_fields_clean = config['forcing_fields']
#         self.lim_idm = config['lim_idm']
#         self.lim_jdm = config['lim_jdm']

   
        
        
    def load_ml_params(self):
        '''returns parameters for machine learning
        '''
        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)

        self.ml_model = config['ml']
        
        try:  # not in old config file
            self.ntrain = config['ntrain']
            self.nval = config['nval']
            self.ntest = config['ntest']

            self.random_state = config['random_state']
            self.epochs = config['epochs']
            self.batch_size = config['batch_size']
            self.ml_name = config['ml_name']            
            
            self.d1 = datetime.datetime.strptime(config['date1_start'], '%Y-%m-%dT%H:%M:%S')
            self.d2  = datetime.datetime.strptime(config['date2_end'], '%Y-%m-%dT%H:%M:%S')
            
            self.num_obs = config['num_obs']
            
            self.var_to_exclude = config['var_to_exclude']
            
            self.nfeat = config['nfeat']
            
        except:
            print('Old config files, some <ml> parameters are not imported.')
            pass
        
 