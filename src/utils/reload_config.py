import os
import yaml
import glob
import subprocess
import datetime
import numpy as np

from src.utils import tardisml_utils
from src.utils import save_name

class Config():
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
        self._load_variable_params()
        self.load_ml_params()  # for ml_model
        self.load_dataset_params()  # for dataset

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

        
        # update ml parameters (on duplicated config file)
        self.define_ml_needs()
                
        
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

    def update_folders(self):
        '''Put corresponding directories name for the followings:
                pca_dir
                res_dir

        Update config file with the corresponding paths
        '''

        self.update_pca_dir()
        self.update_results_dir()

        if self.verbose == 1: print('Config folders updated.')

    def update_pca_dir(self):
        '''Update pca dir for config file
        create new directory if none existant
        '''

        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)

        lim_idm = config['lim_idm']
        lim_jdm = config['lim_jdm']
        str_xy = f"i{lim_idm[0]}-{lim_idm[1]}_j{lim_jdm[0]}-{lim_jdm[1]}"

        pca_dir = f"/results/pca_{str_xy}"


        # if non existant
        path = os.path.join(self.rootdir, config['user']+pca_dir)
        if self.verbose==1:
            print(f'PCA results in: {path}')
        if not os.path.exists(path):
            os.mkdir(path)
            if self.verbose==1:
                print(f'Folder created\n')

        self.modify_in_yaml('pca_dir', pca_dir)


    def update_results_dir(self):
        '''Create folder to store all results
        '''

        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)
        
        # name for results
        str_date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        res_dir = f"/results/{config['ml']}_{str_date}"

        # check if exists
        tmp_res_dir = save_name.check_folder(os.path.join(self.rootdir, config['user']+res_dir))
        res_dir = f'/results/{tmp_res_dir}'

        # create folder
        path = os.path.join(self.rootdir, config['user']+res_dir)
        if self.verbose == 1:  print(f'Results in: {path}')
        if not os.path.exists(path):
            os.mkdir(path)
            if self.verbose == 1:  print(f'Folder created\n')

        # create sub folders
        for subfolder in [config['ml_dir'], config['fig_dir']]:
            path = os.path.join(self.rootdir, config['user']+res_dir+subfolder)
            if not os.path.exists(path):
                os.mkdir(path)
                if self.verbose == 1: print(f'Subfolder created: {path}')

        # res_dir in self, to call copy_config_to_results()
        self.res_dir = config['user'] + res_dir  # config['results_dir']
                    
        self.modify_in_yaml('results_dir', res_dir)



    def modify_in_yaml(self, key, value):
        '''Modify variables in original config file:

        Parameters:
        -----------
            key          : string, name of variable to modify
            value        : new value of the variable

        '''
        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)  # open file

        # changes value:
#         config['pca_dir'] = pca_dir
        config[key] = value

        # write yaml file
        with open(f'{self.ifile}', 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False, explicit_start=True)  # safe_

        if self.verbose == 1:
            print(f"Config file updated '{key}': {self.ifile}")

#         self.copy_config_to_results()
        

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

        self.nosit_dir = config['user'] + config['nosit_dir']
        self.withsit_dir = config['user'] + config['withsit_dir']
        self.forcing_adir = config['user'] + config['forcing_adir']
        self.forcing_bdir = config['user'] + config['forcing_bdir']
        self.freerun_dir = config['user'] + config['freerun_dir']
        self.fig_dir = config['user'] + config['results_dir'] + config['fig_dir']
        self.pca_dir = config['user'] + config['pca_dir']
        self.res_dir = config['user'] + config['results_dir']
        self.ml_dir = self.res_dir + config['ml_dir']

    def _load_variable_params(self):
        '''Load parameters for PCA
        '''
        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)

        self.timeofday = config['timeofday']
        self.target_field = config['target_field']
        self.covar_fields = config['covariable_fields']
        self.forcing_mean_days = config['forcing_mean_days']
        
        if config["forcing_mean_days"]>0:
            self.forcing_fields = [f'{item}_mean{config["forcing_mean_days"]}d' for item in config['forcing_fields']]
        else:  # for forcings at 12h
            self.forcing_fields = config['forcing_fields']
            
        self.forcing_fields_clean = config['forcing_fields']
        self.lim_idm = config['lim_idm']
        self.lim_jdm = config['lim_jdm']

#         self.n_comp = config['para_var_dpd']['n_components']
        self.n_comp = {}
        self.n_comp['tp'] = config['para_var_dpd']['n_components'][0]
        self.n_comp['nc'] = config['para_var_dpd']['n_components'][1]
        self.n_comp['co'] = config['para_var_dpd']['n_components'][2]
        self.n_comp['fo'] = config['para_var_dpd']['n_components'][3]
        
        
    def load_dataset_params(self):
        '''load parameters for dataset characteristics
        '''

        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)
        
#         try:
        self.setup = config['dataset_charac']['setup']
        self.invert_time = config['dataset_charac']['invert_time']
        self.non_ass = config['dataset_charac']['non_ass']
        self.return_raw = config['dataset_charac']['return_raw']
#         except:
#             print('Old config files, <dataset> parameters are not imported.')
            
        
        
        
        
    def load_ml_params(self):
        '''returns parameters for machine learning
        '''
        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)

        self.ml_model = config['ml']
        self.nseq = config['nseq']
        self.params = config['para_var_dpd']
        
        try:  # not in old config file
            self.ntrain = config['ntrain']
            self.nval = config['nval']
            self.ntest = config['ntest']
            self.nsplit = config['nsplit']

            self.random_state = config['random_state']
            self.bayesian_search = config['bayesian_search']
            self.n_iter_bay = config['n_iter_bay']
            self.return_train_score = config['return_train_score']
            self.introspect = config['introspect']
            self.keepvar_threshold = config['keepvar_threshold']
            self.most_imp = config['most_imp']

            self.max_trials = config['max_trials']
            self.epochs = config['epochs']
            self.batch_size = config['batch_size']
            self.ml_name = config['name']  # ['ml_name']
            
            # check if number of epochs == number of PC to predict
            default_epo = 5
            n_PC = self.params['n_components'][0]
            if len(self.epochs) != n_PC:
                self.epochs = [self.epochs[n] if n<len(self.epochs) else default_epo for n in range(n_PC)]
            
            self.needpast = config['needpast']
            self.needfutur = config['needfutur']
            
            # get input features for each PC model
            self.input_fields = config['zz_features']
            # number of input features for each PC
            # self.nfeatures = np.array([len(self.input_fields[ky]) for ky in list(self.input_fields.keys())])
            
        except:
            print('Old config files, some <ml> parameters are not imported.')
            pass
    
    
    
    def define_ml_needs(self):
        '''Define needpast and needfutur used in ML model
        and save them to the .yaml config file
        '''
        
        histo = self.load_history()
        
        maxi = [] # t+ timesteps required for each variable
        mini = [] # t- timesteps required for each variable

        for ky in list(histo.keys()):
            if ky == 'bias':  # is not used
                continue
            elif type(histo[ky]['H']) is list:
                maxi += [np.max(histo[ky]['H'])]
                mini += [np.min(histo[ky]['H'])]
            else:
                for sub_ky in list(histo[ky]['H'].keys()):
                    maxi += [np.max(histo[ky]['H'][sub_ky])]
                    mini += [np.min(histo[ky]['H'][sub_ky])] 
        
        self.needpast = int(np.min(mini))
        self.needfutur = int(np.max(maxi))
                
        #save into .yaml
        self.modify_in_yaml('needpast', self.needpast)
        self.modify_in_yaml('needfutur', self.needfutur)
        
        
        
    def load_history(self):
        '''Load history parameters
        '''
        
        config = yaml.load(open(self.ifile), Loader=yaml.FullLoader)
        
#             yaml.safe_dump(config, file, default_flow_style=False, explicit_start=True)
        
        return config['history']
        
        
    def create_dir_sit_rec(self):
        '''Create subfolder of 'results'/figures/
        where to save SIT reconstruction
        '''
        # create new directory to save new results
        self.sit_rec_dir = f'{self.fig_dir}sit_reconstruct/'

        if not os.path.exists(f'{self.rootdir}{self.sit_rec_dir}'):
            print(f'creating folder: {self.rootdir}{self.sit_rec_dir}')
            os.mkdir(f'{self.rootdir}{self.sit_rec_dir}')

        if self.verbose == 1: print(f'Folder: {self.sit_rec_dir}')

    def copy_config_to_results(self):
        '''Save actual configuration to results folder

        Need to be called after _load_directories()
        '''
        if not hasattr(self, 'res_dir'):
            if self.verbose == 1: print('Trying to call copy_config_to_results() when results folder is not known/created.')
            return
        
        subprocess.run(['cp', f'{self.ifile}', f'{self.rootdir+self.res_dir}'])
        if self.verbose == 1:
            print(f'Config copied to: {self.rootdir+self.res_dir}')
    
    

    
    
    