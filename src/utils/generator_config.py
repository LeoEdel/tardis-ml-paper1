'''Generate configuration files (.yaml) for different variables, parameters, ML algorithm
'''

import os
import yaml

from src.modelling import mdl_history
from src.utils import save_name
from src.utils import tardisml_utils


path = '/users/edelleo1/tardis/tardis-ml/config/config_to_jobs/'

dl_model = ['cnn' , 'lstm', 'ak']


# -----------------------------------------
#     Parameters for config generator
# -----------------------------------------
# ['rf', 'xgb', 'cnn' , 'lstm', 'ak']
ml_models = ['rf', 'xgb', 'cnn' , 'lstm', 'ak']

dl_names = {'cnn':['CNN'],  # , 'CNN_mh'],
            'lstm':['LSTM3_bk'],  # 'LSTM',
            'ak':['AK']}

covar = ['siconc', 'sisnthick', 'zos','vxsi', 'vysi']
forcings = ['2T', 'MSL', 'TP', '10V', '10U', 'SSR', 'STR']

best_co = ['siconc', 'sisnthick']
best_fo = ['2T', 'MSL', 'TP']

nweeks = [0, 1]  # , 2, 3, 4]

dataset_setup = ['no_bias', 'no_bias_sia']  # ['default', 'no_bias']
# history_setup = ['default', 'zero']
# history
# params...
# best at 1 week, 2 weeks, 3 weeks, 4weeks interval



objectives = ['simple', 'recursive', 'recursive sequence']
nseqs = [0, 0, 7]  #  15, 21, 30]


# add in config:

# from ML
random_state = None
bayesian_search = False
n_iter_bay = 100
return_train_score = True
introspect = False
keepvar_threshold = 0 # importance variable
most_imp = None
nsplit = 4 

# ntrain, nval, ntest

# from DL
max_trials  = 100 # AK
epochs = 100
batch_size = 4  # if AK: None
# name model (CNN and LSTM have different models architecture)
# H (History keras) for CNN and LSTM

config_params = {}


# todo: create simple class 
# ------------------------------------

def generate_cfiles():
    print(f'Generating configuration files in: {path}')

    list_configs = []
    
    # loop over parameters
    for mdl in ml_models:
        config_params['ml'] = mdl
        
        for ds_st in dataset_setup:
            config_params['dataset_charac'] = {}
            config_params['dataset_charac']['setup'] = ds_st
            
            if mdl not in dl_model:
                for nseq in nseqs:
                    config_params['nseq'] = nseq
                    
                    list_configs.append(config_params.copy())
                    
            else:
                # loop over max trials / epochs / batch size / models name ?
                for name in dl_names[mdl]:
                   # print(dl_names[mdl], '   :   ',name)
                    config_params['ml_name'] = name
                    if mdl == 'ak':
                        config_params['batch_size'] = None
                        
                    list_configs.append(config_params.copy())
                

                
  #   import pdb; pdb.set_trace()
    for conf in list_configs:  # 
        
        for nw in nweeks:  # loop over histories
        # create history if necessary
            cf = CreateConfig(parameters=conf, verbose=1)
        
            history = mdl_history.History(config=cf)
            history.set_explo_week(nw, best_co, best_fo)  # nweek=1, covar_to_keep=[], fo_to_keep=[]
            history.save_to_config()  # modify in self.config
            
            cf.name_explo = f'{nw}wk'
            cf.generate_name()
            cf.save_config()
        
        
class CreateConfig:
    '''Generate config files
    
    for .yaml file:
        - <None> (in python) should be written as <null> (otherwise will be read as 'None')
    
    '''
    def __init__(self, parameters={}, verbose=0):
        '''Create from an existing file
        
        Parameters
        -------------
        
            parameters         :     dictionary containing (keys: value) to change
        
        
        '''
        self.rootdir =  '/users/edelleo1/' # tardisml_utils.get_rootdir()
        self.default_file = f'{self.rootdir}tardis/tardis-ml/config/data_proc_full.yaml'
        self.opath = f'{self.rootdir}tardis/tardis-ml/config/config_to_jobs/'
        
        # self.ifile = default_file
        self.verbose = verbose
        
        self.parameters = parameters
        
        # load default config
        self.get_config()
        self.modify_params(self.parameters)
        
        
        #self.generate_name()
        #self.save_config()
        
    def get_config(self):
        '''Load default configuration file
        '''
        self.config = yaml.load(open(self.default_file), Loader=yaml.FullLoader)
    
        self.forcing_fields = self.config['forcing_fields'] 
        self.covar_fields = self.config['covariable_fields']
        
    
    
    def generate_name(self):
        '''Define the name of the .yaml
        '''
        #self.name_explo = '1wk'
        self.oname = f"config_"\
            f"{self.config['ml'].upper()}_"\
            f"{self.config['dataset_charac']['setup']}_"\
            f"{self.name_explo}"\
            f".yaml"
               #{random_seed} in case of addition parameters
               #because at some point fuck this shit (messy anyway)
               # .yaml'    
        
        
        # self.oname = 'oui.yaml'
        self.oname = save_name.check(self.opath, self.oname)
        self.ofile = f'{self.opath}{self.oname}'
    
    
    def modify_params(self, params):
        '''
        '''
        # import pdb; pdb.set_trace()

        for key in self.parameters.keys():
            value = self.parameters.get(key)
            if type(value) is dict:  # == "<class 'dict'>":  # dict in dict
                self.config[key][list(value.keys())[0]] = list(value.values())[0]
            elif type(value) is str:
                self.config[key] = value
                
        
        
    def save_config(self):
        '''
        '''
        # write yaml file
        with open(f'{self.ofile}', 'w') as file:
            yaml.safe_dump(self.config, file, default_flow_style=False, explicit_start=True)

        if self.verbose == 1:
            print(f"Config file updated: {self.ofile}")
    

    def modify_in_yaml(self, key, value, save_config=False):
        '''Use by mdl_history.History()
        Modify variables in original config file:

        Parameters:
        -----------
            key          : string, name of variable to modify
            value        : new value of the variable
            save_config  : bool, save to config if True (default)

        '''

        # changes value:
        self.config[key] = value

        # write yaml file
        if save_config:
            self.save_config()
    
    
    
    
    
if __name__ == "__main__":
    generate_cfiles()

        