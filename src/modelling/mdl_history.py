'''Contains all functions for History parameters
'''

class History():
    '''Contains different timesteps that will be used for the dataset for
    different variables types:
            - 'bias'
            - 'noass'
            - 'forcing'
            - 'covar'
            - 'keras'=global, same for history for all variables
    '''
    def __init__(self, config, verbose=0):
        '''
        
        Modify History with the following functions:
            set history based on array/list you give
                set_SIT_history()   : for SIT Bias and SIT without assimilation
                set_history()       : for forcing and covar
        
            will generate history based on parameters you give
                make_SIT_history()  : 
                make_history()      :
        
        Load from config file with:
            load_from_config()
        
        Save to config file with:
            save_to_config()
        
        
        '''
        
        self.config = config
        self.verbose = verbose
        
        # create default thanks to config
        # ADD parameters in config file
        self.params = {}

        self.params['bias'] = {}
        self.params['noass'] = {}
        self.params['forcing'] = {}
        self.params['covar'] = {}
        self.params['keras'] = {}
        
        if config.__class__.__name__ == 'Config':
            self.load_from_config()
        elif config.__class__.__name__ == 'CreateConfig':
            self.set_default()
    
    def load_from_config(self):
        '''Load History parameters from config file
        '''
    
        self.params = self.config.load_history()
        if self.verbose == 1: 
            print(f'History loaded from config file: {self.config.ifile}')
    
    def save_to_config(self):
        '''Save history parameters to config file
        '''
        
        #if self.config.__class__.__name__ == 'Config':  # in reload_config.py
        self.config.modify_in_yaml('history', self.params)
        #elif self.config.__class__.__name__ == 'CreateConfig':  # in generator_config.py
        #    self.config.modify_params('history', self.params)
        #    self.config.save_config()
    
    def set_zeros(self):
        '''Set history to 0 for all
        '''
    
        self.set_SIT_history('bias', [1])
        self.set_SIT_history('noass', [0])
        self.make_history('covar', [], [0], 0, 0)
        self.make_history('forcing', [], [0], 0, 0)
        self.set_SIT_history('keras', [0])
    
    
    def set_default(self):
        '''Default parameters - arbitrary chosen
        '''

        self.set_SIT_history('bias', [1,2,3,8])
        self.set_SIT_history('noass', [0,1,2,3,8])
        
        self.make_history('covar', ['sisnthick'], [0], 7, 2)
        self.make_history('forcing', ['2T', 'TP'], [0], 7, 2)
        
        self.set_SIT_history('keras', [0])
        
   
    def set_explo_week(self, nweek=1, covar_to_keep=[], fo_to_keep=[]):
        '''Basic exploration parameters over several weeks
        
        Parameters:
        -----------
        
            nweek    : int, number of week to explore
        '''

        self.set_zeros()
        
        self.make_SIT_history('bias', [1], interval=7, nexplo=nweek)
        self.make_SIT_history('noass', [0], 7, nweek)
        self.make_history('covar', covar_to_keep, [0], 7, nweek)
        self.make_history('forcing', fo_to_keep, [0], 7, nweek)
        
        self.make_SIT_history('keras', [0], 7, nweek)
        
        
    def get_minmax(self):
        '''Return needfutur and needpast
        that correspond to the highest positive index and the lowest negative index
        '''
        
        self.compute_minmax()
        
        return self.needpast, self.needfutur
    
    def compute_minmax(self):
        '''Compute needpast and needfutur
        that correspond to the highest positive index and the lowest negative index
        '''
        
        needfutur, needpast = 0, 0

        all_idx = []
        for item in self.params:
            if type(self.params[f'{item}']['H']) is list:
                all_idx += self.params[f'{item}']['H']
            elif type(self.params[f'{item}']['H']) is dict:
                for subitem in self.params[f'{item}']['H']:
                    all_idx += self.params[f'{item}']['H'][f'{subitem}']
        
        if max(all_idx)>0:
            needfutur = max(all_idx)
        if min(all_idx)<0:
            needpast = abs(min(all_idx))
        
        self.needpast = needpast
        self.needfutur = needfutur
        
        
    
    def make_SIT_history(self, field, must_be=[1,2,3], interval=3, nexplo=5):
        '''Set history for 'bias' and 'noass' (SIT variables)
        '''
        
        exploration = [must_be[-1]+i*interval for i in range(1, nexplo+1)]
        
        self.params[field]['H'] = must_be + exploration
        
        
    def set_SIT_history(self, field, arr):
        '''
        Set directly the history for 'bias' or 'noass'
        !! Not done for NEGATIVE HISTORY !!
        todo: + save new parameters in config files in changed by hand/ inside script
        
        Parameters:
        -----------
        
            field           :
            arr             : array of integer, index for History
        '''
    
        self.params[field]['H'] = arr
    
        if self.verbose == 1: print(f'History updated: {field}')  #  todo: save change to config file')

    
    
    def set_history(self, var_to_set, arr):
        '''Set directly the history of a 'forcing' or 'covar' variable inside the corresponding dictionary
        
        Paremeters:
        -----------
            var_to_set        : str, variable to define
            arr               : array of integer, history
        
        '''

        if var_to_set in self.config.forcing_fields:
            fields = 'forcing'
        elif var_to_set in self.config.covar_fields:
            fields = 'covar'
        else:
            raise ValueError(f'Exact variable not found in forcings or covariables: {var_to_set}')
        
        self.params[f'{fields}']['H'][f'{var_to_set}'] = arr
        
#         print(f'{var_to_set} updated. todo: save change to config file')
        if self.verbose == 1: print(f'History updated: {var_to_set}')

        
        
        
    def make_history(self, fields, var_to_explore=['2T', 'MSL'], must_be=[1,2,3], interval=3, nexplo=5):
        '''Create dictionary containing future time steps (H) of History
        to use as input in ML
        for forcings and covariables
        
        Parameters:
        -----------
        
            fields            : str. fields to modify, either 'forcing' (for forcing_fields) or  'covar' (for covar_fields)
            var_to_explore    : array of str. variable name in 'fields' that will have an extended history
            must_be           : array of int, History that will be affected to all fields
            interval          : int, number of days between two history values.
                                for variable in var_to_explore, history will be
                                must_be + interval * nexplo
            nexplo            : int, number of history to explore
        '''
        
        dic_hist = {}
        if fields == 'forcing':
            input_fields = self.config.forcing_fields
        elif fields == 'covar':
            input_fields = self.config.covar_fields

        exploration = [must_be[-1]+i*interval for i in range(1,nexplo+1)]  # indexes for long timescale exploration
        thistory = must_be + exploration

        # var in fo_to_explore will have History on long timescale
        for field in input_fields:
            short_field = field.split('_')[0]
            if short_field in var_to_explore:
                dic_hist[f'{short_field}'] = thistory
            else:
                dic_hist[f'{short_field}'] = must_be

        self.params[f'{fields}']['H'] = dic_hist
    
