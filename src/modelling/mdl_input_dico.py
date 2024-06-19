"""inputs dico initialisation 
best parameters for config are also hard coded here

Made for GradientBoosting and Random Forest
"""

import numpy as np


def switch_off(array, to_false):
    '''Put all variables of a input_dico to false if in to_false

    Inputs:
        array    :  input_dico[n] array
        to_false :  array of string containing variables names 

    '''
    new_array = array.copy()
    
    # check that all have been found
    check = np.zeros((len(to_false)))
    
    for idx, item in enumerate(array[:,0]):
        if item in to_false:
            check[to_false.index(item)] = 1
            new_array[idx, 1] = str(False)
            continue
             
    if not check.all():
        print('Variables not found:')
        print(np.array(to_false)[check==0])
                     
    return new_array



def switch_on(array, to_true):
    '''Put all variables of a input_dico to True if in to_true

    Inputs:
        array    :  input_dico[n] array
        to_false :  array of string containing variables names 

    '''
    new_array = array.copy()
    
    # check that all have been found
    check = np.zeros((len(to_true)))
    
    for idx, item in enumerate(array[:,0]):
        if item in to_true:
            check[to_true.index(item)] = 1
            new_array[idx, 1] = str(True)
            continue
             
    if not check.all():
        print('Variables not found:')
        print(np.array(to_true)[check==0])
                     
    return new_array



def all_to_false_except(array, to_true):
    '''Put all variables of a input_dico to false except the ones listed in to_true

    Inputs:
        array    :  input_dico[n] array
        to_true  :  array of string containing variables names 

    '''
    new_array = array.copy()
    for idx, item in enumerate(array[:,0]):
        if item in to_true:
#             print(item + ' True')
            new_array[idx, 1] = str(True)
            continue
        new_array[idx, 1] = str(False)

    return new_array

def all_to_true_except(array, to_false):
    '''Put all variables of a input_dico to true except the ones listed in to_false

    Inputs:
        array    :  input_dico[n] array
        to_false  :  array of string containing variables names 

    '''
    new_array = array.copy()
    for idx, item in enumerate(array[:,0]):
        if item in to_false:
            new_array[idx, 1] = str(False)
            continue
        new_array[idx, 1] = str(True)

    return new_array



def GB_best_variables():
    '''Variables showing the best results (used for TARDIS 1 meeting)
    - for now (20220901)
    '''
    # change input dico 
    to_true = {}

    # from 220818-174505  - meilleure tendance !
    to_true['pc0'] = ['SITf t+0 PC0', 'hsnw00 t+0 PC1', 'SITf t+3 PC1', 'hsnw00 t+3 PC3', 'SITf t+0 PC2',
                      'SITf t+1 PC0', 'hsnw00 t+3 PC1', 'airtmp t+1 PC1', 'hsnw00 t+0 PC2', 'airtmp t+2 PC3',
                      'hsnw00 t+0 PC3', 'shwflx t+1']

    # CA SAUVE LES MEUBLES
    to_true['pc1'] = ['SITf t+2 PC1', 'wndnwd t+3 PC1', 'hsnw00 t+2 PC0',
     'airtmp t+2 PC0', 'hsnw00 t+0 PC1', 'SITf t+2 PC2', 'SITf t+0 PC0', 'wndnwd t+1 PC3']

    # from 220818-174525
    to_true['pc2'] = ['SITf t+0 PC3', 'vapmix t+1 PC0', 'swflx0 t+0 PC0', 'airtmp t+1 PC3', 'SITf t+3 PC3',
                      'SITf t+3 PC1', 'wndewd t+0 PC1', 'airtmp t+3 PC2', 'precip t+1 PC3', 'airtmp t+2 PC1', 
                     'mslprs t+1 PC3', 'wndewd t+1 PC0', 'airtmp t+2 PC2', 'wndewd t+2 PC2', 'vapmix t+3 PC3',]

    # from 220818-174505
    to_true['pc3'] = ['wndewd t+3 PC0', 'fice00 t+3 PC2', 'swflx0 t+0 PC2', 'wndewd t+1 PC0', 'SITf t+0 PC2',
                      'precip t+1 PC1', 'SITf t+2 PC2', 'swflx0 t+1 PC2', 'wndewd t+0 PC0', 'ssh00 t+1 PC0',
                      'swflx0 t+2 PC3', 'precip t+0 PC1']

    return to_true


def GB_annoying_var():
    to_false = {}

    to_false['pc0'] = ['Xe t+1 PC0','Xe t+2 PC0']
    to_false['pc1'] = ['Xe t+1 PC1','Xe t+2 PC1', 'Xe t+1 PC0']
    to_false['pc2'] = ['Xe t+1 PC2','Xe t+2 PC2']
    to_false['pc3'] = ['Xe t+1 PC3','Xe t+2 PC3', 'fice00 t+3 PC2']

    return to_false


class InputDico:
    def __init__(self):
        '''
        '''
    
    
    def load_input_dico(self, idir, ifile):
        '''Load existing input_dico.npy file
        '''

        # if idir/ifile empty
        # return empty input_dico

        dicoback = np.load(f'{idir}{ifile}', allow_pickle=True)
        input_dico = dicoback[None][0]
        self.dico = input_dico



    def init_true(self, ncomp, totlabels):
        '''
        Parameters:
        -----------
            ncomp     : number of PCA for target variable
            totlabels : list of variable names
        
        returns all True dico corresponding for the ML datasets
        '''
        self.ncomp = ncomp

        dico = {}  
        for n in range(ncomp):
            tmp_dico = np.empty((len(totlabels),3), '<U24')
            for idx, var in enumerate(totlabels):
                tmp_dico[idx] = [f'{var}', 'True', '0']

            dico[f'pc{n}'] = tmp_dico
            
        self.dico = dico


    def keep_only_best(self, ml_model):
        '''Only set to True inputs defined as best (hardcoded in this script)
        
        Parameters:
        -----------
            ml_model    : machine learning model used (grdbst, rf, ridge)
        
        '''
        # get best input variables for ML
        if ml_model == 'xgb':
            to_true = GB_best_variables()
        elif ml_model == 'grdbst':
            to_true = GB_best_variables()
        else:
            raise TypeError(f'Best input variables not defined for {ml_model} algo ! Do it !')
        
        # put all other to False except the BEST
        for Nf in range(self.ncomp):
            self.dico[f'pc{Nf}'] = all_to_false_except(self.dico[f'pc{Nf}'].copy(), to_true[f'pc{Nf}'])

    
    def switch_to_false(self, var):
        '''Put var to False in input_dico
        '''
        for Nf in range(self.ncomp):
            self.dico[f'pc{Nf}'] = switch_off(self.dico[f'pc{Nf}'].copy(), var)
        
    
    def switch_to_true(self, var):
        '''Put var to True in input_dico
        '''
        for Nf in range(self.ncomp):
            self.dico[f'pc{Nf}'] = switch_on(self.dico[f'pc{Nf}'].copy(), var)
    
    