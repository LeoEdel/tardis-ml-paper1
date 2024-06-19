"""Optimized parameters hardcoded for Gradient Boosting, Random Forest
"""

# class for ML parameters
from skopt.space import Categorical


def GB_opt_params(random_state=None):
    param_opt = {}
    param_opt['pc0'] = {'gradientboostingregressor__learning_rate': [0.3],
                      'gradientboostingregressor__max_depth': [5], 
                      'gradientboostingregressor__min_samples_leaf': [0.075], 
                      'gradientboostingregressor__n_estimators': [125],
                      'gradientboostingregressor__subsample': [0.27] 
                       }

    param_opt['pc1'] =  {'gradientboostingregressor__learning_rate': [0.333],  # 0.3
                      'gradientboostingregressor__max_depth': [8],  # 8 
                      'gradientboostingregressor__min_samples_leaf': [0.075],  # 0.075
                      'gradientboostingregressor__n_estimators': [200],  # 200
                      'gradientboostingregressor__subsample': [0.27]  # 0.27 
                       }

    param_opt['pc2'] =  {'gradientboostingregressor__learning_rate': [0.62],
                      'gradientboostingregressor__max_depth': [3], 
                      'gradientboostingregressor__min_samples_leaf': [0.03], 
                      'gradientboostingregressor__n_estimators': [260],
                      'gradientboostingregressor__subsample': [0.32]
                       }

    param_opt['pc3'] =  {'gradientboostingregressor__learning_rate': [0.1],
                      'gradientboostingregressor__max_depth': [3], 
                      'gradientboostingregressor__min_samples_leaf': [0.01], 
                      'gradientboostingregressor__n_estimators': [50],  # 300],
                      'gradientboostingregressor__subsample': [0.1]  # 0.1 
                       }

    
    if random_state != None:
        for ky in param_opt.keys():
            param_opt[ky]['gradientboostingregressor__random_state'] = [random_state]
    
    return param_opt



def XGB_opt_params(random_state=None, booster='dart'):
    '''for XGBoost Regressor
    '''
    param_opt = {}
    param_opt['pc0'] = {'xgbregressor__learning_rate': [0.934540199662753],
                      'xgbregressor__max_depth': [7], 
#                       'xgbregressor__min_child_weight': [0.075],
                      'xgbregressor__colsample_bytree': [.2],
                      'xgbregressor__n_estimators': [20],
                      'xgbregressor__subsample': [0.8813580200158647],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.40787863218356557],
                      'xgbregressor__skip_drop': [.4804807260563492]  
                       }

    param_opt['pc1'] =  {'xgbregressor__learning_rate': [0.1],  # 0.3
                      'xgbregressor__max_depth': [5],  # 8 
#                       'xgbregressor__min_child_weight': [0.075],  # 0.075
                      'xgbregressor__colsample_bytree': [.2],
                      'xgbregressor__n_estimators': [178],  # 200
                      'xgbregressor__subsample': [0.253231396202781],  # 0.27 
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.9],
                      'xgbregressor__skip_drop': [.5980437103057147]  
                       }

    param_opt['pc2'] =  {'xgbregressor__learning_rate': [0.1],
                      'xgbregressor__max_depth': [2], 
#                       'xgbregressor__min_child_weight': [0.03], 
                      'xgbregressor__colsample_bytree': [.2],
                      'xgbregressor__n_estimators': [20],
                      'xgbregressor__subsample': [1],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.96192183432204362],
                      'xgbregressor__skip_drop': [.6]  
                       }

    param_opt['pc3'] =  {'xgbregressor__learning_rate': [0.1],
                      'xgbregressor__max_depth': [8], 
#                       'xgbregressor__min_child_weight': [0.01], 
                      'xgbregressor__colsample_bytree': [.2],
                      'xgbregressor__n_estimators': [20],  # 300],
                      'xgbregressor__subsample': [0.1],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.9],
                      'xgbregressor__skip_drop': [.4]  
                       }

    param_opt['pc4'] =  {'xgbregressor__learning_rate': [0.2517266032008606],
                      'xgbregressor__max_depth': [6], 
                      'xgbregressor__colsample_bytree': [.2828800922341152],
                      'xgbregressor__n_estimators': [40],
                      'xgbregressor__subsample': [0.8970505435628179],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.7662421244212114],
                      'xgbregressor__skip_drop': [.4952048725289543]  
                       }
    
    param_opt['pc5'] =  {'xgbregressor__learning_rate': [0.1],
                      'xgbregressor__max_depth': [8], 
                      'xgbregressor__colsample_bytree': [.5085646183074389],
                      'xgbregressor__n_estimators': [20],  # 300],
                      'xgbregressor__subsample': [0.8360066529447301],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.6973449125258281],
                      'xgbregressor__skip_drop': [.9]  
                       }
    
    param_opt['pc6'] =  {'xgbregressor__learning_rate': [0.1],
                      'xgbregressor__max_depth': [4], 
                      'xgbregressor__colsample_bytree': [.2],
                      'xgbregressor__n_estimators': [20],  # 300],
                      'xgbregressor__subsample': [0.1],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.6450810151475734],
                      'xgbregressor__skip_drop': [.6]  
                       }
    
    param_opt['pc7'] =  {'xgbregressor__learning_rate': [1],
                        'xgbregressor__max_depth': [8], 
                      'xgbregressor__colsample_bytree': [1.],
                      'xgbregressor__n_estimators': [175],  # 300],
                      'xgbregressor__subsample': [0.7658450609212719],
                      'xgbregressor__booster': ['dart'],
                      'xgbregressor__sample_type': ['uniform'],
                      'xgbregressor__rate_drop': [.1],
                      'xgbregressor__skip_drop': [.439561953670565]  
                       }
    
    
    
    
    
    
    if random_state != None:
        for ky in param_opt.keys():
            param_opt[ky]['xgbregressor__random_state'] = [random_state]
    
    booster_param = {'xgbregressor__booster': ['dart'],
                        'xgbregressor__sample_type': ['uniform'],
                        'xgbregressor__rate_drop': [.1],
                        'xgbregressor__skip_drop': [.5]}
    
    if booster == 'dart':
        for ky in param_opt.keys():
            for bp in booster_param:
                param_opt[ky][bp] = booster_param[bp]

    return param_opt




def RF_opt_params(random_state=None):
    '''PARAMETERS TOUT NUL: a prendre dun vrai run
    juste pour avoir des params pour run rapidement
    
    todo: add random_state to parameters
    '''
    param_opt = {}
    param_opt['pc0'] = {'randomforestregressor__min_impurity_decrease': [0.3], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [3],
                        'randomforestregressor__n_estimators': [300]}

    param_opt['pc1'] = {'randomforestregressor__min_impurity_decrease': [0.3], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [20],
                        'randomforestregressor__n_estimators': [300]}
    
    param_opt['pc2'] = {'randomforestregressor__min_impurity_decrease': [0.3], 
                        'randomforestregressor__min_samples_leaf': [20],
                        'randomforestregressor__min_samples_split': [3],
                        'randomforestregressor__n_estimators': [50]}
    
    param_opt['pc3'] = {'randomforestregressor__min_impurity_decrease': [0.3], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [20],
                        'randomforestregressor__n_estimators': [50]}
    
    param_opt['pc4'] = {'randomforestregressor__min_impurity_decrease': [0.3], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [20],
                        'randomforestregressor__n_estimators': [50]}
    
    param_opt['pc5'] = {'randomforestregressor__min_impurity_decrease': [0], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [20],
                        'randomforestregressor__n_estimators': [50]}
    
    param_opt['pc6'] = {'randomforestregressor__min_impurity_decrease': [0], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [20],
                        'randomforestregressor__n_estimators': [50]}
    
    param_opt['pc7'] = {'randomforestregressor__min_impurity_decrease': [0.3], 
                        'randomforestregressor__min_samples_leaf': [1],
                        'randomforestregressor__min_samples_split': [20],
                        'randomforestregressor__n_estimators': [50]}
    
    return param_opt

    
    
def Rridge_opt_params():
    pass
    
    

class Params:
    def __init__(self, ml_model, ncomp):
        '''
        Parameters:
        -----------
        
        ml_model       :   string, Machine learning model
        ncomp          :   int, number of PCA for target variable
        
        
        '''
        self.ml_model = ml_model
        self.ncomp = ncomp
        self.param_opt = {}
        
        for Nf in range(self.ncomp):
            self.param_opt[f'pc{Nf}'] = {}
        
        
    def get_search(self, bayesian=True, booster=None):
        '''Get parameters for GridSearch or BayesianSearch
        '''
        if bayesian:
            if self.ml_model == 'xgb':
                param_grid = {'xgbregressor__learning_rate': (0.1,1),
                        'xgbregressor__max_depth': (2,8), 
                        'xgbregressor__colsample_bytree': (.2, 1),
                        'xgbregressor__n_estimators': (20, 300),
                        'xgbregressor__subsample': (0.1, 1),
                        }
                if booster == 'dart':
                    param_grid.update({'xgbregressor__booster': ['dart'],  # Categorical(['dart','gbtree']),
                        'xgbregressor__sample_type': ['uniform'],  # Categorical(['uniform', 'weighted']),
                        'xgbregressor__rate_drop': (.1,.9),
                        'xgbregressor__skip_drop': (.4,.6)})
            
            elif self.ml_model == 'grdbst':
                param_grid = {'gradientboostingregressor__subsample':(0.1, 1),
                      'gradientboostingregressor__learning_rate': (0.1, 1),
                      'gradientboostingregressor__min_samples_leaf': (.01, .1),
                      'gradientboostingregressor__n_estimators': (50, 300),
                      'gradientboostingregressor__max_depth': (3, 10)}
            elif self.ml_model == 'rf':
                param_grid = {'randomforestregressor__min_impurity_decrease': (0, 0.3),
                      'randomforestregressor__min_samples_leaf': (1, 20), 
                      'randomforestregressor__min_samples_split': (3,20),
                      'randomforestregressor__n_estimators': (50, 300)}
            elif self.ml_model == 'ridge':
                param_grid = {'ridge__alpha':(1e-3, 10), 'ridge__random_state':[0]}

        else:
            if self.ml_model == 'xgb':
                param_grid = {'xgbregressor__n_estimators': [100]
                            }
            
            elif self.ml_model == 'grdbst':
                    param_grid = {'gradientboostingregressor__subsample':[0.7,.9], 
                                  'gradientboostingregressor__min_impurity_decrease': [0],
                                  'gradientboostingregressor__min_samples_leaf': [10,50], 
                                  'gradientboostingregressor__min_samples_split': [5,10], 
                                  'gradientboostingregressor__n_estimators': [200], 
                                  'gradientboostingregressor__max_depth': [10,30]}
            elif self.ml_model == 'rf':
                param_grid = {'randomforestregressor__random_state':[0],
                              'randomforestregressor__min_impurity_decrease': [0],
                              'randomforestregressor__min_samples_leaf': [1],
                              'randomforestregressor__min_samples_split': [3],
                              'randomforestregressor__n_estimators': [200]}
            elif self.ml_model == 'ridge':
                param_grid = {'ridge__alpha':[1e-3, 1e-2, 1e-1, 1, 10], 'ridge__random_state':[0]}
            
        for ipca in range(self.ncomp):
            self.param_opt[f'pc{ipca}'] = param_grid
        
        
        
    def optimize(self, random_state=None, booster=None):
        '''Get optimized hardcoded parameters for each PCA
        
        Parameters:
        -----------
            rand0          :   random state, if not None, will be set to INT so results are reproductible
        '''
        
        if self.ml_model == 'grdbst':
            if self.ncomp >4:
                print('Only done for 4 PCA !')
            self.param_opt = GB_opt_params(random_state=random_state)
        elif self.ml_model == 'xgb':
            self.param_opt = XGB_opt_params(random_state=random_state, booster=booster)
        elif self.ml_model == 'rf':
            self.param_opt = RF_opt_params(random_state=random_state)
        else:
            raise TypeError(f'Best ML parameters not defined for <{self.ml_model}> model. Do it !')
   


