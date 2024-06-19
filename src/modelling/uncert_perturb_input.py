"""Class used to create an ensemble of members with perturbed inputs
Some function are from sit_corrected.py
"""

import os
import copy
import numpy as np
import xarray as xr
from random import gauss
from random import seed
from random import random
from sklearn.preprocessing import MinMaxScaler


from src.data_preparation import load_data
from src.feature_extraction import extract_pca
from src.data_preparation import scaling
from src.utils import save_name

class PertPred:
    '''Init with config, dataset, model
    
    Create X perturbation for inputs
    Predict X outputs from model (PC)
    Convert them to SIT
    Gives standard deviation from perturbed ensemble
    '''
    
    
    def __init__(self, config, model, dataset, verbose=1, objective='apply'):
        '''
        
        Parameters:
        -----------
        
            model         :     src.modelling.model_lstm_ModelLSTM, model already loaded
            objective     :     str, 'train' or 'apply', or 'apply91'. 
                                define if ypred is from training (2011-2022), apply (1999-2010), apply91 (1991-1998)
        '''

        self.dataset_ori = copy.deepcopy(dataset)
        self.dataset = copy.deepcopy(dataset)
        self.model = model
        self.config = config
        self.verbose = verbose
        self.objective = objective
        
        self.scalers_file = f'{self.config.rootdir}{self.config.ml_dir}scalers.pkl'
        
        if self.verbose == 1: print('Initialisation...')
            
        self._load_SIT_na()
        self._load_SIC_na()
        self._load_EOF()
        
        # parameters
        self.ndays = dataset['X'].shape[0]
        self.ntimelag = dataset['X'].shape[1]
        self.nvar = dataset['X'].shape[2]
        
        # Contains all results from perturbed inputs
        ypred_perturbed = []
        
    def _load_SIT_na(self):
        '''Load SIT non assimilated (from TOPAZ4b FreeRun or TOPAZ4c)
        to obtain the SIT corrected from ML
        '''
        
        if self.verbose == 1: print('Loading SIT freerun...')
        
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_2011_2022_FreeRun.nc")
        
        if self.objective == 'apply':  # todo: will not work if name is change
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_1999_2010_FreeRun.nc")
        elif self.objective == 'apply91':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_1991_1998_FreeRun.nc")
        
        if self.config.non_ass == 'adjSIC':
            filename = filename[:-3] + '_adjSIC.nc'


        self.sit_na, chronof = load_data.load_nc(filename, 'sithick', X_only=True)
        
        # reshape with original dataset
        _, [self.sit_na] = load_data.trunc_da_to_chrono(self.dataset_ori['chrono'], chronof, [self.sit_na])
    
    def _load_SIC_na(self):
        '''Load SIC from TOPAZ FreeRun
        to average only when SIC > 15%
        '''
        
        if self.verbose == 1: print('Loading SIC...')
        
        
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"siconc_TOPAZ4b23_2011_2022_FreeRun.nc")
        if self.objective == 'apply':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"siconc_TOPAZ4b23_1999_2010_FreeRun.nc")
        elif self.objective == 'apply91':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"siconc_TOPAZ4b23_1991_1998_FreeRun.nc")
        
        if self.config.non_ass == 'adjSIC':
            filename = filename[:-3] + '_adjSIC.nc'
        
        
        self.sic_na, chrono_na = load_data.load_nc(filename, 'siconc', X_only=True)
        
#         if self.objective == 'train':  # should not be necessary
        _, [self.sic_na] = load_data.trunc_da_to_chrono(self.dataset_ori['chrono'], chrono_na, [self.sic_na])
    
    
    def _load_EOF(self):
        
        if self.verbose == 1: print('Loading EOF...')
        
        
        target_field= 'sithick'
        n_components = self.config.n_comp['tp']
        
        filename = os.path.join(self.config.rootdir,self.config.pca_dir,f"pca_{target_field}_{n_components}N_SITerr23_2014_2022.pkl")
        if self.config.non_ass == 'adjSIC':
            filename = os.path.join(self.config.rootdir,self.config.pca_dir,f"pca_{target_field}_{n_components}N_SITerr23_2014_2022_adjSIC.pkl")
            
        
        pca_e = load_data.load_pca(filename)
        
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4err23_2011_2022.nc")
        if self.config.non_ass == 'adjSIC':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4err23_2011_2022_adjSIC.nc")
            
        self.Xe, self.mu_e, RMSE_e, self.chronoe = load_data.load_nc(filename, 'sithick')
    
        maskok = (np.isfinite(self.Xe)).all(dim='time')
        maskok1d = maskok.stack(z=('y','x'))

    # retrieve PC and EOF values
        self.PCs_e = extract_pca.pca_to_PC(pca_e, self.Xe, maskok1d)
        EOF1d_e, self.EOF2d_e = extract_pca.compute_eof(n_components, self.Xe, pca_e, maskok)     

    
    def perturbe_inputs(self, n_pert:int, seed_array=None, max_pert_array=None, noise_color='red'):
        '''
        Add perturbation to initial PCs, and then scale between [0,1]
        
        Parameters:
        -----------
        
        n_pert         :    int, number of pertubations
        seed_array     :    int or array of int, seeds to be passed to noise generator for random state.
        max_pert_array :    float or array of floats, size of n_pert, maximum pertubation allowed for each member of the ensemble
        noise_color      :    string, define which type of noise is added: 'red' (=random walk), or 'white' noise
        '''
        
        ## check function parameters
        if seed_array is None:
            seed_array = np.array([None]*n_pert)
        elif type(seed_array) == int:
            seed_array = np.array([seed_array]*n_pert)
            
        if max_pert_array is None:
            max_pert_array = np.array([None]*n_pert)
        elif type(max_pert_array) == float:
            max_pert_array = np.array([max_pert_array]*n_pert)
            
                        
        ## Check size of input parameters
        assert seed_array.size == n_pert, "Seed array must have the same size as number of perturbations"
        assert max_pert_array.size == n_pert, "Max pert array must have the same size as number of perturbations"
            
        self.n_pert = n_pert
        self.seed_array = seed_array
        self.max_pert_array = max_pert_array
        self.noise_color = noise_color
        
        if self.verbose==1: print(f'Generating {n_pert} perturbations...')
        
        self.dataset_pert = []
        
        ## Create ensemble of n_pert members
        for n in range(n_pert):            
            if noise_color == 'red':
                noise = self.gen_random_walk(a=seed_array[n], max_pert=max_pert_array[n])
            else:
                noise = self.gen_white_noise(a=seed_array[n], max_pert=max_pert_array[n])
                
            
            self.noise = noise
            tmp_X = copy.deepcopy(self.dataset_ori['X']) + copy.deepcopy(self.dataset_ori['X']) * noise

#             tmp_X = scaling.scale_transform(X=tmp_X, scalers_file=self.scalers_file)
            self.dataset['X'] = tmp_X  # scaling.scale_3D(tmp_X)
#             X_scaled = scaling.scale_transform(X=self.X, scalers_file=filename)



            self.dataset_pert += [copy.deepcopy(self.dataset)]

            
        
            
        
    def gen_white_noise(self, a=None, max_pert=0.3):
        '''Generate white noise
        
        Parameters:
        -----------
        
            seed         :  int, initialize internal state from a seed. Default is None, gives random seed.
            max_pert     : float, between [0-1], factor to control the maximum pertubation
        
        '''
        
        if max_pert is None:
            max_pert = 0.3
        
         #seed random number generator
        seed(a)
        
        # Creake white noise with mean and deviation standard
        white_noise = np.array([gauss(0, 1) for i in range(self.ndays*self.ntimelag*self.nvar)])
        
        # reshape to have same size as dataset 'x'
        white_noise = white_noise.reshape((self.ndays, self.ntimelag, self.nvar))
        
        # Scaling between [-1,1]
        scalers = {}
        for ns in range(self.nvar):
            scalers[ns] = MinMaxScaler(feature_range=(-1, 1))
            white_noise[:,:, ns] = scalers[ns].fit_transform(white_noise[:,:, ns])
        
        return white_noise * max_pert


    def gen_random_walk(self, a=None, max_pert=0.3):
        '''Generate random walk time series of length ndays, for ntime lag and nvar
        Resulting format of 3D shape (ndays, ntimelag, nvar)
        Random walk is generate along the time axis (ndays)
        '''
        
        if max_pert is None:
            max_pert = 0.3
        
         #seed random number generator
        seed(a)
        random_walk = np.empty((self.ndays, self.ntimelag*self.nvar))

        for j in range(self.ntimelag*self.nvar):
            random_walk[0, j] = -1 if random() < 0.5 else 1  # first value 
            for i in range(1, self.ndays):  # loop over days
                movement = -1 if random() < 0.5 else 1
                value = random_walk[i-1, j] + movement
                random_walk[i, j] = value
       
        random_walk = random_walk.reshape((self.ndays, self.ntimelag, self.nvar))
        
        # Scaling between [-1,1]
        scalers = {}
        for ns in range(self.nvar):
            scalers[ns] = MinMaxScaler(feature_range=(-1, 1))
            random_walk[:,:, ns] = scalers[ns].fit_transform(random_walk[:,:, ns])

        return random_walk * max_pert


    def predict(self):
        '''Use ML model to predict PCs for each members of the ensemble
        '''
        self.ypred_pert = [] 
        
        ## Predict Input orignial
        self.model.predict_apply(dataset=copy.deepcopy(self.dataset_ori), verbose=0)
        self.ori_ypred_pert = copy.deepcopy(self.model.ypred)
        
        ## Predict ensemble of perturbations
        for n in range(self.n_pert):
            if self.verbose==1: print(f'Predicting PC for perturbation {n}')
            self.model.predict_apply(dataset=self.dataset_pert[n], verbose=0)
            self.ypred_pert += [copy.deepcopy(self.model.ypred)]


    def reconstruct_sit(self):
        '''PC to SIT bias to SIT
        '''        
        
        if self.verbose == 1: print('Reconstructing Sea ice thickness from PCA...')
        
        ## Get SIT from original inputs (no perturbation)
        Xc = xr.dot(self.EOF2d_e, 
                    xr.DataArray(self.ori_ypred_pert, dims=['time','comp'])) + self.mu_e
        self.ori_sit = self.sit_na + Xc.transpose('time', 'y', 'x')
        
        self.sit_pert = []
        
        for n in range(self.n_pert):
            if self.verbose == 1: print(f'For pertubation number {n}')
            
            PCA_est = xr.DataArray(self.ypred_pert[n], dims=['time','comp'])
            Xc = xr.dot(self.EOF2d_e, PCA_est) + self.mu_e
            self.Xc = Xc.transpose('time', 'y', 'x')  # reshape to get time first    

            self.sit_pert += [self.sit_na + self.Xc]

    def compute_std(self, sic_min=0.15):
        '''Return SIT deviation standard from the ensemble of perturbation
        Compute values for Sea Ice concentration > 15%
        '''

        self.sic_min = sic_min
        
        std = np.nanstd([x.where(self.sic_na>sic_min) for x in self.sit_pert], axis=0)
        self.std_t = np.nanmean(std, axis=(1,2))
        std_xy = np.nanmean(std, axis=(0))
        self.std_mean = np.nanmean(std)
        
#         self.std_t = np.nanstd([x.where(self.sic_na>sic_min).mean(('x','y')).to_numpy() for x in self.sit_pert], axis=(0))
#         self.std_mean = np.nanstd([x.where(self.sic_na>sic_min).mean().to_numpy() for x in self.sit_pert], axis=(0))

#         std_xy = np.nanstd([x.where(self.sic_na>sic_min).mean('time').to_numpy() for x in self.sit_pert], axis=(0))
        # convert en DataArray for easy plot
        self.std_xy = xr.DataArray(data=std_xy, dims=["y","x"],
                    coords=dict(y=(["y"], self.ori_sit.y.data), x=(["x"], self.ori_sit.x.data)),
                    attrs=dict(name="Standard Deviation",
                        description="Std of all members of the ensemble of perturbed inputs (PCA)",
                        units="m", standard_name='Std'))
        
        
        
    def compute_mean(self):
        '''Compute average for plot
        '''
        
        sic_min = self.sic_min
        
        self.ori_mean_sit_t = self.ori_sit.where(self.sic_na>sic_min).mean(('x','y'))  # .to_numpy()
        self.ori_mean_sit_xy = self.ori_sit.where(self.sic_na>sic_min).mean(('time'))
        
        
        ## mean time series for each member
        self.means_pert = [x.where(self.sic_na>sic_min).mean(('x','y')).to_numpy() for x in self.sit_pert]
        self.mean_sit_t = np.nanmean(self.means_pert, axis=0)  ## mean time series of ensemble
        
        mean_xy = np.nanmean([x.where(self.sic_na>sic_min).mean('time').to_numpy() for x in self.sit_pert], axis=(0))
        # convert en DataArray for easy plot
        self.mean_xy = xr.DataArray(data=mean_xy, dims=["y","x"],
                    coords=dict(y=(["y"], self.ori_sit.y.data), x=(["x"], self.ori_sit.x.data)),
                    attrs=dict(name="Mean Sea Ice Thickness",
                        description="Ensemble average over all perturbed members",
                        units="m", standard_name='Mean SIT'))
        

        
    def print_stats(self):
        '''Print mean/median/std 
        '''
        
        print(f'Noice color: {self.noise_color} noise')
        print(f'The ensemble of {self.n_pert} perturbed members has the following statistics:')
        print(f'Mean  =  {np.nanmean(self.means_pert):0.03f} m')
        print(f'Std   =  {self.std_mean:0.03f} m')
        
        
        
    def save_uncert(self, odir):
        '''Save ensemble mean and std to .netcdf files
        '''
        
        from datetime import date
        import pandas as pd
                
        
        if not hasattr(self, 'std_t'):
            raise NameError('self.std_t is not defined: call self.compute_std() first.')
     

        std_t = xr.DataArray(data=self.std_t, dims=["time"],
            coords=dict(time=(["time"], self.ori_sit.time.data)),
            attrs=dict(name="Std Sea Ice Thickness",
                description="Spatial standard deviation of the ensemble over all perturbed members",
                units="m", standard_name='Std SIT'))
    
        mean_sit_t = xr.DataArray(data=self.mean_sit_t, dims=["time"],
            coords=dict(time=(["time"], self.ori_sit.time.data)),
            attrs=dict(name="Mean Sea Ice Thickness",
                description="Spatial average of the ensemble over all perturbed members",
                units="m", standard_name='Mean SIT'))
        # -----------------------------------------------------
        # save sit_corrected, sit_non_assimilated, chrono
        first_pred = pd.to_datetime(self.ori_sit['time'][0].data).strftime('%Y-%m-%d')
        last_pred = pd.to_datetime(self.ori_sit['time'][-1].data).strftime('%Y-%m-%d')

        odata = xr.Dataset(data_vars={'sit_mean_t':mean_sit_t,
                                      'sit_mean_xy':self.mean_xy,
                                      'sit_std_t' :std_t,
                                      'sit_std_xy' :self.std_xy,
                                      'ori_sit_mean':self.ori_mean_sit_t,
                                      'ori_sit_mean_xy':self.ori_mean_sit_xy,
                                     },
                   attrs=dict(
                       description='Sea Ice Thickness uncertainty',
                       model_ml=f'LSTM',
                       n_members=f'{self.n_pert}',
                       noise=f'{self.noise_color}',
                       max_pertubation=f'{self.max_pert_array}',
                       first_prediction=f'{first_pred}',
                       last_prediction=f'{last_pred}',
                       author='Leo Edel, Nersc',
                       project='TARDIS',
                       creation_date=f'{date.today()}')
                  )
        
        # -----------------------------------------------------
        
        first_year = pd.to_datetime(self.ori_sit['time'][0].data).strftime('%Y')
        last_year = pd.to_datetime(self.ori_sit['time'][-1].data).strftime('%Y')
        
#         odir = f'{os.path.dirname(self.ifile)}/' 
        oname = f'sit_uncertainty_{first_year}_{last_year}_01.nc'
        
        oname = save_name.check(f"{odir}", oname)
        odata.to_netcdf(f'{odir}{oname}')

        
        # print(self.sit)
        print(f'Saved as: {odir}{oname}')
        
        
        

