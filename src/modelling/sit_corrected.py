"""
This script convert Principal Components prediction (by machien learning/LSTM) into Sea Ice Thickness
"""

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc4
from datetime import date

import src.data_preparation.load_data as load_data
from src.utils import save_name
from src.utils import reload_config
from src.feature_extraction import extract_pca
from src.feature_extraction import baseline
from src.feature_extraction import mean_error

class SITCorrected:
    def __init__(self, ifile, name='ML', verbose=0, objective='train'):
        '''
        Reconstruct SIT corrected by global ML
        
        
        Parameters:
        -----------
            ifile         : string, absolute path to folder created by build_ML.py
            verbose       : int. 0 = no talk, 1 = talk.
            objective     : str, 'train' or 'apply', or 'apply91'. 
                            define if ypred is from training (2011-2022), apply (1999-2010), apply91 (1991-1998)
        
        '''
        
        self.ifile = ifile
        self.verbose = verbose
        self.name = name
        self.objective = objective
        
        self._load_config()        
                
        if self.verbose == 1: print(f'\nInitialisation SIT from {name}')
        self._load_EOF()
        self._load_SIT_na()
        self._load_SIT_a()
        self._load_SIC_na()
        
        self._get_ypred_filename()
        self._load_from_nc()
        
        self._load_baseline()  # update: need self.chrono from .nc > so call last

        
    def _load_config(self):
        '''Import all information from config file
        '''
        
        self.config = reload_config.Config(self.ifile, verbose=self.verbose)
        self.config.create_dir_sit_rec()
        
        
    
    def _get_ypred_filename(self, application=False):
        '''get filename from /ml/ for ypred_xxxxx.nc
        '''
        
#         nc_file = glob.glob(f'{self.ifile}ml/ypred_{self.name}.nc', recursive=False)
        nc_file = glob.glob(f'{self.ifile}ml/ypred_*2022.nc', recursive=False)
        
        if self.objective == 'apply':  # todo change: will not work if 2000 is not written in the filename
            nc_file = glob.glob(f'{self.ifile}ml/ypred*1998*.nc', recursive=False)
        elif self.objective == 'apply91':
            nc_file = glob.glob(f'{self.ifile}ml/ypred*1991*.nc', recursive=False)
            
        
        if len(nc_file)==1:
            self.ifile = nc_file[0]
            if self.verbose == 1:
                print(f'ML prediction .nc file found: {nc_file[0]}')
                
        elif len(nc_file)==0:  # check if .npy (old format)
            if self.verbose == 1: print('.nc file not found. Scanning for .npy...')
            nc_file = glob.glob(f'{self.ifile}ml/yrec*.npy', recursive=False) + glob.glob(f'{self.ifile}ml/ypred*.npy', recursive=False)
            if len(nc_file)>=1:
                self.ifile = nc_file[0]
                if self.verbose == 1: print(f'ML prediction .npy file found: {nc_file[0]}')
            else:
                raise ValueError(f'Too many (>1) or 0 .npy file(s) found: {self.ifile}')
                
        else:
            print(nc_file)
            raise ValueError(f'Too many (>1) or 0 .nc file(s) found: {self.ifile}')
        
        
        
    
    def _load_from_nc(self):
        '''Load PCs prediction
        from .nc file containg ypred, ytrue and chrono
        '''
        
        
        nc = nc4.Dataset(self.ifile, mode='r')
        
        try:  # global prediction
            self.ypred = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['ypred']
            self.ytrue = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['ytrue']
            self.chrono = pd.DataFrame({'date':pd.to_datetime(self.ypred['time'].to_numpy())})
        except:  # ConvLSTM2D
            self.ypred = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['sit_pred']
            self.chrono = pd.DataFrame({'date':pd.to_datetime(self.ypred['time'].to_numpy())})
            
        
    
    def _load_EOF(self):
        target_field= 'sithick'
        n_components = self.config.n_comp['tp']
        
        filename = os.path.join(self.config.rootdir,self.config.pca_dir,f"pca_{target_field}_{n_components}N_SITerr23_2014_2022.pkl")
        if self.config.non_ass == 'adjSIC':
            filename = os.path.join(self.config.rootdir,self.config.pca_dir,f"pca_{target_field}_{n_components}N_SITerr23_2014_2022_adjSIC.pkl")
            
        
        pca_e = load_data.load_pca(filename)
#         pca_e=self.pca_e
        
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4err23_2011_2022.nc")
        if self.config.non_ass == 'adjSIC':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4err23_2011_2022_adjSIC.nc")
            
        self.Xe, self.mu_e, RMSE_e, self.chronoe = load_data.load_nc(filename, 'sithick')
    
        maskok = (np.isfinite(self.Xe)).all(dim='time')
        maskok1d = maskok.stack(z=('y','x'))
    
    # retrieve PC and EOF values
        self.PCs_e = extract_pca.pca_to_PC(pca_e, self.Xe, maskok1d)
        EOF1d_e, self.EOF2d_e = extract_pca.compute_eof(n_components, self.Xe, pca_e, maskok)

    def _load_SIT_na(self):
        '''Load SIT non assimilated (from TOPAZ4b FreeRun or TOPAZ4c)
        to obtain the SIT corrected from ML
        '''
        
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_2011_2022_FreeRun.nc")
        
        if self.objective == 'apply':  # todo: will not work if name is change
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_1999_2010_FreeRun.nc")
        elif self.objective == 'apply91':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_1991_1998_FreeRun.nc")
        
        if self.config.non_ass == 'adjSIC':
            filename = filename[:-3] + '_adjSIC.nc'


        self.sit_na, chronof = load_data.load_nc(filename, 'sithick', X_only=True)
        if self.objective == 'train':
            _, [self.sit_na] = load_data.trunc_da_to_chrono(self.chronoe, chronof, [self.sit_na])
        
    def _load_SIT_a(self):
        '''Load SIT assimilated (from TOPAZ4b)
        for comparison
        '''
        
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_2011_2022.nc")
        if self.config.non_ass == 'adjSIC':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"sithick_TOPAZ4b23_2011_2022_adjSIC.nc")
        
        
        self.sit_a, chrono_a = load_data.load_nc(filename, 'sithick', X_only=True)
        if self.objective == 'train':
            _, [self.sit_a] = load_data.trunc_da_to_chrono(self.chronoe, chrono_a, [self.sit_a])
    
    
    def _load_SIC_na(self):
        '''Load SIC from TOPAZ FreeRun
        to average only when SIC > 15%
        '''
        filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"siconc_TOPAZ4b23_2011_2022_FreeRun.nc")
        if self.objective == 'apply':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"siconc_TOPAZ4b23_1999_2010_FreeRun.nc")
        elif self.objective == 'apply91':
            filename = os.path.join(self.config.rootdir, self.config.pca_dir, f"siconc_TOPAZ4b23_1991_1998_FreeRun.nc")
        
        if self.config.non_ass == 'adjSIC':
            filename = filename[:-3] + '_adjSIC.nc'
        
        
        self.sic_na, chrono_na = load_data.load_nc(filename, 'siconc', X_only=True)
        
        if self.objective == 'train':  # should not be necessary
            _, [self.sic_na] = load_data.trunc_da_to_chrono(self.chronoe, chrono_na, [self.sic_na])
    
    def _load_baseline(self):
        '''load the monthly correction
        and apply it on the time serie of SIT non assimilated
        is better because will work whatever the period (also before 2011-2019)
        '''

        basefile = os.path.join(self.config.rootdir, self.config.pca_dir, f"Baseline_monthly_error_2014_2022.nc")
        if self.config.non_ass == 'adjSIC':
            basefile = os.path.join(self.config.rootdir, self.config.pca_dir, f"Baseline_monthly_error_2014_2022_adjSIC.nc")
            
        nc = nc4.Dataset(basefile, mode='r')
        bl_xe_mm = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))['Xe_mm']
        
        
        # apply to self.sit_na, chrono_a
        self.sit_bl = mean_error.apply_mean_correction(self.chrono, self.sit_na, bl_xe_mm)
        
        # compute rmse_bl: done in self.compute_rmse(), self.compute_corr(), self.compute_bias()
    
    def reconstruct_sit(self):
        '''PCs to SIT bias to SIT
        '''
        
        PC_est = xr.DataArray(self.ypred, dims=['time','comp'])  # 8 PCs estimated by ML
        Xc = xr.dot(self.EOF2d_e,PC_est) + self.mu_e  # transform into SIT bias
        self.Xc = Xc.transpose('time', 'y', 'x')  # reshape to get time first    
        
        self.sit = self.sit_na + self.Xc  # SIT bias is added to SIT from TOPAZ4-Freerun
        
        
        # print('todo: get ntest from config file or results file')
        # BECAUSE different for each model -> not the same parameters -> not the same length
        try:
            self.ntest = self.config.ntest
        except:
            print('Old config, variable <ntest> not found')
            self.ntest = int(self.sit.shape[0]*0.2)
            
        
    def compute_rmse(self):

        
        # real rmse: error reconstructed - error initial
        print(f'Size of the test:{self.ntest}')
        self.RMSE_corr = np.sqrt((np.square(self.Xc.isel(time=slice(None,self.ntest))-
                                            self.Xe.isel(time=slice(None,self.ntest)))).mean(dim='time'))

        # rmse induced by PCA transformation: error reconstructed PCa - error inital 
        # = best possible rmse after correction
        self.Xe_rec = xr.dot(self.EOF2d_e, self.ytrue) + self.mu_e
        self.RMSE_rece = np.sqrt((np.square(self.Xe_rec.isel(time=slice(None,self.ntest))
                                            -self.Xe.isel(time=slice(None,self.ntest)))).mean(dim='time'))
        
        # baseline - simplest correction
       #  import pdb; pdb.set_trace()
        self.rmse_bl = np.sqrt((np.square(self.sit_bl.isel(time=slice(None,self.ntest))-self.sit_na.isel(time=slice(None,self.ntest)))))
        self.RMSE_bl = self.rmse_bl.isel(time=slice(None,self.ntest)).mean(dim='time')
#         self.RMSE_mc_av = self.RMSE_mc.mean()
        
        #  time serie
        self.RMSE_na_t = np.sqrt((np.square(self.Xe)).mean(dim=('x','y')))
        self.RMSE_ml_t = np.sqrt((np.square(self.Xc-self.Xe)).mean(dim=('x','y')))
        self.X_bl = self.sit_bl - self.sit_na  # bias between SIT from baseline correction and SIT from non assimilated
        self.RMSE_bl_t = np.sqrt((np.square(self.X_bl-self.Xe)).mean(dim=('x','y')))
        
        # global average, without RMSE == 0
        self.RMSE_corr_av = self.RMSE_corr.where(self.RMSE_corr>0).mean()
        self.RMSE_rece_av = self.RMSE_rece.where(self.RMSE_rece>0).mean()
        self.RMSE_bl_av = self.RMSE_bl.where(self.RMSE_bl>0).mean()
        
        self.evaluate_bias()
        
    def compute_bias(self):
        ''' Product - reference (=TP4b)
        '''

        self.bias_na = - self.Xe.mean(dim=('x','y'))
        self.bias_bl = (self.sit_bl - self.sit_a).mean(dim=('x','y'))
        self.bias_ml = (self.sit - self.sit_a).mean(dim=('x','y'))
    
    def compute_corr(self):
        '''Compute correlation 
        '''
        
#         self.corr_na = xr.corr(self.sit_na.isel(time=slice(None,self.ntest)), 
#                                self.sit_a.isel(time=slice(None,self.ntest)), dim=('time'))
        
#         self.corr_na_t = xr.corr(self.sit_na.isel(time=slice(None,self.ntest)), 
#                                  self.sit_a.isel(time=slice(None,self.ntest)), dim=('y','x'))
        
#         self.corr_ml = xr.corr(self.sit.isel(time=slice(None,self.ntest)), 
#                                self.sit_a.isel(time=slice(None,self.ntest)), dim=('time'))
#         self.corr_ml_t = xr.corr(self.sit.isel(time=slice(None,self.ntest)), 
#                                  self.sit_a.isel(time=slice(None,self.ntest)), dim=('y','x'))
        
#         self.corr_bl = xr.corr(self.sit_bl.isel(time=slice(None,self.ntest)), 
#                                self.sit_a.isel(time=slice(None,self.ntest)), dim=('time'))
#         self.corr_bl_t = xr.corr(self.sit_bl.isel(time=slice(None,self.ntest)), 
#                                  self.sit_a.isel(time=slice(None,self.ntest)), dim=('y','x'))
        
        self.corr_na = xr.corr(self.sit_na,
                               self.sit_a, dim=('time'))
        
        self.corr_na_t = xr.corr(self.sit_na,
                                 self.sit_a, dim=('y','x'))
        
        self.corr_ml = xr.corr(self.sit,
                               self.sit_a, dim=('time'))
        self.corr_ml_t = xr.corr(self.sit,
                                 self.sit_a, dim=('y','x'))
        
        self.corr_bl = xr.corr(self.sit_bl,
                               self.sit_a, dim=('time'))
        self.corr_bl_t = xr.corr(self.sit_bl,
                                 self.sit_a, dim=('y','x'))
    
    def evaluate_bias(self):
                # save output to .txt file
        print(f'Average of the corrected model error: {self.RMSE_corr_av.values:.2f}')
        print(f'Average of the model error reconstruction (lower bound): {self.RMSE_rece_av.values:.2f}')
        # print(f'Average of the model error (upper bound): {RMSE_e_av.values:.2f}')
        print(f'Average of the baseline error correction (upper bound): {self.RMSE_bl_av.values:.2f}')
        
    def compute_mean(self, sit_min=None, sic_min=None):
        '''Compute time serie
        include ice-free gridpoint
        
        Parameters:
        -----------
        sit_min         : float, if not None, temporal mean will only include grid cell with SIT > sit_min
                          typically 0m to exclude ice-free ocean (or 0.2m ?) 
                          
        sic_min        : float [0,1], if not None, only SIC > sic_min will be averaged. Use SIC from TOPAZ Freerun
        
        '''
        
        
        self.sit_am = self.sit_a.mean(axis=(1,2))
        self.sit_nam = self.sit_na.mean(axis=(1,2))
        self.sit_m = self.sit.mean(axis=(1,2))
        self.sit_blm = self.sit_bl.mean(('y','x'))
        
        # mean only SIT > 0 grid cell
        if sit_min is not None:
            self.sit_am = self.sit_a.where(self.sit_a>sit_min).mean(dim=('y','x')).compute()
            self.sit_nam = self.sit_na.where(self.sit_nam>sit_min).mean(dim=('y','x')).compute()
            self.sit_m = self.sit.where(self.sit_m>sit_min).mean(dim=('y','x')).compute()
            self.sit_blm = self.sit_bl.where(self.sit_bl>sit_min).mean(('y','x'))
            
        if sic_min is not None:
            self.sit_am = self.sit_a.where(self.sic_na>sic_min).mean(dim=('y','x')).compute()
            self.sit_nam = self.sit_na.where(self.sic_na>sic_min).mean(dim=('y','x')).compute()
            self.sit_m = self.sit.where(self.sic_na>sic_min).mean(dim=('y','x')).compute()
            self.sit_blm = self.sit_bl.where(self.sic_na>sic_min).mean(('y','x'))
            
    
    def compute_improvement(self):
        '''Compute skill score: % of improvement
        skill =  1 - [ RMSE (LSTM/ML) / RMSE (no correction) ]

        1: RMSE ML decreases: improvement increases
        0: RMSE ML equals RMSE without assimilation: the improvement is neutral
        '''
        
        # compute for all days
        nt = self.Xe.shape[0]
        
        ### !! only compute if SIT > 0m for TOPAZ freerun and ML !!
        ##  >> even with this, still need to clean
        ##  >> values close to -Inf because rmse_fr small and close to rmse_ml
        
        # rmse machine learning compare to Truth (TOPAZ4b)
##         rmse_ml = np.sqrt(np.square(self.Xe_rec.isel(time=slice(None,nt))-self.Xe.isel(time=slice(None,nt))))
#         rmse_ml = np.sqrt(np.square(self.Xe_rec.where(self.Xe_rec.isel(time=slice(None,nt))).isel(time=slice(None,nt))-self.Xe.where(self.Xe.isel(time=slice(None,nt))).isel(time=slice(None,nt))))
        
        
        # rmse freerun compare to Truth (TOPAZ4b)
#         rmse_fr = np.sqrt(np.square(self.sit_na.where(self.sit_na.isel(time=slice(None,nt))).isel(time=slice(None,nt))-self.Xe.where(self.Xe.isel(time=slice(None,nt))).isel(time=slice(None,nt))))
    
#         self.sk = 1 - (rmse_ml / rmse_fr)
    
    
        # ss = 1 - (mse / mse_clim)
        
        # average on (x,y)
        mse_clim = np.square(self.Xe.isel(time=slice(None,nt))).mean(('x','y'))
        mse_ml =  np.square(self.Xc.isel(time=slice(None,nt))-self.Xe.isel(time=slice(None,nt))).mean(('x','y'))
        self.ss_clim_t = 1 - (mse_ml / mse_clim)
    
        # average on (time)
        mse_clim = np.square(self.Xe.isel(time=slice(None,nt))).mean(('time'))
        mse_ml =  np.square(self.Xc.isel(time=slice(None,nt))-self.Xe.isel(time=slice(None,nt))).mean(('time'))
        self.ss_clim_xy = 1 - (mse_ml / mse_clim)
    
    
    
        # clean skill score:
        # monthly skill score:
        
#         skm = sk.resample(time='1M').mean(dim='time')

        # cap negative value to 0
#         skc = sk.where(sk>=0, 0)

        # put nan when there is not sea ice
#         skc = sk.where(landmask!=1)

        # invalid skill score (-inf)
#         skc = sk.where(sk!=-np.Inf, -1)
    
    
    def compare_bias_ml_da(self):
    
        # Correction by DA
        corr_ass = self.Xe.isel(time=slice(None,self.ntest)).to_numpy().reshape((self.ntest*479*450))
        # Correction by ML
        corr_ml = self.Xc.isel(time=slice(None,self.ntest)).to_numpy().reshape((self.ntest*479*450))
        
        sit_na = self.sit_na.isel(time=slice(None,self.ntest)).to_numpy().reshape((self.ntest*479*450))
        
        # Remove NaN index due to land
        idx_nan = np.where(np.isnan(corr_ass))[0]
        corr_ass = np.delete(corr_ass, idx_nan)
        corr_ml = np.delete(corr_ml, idx_nan)
        sit_na = np.delete(sit_na, idx_nan)
        
        self.diff_corr = (corr_ml - corr_ass)
        
#         # Remove NaN index due to SIT=0
#         idx_nan_diff = np.where(np.isnan(diff_corr))[0]

#         diff_corr = np.delete(diff_corr, idx_nan_diff)
#         sit_na = np.delete(sit_na, idx_nan_diff)
        
#         # Remove NaN index due to substraction
#         idx_inf = np.where(np.isinf(diff_corr))[0]

#         self.diff_corr = np.delete(diff_corr, idx_inf)
#         sit_na = np.delete(sit_na, idx_inf)
        
        self.bias_diff_mean, self.bias_diff_std, self.bins, self.n_pixels = self.compute_corr_af_sit(self.diff_corr, sit_na)
        
        
        
    # get mean and std for each bin of SIT (freerun) of 0.1m

    def compute_corr_af_sit(self, corr, sit_na):
        '''Compute mean correction bias as function of the SIT
        Parameters:
        -----------

            corr    : 1D array, difference of correction between ML and DA
            sit_na  : 1D array, SIT (m) from TOPAZ freerun
        '''

        bin_width = 0.1
        bins_x = np.arange(0, 10+bin_width, bin_width)

        mean_corr = []
        std_corr = []
        n_pixels = []


        for nb, b in enumerate(bins_x):
            idx = np.where((sit_na>b)&(sit_na<=b+bin_width))
            n_pixels += [len(idx[0])]
            mean_corr += [np.nanmean(corr[idx])]
            std_corr += [np.nanstd(corr[idx])]


        return np.array(mean_corr), np.array(std_corr), bins_x, np.array(n_pixels)
        
    
    def save_sit(self):
        '''Save SIT reconstructed with ML as .netcdf
        '''
        
        
        if not hasattr(self, 'sit'):
            raise NameError('self.sit is not defined: call self.reconstruct_sit() first.')
       # elif not hasattr(self, 'bias_ml'):
        #    raise NameError('self.bias_ml is not defined: call self.compute_bias() first.')
        
        
        # -----------------------------------------------------
        # save sit_corrected, sit_non_assimilated, chrono
        first_pred = pd.to_datetime(self.Xc['time'][0].data).strftime('%Y-%m-%d')
        last_pred = pd.to_datetime(self.Xc['time'][-1].data).strftime('%Y-%m-%d')
        
        
        if self.objective == 'train':
            odata = xr.Dataset(data_vars={'sit_ml':self.sit,
                                  'sit_na':self.sit_na,
                                  'bias_ml':self.sit - self.sit_a},
                       attrs=dict(
                           description='Sea Ice Thickness reconstructed by ML',
                           model_ml=f'{self.name}',
                           first_prediction=f'{first_pred}',
                           last_prediction=f'{last_pred}',
                           author='Leo Edel, Nersc',
                           project='TARDIS',
                           creation_date=f'{date.today()}')
                      )
        elif 'apply' in self.objective:
            odata = xr.Dataset(data_vars={'sit_ml':self.sit,
                                  'sit_na':self.sit_na},
                       attrs=dict(
                           description='Sea Ice Thickness reconstructed by ML',
                           model_ml=f'{self.name}',
                           first_prediction=f'{first_pred}',
                           last_prediction=f'{last_pred}',
                           author='Leo Edel, Nersc',
                           project='TARDIS',
                           creation_date=f'{date.today()}')
                      )
        
        # -----------------------------------------------------
        
        first_year = pd.to_datetime(self.sit['time'][0].data).strftime('%Y')
        last_year = pd.to_datetime(self.sit['time'][-1].data).strftime('%Y')
        
        odir = f'{os.path.dirname(self.ifile)}/' 
        oname = f'sit_g{self.name}_{first_year}_{last_year}_01.nc'
        
        oname = save_name.check(f"{odir}", oname)
        odata.to_netcdf(f'{odir}{oname}')

        
        # print(self.sit)
        print(f'Saved as: {odir}{oname}')
        