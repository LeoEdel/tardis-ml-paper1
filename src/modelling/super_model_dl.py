"""
Super Class for LSTM, CNN and AK models
This script contains functions for training, prediction, saving weigths

ML architecture are written in the corresponding scripts:

LSTM >> /src/modelling/model_lstm.py
CNN >> /src/modelling/model_cnn.py 
Auto Keras >> /src/modelling/model_autokeras.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from contextlib import redirect_stdout
import xarray as xr
import netCDF4
import datetime
import copy

from tensorflow.keras import models as km  # import
import tensorflow as tf  # for callback

from src.utils import save_name
from src.visualization import mdl_non_recursive

models = ['cnn', 'lstm', 'ak']

def print_to_txt(model, ofile):
    '''print model.summary() in txt file
    '''
    with open(f'{ofile}','w') as f:
        with redirect_stdout(f):
            model.summary()

class SModelDL():
    
    def __init__(self):
        self.histories = {}
        self.models = {}
        

    def print_summary(self, savefig=False):
         
        if hasattr(self, 'models'):
            for ky in self.models.keys():
                if savefig:
                    ofile = f'model_summary_PC{ky}.txt'
                    print_to_txt(self.models[ky], f'{self.rootdir}{self.ml_dir}{ofile}')
                else:
                    self.models[ky].summary()
                
        if hasattr(self, 'model'):
            if savefig:
                ofile = f'model_summary.txt'
                print_to_txt(self.model, f'{self.rootdir}{self.ml_dir}{ofile}')
            else:
                self.model.summary()
    
    
    def set_model(self, model):
        self.model = model
    
    
    def fit(self, dataset, epochs=250, batch_size=128, print_arch_model=False, print_history=False, suffix=''):
#         '''outdated - to delete - or to keep for local ?
#         '''
        self.epochs = epochs
        self.batch_size = batch_size
        self.suffix = suffix   # name to save (pca number)
        
        if print_arch_model:
            self.print_summary()

    # Create a TensorBoard callback
        logs = "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f'{self.rootdir}{self.ml_dir}{logs}'
        os.mkdir(f'{log_dir}')
        print(f'Creation of {log_dir}')
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                         histogram_freq=1, profile_batch=(10,20))


        self.history = self.model.fit(dataset[0]["Xtrain"], dataset[0]["ytrain"], epochs=epochs, 
                                    batch_size=batch_size, verbose=self.verbose, validation_split=0.2, shuffle=True,
                                    callbacks = [tboard_callback])

        print('Training finished ! suffle true')
        if print_history:
            self.print_history()
    
    
#     def fit_multiple(self, dataset, epochs=None, batch_size=None, print_arch_model=False, print_history=False, suffix=''):
    def fit_multiple(self, dataset=None, epochs=None, batch_size=None, print_arch_model=True, print_history=False, suffix=''):
        '''fit multiple PC
        
        Old default parameters:
        
        epochs = 250
        batch_size = 128
        
        ------------------------------------
        !!!!!!!!!!!!!!!!!
        Since PC model can have different inputs fields:
        dataset is not used anymore 
        
        if correct, remove from parameters
        -------------------------------------
        
        '''
        
        if epochs is not None:  # otherwise define in config file
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        
        self.suffix = suffix
        self.chrono = self.ds.dataset['chrono']
        
#         self.stopped_epochs = np.array([self.epochs]*self.npca)
        self.stopped_epochs = copy.deepcopy(np.array(self.epochs))

        
        # attribut point of the dataset as training point
        if 'local' in self.ds_objective:  # == 'local_apply':
            self.point_train = self.point
        

       

         # Create a TensorBoard callback
        logs = f'logs_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        log_dir = f'{self.rootdir}{self.ml_dir}{logs}'
        os.mkdir(f'{log_dir}')
        print(f'Creation of {log_dir}')
        tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                         histogram_freq=1, profile_batch=(10,20))

#         patience = 10
#         earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2, patience=patience, verbose=1, restore_best_weights=True)

        print('Fitting...')

        for ipca in range(self.npca):
            self.ds.create_dataset_PC(var_to_keep=self.ds.config.input_fields[f'PC{ipca:02d}'])
            self.histories[f'pc{ipca:02d}'] = self.models[f'pc{ipca:02d}'].fit(self.ds.dataset["Xtrain"], 
                                                                       self.ds.dataset["ytrain"][:,ipca],
                                                   epochs=self.epochs[ipca], batch_size=self.batch_size,
                                                   verbose=self.verbose,
                                                   validation_data = (self.ds.dataset['Xval'], self.ds.dataset["yval"][:,ipca]),
                                                   shuffle=False) # , callbacks=[earlystop_callback])

#             best epoch = last epoch checked - patience +1
#             if earlystop_callback.stopped_epoch != self.epochs:  # early stop active
#                 self.stopped_epochs[ipca] = earlystop_callback.stopped_epoch - patience + 1
            
            # [tboard_callback, earlystop_callback])
                                                                       
                                                                       #validation_split=0.2, shuffle=True)
            print(f'pc{ipca:02d} Training finished !')
            
        if print_arch_model:
            self.print_summary()

                #         print(self.stopped_epochs)    
        
        if print_history:
            self.print_histories()
    
    def retrain_wval(self): #, dataset):
        '''Retrain models using best params found in fit_multiple.py
        BUT includes the validation data (20% of 2014-2022)
        '''
        
#         X = np.concatenate((dataset["Xval"], dataset["Xtrain"]))
#         y = np.concatenate((dataset["yval"], dataset["ytrain"]))
        
        
        if self.verbose==1:
            print('Retrain with validation period...')
        
        for ipca in range(self.npca):
            # if callback = earlystop :
#             epochs=self.stopped_epochs[ipca]
            # else:
        #     epochs = self.epochs
            self.ds.create_dataset_PC(var_to_keep=self.ds.config.input_fields[f'PC{ipca:02d}'])
            X = np.concatenate((self.ds.dataset["Xval"], self.ds.dataset["Xtrain"]))
            y = np.concatenate((self.ds.dataset["yval"], self.ds.dataset["ytrain"]))
            
            self.histories[f'pc{ipca:02d}'] = self.models[f'pc{ipca:02d}'].fit(X, y[:,ipca],
                                                   epochs=self.epochs[ipca], batch_size=self.batch_size, 
                                                   verbose=self.verbose, 
                                                   validation_split = 0,
                                                   validation_data = None, 
                                                   shuffle=True, callbacks=None)
    def retrain_wtest(self): # , dataset):
        '''Retrain models using best params found in fit_multiple.py
        BUT includes the validation data (20% of 2014-2022) and test data (2011-2013)
        '''
        
#         X = np.concatenate((dataset["Xtest"], dataset["Xval"], dataset["Xtrain"]))
#         y = np.concatenate((dataset["ytest"], dataset["yval"], dataset["ytrain"]))
        
        
        if self.verbose==1:
            print('Retrain with validation+TEST period...')
        
        for ipca in range(self.npca):
            self.ds.create_dataset_PC(var_to_keep=self.ds.config.input_fields[f'PC{ipca:02d}'])
            X = np.concatenate((self.ds.dataset["Xtest"], self.ds.dataset["Xval"], self.ds.dataset["Xtrain"]))
            y = np.concatenate((self.ds.dataset["ytest"], self.ds.dataset["yval"], self.ds.dataset["ytrain"]))
            
            
            self.histories[f'pc{ipca:02d}'] = self.models[f'pc{ipca:02d}'].fit(X, y[:,ipca],
                                                   epochs=self.epochs[ipca], batch_size=self.batch_size, 
                                                   verbose=self.verbose, 
                                                   validation_split = 0,
                                                   validation_data = None, 
                                                   shuffle=True, callbacks=None)    
        
        
    def print_history(self, savefig=True, showfig=False):
        if savefig:
            ofile = f'{self.rootdir}{self.fig_dir}{self.type}_learning_curves_{self.epochs}_{self.batch_size}{self.suffix}.png'
        else:
            ofile = ''
            
        self.plot_learning_curves(self.history, showfig=showfig, savefile=ofile)
    
    
    def predict_apply(self, dataset=None, verbose=None):
        '''Prediction for multiple PCA for application to unknow SIT (<2010)
        
        Apply for whole dataset (that is not splitted between train/val/test)
        '''
        
        if verbose is None: verbose = self.verbose
        
        # attribute point of the dataset as target point
        self.point_target = self.point
        
#         self.ds_non_assimilated = dataset.non_assimilated
        self.ds_non_assimilated = 'Freerun'
        
        
#         self.x = self.ds.dataset['X']
        self.chrono = self.ds.dataset['chrono']
#         self.ypred = np.zeros((self.x.shape[0], self.npca))
        self.ypred = np.zeros((self.ds.dataset['X'].shape[0], self.npca))
        
        self.ytrue = None  # no ytrue for application
        
        for ipca in range(self.npca):
            self.ds.create_dataset_PC(var_to_keep=self.ds.config.input_fields[f'PC{ipca:02d}'])
            self.x = self.ds.dataset['X']
            
            self.ypred[:,ipca:ipca+1] = self.models[f'pc{ipca:02d}'].predict(self.x, verbose=verbose)
        
        
            
    def print_histories(self, savefig=True, showfig=False):
        
        for ipca in range(self.npca):
            if savefig:
                ofile = f'{self.rootdir}{self.fig_dir}{self.type}_learning_curves_{self.epochs[ipca]}_{self.batch_size}{self.suffix}_PC{ipca:02d}.png'
            else:  ofile = ''
            self.plot_learning_curves(self.histories[f'pc{ipca:02d}'], showfig=showfig, savefile=ofile)
            
            
#     def predict(self, dataset):
#         '''outdated - to delete
#         '''
#         self.x = np.concatenate((dataset[0]["Xtest"], dataset[0]["Xval"], dataset[0]["Xtrain"]))
#         self.ytrue = np.concatenate((dataset[0]["ytest"], dataset[0]["yval"], dataset[0]["ytrain"]))
#         self.chrono = dataset[0]["chrono"]
#         self.ypred = self.model.predict(self.x, verbose=1)

#         print('Prediction finished !')
        

    def predict_multiple(self, dataset=None, point_target=None, point=None):
        '''Prediction for multiple PCA during training phase
        Parameters:
        -----------
            dataset          : dataset from class <mdl_dataset.Dataset>
            point_target     : tuple (y,x), indexes for one point (lat, lon)
            point            : list of points for LOCAL prediction (can be different that points used for training)
        
        
        -- Local prediction might be broken --
        
        '''
        # attribut point of the dataset as target point
        if point_target is not None:
            self.point_target = point_target
        
        self.ds_non_assimilated = 'Freerun'
        
        self.x = np.concatenate((self.ds.dataset["Xtest"], self.ds.dataset["Xval"], self.ds.dataset["Xtrain"]))
        self.ytrue = np.concatenate((self.ds.dataset["ytest"], self.ds.dataset["yval"], self.ds.dataset["ytrain"]))
        self.chrono = self.ds.dataset["chrono"]
        self.ypred = np.zeros((self.ytrue.shape))
        
        
        # for lstm new model
        # self.ypred = np.zeros((self.ytrue.shape[0], dataset['X'].shape[1]))  # , self.ytrue.shape[1]))
        if 'global' in self.ds_objective:
            for ipca in range(self.npca):
                # get inputs for the correct PC
                self.ds.create_dataset_PC(var_to_keep=self.ds.config.input_fields[f'PC{ipca:02d}'])
                # concatenate dataset for prediction
                self.x = np.concatenate((self.ds.dataset["Xtest"], self.ds.dataset["Xval"], self.ds.dataset["Xtrain"]))
                # prediction
                self.ypred[:,ipca:ipca+1] = self.models[f'pc{ipca:02d}'].predict(self.x, verbose=self.verbose)
            
        elif 'local' in self.ds_objective:
            ipca=0  # because no pca - need to change the name ?
            self.ypred = self.models[f'pc{ipca:02d}'].predict(self.x, verbose=self.verbose)
            
            # check that the point in model (self) are the same as in dataset (ds)
            if point is not None:
                self.point = point
        
        # for LSTM NEW ARCHI ONLY
        # self.ypred = self.models[f'pc0'].predict(self.x, verbose=1)
            
        
        print('Prediction finished !! - for lstm_at only')
    
    def predict_multiple_wbias(self):  #, dataset):
        '''Only for global prediction 
        '''
        self.ds_non_assimilated = 'Freerun'
        
        self.x = np.concatenate((self.ds.dataset["Xtest"], self.ds.dataset["Xval"], self.ds.dataset["Xtrain"]))
        self.ytrue = np.concatenate((self.ds.dataset["ytest"], self.ds.dataset["yval"], self.ds.dataset["ytrain"]))
        self.chrono = self.ds.dataset["chrono"]  
#         self.ypred = np.zeros((self.ytrue.shape[0], self.ytrue.shape[1]))
        self.ypred = np.zeros((self.ytrue.shape))
        
        # First feature on self.x is bias
        # need to predict each time step, and upload the bias between prediction and freerun 

        if 'global' in self.ds_objective:
            for ipca in range(self.npca):
                # get inputs for the correct PC
                self.ds.create_dataset_PC(var_to_keep=self.ds.config.input_fields[f'PC{ipca:02d}'])
                # concatenate dataset for prediction
                self.x = np.concatenate((self.ds.dataset["Xtest"], self.ds.dataset["Xval"], self.ds.dataset["Xtrain"]))
                # prediction
                for nt in range(2, self.x.shape[0]):
                    self.ypred[-nt,ipca:ipca+1] = self.models[f'pc{ipca:02d}'].predict(self.x[-nt], verbose=self.verbose)
                    self.x[-nt-1, 0] = self.ypred[-nt, ipca:ipca+1]
    
    
    def save_prediction(self, ds=None):
        '''save results (y predicted and y true)
        '''
        
        print('Save prediction as .nc')  # as dataset .nc with ypred/ytrue/chrono')
        
        if 'local' in self.ds_objective:
            if ds is None:
                print('Need dataset to save results !')
                return
            self.save_prediction_local(ds)
            return
        # else: global
        
        # no change
        # define time valid for netcdf
#         ts = pd.date_range(self.chrono.to_numpy()[0][0], self.chrono.to_numpy()[-1][0])
#         time_unit_out = "days since 1991-01-01 00:00:00"
#         ts_nc = netCDF4.date2num(ts.to_pydatetime(), time_unit_out)

#         ypred = xr.DataArray(self.ypred, 
#                         coords={'time': ts_nc, 'comp': np.arange(self.npca)}, 
#                         dims=["time", "comp"])
        
#         ytrue = xr.DataArray(self.ytrue, 
#                         coords={'time': ts_nc, 'comp': np.arange(self.npca)}, 
#                         dims=["time", "comp"])
        
    
        dt = self.chrono.to_numpy()[:,0]
        
       # import pdb; pdb.set_trace()
        ypred = xr.DataArray(self.ypred, 
                        coords={'time': self.chrono.to_numpy()[:,0], 'comp': np.arange(self.npca)}, 
                        dims=["time", "comp"])
        
        ytrue = xr.DataArray(self.ytrue, 
                        coords={'time': self.chrono.to_numpy()[:,0], 'comp': np.arange(self.npca)}, 
                        dims=["time", "comp"])
        
        osave = xr.Dataset(data_vars={'ypred':ypred,
                                      'ytrue':ytrue},
                           attrs=dict(
                               description='Prediction values for PCA of SIT bias between TOPAZ4b and TOPAZ4b FreeRun',
                               model_ml=f'{self.type}',
                               details=f'Applied to TOPAZ4b 23 {self.ds_non_assimilated}',
                               duration=f'from {dt[0]} to {dt[-1]}',
                               author='Leo Edel, Nersc',
                               project='TARDIS',
                               date=f'{date.today()}')
                          )
        
        first_year = self.chrono.iloc[0].date.year 
        last_year = self.chrono.iloc[-1].date.year
        
        filename = f'ypred_{self.type}_{first_year}_{last_year}.nc'
        if 'apply' in self.ds_objective:  # apply or apply91
            filename = f'ypred_{self.type}_{first_year}_{last_year}_{self.ds_non_assimilated}.nc'
            
            
        ofile = save_name.check(f"{self.rootdir}{self.ml_dir}", filename)
        osave.to_netcdf(f'{self.rootdir}{self.ml_dir}{ofile}')
        print(f'Ytrue saved as: {self.rootdir}{self.ml_dir}{ofile}')
        
    def save_prediction_local(self, ds=None):
        '''Separate function because it is not the same shape
        save directly bias of SIT
        
        
        '''
    
        # put prediction back on 2D maps
        # (time, lat, lon)
        ntimes = ds.outputs[0].shape[0]
        nlat = ds.outputs[0].shape[1]
        nlon = ds.outputs[0].shape[2]
        bias_pred = np.zeros((ntimes, nlat, nlon))

        for idx, pt in enumerate(self.point[::-1]):  # loop on points 
            # [::-1] here because lat upside down, but the projection of y is correct
        #     print(pt)
            bias_pred[:, pt[0], pt[1]] = self.ypred[:, idx, 0]
    
        bpred_xr = xr.DataArray(bias_pred, 
                coords={'time': self.chrono.to_numpy()[:,0], 'y': ds.outputs[0].y, 'x':ds.outputs[0].x}, 
                dims=["time", "y", "x"])
        
        # put back the nan
        bpred_xr = bpred_xr.where(ds.maskok)
        
        btrue_xr = xr.DataArray(ds.outputs[0], 
                coords={'time': self.chrono.to_numpy()[:,0], 'y': ds.outputs[0].y, 'x':ds.outputs[0].x},
                dims=["time", "y", "x"])
    
    
        osave = xr.Dataset(data_vars={'ypred':bpred_xr,
                              'ytrue':btrue_xr},
                   attrs=dict(
                       description='Local Prediction values for bias of SIT between TOPAZ4b and TOPAZ4b FreeRun',
                       extra=f'Trained on {len(self.point)} points',
                       local_model=f'{self.name}',
                       global_model=f'f{self.gname}',
                       author='Leo Edel, Nersc',
                       project='TARDIS',
                       date=f'{datetime.date.today()}')
                  )

        ofile = f'biasSIT_local_{self.name}.nc'  # {self.type}.nc'
        odir = f'{self.rootdir}{self.ml_dir}'
        osave.to_netcdf(f'{odir}{ofile}')
        print(f'Prediction (bias of SIT, .nc) saved as: {odir}{ofile}')
    
    
    def save_model(self):
        '''save model for later use (application)
        '''
        
        print('Saving models...')
        
        for ipca in range(self.npca):
            
            oname = f'model_{self.type}_{self.npca}N_PC{ipca}'  # need more info ?
            ofolder = save_name.check(f"{self.rootdir}{self.ml_dir}", oname)
            ofolder = f"{self.rootdir}{self.ml_dir}{oname}" 
            
            # try .h5 because error with default format
            self.models[f'pc{ipca:02d}'].save(ofolder, save_format='h5')
        
            print(f'Saved as: {ofolder}')
        
        
    def load_model(self, ipath, ifolder_pattern):
        '''Load model
        Parameters:
        -----------
                ipath       : path to sub directory
                ifolder     : pattern of folder name 
                              (without the number of PC: 'model_CNN_8N' when real name is 'model_CNN_8N_PC0')
        
        '''
        
        print('Loading models...')
        
        for ipca in range(self.npca):
            ifolder = f'{ifolder_pattern}_PC{ipca:02d}'
            
#            import pdb; pdb.set_trace()
            self.models[f'pc{ipca:02d}'] = km.load_model(f'{ipath}{ifolder}')
            print(f'Loaded: {ipath}{ifolder}')
 

    def save_model_weights(self, retrained=False):
        '''Save weights for each model for later use
        
        Is easier than saving full models
        
        Parameters:
        -----------
        
            retrained    :   bool, if true, will save model in subfolder called "retrained"
                                   retrained on full dataset 2011-2022
        
        '''
        
        print('Saving models weights...')
        
        ml_dir = self.ml_dir
        if retrained:
            ml_dir += 'retrained/'
        
        for ipca in range(self.npca):
            
            oname = f'model_weights_{self.type}_{self.npca}N_PC{ipca:02d}'  # need more info ?
            ofolder = save_name.check(f"{self.rootdir}{ml_dir}", oname)
            ofolder = f"{self.rootdir}{ml_dir}{oname}" 
            
            self.models[f'pc{ipca:02d}'].save_weights(ofolder)
        
            print(f'Saved as: {ofolder}')
        
    def load_model_weights(self, ipath, ifolder_pattern):
        '''Load model weights
        Parameters:
        -----------
                ipath       : path to sub directory
                ifolder     : pattern of folder name 
                              (without the number of PC: 'model_CNN_8N' when real name is 'model_CNN_8N_PC0')
        
        '''
        
        print('Loading models...')
        
        for ipca in range(self.npca):
            ifolder = f'{ifolder_pattern}_PC{ipca:02d}'
            
            # load weights
            self.models[f'pc{ipca:02d}'].load_weights(f'{ipath}{ifolder}')
            print(f'Loaded: {ipath}{ifolder}')   
        
        
# depreciated : to delete if no script cries  
# no - still used # to keep

# plot

    def plot_learning_curves(self, history, showfig=False, savefile=''):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,4*3))
        
        ax1.plot(history.history['loss'])
        ax1.plot(history.history['val_loss'])
        ax1.legend(['train','val'])
    #     ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')

        ax2.plot(history.history['r_square'])
        ax2.plot(history.history['val_r_square'])
        ax2.legend(['train','val'])
    #     ax2.set_xlabel('Epochs')
        ax2.set_ylabel('r2')

        ax3.plot(history.history['root_mean_squared_error'])
        ax3.plot(history.history['val_root_mean_squared_error'])
        ax3.legend(['train','val'])
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('RMSE')

        plt.tight_layout()

        if savefile != '':
            odir = os.path.dirname(savefile)
            ofile = save_name.check(f"{odir}", os.path.basename(savefile))
            savefile = f'{odir}/{ofile}'
            plt.savefig(f"{savefile}")  # , facecolor='white')
            print(f'Saved as : {savefile}')

        if showfig:
            plt.show()

        plt.close()  

    def draw_prediction(self, ytrue, ypred, chrono, ntest, twin=False, showfig=True, savefile=''):
    
        fig, ax = plt.subplots(figsize=(12,10))

        l1 = ax.plot(chrono, ytrue, label='true error')
        if twin:
            ax_twin = ax.twinx()
            l2 = ax_twin.plot(chrono, ypred, label='prediction', color='orange')
        else:
            l2 = ax.plot(chrono[1:], ypred, label='prediction')
        lab1 = l1+l2
        lab2 = [l.get_label() for l in lab1]
        plt.legend(lab1, lab2)
        plt.axvline(x=chrono.iloc[ntest], linestyle='dotted', color='grey', label='train limit')

        if savefile != '':
            odir = os.path.dirname(savefile)
            ofile = save_name.check(f"{odir}", os.path.basename(savefile))
            savefile = f'{odir}/{ofile}'
            plt.savefig(f"{savefile}")  # , facecolor='white')
            print(f'Saved as : {savefile}')

        if showfig:
            plt.show()

        plt.close()    
