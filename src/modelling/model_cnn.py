import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow_addons as tfa

from tensorflow.keras.models import Model

from src.modelling import super_model_dl


class ModelCNN(super_model_dl.SModelDL):
    
    # def __init__(self, ds, timesteps, features, units=30, reg=None, n_batch=None, rootdir=None, ml_dir=None, fig_dir=None):
    def __init__(self, ds, timesteps, features, units=30, reg=None, n_batch=None, rootdir=None, ml_dir=None, fig_dir=None):
        
        self.timesteps = timesteps
        self.features = features
        self.units = units
        self.reg = reg
        self.rootdir = rootdir
        self.ml_dir = ml_dir
        self.fig_dir = fig_dir
        self.type = 'CNN'
        self.n_batch = n_batch
        
        
        # new vvvvvvvv
        self.name = ds.config.ml_name
        self.epochs = ds.config.epochs
        self.batch_size = ds.config.batch_size
        
        self.ntrain = ds.ntrain
        self.nval = ds.nval
        self.ntest = ds.ntest
        
        # keep point use for training
        # if ds.objective == 'local_train':
        #    self.point_train = ds.point
        #elif ds.objective == 'local_apply':
        #    self.point_apply = ds.point
        self.point = ds.point
        self.ds_objective = ds.objective
        self.verbose = ds.config.verbose
        
        self.cnn_models = [self.mdl_CNN, self.mdl_CNN_mh, self.mdl_CNN_uncert]
        self.mdl_classes_name = ['CNN', 'CNN_mh', 'CNN_uncert']
        
        super().__init__()

    def compile_models(self, name=None, npca=1):
        '''Compile for npca
        
        Parameters:
        -----------
        
            name       : if not None, specify name of the model to use
                         if None (default), use name in config file
                         old default: 'CNN'
                         
        '''
        
        if name is not None:
            self.name = name
        
        self.npca = npca        
        mdl_idx = self.model_selection()

        
        for ipca in range(self.npca):
            self.models[f'pc{ipca}'] = self.cnn_models[mdl_idx]()

        self.type = self.name


    def model_selection(self):
        '''Selection the model required by the name
        '''
        
        # print('!! to try !! + insert in compile_models()')
        print('input change depending on model !!')
        
        # pick the corresponding machine learning model
        mdl_idx = [i for i, cl in enumerate(self.mdl_classes_name) if self.name in cl][0]

        return mdl_idx
        

    def mdl_CNN(self):

        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(self.timesteps, self.features)))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(units = 1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model
        
        
#     def mse_1_point(self, y_true, y_pred):
        
#         return 
        

    def mdl_CNN_mh(self):
        '''Multi Head with different kernel_size, to notice feature of different scales
        '''
        
        print('Warning: Not the same input shape as usually !!')
        
      # head 1
        inputs1 = Input(shape=(self.timesteps, self.features))
        conv1 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=self.reg)(inputs1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # head 2
        inputs2 = Input(shape=(self.timesteps, self.features))
        conv2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', kernel_regularizer=self.reg)(inputs2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # head 3
        inputs3 = Input(shape=(self.timesteps, self.features))
        conv3 = Conv1D(filters=256, kernel_size=11, padding='same', activation='relu', kernel_regularizer=self.reg)(inputs3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(100, activation='relu')(merged)
        outputs = Dense(n_output)(dense1)  # , activation='softmax'

        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model

  
    def mdl_CNN_uncert(self):
        ''' same as mdl_CNN but also returns uncertainty
        '''
        
#         model = Sequential()
#         model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(s
#         model.add(MaxPooling1D(pool_size=4))
#         model.add(Dropout(0.1))

#         model.add(Flatten())
#         model.add(Dense(units = 1))

        
        inputs = Input(shape=(self.timesteps, self.features))
        conv1 = Conv1D(filters=256, kernel_size=5, padding='same', activation='relu')(inputs)
        pool1 = MaxPooling1D(pool_size=4)(conv1)
        drop1 = Dropout(0.1)(pool1)
        
        flat1 = Flatten()(drop1)
        dense1 = Dense(units=1)(flat1)
        
        
        var = Conv1D(1,1, activation='sigmoid')(conv1)  # exponential
        flat2 = Flatten()(var)
        dense2 = Dense(units=1)(flat2)
        merged = concatenate(name='output', inputs=[dense1, dense2])
        
        model = Model(inputs=inputs, outputs=merged)  # dense1) # merged)
        
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        # model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.Root
        # model.compile(optimizer=opt, loss=self.loss_mle_tfp , metrics=[tfa.metrics.RSquare(), tf.keras
        model.compile(optimizer=opt, loss=self.loss_mle, 
                      metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        
        print('Compiled !')
        return model


    def loss_mle(self, y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        mean_true = y_true
        mean_pred, var_pred = y_pred[...,0], tf.maximum(y_pred[...,1], tf.keras.backend.epsilon())
        # Max to prevent NaN's and Inf's
        print('la')
        log_std = tf.math.log(var_pred)
        print('la')
        mse = tf.math.squared_difference(mean_pred, mean_true) / var_pred
        print('la')
        loss = log_std + mse
        print('la')
        loss = tf.reduce_sum(loss, -1)
        return loss


    def loss_mle_tfp(self, y_true, y_pred):
        '''Loss function for predicting uncertainty
        '''
        import tensorflow_probability as tfp
        mean_true = y_true
        mean_pred, std_pred = y_pred[...,0], y_pred[...,1]
        norm_dist = tfp.distributions.Normal(loc = mean_pred, scale = std_pred)
        print('la')
        print(y_true.shape)
        loss = - norm_dist . log_prob (y_true)
        print('la2')
        loss = tf.reduce_mean(loss, -1)
        print('la3')
        return loss  