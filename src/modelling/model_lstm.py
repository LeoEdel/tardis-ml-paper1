"""
Contain several LSTM models and architectures
And function for compiling

Other functions; such as training, prediction, saving weights; are in the super class SModelDL
in the script /src/modelling/super_model_dl.py
"""


import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
import tensorflow_probability as tfp
    
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Attention, Flatten, AdditiveAttention, Permute, Reshape, Add, Dropout
# from tensorflow.keras.layers import ConvLSTM1D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

from src.modelling import super_model_dl
        
def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)
    
        
class ModelLSTM(super_model_dl.SModelDL):
    
    def __init__(self, ds, timesteps, features,
                 npca=1,
                 units=32, 
                 reg=None, n_batch=None, 
                 rootdir=None, ml_dir=None, fig_dir=None,
                 gname=None):
        '''
        
        
        Parameters:
        -----------
        
            timesteps   : int, length of time series. shape[1] of training dataset
            features    : int, number of input of training dataset. shape[1] of training dataset
            
                            >> new definition:
                           array of int, number of input of training dataset for each PC models
                                         must be array of int of length = number of PC or int
                                         if int: will be transform into an array of the given int
            
            
            gname       : str, global name. Type of ML that have been used to correct SIT GLOBALLY
        
        
        
        '''

        self.timesteps = timesteps
        self.features = features
        
        if features is int:
            self.features = np.array([features]*npca)
        
        self.units = units
        self.reg = reg
        self.rootdir = rootdir
        self.ml_dir = ml_dir
        self.fig_dir = fig_dir
        self.type = 'LSTM'
        self.n_batch = n_batch
    
        self.ds = ds
        self.name = ds.config.ml_name
        self.epochs = ds.config.epochs
        self.batch_size = ds.config.batch_size
        
        self.ntrain = ds.ntrain
        self.nval = ds.nval
        self.ntest = ds.ntest
        
        self.point = ds.point
        self.ds_objective = ds.objective
        self.verbose = ds.config.verbose
        self.gname = gname
        
        self.lstm_models = [self.mdl_1LSTM, self.mdl_3LSTM_bk, self.mdl_3LSTM, 
                            self.mdl_LSTM_at, self.mdl_LSTM_local, self.mdl_ConvLSTM,
                           self.mdl_6LSTM_bk, self.mdl_3LSTM_bk_at, self.mdl_LSTM_bi, self.mdl_LSTM_bk, self.mdl_LSTM_test,
                           self.mdl_LSTM_bi2, self.mdl_3LSTM_bk_relu, self.mdl_3LSTM_bk_leaky_relu,
                           self.mdl_3LSTM_bk_relu_WI, self.mdl_3LSTM_bk_relu_GP,
                           self.mdl_3LSTM_res, self.mdl_3LSTM_res_tanh]
        self.mdl_classes_name = ['LSTM1', 'LSTM3_bk', 'LSTM3', 
                                 'LSTM_at', 'LSTM_loc', 'LSTMconv',
                                'LSTM6_bk', 'LSTM3_bk_at', 'LSTM_bi', 'LSTM_bk', 'LSTM_test',
                                'LSTM_bi2', 'LSTM3_bk_relu', 'LSTM3_bk_leaky_relu', 
                                 'LSTM3_bk_relu_WI', 'LSTM3_bk_relu_GP',
                                'LSTM3_bk_residual', 'LSTM3_bk_residual_tanh']
        
#         self.callbacks = None  # default value
        
        super().__init__()
        

    def compile_models(self, name=None, npca=1):
        '''Compile for npca
        
        Parameters:
        -----------
        
            name       : if not None, specify name of the model to use
                         if None (default), use name in config file
                         old default: 'LSTM3_bk'
        '''
        
        
        if name is not None:
            self.name = name
        
        self.npca = npca
        mdl_idx = self.model_selection()  # select model based on name

        # Create one model for each PC
        for ipca in range(self.npca):
            self.models[f'pc{ipca:02d}'] = self.lstm_models[mdl_idx](ipc=ipca)

        self.type = self.name
       
        
    def model_selection(self):
        '''Selection the model required by the name
        '''
        # pick the corresponding machine learning model
        mdl_idx = [i for i, cl in enumerate(self.mdl_classes_name) if self.name in cl][0]

        print(f'ML Architecture selected: {self.mdl_classes_name[mdl_idx]}')
        # print('\t << Caution: inputs may change depending on model >>')
        
        return mdl_idx    
  
    def mdl_LSTM_bk(self, ipc):
        '''
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=320, return_sequences=False, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True, recurrent_dropout=0.1))
#         model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model

    
    def mdl_LSTM_at(self, ipc):
        '''2 LSTM Layer, Bidirectional (explore dependencies in past and future) and 
        an attention layer (help model focus on most important parts ot the time series)
        '''
        # Define the input layer
        inputs = Input(shape=(self.timesteps, self.features[ipc]))

        # Define the LSTM layers
        lstm1 = LSTM(64, return_sequences=True, go_backwards=True)(inputs)
        dropout1= Dropout(0.1)(lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, go_backwards=True)(dropout1)
        dropout2= Dropout(0.1)(lstm2)
        
        lstm3 = LSTM(64, return_sequences=True, go_backwards=True)(dropout2)
        dropout3= Dropout(0.1)(lstm3)        
        
        # Define the bidirectional layer
        bidirectional = Bidirectional(LSTM(64, return_sequences=True))(dropout3)

        # Define the query, value and key
        query = Dense(64, activation='relu')(bidirectional)
        value = Dense(64, activation='relu')(bidirectional)
        key = Dense(64, activation='relu')(bidirectional)

        # Define the attention layer
        attention = Attention()([query, value, key])

        # Define Flatten layer to reduce axes
        flatten = Flatten()(attention)
        
        # Define the output layer
        outputs = Dense(1)(flatten)  # attention)

        # Create the model
        model = Model(inputs=inputs, outputs=outputs)

        # Compile the model
        # model.compile(optimizer='adam', loss='mean_squared_error')
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        
        
        print('Compiled !')
        return model
        
    
    def mdl_3LSTM(self):
        
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features), recurrent_regularizer=self.reg))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        
        print('Compiled !')
        return model

    def mdl_LSTM_bi2(self, ipc):
        '''
        Bidirectional
        
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

                
        model = Sequential()
        model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(self.timesteps, self.features[ipc]))))
        model.add(Dropout(0.1))

        
        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        model.add(Dropout(0.1))
        
        model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
        model.add(Dropout(0.1))
        
        model.add(LSTM(units=32, return_sequences=False, go_backwards=True))
        model.add(Dropout(0.1))
        
        
        model.add(Dense(units=256, activation='selu'))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model
         
    def mdl_LSTM_bi(self, ipc):
        '''
        Bidirectional
        
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        forward_layer = LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]))
        backward_layer = LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), go_backwards=True)
                
        model = Sequential()
        model.add(Bidirectional(layer=forward_layer, backward_layer=backward_layer, merge_mode='concat'))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model

    def mdl_3LSTM_bk(self, ipc):
        '''
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True, recurrent_dropout=0.2))  #activation=None,
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True, recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True, recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model
      
    def mdl_3LSTM_bk_relu(self, ipc):
        '''
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2))  #activation=None,
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model

    def mdl_3LSTM_bk_relu_GP(self, ipc):
        '''Gradient Clipping: clipping gradients during training can help prevent exploding gradients, which can lead to instability and issues with ReLU activations.
        
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2))  #activation=None,
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4, clipvalue=1.0)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model
    
    def mdl_3LSTM_res_tanh(self, ipc):
        '''Use Residual Connections. In case of deep NN, adding residual connections can help mitigate vanishing gradients.
        
        We add residual connections after each LSTM layer by adding the input tensor to the output of the LSTM+Dropout layers.
        
        The residual connections help preserve the original input information, potentially stabilizing the learning process and mitigating issues like vanishing or exploding gradients.
        
        '''
        input_layer = Input(shape=(self.timesteps, self.features[ipc]))

        units = self.features[ipc]
        
        # First LSTM layer
        x = LSTM(units=units, return_sequences=True, kernel_regularizer=self.reg, 
                 go_backwards=True, recurrent_activation='tanh', 
                 recurrent_dropout=0.2, kernel_initializer=HeNormal())(input_layer)
        x = Dropout(0.1)(x)

        # Residual connection for the first LSTM layer
        residual_1 = Add()([input_layer, x])

        # Second LSTM layer
        x = LSTM(units=units, return_sequences=True, go_backwards=True, 
                 recurrent_activation='tanh', kernel_regularizer=self.reg, 
                 recurrent_dropout=0.2, kernel_initializer=HeNormal())(residual_1)
        x = Dropout(0.1)(x)

        # Residual connection for the second LSTM layer
        residual_2 = Add()([residual_1, x])

        # Third LSTM layer
        x = LSTM(units=units, return_sequences=False, go_backwards=True, recurrent_activation='relu', 
                 recurrent_dropout=0.2, kernel_regularizer=self.reg, 
                 kernel_initializer=HeNormal())(residual_2)
        x = Dropout(0.1)(x)

        # Residual connection for the second LSTM layer
        # residual_3 = Add()([residual_2, x])
        # or remove residual_3 and connect output_layer to x and put return_sequence=False in the Third LSTM layer
        
        # Dense output layer
        output_layer = Dense(units=1)(x)
        # output_layer = Dense(units=1)(residual_3)

        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)  # default: 3e-4. 3e-3
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        print('Compiled !')

        return model
    
    def mdl_3LSTM_res(self, ipc):
        '''Use Residual Connections. In case of deep NN, adding residual connections can help mitigate vanishing gradients.
        
        We add residual connections after each LSTM layer by adding the input tensor to the output of the LSTM+Dropout layers.
        
        The residual connections help preserve the original input information, potentially stabilizing the learning process and mitigating issues like vanishing or exploding gradients.
        
        '''
        input_layer = Input(shape=(self.timesteps, self.features[ipc]))

        units = self.features[ipc]
        
        # First LSTM layer
        x = LSTM(units=units, return_sequences=True, kernel_regularizer=self.reg, 
                 go_backwards=True, recurrent_activation='relu', 
                 recurrent_dropout=0.2, kernel_initializer=HeNormal())(input_layer)
        x = Dropout(0.3)(x)  # 0.1

        # Residual connection for the first LSTM layer
        residual_1 = Add()([input_layer, x])

        # Second LSTM layer
        x = LSTM(units=units, return_sequences=True, go_backwards=True, 
                 recurrent_activation='relu', kernel_regularizer=self.reg, 
                 recurrent_dropout=0.2, kernel_initializer=HeNormal())(residual_1)
        x = Dropout(0.3)(x)  # 0.1

        # Residual connection for the second LSTM layer
        residual_2 = Add()([residual_1, x])

        # Third LSTM layer
        x = LSTM(units=units, return_sequences=False, go_backwards=True, recurrent_activation='relu', 
                 recurrent_dropout=0.2, kernel_regularizer=self.reg, 
                 kernel_initializer=HeNormal())(residual_2)
        x = Dropout(0.3)(x)  # 0.1

        # Residual connection for the second LSTM layer
        # residual_3 = Add()([residual_2, x])
        # or remove residual_3 and connect output_layer to x and put return_sequence=False in the Third LSTM layer
        
        # Dense output layer
        output_layer = Dense(units=1)(x)
        # output_layer = Dense(units=1)(residual_3)

        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)  # default: 3e-4. 3e-3
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        print('Compiled !')

        return model

    
    def mdl_3LSTM_bk_relu_WI(self, ipc):
        '''
        With weight initialization. to help prevent neurons from starting with negative inputs that lead to dying ReLUs
        
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), kernel_regularizer=self.reg, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2, kernel_initializer=HeNormal()))  #activation=None,
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True, recurrent_activation='relu', kernel_regularizer=self.reg, recurrent_dropout=0.2, kernel_initializer=HeNormal()))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True, recurrent_activation='relu', recurrent_dropout=0.2, kernel_regularizer=self.reg, kernel_initializer=HeNormal()))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-3) # default: 3e-4
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model

    
    def mdl_3LSTM_bk_leaky_relu(self, ipc):
        ''' Leaky relu has a small slope for negative values, which prevents neurons from dying
        '''
        
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), go_backwards=True, recurrent_activation=leaky_relu, recurrent_dropout=0.2))  #activation=None,
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True, recurrent_activation=leaky_relu, recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True, recurrent_activation=leaky_relu, recurrent_dropout=0.2))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model
    
    
    def mdl_LSTM_test(self, ipc):
        '''
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=320, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True, dropout=0.1, recurrent_dropout=0.2))
#         model.add(Dropout(0.1))

        model.add(LSTM(units=160, return_sequences=True, go_backwards=True,  dropout=0.1, recurrent_dropout=0.2))
#         model.add(Dropout(0.1))

#         model.add(LSTM(units=self.units))
        model.add(LSTM(units=96, go_backwards=True,  dropout=0.1, recurrent_dropout=0.2))
#         model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        # Callbacks: can be define here or in super_model_dl.py
#         self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         self.callbacks = [self.early_stop]
        
        print('Compiled !')
        return model
    
    

    
    def mdl_3LSTM_bk_at(self, ipc):
        '''
        not working
        
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True))
        model.add(Dropout(0.1))

        # Adding self-attention mechanism
        # The attention mechanism
        model.add(Attention())
        # Permute and reshape for compatibility
#         model.add(Permute((2, 1))) 
#         model.add(Reshape((-1, self.timesteps)))
#         attention_result = attention([model.output, model.output])
#         multiply_layer = Multiply()([model.output, attention_result])
#         # Return to original shape
#         model.add(Permute((2, 1))) 
#         model.add(Reshape((-1, self.units)))

        # Adding a Flatten layer before the final Dense layer
#         model.add(tf.keras.layers.Flatten())

        # Final Dense layer
        model.add(Dense(1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        
        print('Compiled !')
        return model
        
    def mdl_6LSTM_bk(self, ipc):
        '''
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=self.units, go_backwards=True))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model


    def mdl_xxLSTM(self, ipc):
        '''
        Parameters:
        -----------
        
            ipc     : int, number of PC to predict (between 0 and maximum PC)
        '''

        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.1))
        
        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.1))
        
        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.1))
        
        model.add(LSTM(units=self.units, return_sequences=True, go_backwards=True))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units, go_backwards=True))
        model.add(Dropout(0.1))

        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model

    def mdl_1LSTM(self):
    
        model = Sequential()
        model.add(LSTM(units = self.units, return_sequences = False, input_shape = (self.timesteps, self.features), dropout=.1,
                        recurrent_dropout=.1, kernel_regularizer=self.reg, recurrent_regularizer=self.reg))
        model.add(Dropout(0.1))

        model.add(Dense(units = 1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)  # 3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')
        return model
        
        
    def mdl_1LSTM_bk(self, ipc):
        '''Same as mdl_1LSTM with go_backwards == True
        '''
        
        model = Sequential()
        model.add(LSTM(units= 32, return_sequences=False, input_shape = (self.timesteps, self.features[ipc]), recurrent_regularizer=self.reg, go_backwards=True, recurrent_dropout=0.2))

        model.add(Dropout(0.1))
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # 3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')    
        return model
        
    def mdl_LSTM_local(self):
        '''Training on many local points
        
        Shape (nsample, times, features)
        
        FOR NOW: works with (times=time serie, nsample=points, features)
        
        # need to add history
        (times=time serie, nsample=points, timestep=history, features)
        
        
        '''
        
        model = Sequential()
        
#         len(self.point)
        model.add(LSTM(units=self.units, return_sequences=True, 
                       input_shape=(None, self.features), 
                       dropout=.1, recurrent_dropout=.1,
                       kernel_regularizer=self.reg, recurrent_regularizer=self.reg, go_backwards=True))

        model.add(Dropout(0.1))
        
        model.add(LSTM(units=self.units*2, return_sequences=True, go_backwards=True,
                      dropout=.1, recurrent_dropout=.1))
        model.add(Dropout(0.1))

        model.add(LSTM(units=self.units*3, return_sequences=True, go_backwards=True,
                      dropout=.1, recurrent_dropout=.1))
        model.add(Dropout(0.1))
        
        
        
        model.add(Dense(units=1))

        opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # 3e-4)
        model.compile(optimizer=opt, loss='mse') # , metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])

        print('Compiled !')    
        return model
        
        
    def mdl_ConvLSTM(self):
        '''input shape : (samples, time, rows, channels)  (with default data_format='channels_last')
        (nsamples=n points, time=time serie, rows=time lag, channels=features)
        (None, None, H, nfeat)
        so the number of points and length of time serie can change
        '''
        
        model = Sequential()
        
        model.add(ConvLSTM1D(input_shape=(None, self.timesteps, self.features), go_backwards=True)) 
        
        model.add(Dense(units=1))
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # 3e-4)
        model.compile(optimizer=opt, loss='mse', metrics=[tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()])
        
        print('Compiled !')    
        return model
        
        
        
        
        
        
        
        
        
        