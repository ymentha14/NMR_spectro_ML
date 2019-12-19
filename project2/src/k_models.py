import numpy as np 

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, LeakyReLU

# Model 1
def model_1(input_data, metric):
    '''return a model based on the input_data size (i.e. nb of features)
       metric: 'mean_absolute_error' OR 'mean_square_error'
    '''
    # instantiate 
    model = Sequential()

    # input Layer
    model.add(Dense(128, kernel_initializer='normal',input_dim = input_data.shape[1], activation='relu'))
    
    # hidden layers
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # output Layer
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # compile the network
    model.compile(loss=metric, optimizer='adam', metrics=[metric])
    
    return model


# Model 2
def model_2(input_data, metric):
    '''return a model based on the input_data size (i.e. nb of features)
       metric: 'mean_absolute_error' OR 'mean_square_error'
    '''
    # instantiate 
    model = Sequential()

    # input Layer
    model.add(Dense(128, kernel_initializer='normal',input_dim = input_data.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    # hidden layers
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # output Layer
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # compile the network
    model.compile(loss=metric, optimizer='adam', metrics=[metric])
    
    return model

# Model 3
def model_3(input_data, metric):
    '''return a model based on the input_data size (i.e. nb of features)
       metric: 'mean_absolute_error' OR 'mean_square_error'
    '''
    # instantiate 
    model = Sequential()

    # input Layer
    model.add(Dense(128, kernel_initializer='normal',input_dim = input_data.shape[1], activation='relu'))
   
    # hidden layers
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    # output Layer
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # compile the network
    model.compile(loss=metric, optimizer='adam', metrics=[metric])
    
    return model

# Model 4
def model_4(input_data, metric):
    '''return a model based on the input_data size (i.e. nb of features)
       metric: 'mean_absolute_error' OR 'mean_square_error'
    '''
    # instantiate 
    model = Sequential()

    # input Layer
    model.add(Dense(128, kernel_initializer='normal',input_dim = input_data.shape[1], activation='relu'))

    # hidden layers
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dense(256, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.2))
    
    # output Layer
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # compile the network
    model.compile(loss=metric, optimizer='adam', metrics=[metric])
    
    return model

# Shallower networks
# Model 5
def model_5(input_data, metric):
    '''return a model based on the input_data size (i.e. nb of features)
       metric: 'mean_absolute_error' OR 'mean_square_error'
    '''
    # instantiate 
    model = Sequential()

    # input Layer
    model.add(Dense(50, kernel_initializer='normal',input_dim = input_data.shape[1], activation='relu')) # test with 50 or 100 here

    # output Layer
    model.add(Dense(1, kernel_initializer='normal',activation='relu')) 

    # compile the network
    model.compile(loss=metric, optimizer='adam', metrics=[metric])

    return model

# Model 6
def model_6(input_data, metric):
    model = Sequential()
    model.add(Dense(100, input_dim=input_data.shape[1], kernel_initializer="uniform")) #best with 50: 2.03 #40: 1.81 #30:1.88
    model.add(Activation('relu'))
    model.add(Dense(1, kernel_initializer="uniform"))
    model.add(Activation('relu'))
    
    adam_opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_absolute_error', optimizer=adam_opt)
    
    return model

# Model 7
def model_7(input_data, metric):
    model = Sequential()
    model.add(Dense(100, input_dim=input_data.shape[1], kernel_initializer="uniform")) #best with 50: 2.03 #40: 1.81 #30:1.88
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer="uniform"))
    model.add(Activation('relu'))

    adam_opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_absolute_error', optimizer=adam_opt)
    
    return model

# Model 8
def model_8(input_data, metric):
    model = Sequential()
    model.add(Dense(100, input_dim=input_data.shape[1], kernel_initializer="uniform")) #best with 50: 2.03 #40: 1.81 #30:1.88
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, kernel_initializer="uniform"))
    #model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))
    
    adam_opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_absolute_error', optimizer=adam_opt)
    
    return model




# Model X
#def keras_model_3(input_data):
#    model = Sequential()

    # we can think of this chunk as the input layer
#    model.add(Dense(100, input_dim=input_data.shape[1], kernel_initializer="uniform")) #best with 50: 2.03 #40: 1.81 #30:1.88
#    model.add(BatchNormalization())
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2)) #best with .2

    # we can think of this chunk as the hidden layer    
    #model.add(Dense(50, kernel_initializer="uniform"))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    # we can think of this chunk as the output layer
#    model.add(Dense(1, kernel_initializer="uniform"))
    #model.add(BatchNormalization())
#    model.add(Activation('relu'))

    # setting up the optimization of our weights 
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # 100 hidden neurons does not work. Try with 100 BUT with Adam
#    adam= Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#    model.compile(loss='mean_absolute_error', optimizer=adam)
    
#    return model