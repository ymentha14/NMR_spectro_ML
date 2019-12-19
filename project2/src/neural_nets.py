import numpy as np 

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, LeakyReLU

import helpers as hl 
import outliers


def train_model(model, train_input, train_target, mini_batch_size, monitor_loss=False):
    '''Train the model using Mini-batch SGD'''
    
    criterion = nn.MSELoss() #regression task
    optimizer = optim.Adam(model.parameters(), lr = 1e-4) #1e-4 normalement
    nb_epochs = 10
    
    # Monitor loss
    losses = []
    
    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, min(mini_batch_size, train_input.shape[0]-b)))
            loss = criterion(output, train_target.narrow(0, b, min(mini_batch_size, train_input.shape[0]-b)))
            model.zero_grad()
            loss.backward()
            
            sum_loss += loss.item() #compute loss for each mini batch for 1 epoch
            
            optimizer.step()
        
        # Monitor loss
        losses.append(sum_loss)
        
        print('[epoch {:d}] loss: {:0.2f}'.format(e+1, sum_loss))
    
    if monitor_loss:
        return losses
    
def k_fold_nn(n, X_total, y_total, iqr=True):
    mini_batch_size = 10
    
    # 4-fold cross-validation
    mse_storage = []
    mae_storage = []
    r2_storage = []

    # Load data, remove outliers but do not split yet into train and test
    X, y = hl.load_data(n, X_total, y_total)

    kf = KFold(n_splits=4, shuffle=True)

    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        print("FOLD {}".format(idx+1))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if iqr:
            X_train, y_train = outliers.IQR_y_outliers(X_train, y_train)

        train_input = torch.Tensor(X_train)
        test_input = torch.Tensor(X_test)
        train_target = torch.Tensor(y_train.reshape(len(y_train), 1))
        test_target = torch.Tensor(y_test.reshape(len(y_test), 1))

        nb_input_neurons = train_input.shape[1]

        model = Net_3(nb_input_neurons) 
        losses = train_model(model, train_input, train_target, mini_batch_size, monitor_loss=True)

        #Make predictions
        y_hat = compute_pred(model, test_input)

        #Compute score
        mse_nn, mae_nn, r2_nn = compute_score(y_test, y_hat.detach().numpy())

        mse_storage.append(mse_nn)
        mae_storage.append(mae_nn)
        r2_storage.append(r2_nn)
        
        print('MSE: {:0.2f} \nMAE: {:0.2f} \nr2: {:0.2f}'.format(mse_nn, mae_nn, r2_nn))
    
    return mse_storage, mae_storage, r2_storage
    
def compute_pred(model, data_input):
    '''Given a trained model, output the prediction corresponding to data_input'''
    y_hat = model(data_input)
    return y_hat

def compute_score(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    return mse, mae, r2

#Architecture 1
class Net1(nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n,101)
        self.fc2 = nn.Linear(100,1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
#Architecture 2
class Net_2(nn.Module):
    def __init__(self, n):
        super(Net_2, self).__init__()
        self.fc1 = nn.Linear(n,100)
        self.fc2 = nn.Linear(100,1)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

#Architecture 3
class Net_3(nn.Module):
    def __init__(self, n):
        super(Net_3, self).__init__()
        self.fc1 = nn.Linear(n,100)
        self.fc2 = nn.Linear(100,1)
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

# Architecture 4
class Net_4(nn.Module):
    def __init__(self, n):
        super(Net_4, self).__init__()
        self.fc1 = nn.Linear(n,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,1)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return x
    
#Architecture 5
class Net_5(nn.Module):
    def __init__(self, n):
        super(Net_5, self).__init__()
        self.fc1 = nn.Linear(n,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,50)
        self.fc5 = nn.Linear(50,1)
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        return x