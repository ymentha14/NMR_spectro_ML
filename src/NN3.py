import numpy as np 
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn.model_selection import KFold

from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, LeakyReLU

class NN3():
    def __init__(self,mod_1,mod_2,mod_3,mini_batch_size = 10 ,
                 apply_iqr = True,apply_scaler = False,apply_pca = False,
                 assemble_y = 'custom',nb_epochs = 150,normalize = False):
        self.mod1 = mod_1
        self.mod2 = mod_2
        self.mod3 = mod_3
        self.minbatchsize = mini_batch_size
        self.assemble_y = assemble_y
        self.nb_epochs = nb_epochs
        self.apply_iqr = apply_iqr
        self.apply_pca = apply_pca
        self.apply_scaler = apply_scaler
        self.normalize = normalize
        
    def train_model(self,model, train_input, train_target, monitor_loss=False):
        criterion = nn.MSELoss() #regression task
        optimizer = optim.Adam(model.parameters(), lr = 1e-4) #1e-4 normalement

        # Monitor loss
        losses = []

        for e in range(self.nb_epochs):
            sum_loss = 0
            N = train_input.size(0)
            for b in range(0, N, self.minbatchsize):
                output = model(train_input.narrow(0, b, min(self.minbatchsize,N - b)))
                loss = criterion(output, train_target.narrow(0, b, min(self.minbatchsize,N - b)))
                model.zero_grad()
                loss.backward()

                sum_loss += loss.item() #compute loss for each mini batch for 1 epoch

                optimizer.step()

            # Monitor loss
            losses.append(sum_loss)

            print('[epoch {:d}] loss: {:0.2f}'.format(e+1, sum_loss))

        if monitor_loss:
            return losses
        
    def IQR_y_outliers(self,X1,X2,X3,y_data):
        ''' aims at removing all rows whose label (i.e. shielding) is considered as outlier.
        output:
         - X_filtered
         - y_filtered
        '''
        q1, q3 = np.percentile(y_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)

        assert(q1 != q3)

        idx = np.where((y_data > lower_bound) & (y_data < upper_bound))
        X1, X2, X3 = X1[idx], X2[idx], X3[idx]
        y_data = y_data[idx]

        assert(X1.shape[0] == y_data.shape[0] and X2.shape[0] == y_data.shape[0] and X3.shape[0] == y_data.shape[0])
        return X1, X2, X3, y_data
    
    def fit(self, X1,X2,X3,y):
        if self.apply_iqr:
            X1,X2,X3,y = self.IQR_y_outliers(X1,X2,X3,y)
        if self.normalize:
            self.trans1 = Normalizer().fit(X1)
            self.trans2 = Normalizer().fit(X2)
            self.tran3 = Normalizer.fit(X3)
        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
        X3 = torch.Tensor(X3)
        y = torch.Tensor(y.reshape(len(y), 1))
        print('#' * 30 + 'Training model 1'+ '#' * 30)
        loss1 = self.train_model(self.mod1, X1, y, monitor_loss=True)
        print('#' * 30 + 'Training model 2'+ '#' * 30)
        loss2 = self.train_model(self.mod2, X2, y, monitor_loss=True)
        print('#' * 30 + 'Training model 3'+ '#' * 30)
        loss3 = self.train_model(self.mod3, X3, y, monitor_loss=True)
        print('#' * 30 + 'TRAINING TERMINATED'+ '#' * 30)
        
    
    def droledemean(self,xs):
        xs = list(xs)
        invs = np.array([[1/np.abs(x -y) for y in xs if y is not x] for x in xs])
        tot = np.sum(invs)
        weights = np.sum(invs,axis = 1)
        return np.sum(weights * xs)/tot
        
    def assemble_ys(self,y1,y2,y3):
        if self.assemble_y == 'mean':
            return np.mean([y1,y2,y3],axis = 0)
        k = np.array([self.droledemean(i) for i in np.array([y1.reshape(y1.shape[0]),
                                                             y2.reshape(y2.shape[0]),
                                                             y3.reshape(y3.shape[0])]).T])
        return k
    
    def predict_indep(self,X1,X2,X3):
        if self.normalize:
            X1 = self.trans1.transform(X1)
            X2 = self.trans2.transform(X2)
            X3 = self.trans3.transform(X3)
        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
        X3 = torch.Tensor(X3)
        y1_hat = self.mod1(X1).detach().numpy()
        y2_hat = self.mod2(X2).detach().numpy()
        y3_hat = self.mod3(X3).detach().numpy()
        return y1_hat,y2_hat,y3_hat
    
    def set_mean(self,meth):
        assert(meth == 'mean' or meth == 'custom')
        self.assemble_y = meth
        
    def predict(self,X1,X2,X3):
        y1_hat,y2_hat,y3_hat = self.predict_indep(X1,X2,X3)
        return self.assemble_ys(y1_hat,y2_hat,y3_hat)      

    
def compute_score(y_actual, y_pred,verbose = False):
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    if verbose:
        print("Obtained MSE on test set %2.2f " % mse)
        print("Obtained MAE on test set %2.2f " % mae)
    return {'mse':mse,'mae':mae}

def KFold3(model3,data_X1,data_X2,data_X3,data_y,n_splits = 4,verbose = True):
    """
    perform Kfold cross validation on an NN3 object
    """
    kf = KFold(n_splits=n_splits, random_state=14, shuffle=True)
    scores_mean = {'mse':[],'mae':[]}
    scores_custom = {'mse':[],'mae':[]}
    for kindx,(train_index, test_index) in enumerate(kf.split(data_y)):
        
        print('%i / %i fold' % (kindx+1,n_splits))
        X1_train, X1_test = data_X1[train_index],data_X1[test_index]
        X2_train, X2_test = data_X2[train_index],data_X2[test_index]
        X3_train, X3_test = data_X3[train_index],data_X3[test_index]
        y_train, y_test = data_y[train_index], data_y[test_index]
        model3.fit(X1_train,X2_train,X3_train,y_train)
        
        model3.set_mean('mean')
        y_hat = model3.predict(X1_test,X2_test,X3_test)
        score_mean = compute_score(y_test,y_hat)
        scores_mean['mse'].append(score_mean['mse'])
        scores_mean['mae'].append(score_mean['mae'])
        
        model3.set_mean('custom')
        y_hat = model3.predict(X1_test,X2_test,X3_test)
        score_custom = compute_score(y_test,y_hat)
        scores_custom['mse'].append(score_custom['mse'])
        scores_custom['mae'].append(score_custom['mae'])
        
        if verbose:
            print('Mean method:{}'.format(score_mean))
            print('Custom method:{}'.format(score_custom))
    return scores_mean,scores_custom


#single training no kfold
def single_test(bibgbibo,data_X1,data_X2,data_X3,data_y):
    train_size = 200
    mask = np.full(data_y.shape[0], False)
    mask[:train_size] = True
    np.random.shuffle(mask)

    y_train = data_y[mask]
    y_test = data_y[~mask]
    X1_train = data_X1[mask]
    X1_test = data_X1[~mask]
    X2_train = data_X2[mask]
    X2_test = data_X2[~mask]
    X3_train = data_X3[mask]
    X3_test = data_X3[~mask]

    bigbibo.fit(X1_train,X2_train,X3_train,y_train)
    
    bigbibo.set_mean('mean')
    y_hat = bigbibo.predict(X1_test,X2_test,X3_test)
    score_mean = compute_score(y_test,y_hat)
    
    bigbibo.set_mean('custom')
    y_hat = bigbibo.predict(X1_test,X2_test,X3_test)
    score_custom = compute_score(y_test,y_hat)
    
    return score_mean,score_custom
