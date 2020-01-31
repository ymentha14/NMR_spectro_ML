import numpy as np 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch 
import torch.nn as nn
#import torch.nn.functional as F
from torch import optim
import neural_nets
import outliers


def load_data(n_samples, tot_data_x, tot_data_y):    
    data_len = tot_data_x.shape[0]
    mask_data = np.random.permutation(data_len)[:n_samples]

    data_X = tot_data_x[mask_data]
    data_y = tot_data_y[mask_data]
    return data_X, data_y

def load_data_train_test(n_samples, tot_data_x, tot_data_y, iqr=False):
    data_X, data_y = load_data(n_samples, tot_data_x, tot_data_y)
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2)

    if iqr:
        X_train, y_train = outliers.IQR_y_outliers(X_train, y_train)

    return X_train, X_test, y_train, y_test


    
def univ_feat_sel(X,y,keep = 0.2,ax = None):
    """
    Plot feature selection using the f_regression from skleanr: 
    it computes an F score based on the correlation of the feature and 
    compute a pvalue based on that.
    X: data
    y: label
    keep: ratio of initial features you want to keep 
    """
    k = int(keep * X.shape[1])
    model = SelectKBest(score_func=f_regression, k=k)
    fit = model.fit(X, y)
    feature_ord_univ = np.argsort(fit.scores_)
    if ax is None:
        plt.bar(feature_ord_univ,fit.scores_)
    else:
        ax.bar(feature_ord_univ,fit.scores_)
    
    
def do_PCA(X_train, X_test, n):
    """
    Useful method to reduce train/test sets outside of the pipeline
    """
    pca = PCA(n_components=n)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test



def load_process_data(n, iqr=False, pca=False, scale=False):
    # load all datas
    # filter (iqr)
    if iqr:
        X_train, X_test, y_train, y_test = load_data_train_test(n, X_tot, y_tot, iqr=apply_iqr)
    else:
        X_train, X_test, y_train, y_test = load_data_train_test(n, X_tot, y_tot)
        
    if scale:
        min_max = MinMaxScaler()
        X_train, X_test = apply_scaler(min_max, X_train, X_test)
        
    if pca:
        n = 80
        X_train, X_test = do_PCA(X_train, X_test, n)
        nb_input_neurons = n

    return X_train, X_test, y_train, y_test

def compute_iqr_effect(n_sample1, n_sample2, n_fold, X_total, y_total, apply_pca=False):
    apply_iqr = [False, True]
    n_samples = [n_sample1, n_sample2] 
    n_iter = n_fold
    
    noiqr_small_n_mse = []
    noiqr_large_n_mse = []
    iqr_small_n_mse = []
    iqr_large_n_mse = []

    noiqr_small_n_mae = []
    noiqr_large_n_mae = []
    iqr_small_n_mae = []
    iqr_large_n_mae = []

    noiqr_small_n_r2 = []
    noiqr_large_n_r2 = []
    iqr_small_n_r2 = []
    iqr_large_n_r2 = []
    
    mini_batch_size = 10
    #nb_input_neurons = 14400

    for b in apply_iqr:
        for n in n_samples:
            for i in range(n_iter):
                # load each time a different set (kind of cross-val)
                X_train, X_test, y_train, y_test = load_data_train_test(n, X_total, y_total, iqr=b)            

                train_input = torch.Tensor(X_train)
                test_input = torch.Tensor(X_test)
                train_target = torch.Tensor(y_train.reshape(len(y_train), 1))
                test_target = torch.Tensor(y_test.reshape(len(y_test), 1))

                model = neural_nets.Net_3(train_input.shape[1]) 
                losses = neural_nets.train_model(model, train_input, train_target, mini_batch_size, monitor_loss=True)

                #Make predictions
                y_hat = neural_nets.compute_pred(model, test_input)

                #Compute score
                mse_nn, mae_nn, r2_nn = neural_nets.compute_score(y_test, y_hat.detach().numpy())

                # Store result
                # Case: iqr, 5000 samples
                if (b and (n == n_samples[0])):
                    iqr_small_n_mse.append(mse_nn)
                    iqr_small_n_mae.append(mae_nn)
                    iqr_small_n_r2.append(r2_nn)
                elif ((not b) and (n == n_samples[0])):
                    noiqr_small_n_mse.append(mse_nn)
                    noiqr_small_n_mae.append(mae_nn)
                    noiqr_small_n_r2.append(r2_nn)
                elif (b and (n == n_samples[1])):
                    iqr_large_n_mse.append(mse_nn)
                    iqr_large_n_mae.append(mae_nn)
                    iqr_large_n_r2.append(r2_nn)
                else:
                    noiqr_large_n_mse.append(mse_nn)
                    noiqr_large_n_mae.append(mae_nn)
                    noiqr_large_n_r2.append(r2_nn)
    
    return noiqr_small_n_mse, noiqr_large_n_mse, iqr_small_n_mse, iqr_large_n_mse, noiqr_small_n_mae, noiqr_large_n_mae, iqr_small_n_mae, iqr_large_n_mae, noiqr_small_n_r2, noiqr_large_n_r2, iqr_small_n_r2, iqr_large_n_r2 






