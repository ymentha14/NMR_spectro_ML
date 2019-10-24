# -*- coding: utf-8 -*-


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def PCA(x):
    mean = np.mean(x, axis=0)
    x_center = x - mean
    sigma = x_center.T.dot(x_center)
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    right_order = np.argsort(np.abs(eigen_values)) # argsort: ascending order (we want descending)
    eigen_vectors = eigen_vectors.T[right_order[::-1]] # revers order to have descending

    return mean, eigen_vectors

def clean_variance(x,x_test):
    """
    Erases the constant columns of x. erases the corresponding columns in x_test 
    """    
    # kill feature 22 first (mandatory if we group categories 2 and 3)
    x = np.delete(x, 22, 1)
    x_test = np.delete(x_test, 22, 1)

    # Compute vars and kill columns with std == 0
    sigmas = np.std(x, axis=0)
    idx_cst_std = np.where(sigmas == 0)
    x = np.delete(x, idx_cst_std, 1)
    x_test = np.delete(x_test, idx_cst_std, 1)
    return x, x_test 

def toy(x):
    x = np.zeros(3)
    
def clean_value(x,value_in,value_out,inplace = False):
    """transforms the value_in entries of x in value_out"""
    #*********TODO***********
    if value_in is np.nan:
        mask  = np.isnan(x)
        x[mask] = value_out
        return x
    x[x == value_in] = value_out
    return x
    
def standardize(x, mean=None, std=None):
    ''' Standardize the data such that it has a mean of 0 and a unitary standard deviation. If mean and std are
        provided (e.g. when we standardize the testing set based on the standardization parameters of the
        training set), the function uses them. '''

    if mean is None:
        mean = np.mean(x, axis=0)
    x -= mean

    if std is None:
        std = np.std(x, axis=0)
    
    x /= std

    return x, mean, std

def standardize_data(x, mean = None,std = None): #replace_nan=True
    #*********TODO***********
    #standardize the data: for x_train it returns x_std, the mean and the std. If mean and std are given, return the same mean, std and standardize x according to them.
    """
    # 1. -999 -> nan
    if np.any(x == -999):
        x[np.where(x == -999)] = np.nan
    """
    # 2. np.nanmean, np.nanstd -> standardize
    if mean is None:
        mean = np.nanmean(x, axis=0)
    x -= mean

    if std is None:
        std = np.nanstd(x, axis=0)
        
    x /= std
    """
    # 3. nan -> 0
    if np.any(np.isnan(x)):
        mask = np.isnan(x)
        x[np.where(mask)] = 0
    """
    return x, mean, std
    

def PCA_visualize(tX):
    #shows the percentage of variance in tX explained by the first x PCs.
    pca = PCA(n_components=30)
    pca.fit(tX)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.cumsum(pca.explained_variance_ratio_))
    ax[0].set_xlabel('Number of components')
    ax[0].set_ylabel('Cumulative explained variance')
    """
    scaler = StandardScaler()
    scaler.fit(tX)
    tX_std = scaler.transform(tX)

    pca.fit(tX_std)
    ax[1].plot(np.cumsum(pca.explained_variance_ratio_))
    ax[1].set_xlabel('Number of components')
    ax[1].set_ylabel('Cumulative explained variance STD')
    """
