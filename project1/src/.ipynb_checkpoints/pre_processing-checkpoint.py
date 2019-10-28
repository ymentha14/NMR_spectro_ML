# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np


def get_PCA(x,mean= None):
    if mean is None:
        mean = np.mean(x, axis=0)
    x_center = x - mean
    sigma = x_center.T.dot(x_center)
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    right_order = np.argsort(np.abs(eigen_values)) # argsort: ascending order (we want descending)
    eigen_vectors = eigen_vectors.T[right_order[::-1]] # revers order to have descending
    coeffs = x_center @ eigen_vectors
    return mean,eigen_vectors,eigen_values

def reduce_PCA(eigvecs,x,factor):
    #n x q
    coeffs = x @ eigvecs
    #q x k
    eigvecs2 = eigvecs[:,:factor]
    #n x k
    coeffs2 = coeffs[:,:factor]
    #n x q
    #x = coeffs2 @ eigvecs2.transpose()
    return np.array(coeffs2)

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

def build_poly(x, degree):
    assert(all(x[:,0] == 1))
    assert(x.shape[1] <= 31)
    
    x = np.delete(x, 0, axis = 1)

    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.concatenate([np.power(x,deg) for deg in range(1,degree+1)],axis = 1)
    x = np.c_[np.ones((x.shape[0], 1)), x]
    return x


    
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
    pca = get_PCA(tX)
    plt.plot(pca)