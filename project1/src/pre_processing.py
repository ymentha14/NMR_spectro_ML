# -*- coding: utf-8 -*-


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA(x):
    mean = np.mean(x, axis=0)
    x_center = x - mean
    sigma = x_center.T.dot(x_center)
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    right_order = np.argsort(np.abs(eigen_values)) # argsort: ascending order (we want descending)
    eigen_vectors = eigen_vectors.T[right_order[::-1]] # revers order to have descending

    return mean, eigen_vectors

def clean_variance(x,x_test,inplace = False):
"""erases the constant columns of x. erases the corresponding columns in x_test """
    #*********TODO***********

    if inplace:
        x = x.copy()
        x_test = x_test.copy()
    raise Exception('Not implemented.')
    return x,x_test #dummy
    
    
    
def clean_value(x,value_in,value_out,inplace = False):
"""transforms the value_in entries of x in value_out"""
    #*********TODO***********

    if inplace:
        x = x.copy()
    raise Exception('Not implemented.')

    
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

def standardize_data(x, mean = None,std = None,inplace = True): #replace_nan=True
    #*********TODO***********
    #standardize the data: for x_train it returns x_std, the mean and the std. If mean and std are given, return the same mean, std and standardize x according to them.
    if inplace:
        x = x.copy()
    raise Exception("Not coded yet.")
    """
    
    """
    #Standardize data without taking nan (i.e. -999) into account
    #Replace -999 by 0 if replace_nan=True
    """
    x_stats = x.copy()
    x_std = x

    # Step 1: check for the presence of -999 and replace by nan
    if np.any(x == -999):
        x_stats[np.where(x_stats == -999)] = np.nan
    
    # Step 2: compute stats without taking nan into account
    sigma = np.nanstd(x_stats, axis=0)
    mu = np.nanmean(x_stats, axis=0)

    # Step 3: replace -999 by 0
    x_std[x_std == -999] = 0.0

    # Step 4: check if a col has all the same entries (i.e. std = 0)
    if np.all(sigma != 0):
        x_std = (x_std - mu) / sigma
    else:
        # ne pas standardiser seulement la colonne qui a std=0 
        # (notamment le cas pour la dernière colonne de la catégorie 0)
        idx = np.arange(0, x_std.shape[1])
        table = np.concatenate((idx.reshape(1, len(idx)), sigma.reshape(1, len(sigma))), axis=0)

        # idx   | 0             ...                28
        # --------------------------------------------
        # sigma | s0            ...                s28

        for i in idx: #range(len(idx)):
            if table[1][i] == 0:
                x_std[:, i] = (x_std[:, i] - mu[i])
            else:
                x_std[:, i] = (x_std[:, i] - mu[i]) / sigma[i] 

    return x_std, mu, sigma 
"""
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
