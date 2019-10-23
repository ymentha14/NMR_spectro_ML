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
