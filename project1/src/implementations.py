'''
Machine Learning (CS-433)

@author: Gianni Giusto, Maxime Epars, Yann Mentha 
'''


### - IMPORTS - ###
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


### - FUNCTIONS - ###
# - Helpers
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# - Linear regression            
def compute_mse(y, tx, w):
    '''Compute MSE loss.'''
    e = y - tx.dot(w)
    mse = (1/2) * np.mean(e**2)
    return mse

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss_ls(y, tx, w):
    """
    Calculate the least-square loss using mse.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient_ls(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    w = initial_w
    
    for i in range(max_iters):
        # gradient
        gradient = compute_gradient_ls(y, tx, w)
        
        # loss
        loss = compute_loss_ls(y, tx, w)
        
        # update rule
        w -= gamma * gradient
        
    return w, loss

def least_square_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using SGD.
    Compute MSE loss
    """
    w = initial_w
    
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # gradient
            gradient = compute_gradient_ls(y_batch, tx_batch, w)
           
            # update rule
            w -= gamma * gradient
            
            # loss
            loss = compute_loss_ls(y_batch, tx_batch, w)
            
    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations.
    Note: better use solve. Problems occur when using np.linalg.inv()
    np.linalg.solve(a,b): solve problems of type ax=b
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss_ls(y, tx, w)
    
    return w, loss

# - Ridge regression
def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N = tx.shape[0]
    D = tx.shape[1]
    a = tx.T.dot(tx) + 2*N*lambda_*np.identity(D)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a, b)
    loss = compute_loss_ls(y, tx, w)
        
    return w, loss

# - Logistic regression
def sigmoid(s):
    """apply sigmoid function."""
    return 1 / (1 + np.exp(-s))

def compute_loss_lr(y, tx, w):
    """
    Compute the loss of the logistic regression
    by negative log likelihood.
    """
    prediction = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(prediction)) + (1 - y).T.dot(np.log(1 - prediction))
    return np.squeeze(-loss)

def compute_gradient_lr(y, tx, w):
    """compute the gradient of loss for the logistic regression."""
    prediction = sigmoid(tx.dot(w))
    gradient = tx.T.dot(prediction - y)
    return gradient

# GD version
#def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    # Note: we don't have to keep track of the weithts and losses
#    w = initial_w
    
    # as we assume that the constant term is contained in x
#    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # gradient descent
#    for i in range(max_iters):
        # loss
#        loss = compute_loss_lr(y, tx, w)

        # gradient
#        gradient = compute_gradient_lr(y, tx, w)
        
        # update rule
#        w -= gamma * gradient
    
#    return w, loss

# SGD version
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""   
    w = initial_w
    
    # as we assume that the constant term is contained in x
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # stochastic gradient descent
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1): 
            # loss
            loss = compute_loss_lr(y_batch, tx_batch, w)

            # gradient
            gradient = compute_gradient_lr(y_batch, tx_batch, w)

            # update rule
            w -= gamma * gradient
    
    return w, loss

# GD version
#def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""
    # same as before except that we have the additional penalty term
#    w = initial_w
    
    # constant term contained in x
#    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # gradient descent
#    for i in range(max_iters):
        # loss
#        loss =  compute_loss_lr + lambda_/2 *  np.squeeze(w.T.dot(w))
        
        # gradient
#        gradient =  compute_gradient_lr + lambda_ * w
        
        # update rule
#        w -= gamma * gradient
    
#    return w, loss

# SGD version
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""  
    w = initial_w
    
    # constant term contained in x
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # gradient descent
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1): 
            # loss
            loss = compute_loss_lr(y_batch, tx_batch, w) + lambda_/2 *  np.squeeze(w.T.dot(w))

            # gradient
            gradient = compute_gradient_lr(y_batch, x_batch, w) + lambda_ * w

            # update rule
            w -= gamma * gradient
    
    return w, loss


### - MORE METHODS - ###
def PCA(x):
    mean = np.mean(x, axis=0)
    x_center = x - mean
    sigma = x_center.T.dot(x_center)
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    right_order = np.argsort(np.abs(eigen_values)) # argsort: ascending order (we want descending)
    eigen_vectors = eigen_vectors.T[right_order[::-1]] # revers order to have descending

    return mean, eigen_vectors
def train_test_split(X, y, ratio, seed=12):
    """Split the X and y sets into a training and a validation/testing one."""
    
    np.random.seed(seed)

    permutations = np.random.permutation(len(X))
    x = X[permutations]
    y = y[permutations]
    
    idx = int(np.floor(ratio * len(X)))
    x_train = x[:idx]
    y_train = y[:idx]
    x_test = x[idx:]
    y_test = y[idx:]
    
    return x_train, x_test, y_train, y_test

def accuracy(y, y_hat):
    """
    Compute the accuracy between the real values (y) 
    and the predicted ones (y_hat).
    Note: for binary predictions with values {0, 1} only.
    """
    return np.sum(y == y_hat) / len(y)

def k_fold_cv(y, X, k):
    """
    k-fold cross-validation for model selection.
    
    y: labels
    X: features matrix
    k: number of folds
    
    output: mean accuracy and std across the k-folds
    """
    
    # Before partitionning the data we shuffle it randomly
    permutations = np.random.permutation(len(y))
    X = X[permutations]
    y = y[permutations]
    
    # array to store the outcome accuracy  of the different folds
    accuracies = np.zeros(k)
    
    # Indexes for the k different intervals
    idxs = np.linspace(0, len(y), k+1, dtype=int) # +1 for 'upper bound'
     
    for i in range(k):
        X_te = X[idxs[i]:idxs[i+1]] #test indexes
        y_te = y[idxs[i]:idxs[i+1]]
        
        X_tr = np.delete(X, list(np.arange(idxs[i], idxs[i+1])), axis=0)
        y_tr = np.delete(y, list(np.arange(idxs[i], idxs[i+1])), axis=0)
        
        ## TODO: implement the classifier part 
        # 1. fit data
        # 2. compute y_hat
        # 3. compute accuracy
    
    return np.mean(accuracies), np.std(accuracies)


### - DATA PRE-PROCESSING - ###
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

def standardize_data(x):
    '''
    TODO: deal with -999 
    '''
    mu = np.mean(x, axis=0)
    sigma = np.std(x_centered, axis=0) 
    std_x = (x - mu) / sigma

    return std_x, mu, sigma

def split_dataset(y, x):
    """Split the dataset into subsets, each subset containing all datapoints that have the same value for 
       the categorical feature PRI_jet_num. The function returns each subset and its corresponding labels """
    PRI_jet_num_values = np.unique(x[:,22])
    sub_group_indices = []
    x_subgroup = []
    y_subgroup = []
    for i in range(int(np.amax(PRI_jet_num_values))+1):
        sub_group_indices.append(np.where(x[:,22]==i))
        x_subgroup.append(x[sub_group_indices[i],:])
        y_subgroup.append(y[sub_group_indices[i]])
        x_subgroup[i] = np.squeeze(x_subgroup[i])
        x_subgroup[i] = np.delete(x_subgroup[i], 22, axis=1)
    
    return x_subgroup[0], y_subgroup[0], x_subgroup[1], y_subgroup[1], x_subgroup[2], y_subgroup[2], x_subgroup[3], y_subgroup[3]

def split_categorical_data(data, labels, feature_nb):
    '''
    Split the dataset and its labels into 3 distincts subsets:
        - PRI_jet_num = 0 (= data_0)
        - PRI_jet_num = 1 (= data_1)
        - PRI_jet_num = 2, 3 (= data_3)

    Note: the PRI_jet_num corresponds to the 22th if we use the original data (i.e. tX)
    '''
    # not necessary to create a copy
    # category 0
    data_0 = data[np.where(data[:, feature_nb] == 0)]
    labels_0 = labels[np.where(data[:, feature_nb] == 0)]

    # category 1
    data_1 = data[np.where(data[:, feature_nb] == 1)]
    labels_1 = labels[np.where(data[:, feature_nb] == 1)]

    # category 2 & 3
    data_2 = data[np.where(data[:, feature_nb] >= 2)]
    labels_2 = labels[np.where(data[:, feature_nb] >= 2)]

    return data_0, data_1, data_2, labels_0, labels_1, labels_2

### - TRAINING & TESTING - ###







