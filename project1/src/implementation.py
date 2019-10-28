#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv

## IMPLEMENTATION


T_pos = lambda a,b: sum(1 for x,y in zip(a,b) if x == y and x == 1)
F_pos = lambda a,b: sum(1 for x,y in zip(a,b) if x != y and x == 1)
T_neg = lambda a,b: sum(1 for x,y in zip(a,b) if x == y and x == -1)
F_neg = lambda a,b: sum(1 for x,y in zip(a,b) if x != y and x == -1)


def accuracy(y,y_corr):
    return (T_pos(y,y_corr) + T_neg(y,y_corr))/len(y)
    
def precision(y,y_corr):
    Tp = T_pos(y,y_corr)
    Fp = F_pos(y,y_corr)
    if (Tp + Fp == 0):
        return 0
    return Tp/(Tp + Fp)

def recall(y,y_corr):
    Tp = T_pos(y,y_corr)
    Fn = F_neg(y,y_corr)
    return Tp / (Tp + Fn)
    
def f1(y,y_corr):
    prec = precision(y,y_corr)
    rec = recall(y,y_corr)
    if (prec + rec == 0):
        return 0
    return 2*prec*rec / (prec + rec)

dico = {accuracy:"accuracy",f1:"F1 score"}


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


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

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N = tx.shape[0]
    D = tx.shape[1]
    a = tx.T.dot(tx) + 2*N*lambda_*np.identity(D)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss_ls(y, tx, w)

    return w, loss

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

# SGD version

def logistic_regression(y, tx, initial_w, max_iters = 1000, gamma = 0.2):
    """Logistic regression using gradient descent or SGD."""   
    w = initial_w
    loss = 0
    # stochastic gradient descent
    for i in range(max_iters):
        if i % 10 == 0 :print("logreg {i}th iteration:,  loss = {loss}".format(i = i,loss = loss))
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=100, num_batches=1): 
            # loss
            loss = compute_loss_lr(y_batch, tx_batch, w)

            # gradient
            gradient = compute_gradient_lr(y_batch, tx_batch, w)

            # update rule
            w -= gamma * gradient
    
    return w, loss


# SGD version
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""  
    w = initial_w
        
    # gradient descent
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=100, num_batches=1): 
            # loss
            loss = compute_loss_lr(y_batch, tx_batch, w) + lambda_/2 *  np.squeeze(w.T.dot(w))

            # gradient
            gradient = compute_gradient_lr(y_batch, tx_batch, w) + lambda_ * w

            # update rule
            w -= gamma * gradient
    
    return w, loss
##DATA PROCESSING



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

def get_best_cutoff(X,y,w,metric):
    cutoffs = np.linspace(-5,5,40)
    y_hat = X @ w
    y_hat_tries = [[-1 if i < cut else 1.0 for i in y_hat] for cut in cutoffs]
    metr_per_cut = [metric(j,y) for j in y_hat_tries]
    opt_indx = np.argmax(metr_per_cut)
    opt_cutoff = cutoffs[opt_indx]
    assert(opt_indx != 0 and opt_indx != len(cutoffs)-1)
    return opt_cutoff


def k_fold_cv(y, X, k,f,metric = accuracy,verbose = True):
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
    cutoffs = np.linspace(-5,5,40)

    # array to store the outcome accuracy  of the different folds
    metric_train = []
    metric_test = []
    opt_cutoffs = []
    # Indexes for the k different intervals
    idxs = np.linspace(0, len(y), k+1, dtype=int) # +1 for 'upper bound'
    for i in range(k):
        if verbose: print("{i}/{k} round for the kfold:".format(i = i + 1, k = k))
        try:
            X_te = X[idxs[i]:idxs[i+1]] #test indexes
            y_te = y[idxs[i]:idxs[i+1]]

            X_tr = np.delete(X, list(np.arange(idxs[i], idxs[i+1])), axis=0)
            y_tr = np.delete(y, list(np.arange(idxs[i], idxs[i+1])), axis=0)

            ## TODO: implement the classifier part
            # 1. fit data

            w_opt, loss_train = f(y_tr,X_tr)
            # 2. compute y_hat
            y_hat_test = X_te @ w_opt
            y_hat_train = X_tr @ w_opt
            y_hat_trains = [[-1 if i < cut else 1.0 for i in y_hat_train] for cut in cutoffs]
            metr_per_cut = [metric(j,y_tr) for j in y_hat_trains]
            opt_indx = np.argmax(metr_per_cut)
            opt_cutoff = cutoffs[opt_indx]

            y_hat_train = [-1 if i < opt_cutoff else 1.0 for i in y_hat_train]
            y_hat_test = [-1 if i < opt_cutoff else 1.0 for i in y_hat_test]

            # 3. compute accuracy
            obtained_met_test = metric (y_hat_test,y_te)
            obtained_met_train = metric(y_hat_train,y_tr)

            if verbose: print("obtained " + dico[metric] + " on {i}/{k} round of kfold:{acc}".format(i = i + 1,k = k, acc = obtained_met_test))
            metric_train.append(obtained_met_train)
            metric_test.append(obtained_met_test)
            opt_cutoffs.append(opt_cutoff)
        except:
            print("Singular Matrix in KFOLD!")

    return metric_train,metric_test,opt_cutoffs







def split_categorical_data(data,feature_indx,labels = None, split = True):
    '''
    Split the dataset and its labels into 3 distincts subsets:
        - PRI_jet_num = 0 (= data_0)
        - PRI_jet_num = 1 (= data_1)
        - PRI_jet_num = 2, 3 (= data_3)

    Note: the PRI_jet_num corresponds to the 22th if we use the original data (i.e. tX)
    '''
    if (not split):
        #case xtest, we return the indexes corresponding to the matrix
        if labels is None: labels = np.array(range(len(data)))
        return [(data,labels)]
    # not necessary to create a copy
    # category 0
    index_0 = np.array(np.where(data[:, feature_indx] == 0))
    index_0 = np.squeeze(index_0)
    data_0 = data[index_0]
    index_1 = np.array(np.where(data[:, feature_indx] == 1))
    index_1 = np.squeeze(index_1)
    data_1 = data[index_1]
    index_2 = np.array(np.where(data[:, feature_indx] >= 2))
    index_2 = np.squeeze(index_2)
    data_2 = data[index_2]
    if labels is not None:
        labels_0 = labels[index_0]
        labels_1 = labels[index_1]
        labels_2 = labels[index_2]
        return [(data_0,labels_0), (data_1,labels_1), (data_2,labels_2)]
    return [(data_0,index_0),(data_1,index_1),(data_2,index_2)]


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


##DATA VIZUALIZATION
def cross_validation_visualization(lambds, metric_tr, metric_te, ax, index):
    """visualization the curves of mse_tr and mse_te."""
    ax.set_title("Cross validation of subset {i}".format(i=index))
    ax.semilogx(lambds, metric_tr, marker=".", color='b', label='train accuracy')
    ax.semilogx(lambds, metric_te, marker=".", color='r', label='test accuracy')
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.legend(loc='upper right')
    ax.grid(True)

def visualize_boxplot_cross_validation(k_data, pos = None):
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.boxplot(k_data, positions = pos, sym='+')

def visualize_boxplot_cross_validation2(k_data, pos, ax, index, training=True):
    ax.set_xlabel("lambda")
    ax.set_ylabel("accuracy")
    ax.grid(True, axis='x', which='major', linestyle='--')
    if training:
        ax.set_title("Cross validation for training of categorical subset {i}".format(i=index))
    else:
        ax.set_title("Cross validation for testing of categorical subset {i}".format(i=index))

    ax.boxplot(k_data, positions = pos, sym='+')




##UTILITARIES
