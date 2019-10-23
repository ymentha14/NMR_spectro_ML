# -*- coding: utf-8 -*-

# As we'redealing with large arrays, we will use the batch iter method in order to avoid memory access problems
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


