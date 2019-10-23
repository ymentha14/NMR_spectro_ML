# -*- coding: utf-8 -*-
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
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD."""
    # Note: we don't have to keep track of the weithts and losses
    w = initial_w
    
    # as we assume that the constant term is contained in x
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # gradient descent
    for i in range(max_iters):
        # loss
        loss = compute_loss_lr(y, tx, w)

        # gradient
        gradient = compute_gradient_lr(y, tx, w)
        
        # update rule
        w -= gamma * gradient
    
    return w, loss

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
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD."""
    # same as before except that we have the additional penalty term
    w = initial_w
    
    # constant term contained in x
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    
    # gradient descent
    for i in range(max_iters):
        # loss
        loss =  compute_loss_lr + lambda_/2 *  np.squeeze(w.T.dot(w))
        
        # gradient
        gradient =  compute_gradient_lr + lambda_ * w
        
        # update rule
        w -= gamma * gradient
    
    return w, loss

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
