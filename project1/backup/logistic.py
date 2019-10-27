# -*- coding: utf-8 -*-
from split import *

def sigmoid(s):
    """apply sigmoid function."""
    return 1 / (1 + np.exp(-s))

def compute_loss_lr(y, tx, w):
    """
    Compute the loss of the logistic regression
    by negative log likelihood.
    """
    prediction = sigmoid(tx.dot(w))

    # to avoid log(0)
    eps = 1e-10

    loss = y.T.dot(np.log(prediction + eps)) + (1.0 - y).T.dot(np.log((1.0 - prediction) + eps))
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
