# -*- coding: utf-8 -*-

from split import *

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

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # kill cst columns (as this is our convention, it is not compatible with build_poly) 
    x_red = np.delete(x, 0, 1)
    poly = np.ones((len(x_red), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x_red, deg)]
    return poly #np.concatenate([np.power(x,deg) for deg in range(degree+1)],axis = 1)

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
