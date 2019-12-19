import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def cross_validation_visualization(lambds, metric_te, ax):
    """visualization the curves of mse_tr and mse_te."""
    #ax.semilogx(lambds, metric_tr, marker=".", color='b', label='train accuracy')
    ax.semilogx(lambds, metric_te, label='test accuracy')
    ax.grid(True)


def visualize_boxplot_cross_validation2(k_data, pos, ax):
    ax.grid(True, axis='x', which='major', linestyle='--')
    ax.boxplot(k_data,positions = [np.exp(i+1) for i in range(6)], sym='+')


def test_alphas_meth(meth,alphas,X,y,k = 4):
    """
    Test the different values of alpha for a given method and return the best alpha according to the mse criterion.
    """
    res_mse = []
    for alpha in alphas: 
        rid = meth(alpha = alpha)
        res_mse.append(-cross_val_score(rid,X,y,cv = k,scoring='neg_mean_squared_error'))
    fig,ax = plt.subplots(1)
    ax.semilogx(alphas,res_mse)
    ax.set_ylabel('Mse')
    ax.set_xlabel('alphas')
    rse_mse_means = np.mean(res_mse,axis = 1)
    cross_validation_visualization(alphas,rse_mse_means,ax = ax)
    visualize_boxplot_cross_validation2(res_mse,alphas,ax)
    return alphas[np.argmin(rse_mse_means)]



