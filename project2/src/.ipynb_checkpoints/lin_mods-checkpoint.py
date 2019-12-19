import numpy as np 
import matplotlib.pyplot as plt
import datetime
import json
import pickle

from sklearn.model_selection import cross_val_score


def cross_validation_visualization(lambds, metric_te, ax):
    """visualization the curves of mse_tr and mse_te."""
    #ax.semilogx(lambds, metric_tr, marker=".", color='b', label='train accuracy')
    ax.semilogx(lambds, metric_te, label='test accuracy')


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
    ax.set_xscale('log')
    ax.semilogx(alphas,res_mse)
    ax.set_ylabel('Mse')
    ax.set_xlabel('alphas')
    ax.grid(True)
    rse_mse_means = np.mean(res_mse,axis = 1)
    #cross_validation_visualization(alphas,rse_mse_means,ax = ax)
    #visualize_boxplot_cross_validation2(res_mse,alphas,ax)
    return alphas[np.argmin(rse_mse_means)]


def log_model(results,pipeline,param_grid,datapath):
    """
    Write a log file of the model in order to keep trace of it
    """
    
    date = (datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    mse = -results['test_neg_mean_squared_error'].mean()
    mae = -results['test_neg_mean_absolute_error'].mean()
    r2 = results['test_r2'].mean()
    file_name = 'mae=%.2f_mse=%.2f_R2%.2f='%(mae,mse,r2) + date 
    file_path = './log'
    
    res = {'mae':mae,'mse':mse,'r2':r2}
    
    pipeline = [str(i) for i in list(pipeline)]

    def defo(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError('Not serializable')
    with open('../log/' + datapath + '/' + file_name + '.txt', 'w',encoding="utf-8",newline='\r\n') as file:
        json.dump(res,file,indent=4,ensure_ascii=False)
        json.dump(pipeline,file,default=defo)
    with open('../log/' + datapath + '/' + file_name + '.pickle','wb') as file:
        pickle.dump([pipeline,param_grid],file)
    print('Log saved')

def display_score(cv_results):
    """
    cv_results:dictionarry having
    test_neg_mean_squared_error,test_neg_mean_absolute_error and test_r2 as its key
    """
    K = len(cv_results['test_neg_mean_squared_error'])
    mse = -cv_results['test_neg_mean_squared_error'].mean()
    mae = -cv_results['test_neg_mean_absolute_error'].mean()
    r2 = cv_results['test_r2'].mean()
    print("On %i folds" % K)
    print("Obtained MSE on test set %2.2f " % mse)
    print("Obtained MAE on test set %2.2f " % mae)
    print("Obtained r2 on test set %2.2f " % r2)

    
def bias_variance_visualization(scoring_train, scoring_test, mean_scoring_train, mean_scoring_test, data_range, axis, scoring):
    """
    Visulize the bias-variance decomposition on 3 subplots, one for each scoring.
    """ 
    axis.plot(
        data_range,
        scoring_train,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    axis.plot(
        data_range,
        scoring_test,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    axis.plot(
        data_range,
        mean_scoring_train,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    axis.plot(
        data_range,
        mean_scoring_test,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    axis.set_xlabel("data size")
    axis.set_ylabel("error")
    axis.set_title(scoring)
    #axis.legend(loc='best')

def bias_variance_decomposition(data_range, results, seeds):
    """
    Decompose the results and triggers their visualization.
    """
    
    print("Start printing... \n")
    
    k = len(results[0][0]['fit_time'])
    mse_tr, mae_tr, r2_tr, mse_te, mae_te, r2_te = np.zeros((6, len(data_range), len(seeds), k))
    
    # Splitting the results into the different scorings and training and testing errors.
    for i, size in enumerate(data_range):
        for index, seed in enumerate(seeds):
            
            mse_tr[i][index] = -results[i][index]['train_neg_mean_squared_error']
            mse_te[i][index] = -results[i][index]['test_neg_mean_squared_error']
            
            mae_tr[i][index] = -results[i][index]['train_neg_mean_absolute_error']
            mae_te[i][index] = -results[i][index]['test_neg_mean_absolute_error']
            
            r2_tr[i][index] = results[i][index]['train_r2']
            r2_te[i][index] = results[i][index]['test_r2']
    
    # averaging the results over k-fold and then random seeds
    mse_tr_mean_kfold = mse_tr.mean(axis=2)
    mse_tr_mean_seeds = mse_tr_mean_kfold.mean(axis=1)
    mse_te_mean_kfold = mse_te.mean(axis=2)
    mse_te_mean_seeds = mse_te_mean_kfold.mean(axis=1)
    
    mae_tr_mean_kfold = mae_tr.mean(axis=2)
    mae_tr_mean_seeds = mae_tr_mean_kfold.mean(axis=1)
    mae_te_mean_kfold = mae_te.mean(axis=2)
    mae_te_mean_seeds = mae_te_mean_kfold.mean(axis=1)
    
    r2_tr_mean_kfold = r2_tr.mean(axis=2)
    r2_tr_mean_seeds = r2_tr_mean_kfold.mean(axis=1)
    r2_te_mean_kfold = r2_te.mean(axis=2)
    r2_te_mean_seeds = r2_te_mean_kfold.mean(axis=1)
    
    # Visualization
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fig.set_size_inches(14,9)
    
    bias_variance_visualization(mse_tr_mean_kfold,mse_te_mean_kfold, mse_tr_mean_seeds, mse_te_mean_seeds, data_range, axes[0], 'mse')
    bias_variance_visualization(mae_tr_mean_kfold, mae_te_mean_kfold, mae_tr_mean_seeds, mae_te_mean_seeds,data_range, axes[1], 'mae')
    bias_variance_visualization(r2_tr_mean_kfold, r2_te_mean_kfold, r2_tr_mean_seeds, r2_te_mean_seeds, data_range, axes[2], 'r2')
    
    plt.savefig("bias-variance")
    
    return mse_tr_mean_seeds, mae_tr_mean_seeds, r2_tr_mean_seeds, mse_te_mean_seeds, mae_te_mean_seeds, r2_te_mean_seeds

def bias_variance(pipeline, start, stop, number, seed_number):
    """
    Bias-variance decomposition to test the predictive power of a pipeline with subsets of different sizes.
    """
    data_range = np.logspace(np.log10(start),np.log10(stop),number, dtype=int)
    results = []
    
    seeds = range(seed_number)
    
    for iter_, size in enumerate(data_range):
        print('Data size of iteration {i}: {s} \n'.format(i=iter_, s=size))
        results.append(pipeline(size, seeds))
    
    print('Finished cross-validation...\n')
    
    return bias_variance_decomposition(data_range, results, seeds)