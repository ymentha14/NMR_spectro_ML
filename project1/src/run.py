#!/usr/bin/env python3

## Loading the Data:

import implementations as imp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd 


DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'


y, tX, ids = imp.load_csv_data(DATA_TRAIN_PATH)
_, tX_test, ids_test = imp.load_csv_data(DATA_TEST_PATH)

print(tX.shape)
print(y.shape)

## Parameters

KFOLD = 5
METRIC = imp.accuracy
LAMBDAS_OPTS = []
DEGREES_POLY = []
degrees = np.arange(1, 14)
lambdas = np.logspace(-5, 0, 15)

## Data exploration
print("\n ############################# DATA EXPLORATION ###############################")

data = pd.read_csv(DATA_TRAIN_PATH)
test_data = pd.read_csv(DATA_TEST_PATH)
dic = {'s':1,'b':-1}
data.Prediction = data.Prediction.map(dic)
test_data.Prediction = test_data.Prediction.map(dic)
data.head(10)

mask = data.isin([-999]).any(axis = 1)

#_The vast majoriy of our data has -999 values: we'd better handle it carefully_

std = np.nanstd(tX,axis = 0)
mean = np.nanmean(tX,axis = 0)

print('Train set size: {} samples x {} features'.format(pd.DataFrame(tX).shape[0], pd.DataFrame(tX).shape[1]))
print('Test set size: {} samples x {} features'.format(test_data.shape[0], pd.DataFrame(tX).shape[1]))


#### Class separation - Justification
print("\n ############################# CLASS SEPARATION ###############################")

col_names = list(data.columns)[2:]

data_0 = data[data['PRI_jet_num'] == 0]
data_1 = data[data['PRI_jet_num'] == 1]
data_2 = data[data['PRI_jet_num'] >= 2]


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist(data_0['DER_mass_transverse_met_lep'], bins=50, density=True, alpha=0.3, label='class 0', edgecolor='w', lw=.5)
ax0.hist(data_1['DER_mass_transverse_met_lep'], bins=50, density=True, alpha=0.3, label='class 1', edgecolor='w', lw=.5)
ax0.hist(data_2['DER_mass_transverse_met_lep'], bins=50, density=True, alpha=0.3, label='class 2 & 3', edgecolor='w', lw=.5)
ax0.legend()
ax0.set_xlabel('Value', fontsize=10)
ax0.set_ylabel('Proportion', fontsize=10)
ax0.set_title('DER_mass_transverse_met_lep', fontsize=12)

ax1.hist(data_0['DER_deltar_tau_lep'], bins=50, density=True, alpha=0.3, label='class 0', edgecolor='w', lw=.5)
ax1.hist(data_1['DER_deltar_tau_lep'], bins=50, density=True, alpha=0.3, label='class 1', edgecolor='w', lw=.5)
ax1.hist(data_2['DER_deltar_tau_lep'], bins=50, density=True, alpha=0.3, label='class 2 & 3', edgecolor='w', lw=.5)
ax1.legend()
ax1.set_xlabel('Value', fontsize=10)
ax1.set_ylabel('Proportion', fontsize=10)
ax1.set_title('DER_deltar_tau_lep',fontsize=12)

ax2.hist(data_0['DER_sum_pt'], bins=50, density=True, alpha=0.3, label='class 0', edgecolor='w', lw=.5)
ax2.hist(data_1['DER_sum_pt'], bins=50, density=True, alpha=0.3, label='class 1', edgecolor='w', lw=.5)
ax2.hist(data_2['DER_sum_pt'], bins=50, density=True, alpha=0.3, label='class 2 & 3', edgecolor='w', lw=.5)
ax2.legend()
ax2.set_xlabel('Value', fontsize=10)
ax2.set_ylabel('Proportion', fontsize=10)
ax2.set_title('DER_sum_pt', fontsize=12)


ax3.hist(data_0['PRI_met_sumet'], bins=50, density=True, alpha=0.3, label='class 0', edgecolor='w', lw=.5)
ax3.hist(data_1['PRI_met_sumet'], bins=50, density=True, alpha=0.3, label='class 1', edgecolor='w', lw=.5)
ax3.hist(data_2['PRI_met_sumet'], bins=50, density=True, alpha=0.3, label='class 2 & 3', edgecolor='w', lw=.5)
ax3.legend()
#ax3.ticklabel_format(useMathText=True)
#ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))
ax3.set_xlabel('Value', fontsize=10)
ax3.set_ylabel('Proportion', fontsize=10)
ax3.set_title('PRI_met_sumet', fontsize=12)

fig.subplots_adjust(hspace=0.3)

#Save figure
fig.savefig("../results/classes_comparison.png")



## Data Cleaning
print("\n ############################# DATA CLEANING ###############################")

#totrash before submit: we use pandas to know to which index PRI_jet_num does correspond.
np.where(data.columns.values == "PRI_jet_num")

data_trains = imp.split_categorical_data(tX,22,labels = y,split = True)
data_tests = imp.split_categorical_data(tX_test,22,split = True)

mean = 0
stdev = 0
clean_data_trains = []
clean_data_tests = []
for i,((x_train,y_train),(x_test,test_indx)) in enumerate(zip(data_trains,data_tests)):
    x_train,x_test = imp.clean_variance(x_train,x_test)
    
    x_train = imp.clean_value(x_train,-999,np.nan)
    x_test = imp.clean_value(x_test,-999,np.nan)
    
    x_train,mean,stdev =  imp.standardize_data(x_train)
    x_test,_,_ = imp.standardize_data(x_test, mean,stdev)
    
    """    
    mean,eigvecs,eigvals = imp.get_PCA(x_train)
    x_test = x_test - mean
    
    x_train = imp.reduce_PCA(eigvecs,x_train,10)
    x_test = imp.reduce_PCA(eigvecs,x_test,10)
    """
    
    x_train = imp.clean_value(x_train,np.nan,0,inplace = True)
    x_test = imp.clean_value(x_test,np.nan,0,inplace = True)
    
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    
    clean_data_trains.append((x_train,y_train))
    clean_data_tests.append((x_test,test_indx))

## Optimization
print("\n ############################OPTIMIZATION############################")

#### Ridge

# for ridge: for every models test different lambdas and degrees

D = len(degrees)
L = len(lambdas)
#averages of the f1/accuracy over the kfold for each cell
metrics_tot = []

#higher level: we keep the k_metrics_train,k_metrics_test and optcutoffs in a similar table
save_metrics = []

for idx_subset, (x_train, y_train) in enumerate(clean_data_trains):
    print('##### START SUBSET {} #####'.format(idx_subset))
    save_metric  = []
    
    for idx_deg, deg in enumerate(degrees):
        x_poly = imp.build_poly(x_train, deg)
        temp1 = []
        print("{d}/{D} row".format(d = idx_deg,D = D))
        for idx_lambda, lambda_ in enumerate(lambdas):
            ridge = lambda y, x: imp.ridge_regression(y,x,lambda_)
            
            start = datetime.datetime.now()

            k_metrics_train, k_metrics_test,_ = imp.k_fold_cv(y_train, x_poly, KFOLD, ridge,METRIC,verbose = False)
            
            end = datetime.datetime.now()
            remain = end - start
            remain = float(remain.total_seconds())* (L * D * (3 - idx_subset) - idx_deg * L - idx_lambda)/60
            print("{i}/{L} column".format(i = idx_lambda,L = L))
            print("remaining time {} min ".format(remain))
            temp1.append([k_metrics_train,k_metrics_test])
            
            # update table
        save_metric.append(temp1)
    save_metrics.append(save_metric)
    print('##### END SUBSET {} #####'.format(idx_subset))

for group in save_metrics:
    metrics_tot.append(np.array([[np.mean(lam[1]) for lam in deg]for deg in group]).T)
 

# save accuracies

with open('../backup/'+ imp.dico[METRIC] + '_metrics_tot.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(metrics_tot, f)

pickle.dump( save_metrics, open( "../backup/"+ imp.dico[METRIC] + "_save_metrics.pkl", "wb" ) )

save_metrics = pickle.load(open( "../backup/"+ imp.dico[METRIC] + "_save_metrics.pkl", "rb" ) )

# Version matplotlib

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
ax0, ax1, ax2 = axes.flatten()

d0 = ax0.imshow(metrics_tot[0], cmap='GnBu', aspect='auto')
ax0.set_yticklabels(np.round(lambdas, 5), rotation=60)
#ax0.set_xticklabels(degrees)
plt.xticks(degrees)
ax0.set_xlabel('degree', fontsize=12)
ax0.set_ylabel('lambda', fontsize=12)
ax0.set_title('Subset 0', fontsize=15)
fig.colorbar(d0, ax=ax0)

d1 = ax1.imshow(metrics_tot[1], cmap='GnBu', aspect='auto')
ax1.set_yticklabels(np.round(lambdas, 5), rotation=60)
#ax1.set_xticklabels(degrees)
#plt.xticks(degrees)
#plt.gca().set_xticks(degrees)
ax1.set_xlabel('degree', fontsize=12)
ax1.set_ylabel('')
ax1.set_title('Subset 1', fontsize=15)
fig.colorbar(d1, ax=ax1)

d2 = ax2.imshow(metrics_tot[2], cmap='GnBu', aspect='auto')
ax2.set_yticklabels(np.round(lambdas, 5), rotation=60)
#ax2.set_xticklabels(degrees)
#plt.xticks(degrees)
ax2.set_xlabel('degree', fontsize=12)
ax2.set_ylabel('')
ax2.set_title('Subset 2', fontsize=15)
fig.colorbar(d2, ax=ax2)

plt.subplots_adjust(wspace=0.3)

# y: ↓ (lambdas), x: → (degree)

#we save the 3 best indexes for the degree
best_degrees = []
for nb, met in enumerate(metrics_tot):
    print('SUBSET {}'.format(nb))
    ymax = np.asscalar(np.where(met == np.max(met))[0])
    xmax = np.asscalar(np.where(met == np.max(met))[1])
    best_degrees.append(xmax)
    
    print('Best degree for subset {}: {}'.format(nb, degrees[xmax]))
    DEGREES_POLY.append(degrees[xmax])
    
    print('Best lambda for subset {}: {}'.format(nb, lambdas[ymax]))
    LAMBDAS_OPTS.append(lambdas[ymax])
    print(imp.dico[METRIC] + ': {}'.format(met[ymax][xmax]))


## Visualization
print("\n ############################## VIZUALIZATION ###############################")


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16,16))
ax = axes.flatten()

for idx_subset,(deg,save_metric) in enumerate(zip(best_degrees,save_metrics)):
    accuracy_train = []
    accuracy_test = []
    best_column = np.array(save_metric)[deg,:]
    
    for idx_lambda,k_result in enumerate(best_column):
        k_accuracies_train, k_accuracies_test = k_result
        imp.visualize_boxplot_cross_validation2(k_accuracies_train,[idx_lambda], ax[idx_subset], idx_subset, True)
        imp.visualize_boxplot_cross_validation2(k_accuracies_train,[idx_lambda], ax[idx_subset+3], idx_subset, False)

        accuracy_train.append(np.mean(k_accuracies_train))
        accuracy_test.append(np.mean(k_accuracies_test))
    imp.cross_validation_visualization(lambdas, accuracy_train, accuracy_test, ax[idx_subset+6], idx_subset)
    

## Final Model
#_We now need to standardize the function so that they all take the same type of parameters as inputs_

#Kfold for the methods


#-----------------------------------------------------------------------
init_w = np.random.rand(clean_data_tests[0][0].shape[1])
maxiters = 100
gamma = 0.01

#method 1
meth1 = lambda  y, x: imp.ridge_regression(y,x,5.1794746792312125e-05)

if (len(clean_data_tests) > 1):
    #method 2
    init_w2 = np.random.rand(clean_data_tests[1][0].shape[1])
    #reg_log_reg = lambda y,x : imp.reg_logistic_regression(y, x, lambda_, init_w2, maxiters, gamma)
    #meth2 = lambda  y, x: imp.logistic_regression(y,x,init_w2,5,gamma)
    meth2 = lambda y,x : imp.ridge_regression(y,x,0.0013894954943731374)

    #method 3
    init_w3 = np.random.rand(clean_data_tests[2][0].shape[1])
    lambda_ = 0.1
    meth3 = lambda y, x: imp.ridge_regression(y,x,0.00138944954943731374)
    #log_reg3 = lambda  y, x: imp.logistic_regression(y,x,init_w3,5,gamma)

methods = [meth1,meth2,meth3]
#-----------------------------------------------------------------------

metrics_group_means = []
metrics_group_stds = []
cutoffs_group = []
degre_polys = [12,12,13]
for round_,((x_train,y_train),meth,deg) in enumerate(zip(clean_data_trains,methods,degre_polys)):
    print("**********treating the {i}th group of data:**************".format(i = round_+1))
    x_poly = imp.build_poly(x_train, deg)
    metrics, metric_stds,opt_cutoffs = imp.k_fold_cv(y_train,x_poly,KFOLD,meth,metric = METRIC)
    metrics_group_means.append(metrics)
    metrics_group_stds.append(metric_stds)
    cutoffs_group.append(opt_cutoffs)
print("\n done! Obtained :" + imp.dico[METRIC],[np.mean(i) for i in metrics_group_means])
print("ideal cutoffs for these groups-methods pairs:",[np.mean(i) for i in cutoffs_group])

## Submission
print("\n ################################# SUBMISSION ################################")

#_We now interpolate the data thanks to the model defined 2 cells higher..._

y_submit = np.zeros(len(tX_test))
assert(len(tX_test) == sum([i[0].shape[0] for i in clean_data_tests]))
for (x_test,y_indx),(x_train,y_train),meth,deg in zip(clean_data_tests,clean_data_trains,methods,degre_polys):
    x_poly_tr = imp.build_poly(x_train, deg)
    x_poly_te = imp.build_poly(x_test,deg)
    w_fin,loss = meth(y_train,x_poly_tr)
    y_test = x_poly_te @ w_fin
    cut = imp.get_best_cutoff(x_poly_tr,y_train,w_fin,METRIC)
    y_test = [-1 if i < 0 else 1.0 for i in y_test]
    y_submit[y_indx] = y_test

#_And finally save the results to csv._

imp.create_csv_submission(ids_test,y_submit,"../darth_mole.csv")

print("The program terminated successfully. We hope you enjoyed the run.")
exit(0)
