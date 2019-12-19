import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, LeakyReLU

def plot_distribs(X,feat_indxes = None):
    """
    X: pandas dataframe to plot the values for
    feat_indxes: columns you want to plot
    """
    if feat_indxes is None:
        feat_indxes = np.random.permutation(X.shape[1])[:9]
    else:
        assert(feat_indxes.shape[0] == 9)
    fig, axes = plt.subplots(nrows=3, ncols=3)
    fig.set_size_inches(11,11)
    for ind,i in enumerate(feat_indxes):
        index = np.unravel_index(ind,(3,3))
        axes[index].ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        X.iloc[:,i].hist(ax = axes[index],bins = 80)
        axes[index].title.set_text('col {}'.format(str(i)))
        #data_X_df.iloc[:,i].plot.box(ax = axes[index])

def plot_feature_dist(data):
    ''' Display the distributions of 9 features. data should be X_tot'''
    feature_number = [12711, 1457, 4502, 1321, 10978, 9356, 2846, 9976, 11278]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for (ax, feature_nb) in zip(axes, feature_number):
        feature = data[:, feature_nb]
        
        sns.distplot(feature, ax=ax, kde=False, hist_kws=dict(edgecolor="w", linewidth=1))
        ax.axes.set_title("Feature {}".format(str(feature_nb), fontsize=20))
        ax.set_xlabel("", fontsize=17)
        ax.set_ylabel("",fontsize=17)
        ax.grid(False)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

def plot_PCA(n_comp, X_train):
    """
    displays the 'elbow' of the PCA, ie the screeplot"""
    pca = PCA(n_components = n_comp)
    pca.fit(X_train)
    plt.subplots(figsize=(10,8))
    plt.figure(1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

def plot_iqr(y, y_iqr):
    fig, ax = plt.subplots(figsize=(10,7))
    sns.set(font_scale=1) 
    plt.subplot(2,2,1)
    plt.title('Shielding distribution \n (w/o outlier removal)', fontsize=15)
    sns.distplot(y).set(xlim=(-5,65),ylim=(0,0.14))
    plt.subplot(2,2,2)
    plt.title('Shielding distribution \n (after applying IQR)', fontsize=15)
    sns.distplot(y_iqr).set(xlim=(-5,65),ylim=(0,0.14))
    plt.subplot(2,2,3)
    b1 = sns.boxplot(x=y, linewidth=1.5).set(xlabel='Shielding', xlim=(-5,65))
    plt.subplot(2,2,4)
    sns.boxplot(x=y_iqr, linewidth=1.5).set(xlabel='Shielding', xlim=(-5,65))

def plot_iqr_effect(noiqr_small_n, noiqr_large_n, iqr_small_n, iqr_large_n):
    iqr_data = noiqr_small_n + noiqr_large_n + iqr_small_n + iqr_large_n

    iqrs_labels = ['no'] * 8 + ['yes'] * 8 
    n_labels = [5000] * 4 + [20000] * 4 + [5000] * 4 + [20000] * 4

    my_df = pd.DataFrame({
        'mse': iqr_data,
        'iqr': iqrs_labels,
        'n': n_labels
    })

    iqr_vs_n, _ = plt.subplots(figsize=(10,8))
    sns.set(style="whitegrid")

    ax = sns.boxplot(data=my_df, x='iqr', y='mse', hue='n', linewidth=1.)
    ax.set_xticklabels(['No IQR', 'IQR'], fontsize=20)
    ax.set_xlabel('', fontsize=20)
    ax.set_ylabel('MAE', fontsize=20)

    plt.setp(ax.get_legend().get_texts(), fontsize='20')
    plt.setp(ax.get_legend().get_title(), fontsize='28')

    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .75))
    