import numpy as np 
import matplotlib.pyplot as plt

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