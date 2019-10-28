# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
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


