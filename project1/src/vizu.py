# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
def cross_validation_visualization(lambds, metric_tr, metric_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, metric_tr, marker=".", color='b', label='train accuracy')
    plt.semilogx(lambds, metric_te, marker=".", color='r', label='test accuracy')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig("cross_validation")

def visualize_boxplot_cross_validation(k_data, pos = None):
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.boxplot(k_data, positions = pos, sym='+')


