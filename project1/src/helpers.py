# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np




#def clean_uncomplete_features(x, threshold):
#    """This function removes features that have a number of non-defined values above a given threshold.
#       In this dataset, the meaningless values or the ones that cannot be computed are given the values -999.0"""
#    raise Exception('not implemented')

T_pos = lambda a,b: sum(1 for x,y in zip(a,b) if x == y and x == 1)
F_pos = lambda a,b: sum(1 for x,y in zip(a,b) if x != y and x == 1)
T_neg = lambda a,b: sum(1 for x,y in zip(a,b) if x == y and x == -1)
F_neg = lambda a,b: sum(1 for x,y in zip(a,b) if x != y and x == -1)


def accuracy(y,y_corr):
    return (T_pos(y,y_corr) + T_neg(y,y_corr))/len(y)
    
def precision(y,y_corr):
    Tp = T_pos(y,y_corr)
    Fp = F_pos(y,y_corr)
    if (Tp + Fp == 0):
        return 0
    return Tp/(Tp + Fp)

def recall(y,y_corr):
    Tp = T_pos(y,y_corr)
    Fn = F_neg(y,y_corr)
    return Tp / (Tp + Fn)
    
def f1(y,y_corr):
    prec = precision(y,y_corr)
    rec = recall(y,y_corr)
    if (prec + rec == 0):
        return 0
    return 2*prec*rec / (prec + rec)

dico = {accuracy:"accuracy",f1:"F1 score"}


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
