import math
import time
from math import pow, log2

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Utility functions

def get_label(dataset):
    if 'adult' in dataset:
        return 'income'
    if 'cmc' in dataset:
        return 'contr_use'
    if 'compas' in dataset:
        return 'two_year_recid'
    if 'crime' in dataset:
        return 'ViolentCrimesClass'
    if 'drug' in dataset:
        return 'y'
    if 'german' in dataset:
        return 'credit'
    if 'law' in dataset:
        return 'gpa'
    if 'obesity' in dataset:
        return 'y'
    if 'park' in dataset:
        return 'score_cut'
    if 'trump' in dataset:
        return 'POLITICAL_VIEW'
    if 'wine' in dataset:
        return 'quality'
    if 'antivirus' in dataset:
        return 'y'
    if 'aps' in dataset:
        return 'class'
    if 'arcene' in dataset:
        return 'y'
    if 'dexter' in dataset:
        return 'y'

# FPTC functions

def get_training_time(data, label, model='logreg'):
    adult_x = data.drop(columns=label)
    adult_y = data[label]
    if model == 'logreg':
        classifier = LogisticRegression(
            penalty='l2', solver='sag', max_iter=10000)
    else:
        classifier = RandomForestClassifier(
            n_estimators=80, max_depth=2, random_state=0)
    scaler = StandardScaler()
    adult_x = scaler.fit_transform(adult_x)
    start = time.time()
    classifier.fit(adult_x, adult_y)
    stop = time.time()
    training_time = round(stop - start, 2)
    return training_time, classifier


def train_fptc_logreg(rows, cols, classes, slope, iters, intercept=''):
    n = rows
    v = cols
    m = classes
    if slope == '' and intercept == '':
        f_classifier = iters * n * v * pow(m, 2)
    elif intercept == '':
        f_classifier = (iters * n * v * pow(m, 2) * slope)
    else:
        f_classifier = (iters * n * v * pow(m, 2) * slope) + intercept 
    return f_classifier

def train_fptc_rf(rows, cols, classes, slope, trees, intercept=None):
    n = rows
    v = cols
    m = classes
    if slope == '' and intercept == None:
        f_classifier = trees * (m + 1) * n * v * log2(n)
    elif intercept == None:
        f_classifier = (trees * (m + 1) * n * v * log2(n)) * slope
    else:
        f_classifier = ((trees * (m + 1) * n * v * log2(n)) * slope) + intercept
    return f_classifier

def compute_rmse(fptc, training_time):
    error = math.sqrt(mean_squared_error(training_time, fptc))
    return error

def get_mape(tt, fptc):
    return mean_absolute_percentage_error(tt, fptc)*100