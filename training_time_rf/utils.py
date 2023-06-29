import math
import time
from math import pow

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
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
    if 'kickstarter' in dataset:
        return 'State'

# FPTC functions


def get_training_time(data, label):
    adult_x = data.drop(columns=label)
    adult_y = data[label]
    classifier = RandomForestClassifier(n_estimators=80)
    scaler = StandardScaler()
    adult_x = scaler.fit_transform(adult_x)
    start = time.time()
    classifier.fit(adult_x, adult_y)
    stop = time.time()
    training_time = round(stop - start, 2)
    return training_time, classifier


def train_fptc(data, label, slope, classifier):
    adult_x = data.drop(columns=label)
    adult_y = data[label]
    n = adult_x.shape[0]
    v = adult_x.shape[1]
    m = len(np.unique(adult_y))
    if slope == '':
        f_classifier = classifier.n_iter_[0] * n * v * pow(m, 2)
    else:
        f_classifier = classifier.n_iter_[0] * n * v * pow(m, 2) * slope
    return f_classifier


def compute_rmse(fptc, training_time):
    error = math.sqrt(mean_squared_error(fptc, training_time))
    return error
