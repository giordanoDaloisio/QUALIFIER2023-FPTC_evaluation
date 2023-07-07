import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import re
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time
from math import pow, log2
from scipy.stats import linregress
import numpy as np

# Dataset creation functions

def create_training_sets(filename, start, step, dataset=None):
    if dataset is None:
        dataset = pd.read_csv(filename, index_col=0)
    number_rows = dataset.shape[0]
    sets = []
    for i in range(start, number_rows + 1, step):
        df = dataset.sample(n=i)
        sets.append(df)
    return sets

def open_training_sets(directory):
    df_dictionary = {}
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    onlyfiles.sort(key=lambda f: int(re.sub('\D', '', f)))
    for file in onlyfiles:
        df = pd.read_csv(directory + '/' + file)
        df_dictionary[file] = df

    return df_dictionary

# Additional info functions

def get_imbalance(label_col):
    vals = label_col.value_counts()
    disp = vals.iloc[0]/vals.sum()
    for v in vals.iloc[1:]:
        disp = disp - (v/vals.sum())
    return abs(disp)

def get_binary_continous_cols(dataset):
    binary_cols = 0
    for col in range(len(dataset.columns)):
        if len(dataset.iloc[:, col].unique()) == 2:
            binary_cols += 1
    bin_cols = binary_cols/len(dataset.columns)
    cont_cols = (len(dataset.columns) - binary_cols) / len(dataset.columns)
    return bin_cols, cont_cols

def get_sparsity(dataset):
    return dataset[dataset == 0].count().sum() / (dataset.shape[0] * dataset.shape[1])

# FPTC functions

def get_slope(df_names, label, model='logreg'):
    data = _compute_fptc(df_names, label, model)
    slope, intercept, r_value, p_value, std_err = linregress(data['FPTC'], data['Training time'])
    training_time = data['Training time'].mean()
    fptc = data['FPTC'].mean()
    return slope, intercept, training_time, fptc, data['Training time'], data['FPTC']

def _compute_fptc(df_names, label, model='logreg'):
    fs_classifier = []
    training_times = []

    for data in df_names:
        adult_x = data.drop(columns=label)
        adult_y = data[label]
        if model == 'logreg':
            classifier = LogisticRegression(penalty='l2', solver='sag', max_iter=10000)
        else:
            classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        start = time.time()
        classifier.fit(adult_x, adult_y)
        stop = time.time()
        training_time = round(stop - start, 2)
        training_times.append(training_time)
        n = adult_x.shape[0]
        v = adult_x.shape[1]
        m = len(np.unique(adult_y))
        if model == 'logreg':
            f_classifier = _train_fptc_logreg(n, v, m, classifier.n_iter_[0])
        else:
            f_classifier = _train_fptc_rf(n, v, m, classifier.n_estimators)
        fs_classifier.append(round(f_classifier, 2))
    return pd.DataFrame({'Training time': training_times, 'FPTC': fs_classifier})

def _train_fptc_logreg(rows, cols, classes, iters, intercept=''):
    n = rows
    v = cols
    m = classes
    f_classifier = iters * n * v * pow(m, 2)
    return f_classifier

def _train_fptc_rf(rows, cols, classes, trees, intercept=None):
    n = rows
    v = cols
    m = classes
    f_classifier = trees * (m + 1) * n * v * log2(n)
    return f_classifier

