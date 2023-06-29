import pandas as pd
from os import listdir
import os
from os.path import isfile, join
import re
from sklearn.linear_model import LogisticRegression
import time
from math import pow
from scipy.stats import linregress
import numpy as np

# Dataset creation functions

def create_training_sets(inpup_dir, output_dir, filename, start, step, dataset=None):
    if dataset is None:
        dataset = pd.read_csv(inpup_dir + filename, index_col=0)
    number_rows = dataset.shape[0]
    sets = []
    #os.makedirs(output_dir, exist_ok=True)
    for i in range(start, number_rows + 1, step):
        df = dataset.sample(n=i)
        sets.append(df)
        #df.to_csv(output_dir + filename[0:len(filename) - 4] + '_' + str(i) + '.csv', index=False)
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

def compute_fptc(path, df_names, label, slope=''):
    fs_classifier = []
    training_times = []
    iterations = []
    classes = []
    features = []
    attributes = []
    class_imbalance = []
    continuous_cols = []
    bin_cols = []
    maj_class = []
    min_class = []
    sparsity = []
    for data in df_names:
        #data = pd.read_csv(os.path.join(path, key))
        adult_x = data.drop(columns=label)
        adult_y = data[label]
        classifier = LogisticRegression(penalty='l2', solver='sag', max_iter=2000)
        start = time.time()
        classifier.fit(adult_x, adult_y)
        stop = time.time()
        training_time = round(stop - start, 2)
        training_times.append(training_time)
        iterations.append(classifier.n_iter_[0])
        n = adult_x.shape[0]
        v = adult_x.shape[1]
        m = len(np.unique(adult_y))
        classes.append(m)
        features.append(v)
        attributes.append(n)
        if slope == '':
          f_classifier = classifier.n_iter_[0] * n * v * pow(m, 2)
        else:
          f_classifier = classifier.n_iter_[0] * n * v * pow(m, 2) * slope
        fs_classifier.append(round(f_classifier, 2))

        # Other values
        binary_cols, cont_cols = get_binary_continous_cols(adult_x)
        continuous_cols.append(cont_cols)
        bin_cols.append(binary_cols)
        class_imbalance.append(get_imbalance(adult_y))
        class_values = adult_y.value_counts(normalize=True).sort_values(ascending=True)
        maj_class.append(class_values.iloc[-1])
        min_class.append(class_values.iloc[0])
        sparsity.append(get_sparsity(adult_x))
    return pd.DataFrame({'Training time': training_times, 'FPTC': fs_classifier, 'Iterations': iterations, 'Classes': classes, 'Slope': slope, 'Features': features, 'Instances Size': attributes, 'Imbalance': class_imbalance, 'Continous Columns': continuous_cols, 'Binary Columns': bin_cols, 'Majority Classes': maj_class, 'Minority Classes': min_class, 'Sparsity': sparsity})

def get_slope(path, df_names, label):
    data = compute_fptc(path, df_names, label)
    slope, intercept, r_value, p_value, std_err = linregress(data['FPTC'], data['Training time'])
    training_time = data['Training time'].mean()
    fptc = data['FPTC'].mean()
    return slope, intercept, training_time, fptc, data['Training time'], data['FPTC']

def train_fptc(path, df_names, label, slope, name):
    ris = compute_fptc(path, df_names, label, slope)
    ris.to_csv(name)
