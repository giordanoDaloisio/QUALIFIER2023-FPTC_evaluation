import pandas as pd
import utils
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-r')
parser.add_argument('-d')

args = parser.parse_args()
rounds = int(args.r)
folder = args.d

slopes = pd.read_csv('median_slopes_dell.csv')

for f in os.listdir(folder):
    data = pd.read_csv(os.path.join(folder, f))
    label = utils.get_label(f)
    errors = pd.DataFrame()
    for i in range(rounds):
        training_time, classifier = utils.get_training_time(data, label)
        for i, row in slopes.iterrows():
            fptc = utils.train_fptc(data, label, row['Slope'], classifier)
            error = utils.compute_rmse([fptc], [training_time])
            slope = row['Slope']
            errors = pd.concat([errors, pd.DataFrame({
                'Round': [i],
                'Slope': [slope],
                'RMSE': [error],
                'Slope Columns': [row['Columns']],
                'Dataset Columns': [data.shape[1]],
                'Training Time': [training_time],
                'FPTC': [fptc]})], ignore_index=True)
    errors.to_csv(os.path.join('errors', f), index=False)
    print(f + " completed")
