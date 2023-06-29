from argparse import ArgumentParser
import os
import pandas as pd
import utils

parser = ArgumentParser()
parser.add_argument('-r')
parser.add_argument('-d')

args = parser.parse_args()
rounds = int(args.r)
folder = args.d

for f in os.listdir(folder):
    data = pd.read_csv(os.path.join(folder, f))
    label = utils.get_label(f)
    training_times = pd.DataFrame(columns=['Round', 'Training Time', 'Trees'])
    for i in range(rounds):
        training_time, classifier = utils.get_training_time(data, label)
        training_times = pd.concat([training_times, pd.DataFrame(
            {'Round': [i], 'Training Time': [training_time], 'Trees': [len(classifier.estimators_)]}
            )], ignore_index=True)
    training_times.to_csv(f'training_time/training_time_{f}', index=False)
    print(f'Finished {f}')
