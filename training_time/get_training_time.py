from argparse import ArgumentParser
import os
import pandas as pd
import utils

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-r', '--rounds', default=20)
    parser.add_argument('-d', '--data', default='data')
    parser.add_argument('-m', '--model', default='logreg', choices=['logreg', 'rf'])


    args = parser.parse_args()
    rounds = int(args.rounds)
    folder = args.data
    model = args.model

    for f in os.listdir(folder):
        data = pd.read_csv(os.path.join(folder, f))
        label = utils.get_label(f)
        if model == 'logreg':
            training_times = pd.DataFrame(columns=['Round', 'Training Time', 'Iterations'])
        else:
            training_times = pd.DataFrame(columns=['Round', 'Training Time', 'Trees'])
        for i in range(rounds):
            training_time, classifier = utils.get_training_time(data, label, model)
            if model == 'logreg':
                training_times = pd.concat([training_times, pd.DataFrame(
                    {'Round': [i], 'Training Time': [training_time], 'Iterations': [classifier.n_iter_[0]]}
                    )], ignore_index=True)
            else:
                training_times = pd.concat([training_times, pd.DataFrame(
                    {'Round': [i], 'Training Time': [training_time], 'Trees': [classifier.n_estimators]}
                    )], ignore_index=True)
        os.makedirs(f'training_time_{model}', exist_ok=True)
        training_times.to_csv(f'training_time_{model}/training_time_{f}', index=False)
        print(f'Finished {f}')
