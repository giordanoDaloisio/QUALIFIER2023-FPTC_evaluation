import pandas as pd
from utils import create_training_sets, get_slope
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("-r", "--rounds", type=int)
parser.add_argument("-s", "--samples", type=int)
parser.add_argument("-n", "--dataset_name", default="slope_test_dataset")
args = parser.parse_args()
file_name = args.dataset_name
label_name = "y"
data = pd.read_csv(f"raw_data/{file_name}.csv", index_col=0)
for i in range(args.rounds):
    number_of_cols = args.samples
    slope_data = data.drop(columns=label_name).iloc[:, :number_of_cols]
    slope_data[label_name] = data[label_name]
    slopes = pd.DataFrame(columns=["Columns", "Slope", "Intercept", "Training Time Mean", "Raw FPTC Mean", "Training Time Full", "FPTC Full"])
    while slope_data.shape[1] < data.shape[1]:
        folder_name = f"{file_name}_data/{file_name}_{slope_data.shape[1]}"
        # os.makedirs(folder_name, exist_ok=True)
        sets = create_training_sets(
            "", folder_name + "/", file_name, 100, 1000, slope_data
        )
        slope, intercept, tt_mean, fptc_mean, tts, fptcs = get_slope(folder_name, sets, label_name)
        slopes = pd.concat([slopes, pd.DataFrame({"Columns": len(slope_data.columns), "Slope": slope, "Intercept": intercept, "Training Time Mean": tt_mean, "Raw FPTC Mean": fptc_mean, "Training Time Full": tts, "FPTC Full": fptcs }, index=[0])], ignore_index=True)
        os.makedirs('slopes_intercept', exist_ok=True)
        slopes.to_csv(f"slopes_intercept/slopes_{file_name}_{i}.csv")
        # Add new columns to data
        number_of_cols = slope_data.shape[1] + args.samples
        slope_data = data.drop(columns=label_name).iloc[:, :number_of_cols]
        slope_data[label_name] = data[label_name]
    print(f"Round {i} completed")
