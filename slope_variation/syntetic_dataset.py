import pandas as pd
from sklearn.datasets import make_classification
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("-r", "--rows", type=int, default=1000)
parser.add_argument("-c", "--cols", type=int, default=6167)
parser.add_argument("-n", "--name", default="slope_test_dataset")
args = parser.parse_args()

n_cols = args.cols
name = args.name

x, y = make_classification(n_samples=args.rows, n_features=n_cols)

dataset = pd.DataFrame(x)
dataset["y"] = y
os.makedirs("raw_data", exist_ok=True)
dataset.to_csv(f"raw_data/{name}.csv")
