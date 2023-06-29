import pandas as pd
from sklearn.datasets import make_classification

n_cols = 10000

x, y = make_classification(n_samples=6167, n_features=n_cols)

half_inf = pd.DataFrame(x)
half_inf['y'] = y
half_inf.to_csv('raw_data/slope_test_dataset.csv')
#create_training_sets('raw_data/', 'data/very_less_cols/', 'very_less_columns.csv', 100, 1000)