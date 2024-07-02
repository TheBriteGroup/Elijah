# File: main.py
# Date: 2024-06-03
# Author: Elijah Widener Ferreira
#
# Brief: This file demonstrates the full process of finding the best
# features, then training and testing the best fit model to impute missing data.

from feature_selection import find_features
from data_curation import split_data
import data_preprocessing as dp

# Load your DataFrame
df = pd.read_csv('your_data.csv')

dp.preprocess_data(df)

# Specify the column to simulate missing data
impute_col = 'column_to_impute'

# Find feature importances
importances = find_features(df, impute_col)


# Split the data and drop less important columns
train_df, test_df = split_data(df, impute_col, importances, threshold=0.01)