# File: data_preprocessing.py
# Date: 2024-06-03
# Author: Elijah Widener Ferreira
#
# Brief: This file removes unnecessary columns from the dataset and preprocesses the data for imputation.


import pandas as pd

def preprocess_data(df):


    # drop unnecessary columns
    columns_to_remove = ['SHIPMT_ID', 'EXPORT_CNTRY']
    df = df.drop(columns=columns_to_remove)

    # Convert categorical columns to numerical
    Y_N_map = {'Y': 1, 'P': 1, 'O': 1, 'N': 0,}
    df['TEMP_CNTL_YN'] = df['TEMP_CNTL_YN'].map(Y_N_map)
    df['EXPORT_YN'] = df['EXPORT_YN'].map(Y_N_map)
    df['HAZMAT'] = df['HAZMAT'].map(Y_N_map)
