import numpy as np
import pandas as pd
import os

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):

    # Drop unnecessary columns
    data = data.drop(columns=['SampleId'])
        
    # Normalize the features
    features = ['Time Measured(Sec)', 'Voltage Measured(V)', 'Current Measured', 'Temperature Measured']
    data[features] = (data[features] - data[features].mean()) / data[features].std()
    
    return data

def create_sequences(data, window_size, target_col='Capacity(Ah)'):

    sequences = []
    targets = []
    
    for i in range(len(data) - window_size):
        sequence = data[i:i+window_size].drop(columns=[target_col]).values
        target = data.iloc[i+window_size][target_col]
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def split_data(data, train_ratio=0.8):

    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def prepare_data(file_path, window_size):

    data = load_data(file_path)
    data = preprocess_data(data)
    
    train_data, test_data = split_data(data)
    
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)
    
    return X_train, y_train, X_test, y_test