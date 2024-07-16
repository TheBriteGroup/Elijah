import os
import pandas as pd
import numpy as np

def load_vibration_data(directory):
    data_list = []
    for file in os.listdir(directory):
        if file.startswith('acc_'):
            file_path = os.path.join(directory, file)
            temp_data = pd.read_csv(file_path, sep=',', header=None)
            temp_data.columns = ['hour', 'minute', 'second', 'microsecond', 'horizontal_vibration', 'vertical_vibration']
            data_list.append(temp_data)
    data = pd.concat(data_list, ignore_index=True)
    return data

def load_temperature_data(directory):
    data_list = []
    for file in os.listdir(directory):
        if file.startswith('temp_'):
            file_path = os.path.join(directory, file)
            print(f"Loading temperature data from: {file_path}")
            temp_data = pd.read_csv(file_path, sep=',', header=None)
            print("Temperature data structure:")
            print(temp_data.head())
            temp_data.columns = ['hour', 'minute', 'second', 'decisecond', 'temperature']
            data_list.append(temp_data)
    data = pd.concat(data_list, ignore_index=True)
    return data

def calculate_rul(total_time, data):
    data['timestamp'] = pd.to_datetime(data['hour'].astype(str) + ':' + data['minute'].astype(str) + ':' + data['second'].astype(str), format='%H:%M:%S')
    data['timestamp'] = data['timestamp'] + pd.to_timedelta(data['microsecond'], unit='us')
    last_timestamp = data['timestamp'].iloc[-1]
    total_time = pd.to_timedelta(total_time, unit='s')
    data['rul'] = total_time - (last_timestamp - data['timestamp'])
    data['rul'] = data['rul'].dt.total_seconds()
    return data[['rul']]

def normalize_data(data):
    min_temp = data['temperature'].min()
    max_temp = data['temperature'].max()
    data['temperature'] = (data['temperature'] - min_temp) / (max_temp - min_temp)
    return data


def downsample_data(data, downsample_factor):
    return data.iloc[::downsample_factor]


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    total_sequences = len(data) - sequence_length
    for i in range(total_sequences):
        if i % 1000 == 0:
            print(f"Creating sequence {i+1}/{total_sequences}")
        sequence = data.iloc[i:i+sequence_length][['horizontal_vibration', 'vertical_vibration']].values
        target = data.iloc[i+sequence_length]['rul']
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def save_preprocessed_data(preprocessed_data, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for bearing, data in preprocessed_data.items():
        sequences_path = os.path.join(output_directory, f'{bearing}_sequences.npy')
        targets_path = os.path.join(output_directory, f'{bearing}_targets.npy')
        np.save(sequences_path, data['sequences'])
        np.save(targets_path, data['targets'])

def load_preprocessed_data(directory_paths, output_directory):
    preprocessed_data = {}
    for bearing in directory_paths.keys():
        sequences_path = os.path.join(output_directory, f'{bearing}_sequences.npy')
        targets_path = os.path.join(output_directory, f'{bearing}_targets.npy')
        if os.path.exists(sequences_path) and os.path.exists(targets_path):
            sequences = np.load(sequences_path)
            targets = np.load(targets_path)
            preprocessed_data[bearing] = {
                'sequences': sequences,
                'targets': targets
            }
    return preprocessed_data

def main(directory_paths, total_times, sequence_length, downsample_factor, output_directory):
    preprocessed_data = load_preprocessed_data(directory_paths, output_directory)
    
    if not preprocessed_data:
        preprocessed_data = {}
        for bearing, directory_path in directory_paths.items():
            vibration_data = load_vibration_data(directory_path)
            vibration_data['rul'] = calculate_rul(total_times[bearing], vibration_data)
            data = vibration_data[['horizontal_vibration', 'vertical_vibration', 'rul']]
            data = downsample_data(data, downsample_factor)
            sequences, targets = create_sequences(data, sequence_length)
            preprocessed_data[bearing] = {
                'sequences': sequences,
                'targets': targets
            }
        
        save_preprocessed_data(preprocessed_data, output_directory)
    
    return preprocessed_data