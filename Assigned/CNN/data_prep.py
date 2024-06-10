# File: data_prep.py
# Date: 2024-05-27
# Author: Elijah Widener Ferreira
#
# Brief: This file contains the functions used to prepare the data for the model.

import numpy as np
import pandas as pd
import os
import random as rd


# ----------------------------------------------------------------------------------

def normalize_data(data_file):
    # Read the data file
    data = pd.read_csv(data_file, sep=' ', header=None)
    
    # Remove the last two columns
    data = data.iloc[:, 0:-2]
    
    # Name the columns
    """
    - ID: Engine ID
    - C: The number of cycles or time steps for each engine unit.
    - O1, O2, O3: Operational settings
    - 'S1' to 'S21': Sensor measurements for each cycle. These columns contain the values 
    recorded by various sensors monitoring the engine's performance and health.
    """
    data.columns = ['ID', 'Cycle', 'O1', 'O2', 'O3', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                    'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']
    
    engines = np.unique(data['ID'])  # list of unique engine IDs
    sensor_readings = list(data.columns[5:])  # list of sensor readings

    
    # Calculate RUL
    data_grouped = data.groupby(['ID'], sort=False)['Cycle'].max().reset_index().rename(columns={'Cycle': 'MaxCycleID'})
    data = pd.merge(data, data_grouped, how='inner', on='ID')
    data['RUL'] = data['MaxCycleID'] - data['Cycle']
    
    # Normalize the data
    '''
    Normalization ensures that each feature contributes proportionally to the learning process and prevents 
    features with larger magnitudes from dominating the learning algorithm.

    TODO: Look into how the CNN model handles data without normalization

    The paper states the data should be normalized following this equation:

        2(𝑚_ij^o - 𝑚_jo^min)
    𝑚^ = -----------------------  - 1
        (𝑚_jo^max - 𝑚_jo^min)

    Where m_ij^o is the original value of the jth sensor during flight cycle i, under operating condition o. 

    m_jo^min and m_jo^max are the minimum and maximum values of the jth sensor under operating condition o, respectively.

    m^ is the normalized value of the jth sensor during flight cycle i.
    '''
    min_values = data.min()
    max_values = data.max()
    
    for sensor in sensor_readings:
        mini = min_values.loc[sensor]
        maxi = max_values.loc[sensor]
        
        if maxi == mini:
            maxi = mini + 0.0001
        
        data[sensor] = ((2 * (data[sensor] - mini)) / (maxi - mini)) - 1
    

    
    # Drop unnecessary columns
    data = data.drop(columns=['O1', 'O2', 'O3', 'Cycle', 'MaxCycleID'])

    return data

def normalize_all_data():

    current_dir = os.getcwd()
    output_dir = 'Normalized Data'

    project_folder = "CNN"
    project_path = os.path.join(current_dir, project_folder)
    os.chdir(project_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all .txt files in the Data folder
    data_files = [file for file in os.listdir('./Data') if file.endswith('.txt') and (file.startswith('train') or file.startswith('test'))]
    
    for data_file in data_files:
        # Normalize the data
        normalized_data = normalize_data(os.path.join('./Data', data_file))
        
        # Save the normalized data to a new file
        output_file = os.path.join(output_dir, 'normalized_' + data_file)
        np.savetxt(output_file, normalized_data.values, fmt='%1.6f')
    
    print("Normalization completed. Normalized data saved to the 'Normalized Data' folder.")

def prepare_training_data(instance, window_size):
    """
    returns X and y, where X is a list of input samples and y is a list of target RULs.
    Preprocessed Training Data
+-----------+--------+--------+-----+--------+
| Engine ID | Sensor | Sensor | ... |  RUL   |
|           |   1    |   2    |     |        |
+-----------+--------+--------+-----+--------+
|     1     |  0.5   |  0.8   | ... |  100   |
|     1     |  0.6   |  0.7   | ... |   99   |
|     2     |  0.4   |  0.9   | ... |  150   |
|    ...    |  ...   |  ...   | ... |  ...   |
+-----------+--------+--------+-----+--------+

             |
             |  Creating Input Samples and Target RULs
             v

Input Samples                   Target RULs
+--------+--------+-----+-----+
| Sensor | Sensor | ... | RUL |
|   1    |   2    |     |     |
+--------+--------+-----+-----+
|  0.5   |  0.8   | ... | 100 |
|  0.6   |  0.7   | ... |  99 |
|  0.4   |  0.9   | ... | 150 |
|  ...   |  ...   | ... | ... |
+--------+--------+-----+-----+

             |
             |  Combining Input Samples and Target RULs
             v

Training Data Points
+-----------------------------------+
| (Input Sample, Target RUL)        |
+-----------------------------------+
| ([0.5, 0.8, ...], 100)            |
| ([0.6, 0.7, ...], 99)             |
| ([0.4, 0.9, ...], 150)            |
| ...                               |
+-----------------------------------+

             |
             |  Shuffling the Training Data
             v

Shuffled Training Data Points
+-----------------------------------+
| (Input Sample, Target RUL)        |
+-----------------------------------+
| ([0.4, 0.9, ...], 150)            |
| ([0.5, 0.8, ...], 100)            |
| ([0.6, 0.7, ...], 99)             |
| ...                               |
+-----------------------------------+

             |
             |  Separating Features and Labels
             v

Input Features (X)                 Target Labels (y)
+--------+--------+-----+          +-----+
| Sensor | Sensor | ... |          | RUL |
|   1    |   2    |     |          |     |
+--------+--------+-----+          +-----+
|  0.4   |  0.9   | ... |          | 150 |
|  0.5   |  0.8   | ... |          | 100 |
|  0.6   |  0.7   | ... |          |  99 |
|  ...   |  ...   | ... |          | ... |
+--------+--------+-----+          +-----+
    
    """
    data_targets = pd.read_csv(f"Assigned/CNN/Normalized Data/normalized_train_{instance}.txt", sep=" ", header=None)
    data_targets.columns = ["ID"] + [f"S{i}" for i in range(1, 22)] + ["RUL"]
    sensors_to_drop = ["S1", "S5", "S6", "S10", "S16", "S18", "S19"]
    data_targets = data_targets.drop(columns=sensors_to_drop)

    seed = 1
    all_IDs = np.unique(data_targets["ID"])

    target_ruls = []
    samples = []
    
    for engine_id in all_IDs:
        row_nums = list(data_targets[data_targets["ID"] == engine_id].index)
        last_index = row_nums[-1]  # last index where this condition holds
        first_index = row_nums[0]  # first index where this condition holds
        
        rownumber = first_index

        while (rownumber + window_size) <= last_index:
            input_list = [] #sensor measurements
            
            current_row = rownumber + window_size

            target_rul = min(data_targets["RUL"].iloc[current_row -1 ], 125)
            target_ruls.append(target_rul)

            
            for column in data_targets.columns:
                if column =="ID" or column == "RUL":
                    continue
                if rownumber < first_index:
                    input_column = list(data_targets[column].iloc[first_index: current_row])
                    length_missing = first_index - rownumber
                    zeros = [0] * length_missing
                    input_column = zeros + input_column
                else:
                    input_column = list(data_targets[column].iloc[rownumber: current_row])
                input_list.append(input_column)
            
            samples.append(input_list)
            rownumber += 1
            
    samples = np.array(samples)
    training_data = []

    # combine input sample with corresponding target RUL
    for i in range(0, len(samples), 1):
        sample = samples[i]

        RUL = target_ruls[i]
        training_data.append([sample, RUL])

    rd.seed(seed)
    rd.shuffle(training_data)

    X = []
    y = []

    # append samples to input list X and target vector y

    for features, label in training_data:
        X.append(features)

        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

def prepare_testing_data(instance, window_size, skipped_sensors):
    test_data = pd.read_csv(f"Assigned/CNN/Normalized Data/normalized_test_{instance}.txt", sep=" ", header=None)
    test_data.columns = ["ID"] + [f"S{i}" for i in range(1, 22)] + ["RUL"]
    test_data = test_data.drop(columns=skipped_sensors)

    test_data = prepare_data_model_endpoints(test_data, window_size)

    x_test = test_data[0] #samples
    y_test = test_data[1] #target_ruls

    return x_test, y_test

def prepare_data_model_endpoints(data, window_size):
    samples = []
    target_ruls = []

    for engine_id in data["ID"].unique():
        engine_data = data[data["ID"] == engine_id]
        
        if len(engine_data) >= window_size:
            sample = []
            for sensor in data.columns[2:-1]:
                sample.append(engine_data[sensor].values[-window_size:])
            samples.append(sample)
            
            target_rul = min(engine_data["RUL"].values[-1], 125)
            target_ruls.append(target_rul)
        else:
            # Zero padding for engines with less than window_size cycles
            sample = []
            for sensor in data.columns[2:-1]:
                padded_data = np.zeros(window_size)
                padded_data[-len(engine_data):] = engine_data[sensor].values
                sample.append(padded_data)
            samples.append(sample)
            
            target_rul = min(engine_data["RUL"].values[-1], 125)
            target_ruls.append(target_rul)

    samples = np.array(samples)
    target_ruls = np.array(target_ruls)

    return samples, target_ruls


# def main():
#     normalize_all_data()

# main()