# File: data_prep.py
# Date: 2024-05-27
# Author: Elijah Widener Ferreira
#
# Brief: This file contains the functions used to prepare the data for the model.

import numpy as np
import pandas as pd

# manually set instance number FD001-4
instance = "FD004"

print("Starting Data Prep")


# ----------------------------------------------------------------------------------
# reading the training data file

# ID, Cycle, O1, O2, O3, S1, S2, S3, S4...S21
data_train = pd.read_csv(r'Assigned\Dynamic predictive maintenance for multiple components using data-driven probabilistic RUL prognostics\Data\train_' + instance
 + '.txt', sep=' ', header=None)

print(data_train)
input("Press Enter to continue...")

# The last two columns only contain NaN: delete these
data_train = data_train.iloc[:, 0:-2] 

# Name the colums for debugging/visualization
"""
- ID: Engine ID
- C: The number of cycles or time steps for each engine unit.
- O1, O2, O3: Operational settings
- 'S1' to 'S21': Sensor measurements for each cycle. These columns contain the values 
recorded by various sensors monitoring the engine's performance and health.
"""
data_train.columns = ['ID', 'Cycle', 'O1', 'O2', 'O3', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9',
                             'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21']

engines = np.unique(data_train['ID'])  # list of unique engine IDs
sensor_readings = list(data_train.columns[5:])  # list of sensor readings


print("Training data loaded")
input("Press Enter to continue...")


# ID, MaxCycleID
data_train_grouped = data_train.groupby(['ID'], sort=False)['Cycle'].max().reset_index().rename(columns={'Cycle': 'MaxCycleID'})  

# ID, Cycle, O1, O2, O3, S1, S2, S3, S4...S21, MaxCycleID
include_RUL_train = pd.merge(data_train, data_train_grouped, how='inner', on='ID')

# ID, Cycle, O1, O2, O3, S1, S2, S3, S4...S21, MaxCycleID, RUL
include_RUL_train['RUL'] = include_RUL_train['MaxCycleID'] - include_RUL_train['Cycle']
normalized_data = include_RUL_train


print(normalized_data)
input("Press Enter to continue...")


# Normalize the data with respect to operating conditions
print("Normalizing the data")

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
min_train = normalized_data.min()
max_train = normalized_data.max()

for sensor in sensor_readings:
    print("Sensor is ", sensor)
    mini = min_train.loc[sensor]
    maxi = max_train.loc[sensor]

    if maxi == mini:
        maxi = mini + 0.0001

    for i in range(0, normalized_data.shape[0], 1):
        normalized_data.loc[i, sensor] = ((2 * (normalized_data.loc[i, sensor] - mini)) / (maxi - mini)) - 1
        # print("Normalized value is ", normalized_data.loc[i, sensor])
        # input("Press Enter to continue...")

print("Data normalization completed")

print("Drop unnecessary columns")
normalized_data = normalized_data.drop(columns=['O1', 'O2', 'O3', 'Cycle', 'MaxCycleID'])

print("Samples of normalized training data:")
print(normalized_data.head())

np.savetxt(r'.\normalized_train_' + instance + '.txt', normalized_data.values, fmt='%1.6f')
print("Normalized training data saved to file")

