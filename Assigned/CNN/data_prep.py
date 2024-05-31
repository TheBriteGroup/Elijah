# File: data_prep.py
# Date: 2024-05-27
# Author: Elijah Widener Ferreira
#
# Brief: This file contains the functions used to prepare the data for the model.

import numpy as np
import pandas as pd
import os


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



'''
Normalized data follows this form:
COL 1: Engine ID
COL 2-22: Normalized sensor readings
COL 23  : RUL

'''

normalize_all_data()