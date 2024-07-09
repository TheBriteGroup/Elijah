# File: batteries_evaluate.py
# Date: 2024-06-08
# Author: Elijah Widener Ferreira
#
# Brief: This script loads a saved model and evaluates it on the test data using RUL metrics.

import numpy as np
from keras._tf_keras.keras.models import load_model
from Bat_CNN import MCDropout, CNN
import data_prep as dp
import sys
sys.path.append(r'C:\Users\elija\Desktop\BriteGroup\Assigned\CNN')
import RUL_metrics as rm

def create_sequences(file_path, window_size):
    X_train, y_train, X_test, y_test = dp.prepare_data(file_path, window_size)
    return X_test, y_test


def main():
    file_path = 'Assigned/CNN/Batteries/Input n Capacity.csv'
    window_size = 20

    # Load the saved model
    model = load_model('batteries_best.keras', custom_objects={'MCDropout': MCDropout})
    print("Model loaded successfully.")
    
    # Print information about the model
    print(model.summary())
    input("Press Enter to continue...")
    
    # Process the test data
    X_test, y_test = create_sequences(file_path, window_size)

    # Evaluate the model using RUL metrics
    number_of_runs = 20
    name = 'Batteries'
    rm.test_montecarlo_output(model, X_test, y_test, number_of_runs, name)


if __name__ == '__main__':
    main()