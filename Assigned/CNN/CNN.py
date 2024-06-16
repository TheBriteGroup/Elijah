# File: CNN.py
# Date: 2024-05-31
# Author: Elijah Widener Ferreira
#
# Brief: This is the implementation of the CNN for the normalized data.


# There is a bug with tensorflow importing that is described here, which is why the import is done this way.
# https://stackoverflow.com/questions/71000250/import-tensorflow-keras-could-not-be-resolved-after-upgrading-to-tensorflow-2

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras._tf_keras.keras.layers import Conv2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import data_prep as dp


class MCDropout(Dropout):
    """The Monte Carlo dropout function from the regular keras dropout function."""

    def call(self, inputs):
        return super().call(inputs, training=True)

def get_target_rul(actual_rul, R_early):
    if actual_rul > R_early:
        return 125
    else:
        return actual_rul
    
def create_sequences(instance, window_size):
    data = dp.prepare_training_data(instance, window_size)

    input_array = data[0]
    target_array = data[1]
    return input_array, target_array


def create_model():
        
    N = 30
    L = 5
    K = 10
    S = (1,10)
    S_prime = (1,3)
    nodes = 100
    p = 0.5
    M = 1000
    R_early = 125
    num_sensors = 14 # 21 sensors in total, 14 are used in the model
    """
    Table 2: Considered hyperparameters of the CNN.
    +--------------------------------------------------+----------+
    | Hyperparameter                                   | Value    |
    +--------------------------------------------------+----------+
    | Hyperparameters — architecture                   |          |
    | Window-size N                                    | 30       |
    | Convolutional layers L                           | 5        |
    | Number of filters K                              | 10       |
    | Kernel size S                                    | 10       |
    | Kernel size S' last convolutional layer          | 3        |
    | Number of nodes fully connected layer            | 100      |
    | Monte Carlo dropout rate p                       | 0.5      |
    | Number of passes M                               | 1000     |
    | R_early                                          | 125      |
    +--------------------------------------------------+----------+
    | Hyperparameters — optimization                   |          |
    | Optimizer                                        | Adam     |
    | Number of epochs                                 | 250      |
    | Training-Validation split                        | 80%-20%  |
    | Initial learning rate                            | 0.001    |
    | Decrease learning rate when no improvement in    |          |
    | validation loss for ... epochs in a row          | 10       |
    | Decrease learning rate by                        | 1/2      |
    +--------------------------------------------------+----------+
    """
    # ----------------------------------------------------------------------------------

    """
    Window-size N: The number of past flight cycles included in each input sample.

    Convolutional layers L: The number of convolutional layers in the CNN.

    Number of filters K: The number of filters in each convolutional layer.

    Kernel size S: The size of the kernel (filter) in each convolutional layer.

    Kernel size S' last convolutional layer: The size of the kernel in the last convolutional layer.

    Number of nodes fully connected layer: The number of nodes in the fully connected layer.

    Monte Carlo dropout rate ρ: The dropout rate used for Monte Carlo dropout.

    Number of passes M: The number of forward passes through the network during Monte Carlo dropout.

    R_early: The threshold for the piece-wise linear RUL target function.

    Optimizer: The optimization algorithm used for training the CNN.

    Number of epochs: The number of training epochs.

    Training–Validation split: The ratio of data used for training and validation.

    Initial learning rate: The initial learning rate for the optimizer.

    Decrease learning rate: The number of epochs to wait before reducing the learning rate if there is no improvement in validation loss.

    Decrease learning rate by: The factor by which the learning rate is reduced.

    """


    '''
        Input Layer: Sequences of sensor data.
        Convolutional Layers: For feature extraction.
        Flatten Layer: To convert the 2D outputs to 1D.
        Dense Layers: For learning complex patterns and outputting the final result
    '''
    model = Sequential([
        Conv2D(filters=K, kernel_size=S, activation='tanh', input_shape=(num_sensors, N, 1), padding="same", kernel_initializer="glorot_normal"),
        MCDropout(rate=0.5),
        Conv2D(filters=K, kernel_size=S, activation='tanh', padding="same", kernel_initializer="glorot_normal"),
        MCDropout(rate=0.5),
        Conv2D(filters=K, kernel_size=S, activation='tanh', padding="same", kernel_initializer="glorot_normal"),
        MCDropout(rate=0.5),
        Conv2D(filters=K, kernel_size=S, activation='tanh', padding="same", kernel_initializer="glorot_normal"),
        MCDropout(rate=0.5),
        Conv2D(filters=1, kernel_size=S_prime, activation='tanh', padding="same", kernel_initializer="glorot_normal"),
        MCDropout(rate=0.5),
        Flatten(),
        MCDropout(rate=0.5),
        Dense(100, activation="tanh", kernel_initializer="glorot_normal"),
        MCDropout(rate=0.5),
        Dense(1, activation="relu", kernel_initializer="glorot_normal")
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def train_model(model, X_train, y_train, engine_set):
    # Define callbacks
    mcp_save = ModelCheckpoint("cnn_model_FD001.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
    reduce_lr_loss = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0000001, verbose=1, min_delta=1e-4, mode="auto")
    model.save(f'cnn_model_{engine_set}.keras')


    print("The dimensions of y are ", len(y_train))
    print("with type ", type(y_train))
    print(y_train[0])
    print("The dimensions of x are ", X_train.shape)
    print("the type of x is ", type(X_train))
    print("The length of X is ", len(X_train))
    

    # Train the model
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, callbacks=[mcp_save, reduce_lr_loss], validation_split=0.2, verbose=1)
    
    return history
# ----------------------------------------------------------------------------------

def main():

    # Load and preprocess the data for each dataset
    engine_sets = ['FD001']
    # , 'FD002', 'FD003', 'FD004']
    models = {}  # Dictionary to store trained models
    N = 30  # Window size
    num_sensors=14

    for engine_set in engine_sets:
        print(f"Evaluating model for engine set: {engine_set}")
        
        # Load and preprocess the data
        X_train, y_train = create_sequences(engine_set, window_size=N)
        
        # Create and train the model
        model = create_model()
        history = train_model(model, X_train, y_train, engine_set)
        
        # Create a folder for cnn_history if it doesn't exist
        if not os.path.exists('cnn_history'):
            os.makedirs('cnn_history')

        # Save the training history
        # This gets used by the evaluate.py script to plot the loss and MAE against the number of epochs.
        np.save(f'cnn_history/cnn_history_{engine_set}.npy', history.history)

    

if __name__ == '__main__':
    main()