# File: CNN.py
# Date: 2024-05-31
# Author: Elijah Widener Ferreira
#
# Brief: This is the implementation of the CNN for the normalized data.

import TensorFlow as tf
import Keras as keras
import numpy as np


# Initialize the hyperparameters
N = 30
L = 5
K = 10
S = 10
S_prime = 3
nodes = 100
p = 0.5
M = 1000
R_early = 125
num_sensors = 21

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
def create_sequences(data, window_size):
    X = [] # Input sequences 
    # 3D array with shape (num_samples, window_size, num_sensors)

    y = [] # RUL values
    # 1D array with shape (num_samples)


    for engine_id in np.unique[data[:, 0]]:
        engine_data = data[data[:, 0] == engine_id] # data from all rows that match the engine_id

        for i in range(len(engine_data) - window_size):
            window = engine_data[i:i + window_size, 1:-1] # Exclude first and last columns

            X.append(window) # Sensor readings
            y.append(engine_data[i+window_size-1, -1]) # RUL values in the last column

    return np.array(X), np.array(y)




model = keras.Sequential([

    '''
    Input Layer: Sequences of sensor data.
    Convolutional Layers: For feature extraction.
    Flatten Layer: To convert the 2D outputs to 1D.
    Dense Layers: For learning complex patterns and outputting the final result
    '''

    # Layers
    keras.layers.Conv1D(filters=K, kernel_size=S, activation='tanh', input_shape=(N,num_sensors)),
    keras.layers.Conv1D(filters=K, kernel_size=S_prime),
    keras.layers.Flatten(),
    keras.layers.Dense(units=100, activation='tanh'),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

X_train, y_train = create_sequences(train_data, window_size=N)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

loss, mae = model.evaluate(X_test, y_test)


predictions = model.predict(X_new)





