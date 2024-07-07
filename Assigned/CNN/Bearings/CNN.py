# File: CNN.py
# Date: 2024-07-04
# Author: Elijah Widener Ferreira
#
# Brief: Contains the CNN for the Bearings Dataset


import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras._tf_keras.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import data_prep as dp

# Set the directory paths and total times for each bearing
directory_paths = {
    'Bearing1_4': 'C:/Users/elija/Downloads/archive/ieee-phm-2012-data-challenge-dataset-master/Test_set\Bearing1_4',
    #'Bearing2_4': 'C:/Users/elija/Downloads/archive/ieee-phm-2012-data-challenge-dataset-master/Test_set\Bearing2_4', # Change temp separator to ;
    #'Bearing3_3': 'C:/Users/elija/Downloads/archive/ieee-phm-2012-data-challenge-dataset-master/Test_set\Bearing3_3'
}

total_times = {
    'Bearing1_4': 11380,
    'Bearing2_4': 6110,
    'Bearing3_3': 3510
}

output_directory = 'Assigned\CNN\Bearings\Data'


# Set the sequence length
sequence_length = 30
num_sensors = 3
downsample_factor = 10
validation_split = 0.1
test_split = 0.2


# Preprocess the data using the main function from data_prep.py
preprocessed_data = dp.main(directory_paths, total_times, sequence_length, downsample_factor, output_directory)

# Prepare the data for training, validation, and testing
bearing_data = preprocessed_data['Bearing1_4']  # Select the desired bearing data
print(bearing_data)
sequences = bearing_data['sequences']
print(sequences)
targets = bearing_data['targets']
print(targets)

input("enter...")



# Split the data into training, validation, and testing sets
train_sequences, test_sequences, train_targets, test_targets = train_test_split(
    sequences, targets, test_size=test_split, random_state=42)
train_sequences, val_sequences, train_targets, val_targets = train_test_split(
    train_sequences, train_targets, test_size=validation_split / (1 - test_split), random_state=42)

# Create the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, num_sensors)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])

# Check for NaN values in sequences and targets
print(f"NaN values in sequences: {np.isnan(sequences).sum()}")
print(f"NaN values in targets: {np.isnan(targets).sum()}")

# Remove sequences with NaN values
sequences = sequences[~np.isnan(sequences).any(axis=(2, 2))]
targets = targets[~np.isnan(targets)]

# Check again for NaN values
print(f"NaN values in cleaned sequences: {np.isnan(sequences).sum()}")
print(f"NaN values in cleaned targets: {np.isnan(targets).sum()}")

input("Press enter to continue")

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("Sequences shape:", sequences.shape)

# Train the model
mcp_save = ModelCheckpoint(f"cnn_model_{"Bearing_1_4"}.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
reduce_lr_loss = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.0000001, verbose=1, min_delta=1e-4, mode="auto")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(train_sequences, train_targets, epochs=150, batch_size=32, callbacks=[mcp_save, reduce_lr_loss, early_stopping], validation_split=0.2, verbose=1)


# Evaluate the model on the testing set
loss = model.evaluate(test_sequences, test_targets)
print(f'Test loss: {loss}')