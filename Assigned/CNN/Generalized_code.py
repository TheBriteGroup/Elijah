import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras._tf_keras.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import data_prep as dp
import evaluate as ev

class GeneralizedRULModel:
    def __init__(self, input_shape, output_shape=1, filters=64, kernel_size=3, 
                 dense_units=128, dropout_rate=0.5, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=self.filters*2, kernel_size=self.kernel_size, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(self.dense_units, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.output_shape, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                      loss='mse', metrics=['mae'])
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, dataset_name='Generalized'):
        callbacks = [
            ModelCheckpoint(f'best_model_{dataset_name}.keras', save_best_only=True, monitor='val_loss', mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

# Example Create Sequences
def create_sequences(instance, window_size):
    data = dp.prepare_training_data(instance, window_size)

    input_array = data[0]
    target_array = data[1]
    return input_array, target_array


def main():
    # Example usage
    datasets = ['FD001']
    
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset")
        
        # This function varies depending on the dataset (Example C-MAPSS)
        X_train, y_train = create_sequences(dataset_name, window_size=30)

        # Define model parameters
        input_shape = X_train.shape[1:]
        
        # Create and train model
        model = GeneralizedRULModel(input_shape)
        history = model.train(X_train, y_train, dataset_name=dataset_name)

        ev.plot_CNN_history_statistics(history)
        
        # How you create the test data varies depending on the dataset (Example C-MAPSS)
        X_test, y_test = ev.create_sequences(dataset_name, window_size=30)

        # Evaluate model
        ev.plot_predictions(model, X_test, y_test)
        

if __name__ == "__main__":
    main()