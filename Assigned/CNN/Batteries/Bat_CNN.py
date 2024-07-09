import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras._tf_keras.keras.layers import Conv1D, Conv2D, Flatten, Dense, Dropout, MaxPooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import data_prep as dp
import tensorflow as tf


class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class CNN:
    def __init__(self, input_shape, output_shape, dropout_rate=0.5, num_mc_samples=100):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dropout_rate = dropout_rate
        self.num_mc_samples = num_mc_samples
        self.model = self.build_model()
        self.model_2d = self.build_2d_model()


    def build_model(self):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            MCDropout(self.dropout_rate),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MCDropout(self.dropout_rate),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            MCDropout(self.dropout_rate),
            Dense(self.output_shape, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def build_2d_model(self, K=64, S=3, S_prime=3):
        num_sensors = self.input_shape[1]
        N = self.input_shape[0]
        
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
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model

    def train(self, X_train, y_train, epochs=200, batch_size=32, validation_split=0.1):
        checkpoint = ModelCheckpoint('batteries_best.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode="min", verbose= 1, restore_best_weights=True)
        
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split, callbacks=[checkpoint, early_stopping])

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test MAE: {mae:.4f}')

    def predict(self, X):
        mc_predictions = []
        for _ in range(self.num_mc_samples):
            mc_predictions.append(self.model.predict(X))
        mc_predictions = np.array(mc_predictions)
        return np.mean(mc_predictions, axis=0), np.std(mc_predictions, axis=0)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path, custom_objects={'MCDropout': MCDropout})

def main():
    file_path = 'Assigned/CNN/Batteries/Input n Capacity.csv'
    window_size = 10
    X_train, y_train, X_test, y_test = dp.prepare_data(file_path, window_size)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    cnn = CNN(input_shape=(window_size, 5), output_shape=1)
    cnn.train(X_train, y_train)
    cnn.evaluate(X_test, y_test)
        
    cnn.load_model('batteries_best.keras')
    cnn.evaluate(X_test, y_test)
    
    X_new = np.random.rand(10, window_size, 5)
    mean, std = cnn.predict(X_new)
    print(f'Mean: {mean}')
    print(f'Std: {std}')


if __name__ == '__main__':
    main()


