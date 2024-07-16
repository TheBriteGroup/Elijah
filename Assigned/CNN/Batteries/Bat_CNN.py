import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras._tf_keras.keras.layers import Conv1D, Conv2D, Flatten, Dense, Dropout, MaxPooling1D, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import data_prep as dp
import tensorflow as tf
import sys
sys.path.append('Assigned/CNN/')
from RUL_metrics import compute_reliability_score, compute_coverage, compute_mean_variance
import matplotlib.pyplot as plt


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
            BatchNormalization(),
            MCDropout(self.dropout_rate),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MCDropout(self.dropout_rate),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation='relu'),
            MCDropout(self.dropout_rate),
            Dense(1, activation='linear')
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
        checkpoint = ModelCheckpoint('models/batteries_best.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, mode="min", verbose= 1, restore_best_weights=True)
        
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split, callbacks=[checkpoint, early_stopping])
        return history
        
    def train_2d(self, X_train, y_train, epochs=200, batch_size=32, validation_split=0.1):
        checkpoint = ModelCheckpoint('models/batteries_best_2d.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode="min", verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='min', min_lr=1e-6)
        
        self.model_2d.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_split=validation_split, callbacks=[checkpoint, early_stopping, reduce_lr])    

    def evaluate(self, X_test, y_test):
        loss, mae = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.4f}')
        print(f'Test MAE: {mae:.4f}')

    def evaluate_2d(self, X_test, y_test):
        loss, mae = self.model_2d.evaluate(X_test, y_test)
        print(f'Test Loss (2D): {loss:.4f}')
        print(f'Test MAE (2D): {mae:.4f}')    

    def predict(self, X):
        mc_predictions = []
        for _ in range(self.num_mc_samples):
            mc_predictions.append(self.model.predict(X))
        mc_predictions = np.array(mc_predictions)
        return np.mean(mc_predictions, axis=0), np.std(mc_predictions, axis=0)
    
    def predict_2d(self, X):
        mc_predictions = []
        for _ in range(self.num_mc_samples):
            mc_predictions.append(self.model_2d.predict(X))
        mc_predictions = np.array(mc_predictions)
        return np.mean(mc_predictions, axis=0), np.std(mc_predictions, axis=0)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path, custom_objects={'MCDropout': MCDropout})

    def load_model_2d(self, file_path):
        self.model_2d = tf.keras.models.load_model(file_path, custom_objects={'MCDropout': MCDropout})    

def plot_CNN_history_statistics(history):
    """Plots the loss and MAE (Mean Absolute Error) against the number of epochs."""
    # Collect the loss and MAE (Mean Absolute Error).
    train_loss = history['loss']
    train_mean_abs_error = history['mae']
    val_loss = history['val_loss']
    val_mean_abs_error = history['val_mae']

    # Plot the loss and MAE against the number of epochs.
    xrange = range(1, len(train_loss) + 1)

    plt.figure("Training and Validation loss (MSE) vs. number of epochs:")
    plt.plot(xrange, train_loss, '-', label='Training loss (MSE)')
    plt.plot(xrange, val_loss, '-', label='Validation loss (MSE)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure("Training and Validation Mean Abs. Error (MAE) vs. number of epochs:")
    plt.plot(xrange, train_mean_abs_error, 'r-', label='Training (MAE)')
    #plt.plot(xrange, val_mean_abs_error, 'b-', label='Validation loss (MAE)')
    plt.grid()
    plt.legend()
    plt.show()

def main():
    file_path = 'Assigned/CNN/Batteries/Input n Capacity.csv'
    window_size = 10

    window_size = 10

    data = dp.load_data(file_path)
    data = dp.preprocess_data(data)

    X, y = dp.create_sequences(data, window_size)

    print(f"X_train shape: {X.shape}")
    print(f"X_test shape: {y.shape}")
    
    cnn = CNN(input_shape=(window_size, 5), output_shape=1)
    history = cnn.train(X, y)
    np.save(f'Assigned\cnn_figures\cnn_history/cnn_history_batteries.npy', history.history)
    plot_CNN_history_statistics(history.history)

    cnn.evaluate(X, y)

    y_pred = cnn.model.predict(X)

    true_RULs = {i: y[i] for i in range(len(y))}
    RUL_distributions = {i: [y_pred[i]] for i in range(len(y_pred))}

    RS_total, RS_under, RS_over, curve = compute_reliability_score(true_RULs, RUL_distributions, "blablabla")
    coverage = compute_coverage(true_RULs, RUL_distributions, alpha=0.9)
    mean_variance = compute_mean_variance(RUL_distributions, number_of_runs=1)[0]

    print(f"Total Reliability Score: {RS_total:.4f}")
    print(f"Mean Variance: {mean_variance}")
    
    # Prepare 2D data
    # X_train_2d = X_train.reshape((X_train.shape[0], X_train.shape[2], X_train.shape[1], 1))
    # X_test_2d = X_test.reshape((X_test.shape[0], X_test.shape[2], X_test.shape[1], 1))
    
    # cnn.train_2d(X_train_2d, y_train)
    # cnn.evaluate_2d(X_test_2d, y_test)
        
    # cnn.load_model('models/batteries_best.keras')
    # cnn.evaluate(X_test, y_test)
    
    # # cnn.load_model_2d('models/batteries_best_2d.keras')
    # # cnn.evaluate_2d(X_test_2d, y_test)
    
    # X_new = np.random.rand(10, window_size, 5)
    # mean, std = cnn.predict(X_new)
    # print(f'Mean: {mean}')
    # print(f'Std: {std}')
    
    # X_new_2d = X_new.reshape((X_new.shape[0], X_new.shape[2], X_new.shape[1], 1))
    # mean_2d, std_2d = cnn.predict_2d(X_new_2d)
    # print(f'Mean (2D): {mean_2d}')
    # print(f'Std (2D): {std_2d}')


if __name__ == '__main__':
    main()