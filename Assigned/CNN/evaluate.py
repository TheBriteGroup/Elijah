# File: evaluate.py
# Date: 2024-06-08
# Author: Elijah Widener Ferreira
#
# Brief: This script loads a saved model and evaluates it on the test data. It then plots the predictions against the actual RUL values.


import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model
from CNN import create_sequences
from CNN import MCDropout

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
    plt.plot(xrange, val_mean_abs_error, 'b-', label='Validation loss (MAE)')
    plt.grid()
    plt.legend()
    plt.show()


def plot_predictions(model, test_data, N):
    # Load and preprocess the test data
    X_test, y_test = create_sequences(test_data, window_size=N)
    
    # Evaluate the model on the test data
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test MAE: {mae:.4f}')
    
    # Make predictions on new data
    """
    Returns:
    Scalar test loss (if the model has a single output and no metrics) or list of scalars 
    (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will
    give you the display labels for the scalar outputs.
    """
    predictions = model.predict(X_test)
    
    # Visualize the predictions
    plt.figure("Predicted RUL vs. Actual RUL")
    plt.scatter(y_test, predictions, c='blue', label='Predictions')
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label='Ideal Line')
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.legend()
    plt.show()


# ----------------------------------------------------------------------------------

def main():
    # List of engine sets
    engine_sets = ['FD001']
    #, 'FD002', 'FD003', 'FD004']
    
    for engine_set in engine_sets:
        print(f"Visualizing model for engine set: {engine_set}")

        # Load and plot the training history
        history = np.load(f'cnn_history/cnn_history_{engine_set}.npy', allow_pickle=True).item()
        plot_CNN_history_statistics(history)

        # Load the saved model
        model = load_model(f'cnn_model_{engine_set}.keras', custom_objects={'MCDropout': MCDropout})
        
        # Load the test data
        test_data = np.loadtxt(f'Assigned/CNN/Normalized Data/normalized_test_{engine_set}.txt')
        
        # Process the test data and plot predictions
        plot_predictions(model, test_data, N=30)  
    

if __name__ == '__main__':
    main()