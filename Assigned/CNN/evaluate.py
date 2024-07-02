# File: evaluate.py
# Date: 2024-06-08
# Author: Elijah Widener Ferreira
#
# Brief: This script loads a saved model and evaluates it on the test data. It then plots the predictions against the actual RUL values.


import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.models import load_model
from CNN import MCDropout
import data_prep as dp
import RUL_metrics as rm 
import sys


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


def create_sequences(instance, window_size):
    skipped_sensors = ["S1", "S5", "S6", "S10", "S16", "S18", "S19"]

    data = dp.prepare_testing_data(instance, window_size, skipped_sensors)

    input_array = data[0]
    target_array = data[1]
    target_array = np.array(target_array)

    print(f'input_array type: {type(input_array)}, shape: {input_array.shape}')
    print(f'target_array type: {type(target_array)}, shape: {target_array.shape}')

    input("Press Enter to continue...")
    return input_array, target_array

def plot_predictions(model, instance, X_test, y_test):
    
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
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.title('Predicted vs. Actual RUL')
    plt.show()


# ----------------------------------------------------------------------------------

def main():
    # List of engine sets
    engine_sets = ['FD001']
    #, 'FD002', 'FD003', 'FD004']
    reliability_curves = []
    
    
    for engine_set in engine_sets:
        print(f"Visualizing model for engine set: {engine_set}")

        # Load and plot the training history
        history = np.load(f'cnn_figures/cnn_history_{engine_set}.npy', allow_pickle=True).item()
        plot_CNN_history_statistics(history)

        # Load the saved model
        model = load_model(f'cnn_model_{engine_set}.keras', custom_objects={'MCDropout': MCDropout})
        
        print("Model loaded successfully.")
        # Print information about the model
        print(model.summary())
        input("Press Enter to continue..."  )
        # Process the test data and plot predictions
        x_test, y_test = create_sequences(engine_set, window_size=30)

        plot_predictions(model, engine_set, x_test, y_test)  

        # Evaluate the model using metrics from the paper (copied)
        curve = rm.test_montecarlo_output(model,x_test, y_test, 20, [3], engine_set)
        reliability_curves.append(curve)

    #ideal_curve = list(np.arange(0, 1 + sys.float_info.epsilon, 0.01))  # ideal curve, where y = x

    #rm.plot_combined_reliability_diagram(ideal_curve, reliability_curves, engine_sets)
        

if __name__ == '__main__':
    main()


# There are a total of 100 predictions.
# The reliability score (under) is 0.011492820512820526
# The reliability score (over) is 0.03014282051282053
# The total reliability score is 0.04163564102564106
# The coverage at alpha = 0.5 is 0.52
# The mean width at 0.5 is 16.37
# The coverage at 0.alpha = 0.9 is 0.9
# The mean width at 0.9 is 37.98
# The coverage at 0.alpha = 0.95 is 0.9
# The mean width at 0.95 is 37.98
# The RMSE is 12.939067069714586
# The MAE is 9.913000106811523
# The mean variance is  140.43755004097548
# The mean std is  11.594259893950055


# FD002:
# There are a total of 259 predictions.
# The reliability score (under) is 0.05407634543279984
# The reliability score (over) is 0.004423835780290149
# The total reliability score is 0.05850018121308999
# The coverage at alpha = 0.5 is 0.4671814671814672
# The mean width at 0.5 is 15.32046332046332
# The coverage at 0.alpha = 0.9 is 0.8185328185328186
# The mean width at 0.9 is 36.13899613899614
# The coverage at 0.alpha = 0.95 is 0.8185328185328186
# The mean width at 0.95 is 36.13899613899614
# The RMSE is 14.177627664564074
# The MAE is 10.785521227420528
# The mean variance is  123.41707530658094
# The mean std is  10.90548156665958
# the true RUL is  110.0  and the mean RUL is  64.65


# FD003:
# The reliability score (under) is 0.026019722222222263
# The reliability score (over) is 0.008569722222222227
# The total reliability score is 0.03458944444444449
# The coverage at alpha = 0.5 is 0.47
# The mean width at 0.5 is 17.02
# The coverage at 0.alpha = 0.9 is 0.9
# The mean width at 0.9 is 39.84
# The coverage at 0.alpha = 0.95 is 0.9
# The mean width at 0.95 is 39.84
# The RMSE is 13.056943680201702
# The MAE is 10.385500071048737
# The mean variance is  155.01372496283398
# The mean std is  12.156652526048212
# the true RUL is  120.0  and the mean RUL is  113.9
# PS C:\Users\elija\Desktop\BriteGroup> 


# FD004:
# There are a total of 248 predictions.
# The reliability score (under) is 0.060917942631055946
# The reliability score (over) is 0.0036397168246042762
# The total reliability score is 0.06455765945566022
# The coverage at alpha = 0.5 is 0.4274193548387097
# The mean width at 0.5 is 16.092741935483872
# The coverage at 0.alpha = 0.9 is 0.8104838709677419
# The mean width at 0.9 is 38.850806451612904
# The coverage at 0.alpha = 0.95 is 0.8104838709677419
# The mean width at 0.95 is 38.850806451612904
# The RMSE is 15.970927834940067
# The MAE is 12.005443653752726
# The mean variance is  139.9042642643982
# The mean std is  11.594696465505427
# the true RUL is  75.0  and the mean RUL is  98.85
