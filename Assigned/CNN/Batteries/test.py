import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras._tf_keras.keras.layers import Conv1D, Conv2D, Flatten, Dense, Dropout, MaxPooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Assigned\CNN\Batteries\Input n Capacity.csv')

# Preprocess the data
data = data.drop(columns=['SampleId'])
data = data.dropna()

# Split the data into features and target
X = np.array(data.iloc[:,1:5].values)  # Exclude the 'Cycle' column from features
y = np.array(data.iloc[:,0].values)   # Use 'Cycle' as the target variable


# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data for CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=4, kernel_initializer='uniform', activation='tanh'))  # Adjust input_dim to match the number of features
    model.add(Dropout(0.15))
    model.add(Dense(7, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))  # Use a single output neuron with linear activation for regression
    
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

model = create_model()

print(model.summary())


# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1, callbacks=[early_stop])

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test MAE: {mae:.4f}')

# Make predictions on new data
X_new = np.random.rand(10, X_train.shape[1])
X_new = scaler.transform(X_new)
X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
predictions = model.predict(X_new)
print(f'Predictions: {predictions.flatten()}')

y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Cycle')
plt.ylabel('Predicted Cycle')
plt.title('Predicted vs. Actual Cycle')
plt.show()
plt.pause(3)
plt.close()