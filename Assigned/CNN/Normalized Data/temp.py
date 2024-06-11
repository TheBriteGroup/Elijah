import pandas as pd
from sklearn.model_selection import train_test_split

# Specify the file path
file_path = r'Assigned/CNN/Normalized Data/normalized_train_FD001.txt'

# Read the dataset into a pandas DataFrame
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

# Splitting data into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and target for saving
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Specify output file paths
train_file_path = r'C:/Users/elija/Desktop/BriteGroup/Assigned/CNN/Normalized Data/normalized_train_FD001.txt'
test_file_path = r'C:/Users/elija/Desktop/BriteGroup/Assigned/CNN/Normalized Data/normalized_test_split_FD001.txt'

# Save the training and test sets to separate files
train_data.to_csv(train_file_path, sep=' ', header=False, index=False)
test_data.to_csv(test_file_path, sep=' ', header=False, index=False)

print("Training and testing data have been saved to separate files.")
