# File: feature_selection.py
# Date: 2024-06-03
# Author: Elijah Widener Ferreira
#
# Brief: This file performs feature selection using a Random Forest classifier.


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def find_features(df, target_column):
    # Separate features (X) and target variable (y)
    X = df.drop(target_column, axis=1)  
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest classifier
    rf.fit(X_train, y_train)

    # Get the feature importances
    importances = rf.feature_importances_

    # Print the feature importances
    print("Feature Importances:")
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance}")
              
    return importances


