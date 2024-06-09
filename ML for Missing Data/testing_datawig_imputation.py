import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer as SklearnSimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from datawig import SimpleImputer as DataWigSimpleImputer
from datawig import Imputer as DataWigDeepImputer

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Specify the column to impute
column_to_impute = 'target_column'

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# List of imputers to compare
imputers = [
    ('SklearnSimpleImputer', SklearnSimpleImputer(strategy='mean')),
    ('KNNImputer', KNNImputer(n_neighbors=5)),
    ('IterativeImputer', IterativeImputer(max_iter=10, random_state=0)),
]
   



'''
Data Encoding: 
DataWig automatically encodes categorical variables using techniques like one-hot encoding or label encoding. 
This step converts categorical data into numerical representations suitable for machine learning algorithms.

Feature Selection: 
DataWig allows you to specify the input columns (features) that will be used for imputation. 
It can handle both numerical and categorical features.

Model Training: 
Based on the chosen imputer (e.g., SimpleImputer, LinearImputer, DeepImputer), 
DataWig trains a machine learning model using the input features to predict the missing values in the target column. 
The model learns patterns and relationships from the available data.

Imputation: 
Once the model is trained, DataWig uses it to impute the missing values in the target column. 
It makes predictions for the missing values based on the learned patterns and relationships from the input features.

Evaluation: 
DataWig provides methods to evaluate the imputation performance, such as calculating the mean squared error (MSE) between the imputed values and the actual values (if available). 
This helps assess the quality of the imputation.
'''


# Load the dataset
df = pd.read_csv('')

# Specify the column to impute
column_to_impute = 'target_column'

# Split the data into train and test sets
train_df, test_df = random_split(df)

# List of imputers to compare
imputers = [
    ('SklearnSimpleImputer', SklearnSimpleImputer(strategy='mean')),
    ('KNNImputer', KNNImputer(n_neighbors=5)),
    ('IterativeImputer', IterativeImputer(max_iter=10, random_state=0)),
    ('DataWigSimpleImputer', DataWigSimpleImputer(input_columns=['fill'], output_column=column_to_impute)),
    ('DataWigDeepImputer', DataWigDeepImputer(input_columns=['fill'], output_column=column_to_impute))
]


# Evaluate each imputer
for name, imputer in imputers:
    
    if 'DataWig' in name:
        # For DataWig imputers
        imputer.fit(train_df=train_df_datawig)
        imputed_test_df = imputer.predict(test_df_datawig).to_pandas_dataframe()
    
    else:
    # For scikit-learn imputers
        imputer.fit(train_df.drop(columns=[column_to_impute]))
        imputed_values = imputer.transform(test_df.drop(columns=[column_to_impute]))
        imputed_test_df = test_df.copy()
        imputed_test_df[column_to_impute] = imputed_values[:, test_df.columns.get_loc(column_to_impute)]


    # overall measure of the imputation accuracy, lower values are better
    mse = mean_squared_error(test_df[column_to_impute].dropna(), imputed_test_df[column_to_impute].loc[test_df[column_to_impute].notna()])


    # Linear correlation between the imputed values and the actual values
    # -1 to 1, where 1 indicates a strong positive correlation. We are looking for close to 1
    pearson_corr, _ = pearsonr(test_df[column_to_impute].dropna(), imputed_test_df[column_to_impute].loc[test_df[column_to_impute].notna()])

    print(f"Evaluating {name}:")
    print(f"Imputation MSE: {mse:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print()