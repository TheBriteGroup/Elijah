import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from datawig import SimpleImputer, LinearImputer, DeepImputer, IterativeImputer, random_split


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
    SimpleImputer,
    LinearImputer,
    DeepImputer,
    IterativeImputer
]

# Evaluate each imputer
for Imputer in imputers:
    
    # Initialize the imputer
    imputer = Imputer(
        input_columns=['fill'],
        output_column=column_to_impute,
        output_path=f'imputer_model_{Imputer.__name__}'
    )
    
    # Fit the imputer on the training data
    imputer.fit(train_df=train_df)
    
    # Impute missing values in the test data
    imputed_test_df = imputer.predict(test_df)
    
    # overall measure of the imputation accuracy, lower values are better
    mse = mean_squared_error(imputed_test_df[column_to_impute], test_df[column_to_impute])
    
    # Linear correlation between the imputed values and the actual values
    # -1 to 1, where 1 indicates a strong positive correlation. We are looking for close to 1
    pearson_corr, _ = pearsonr(imputed_test_df[column_to_impute], test_df[column_to_impute])

    print(f"Evaluating {Imputer.__name__}:")
    print(f"Imputation MSE: {mse:.4f}")
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    print()