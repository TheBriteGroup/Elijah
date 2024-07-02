# File: test_train_curation
# Date: 2024-06-03
# Author: Elijah Widener Ferreira
#
# Brief: This file contains the functions to curate the data for the test and train sets



from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

def split_data(df, impute_col, feature_importances, threshold=0.01):
    """
    Drop columns based on feature importances and split a DataFrame into a training and testing set.
    
    Args:
        df: A pandas DataFrame
        impute_col: A column to simulate missing data
        feature_importances: A list or array of feature importances obtained from the Random Forest classifier
        threshold: The importance threshold below which columns will be dropped (default: 0.01)
        
    Returns:
        train_df: A pandas DataFrame (80% of the input DataFrame)
        test_df: A pandas DataFrame (20% of the input DataFrame)
    """
    
    # Drop columns with low feature importance
    important_features = [feature for feature, importance in zip(df.columns, feature_importances) if importance > threshold]
    df = df[important_features]
    
    # Randomly remove values to simulate missing data
    df[impute_col] = df[impute_col].sample(frac=0.8, random_state=42).reset_index(drop=True)

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

# manual implementation, no dropping cols
# def split_data(df, impute_col):
#     """
#     Split a DataFrame into a training and testing set. after
#     
#     Args:
#         df: A pandas DataFrame
#         impute_col : A column to simulate missing data
#         
#     Returns:
#         train_df: A pandas DataFrame (80% of the input DataFrame)
#         test_df: A pandas DataFrame (20% of the input DataFrame)
#     """
# 
#     # Randomly remove values to simulate missing data
#     df[impute_col] = df[impute_col].sample(frac=0.8, random_state=42).reset_index(drop=True)
# 
#     # Split data
#     train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#     return train_df, test_df
# 