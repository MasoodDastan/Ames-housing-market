
import pandas as pd
import numpy as np

def outlier_censoring(df, upper_limit=95):
    """
    Caps the values of numeric features in a DataFrame to a specified upper limit percentile.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        upper_limit (float, optional): The percentile value to use as the upper limit. Defaults to 95.
    
    Returns:
        pd.DataFrame: The DataFrame with capped values for numeric features.
    """
    if upper_limit <= 0 or upper_limit >= 100:
        raise ValueError("The upper_limit parameter must be between 0 and 100.")
    
    data = df.copy()
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_features) == 0:
        raise ValueError("No numeric features found in the DataFrame.")
    
    for col in numeric_features:
        percentile_value = np.percentile(data[col], upper_limit)
        data.loc[data[col] > percentile_value, col] = percentile_value
    
    return data
