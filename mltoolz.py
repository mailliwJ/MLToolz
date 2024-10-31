# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, chi2_contingency

# ----------------------------------------------------------------------------------------------------------------

def describe_and_suggest(df, cat_threshold=10, cont_threshold=10.0, count=False, transpose=False):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Input must be a pandas DataFrame, but received {type(df).__name__}.')
    
    print(f'{type(df)}')
    print(f"RangeIndex: {len(df)} entries, 0 to {len(df)-1}")
    print(f"Data columns (total {df.shape[1]} columns)")
    dtypes_summary = df.dtypes.value_counts()
    dtypes_string = ', '.join([f'{dtype.name}({count})' for dtype, count in dtypes_summary.items()])
    print(f"dtypes: {dtypes_string}")
    mem_usage = df.memory_usage(deep=True).sum() / 1024
    print(f"memory usage: {mem_usage:.1f} KB")
    print()
    total_missing_percentage = (df.isna().sum().sum())/len(df) *100
    print(f'Total Percentage of Null Values: {total_missing_percentage:.2f}%')
    
    if not isinstance(cat_threshold, int):
        raise TypeError(f'cat_threshold must be an int, but received {type(cat_threshold).__name__}.')
    if not isinstance(cont_threshold, float):
        raise TypeError(f'cont_threshold must be a float, but received {type(cont_threshold).__name__}.')

    num_rows = len(df)
    if num_rows == 0:
        raise ValueError('The DataFrame is empty.')

    data_type = df.dtypes
    null_count = df.notna().sum()
    missings = df.isna().sum()
    missings_perc = round(df.isna().sum() / num_rows * 100, 2)
    unique_values = df.nunique()
    cardinality = round(unique_values / num_rows * 100, 2)

    df_summary = pd.DataFrame({
        'Data Type': data_type,
        'Not-Null': null_count,
        'Missing': missings,
        'Missing (%)': missings_perc,
        'Unique': unique_values,
        'Cardinality (%)': cardinality
    })

    suggested_types = []

    for col in df.columns:
        card = unique_values[col]
        percent_card = card / num_rows * 100

        if card == 2:
            suggested_type = 'Binary'
        elif df[col].dtype == 'object':
            suggested_type ='Categorical'
        elif card < cat_threshold:
            suggested_type = 'Categorical'
        else:
            suggested_type = 'Numerical Continuous' if percent_card >= cont_threshold else 'Numerical Discrete'
        
        suggested_types.append(suggested_type)
    
    df_summary['Suggested Type'] = suggested_types

    if transpose:
        return df_summary.T
    return df_summary

# -----------------------------------------------------------------------------------------------------------------------------------

def select_num_features(data, target_col, target_type='num', corr_threshold=0.5, pvalue=0.05, cardinality=20):
    """
    Identifies numeric columns in a DataFrame that are significantly related to the 'target_col' based on
    correlation for numeric targets or Chi-square for categorical targets. 

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data.
    target_col : str
        Target column to correlate with other numeric columns.
    target_type : {'num', 'cat'}
        Type of target column. 'num' for numeric, 'cat' for categorical.
    corr_threshold : float, optional
        Correlation threshold for numeric targets (absolute value between 0 and 1).
    pvalue : float, optional
        Significance level for filtering statistically significant features (default 0.05).
    cardinality : int, optional
        Minimum unique values required for a numeric feature to be considered continuous.

    Returns
    -------
    features_num : list
        A list of numeric column names that are significantly associated with 'target_col' 
        based on the selected method.
    """
    
    # Validate the DataFrame
    if not isinstance(data, pd.DataFrame):
        print('The "data" parameter must be a pandas DataFrame.')
        return None
    
    # Validate target_col exists in the DataFrame
    if target_col not in data.columns:
        print(f'The column "{target_col}" is not present in the DataFrame.')
        return None

    # Validate target_type
    if target_type not in ('num', 'cat'):
        print('The "target_type" parameter must be either "num" or "cat".')
        return None
    
    # Additional check for pvalue when target_type is 'cat'
    if target_type == 'cat' and pvalue is None:
        print('For target_type "cat", "pvalue" must have a specified value.')
        return None

    # Initialize the list to store selected features
    features_num = []

    # Select numeric columns excluding the target column
    numeric_cols = data.select_dtypes(include=[int, float]).columns.difference([target_col])
    
    # If target is numeric, use Pearson correlation
    if target_type == 'num':
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            print(f'For target_type "num", "{target_col}" must be numeric.')
            return None
        
        # Calculate Pearson correlation and filter by threshold and cardinality
        for col in numeric_cols:
            if data[col].nunique() >= cardinality:  # Only include features with sufficient cardinality
                corr, p_val = pearsonr(data[col], data[target_col])
                if abs(corr) >= corr_threshold and (pvalue is None or p_val <= pvalue):
                    features_num.append(col)
    
    # If target is categorical, use Chi-square test
    elif target_type == 'cat':
        if pd.api.types.is_numeric_dtype(data[target_col]):
            print(f'For target_type "cat", "{target_col}" should be categorical.')
            return None
        
        # Calculate Chi-square statistic for each numeric feature against the categorical target
        for col in numeric_cols:
            if data[col].nunique() >= cardinality:  # Only include features with sufficient cardinality
                contingency_table = pd.crosstab(data[col].apply(pd.cut, bins=5, labels=False), data[target_col])
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                if p_val <= pvalue:
                    features_num.append(col)

    # Return the list of selected numeric features
    return features_num

# ----------------------------------------------------------------------------------------------------------------
