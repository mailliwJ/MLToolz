# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, get_scorer                           
from sklearn.base import is_classifier, is_regressor

# ----------------------------------------------------------------------------------------------------------------

def describe_and_suggest(df, cat_threshold=10, cont_threshold=10.0, count=False, transpose=False):
    """
    Provides an overview of a DataFrame's structure and suggests appropriate data types for each column 
    based on cardinality thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be analyzed.
    cat_threshold : int, optional
        Maximum number of unique values for a column to be considered categorical (default is 10).
    cont_threshold : float, optional
        Minimum cardinality percentage for a numeric column to be considered continuous (default is 10.0).
    count : bool, optional
        If True, displays count information (currently not used in this version).
    transpose : bool, optional
        If True, returns the summary DataFrame transposed.

    Returns
    -------
    df_summary : pandas.DataFrame
        A summary DataFrame with information on each column's data type, missing values, cardinality, 
        and suggested data type.
    """

    # Validate input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Input must be a pandas DataFrame, but received {type(df).__name__}.')
    
    # Display DataFrame basic info (type, entries, and memory usage)
    print(f'{type(df)}')
    print(f"RangeIndex: {len(df)} entries, 0 to {len(df)-1}")
    print(f"Data columns (total {df.shape[1]} columns)")
    
    # Generate a summary of data types in the DataFrame
    dtypes_summary = df.dtypes.value_counts()
    dtypes_string = ', '.join([f'{dtype.name}({count})' for dtype, count in dtypes_summary.items()])
    print(f"dtypes: {dtypes_string}")
    
    # Display memory usage
    mem_usage = df.memory_usage(deep=True).sum() / 1024
    print(f"memory usage: {mem_usage:.1f} KB")
    print()
    
    # Calculate and display the total percentage of missing values
    total_missing_percentage = (df.isna().sum().sum())/len(df) *100
    print(f'Total Percentage of Null Values: {total_missing_percentage:.2f}%')
    
    # Validate cat_threshold and cont_threshold types
    if not isinstance(cat_threshold, int):
        raise TypeError(f'cat_threshold must be an int, but received {type(cat_threshold).__name__}.')
    if not isinstance(cont_threshold, float):
        raise TypeError(f'cont_threshold must be a float, but received {type(cont_threshold).__name__}.')

    # Number of rows in the DataFrame
    num_rows = len(df)
    if num_rows == 0:
        raise ValueError('The DataFrame is empty.')

    # Prepare column-wise summary information
    data_type = df.dtypes  # Column data types
    null_count = df.notna().sum()  # Count of non-null values per column
    missings = df.isna().sum()  # Count of missing values per column
    missings_perc = round(df.isna().sum() / num_rows * 100, 2)  # Percentage of missing values per column
    unique_values = df.nunique()  # Number of unique values per column
    cardinality = round(unique_values / num_rows * 100, 2)  # Cardinality percentage per column

    # Create DataFrame summarizing the above metrics for each column
    df_summary = pd.DataFrame({
        'Data Type': data_type,
        'Not-Null': null_count,
        'Missing': missings,
        'Missing (%)': missings_perc,
        'Unique': unique_values,
        'Cardinality (%)': cardinality
    })

    # List to store the suggested type for each column based on cardinality and thresholds
    suggested_types = []

    for col in df.columns:
        card = unique_values[col]  # Unique values count for the column
        percent_card = card / num_rows * 100  # Cardinality percentage for the column

        # Determine the suggested type based on thresholds and value distributions
        if card == 2:
            suggested_type = 'Binary'  # Two unique values typically indicate a binary column
        elif df[col].dtype == 'object':
            suggested_type = 'Categorical'  # Object columns are likely categorical
        elif card < cat_threshold:
            suggested_type = 'Categorical'  # Few unique values may indicate categorical data
        else:
            # If cardinality percentage meets cont_threshold, it's considered continuous
            suggested_type = 'Numerical Continuous' if percent_card >= cont_threshold else 'Numerical Discrete'
        
        # Append the suggested type to the list
        suggested_types.append(suggested_type)
    
    # Add the suggested types to the summary DataFrame
    df_summary['Suggested Type'] = suggested_types

    # Return the summary, transposed if requested
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

def select_cat_features(data, target_col, target_type='cat', cat_threshold=10, mi_threshold=0.1, pvalue=0.05):
    """
    Identifies categorical columns in a DataFrame that are significantly related to the 'target_col' based on
    Chi-square for numeric targets or mutual information for categorical targets.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data.
    target_col : str
        Target column to correlate with other categorical columns.
    target_type : {'num', 'cat'}
        Type of target column. 'num' for numeric, 'cat' for categorical.
    cat_threshold : int, optional
        Maximum unique values for a numeric column to be considered categorical (default is 10).
    mi_threshold : float, optional
        Mutual information score threshold for selecting categorical features (default is 0.1).
    pvalue : float, optional
        Significance level for filtering statistically significant features with Chi-square (default is 0.05).

    Returns
    -------
    features_cat : list
        A list of categorical column names that are significantly associated with 'target_col' 
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
    
    # Additional check for mi_threshold when target_type is 'cat'
    if target_type == 'cat' and mi_threshold is None:
        print('For target_type "cat", "mi_threshold" must have a specified value.')
        return None

    # Initialize the list to store selected features
    features_cat = []

    # Select categorical-like columns based on data type and cardinality
    candidate_cols = [
        col for col in data.columns if col != target_col and (
            data[col].dtype == 'object' or data[col].nunique() <= cat_threshold
        )
    ]

    # If target is numeric, use Chi-square test for categorical features
    if target_type == 'num':
        if not pd.api.types.is_numeric_dtype(data[target_col]):
            print(f'For target_type "num", "{target_col}" must be numeric.')
            return None
        
        # Calculate Chi-square statistic for each categorical-like feature against the numeric target
        for col in candidate_cols:
            contingency_table = pd.crosstab(data[col], pd.cut(data[target_col], bins=5, labels=False))
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            if p_val <= pvalue:
                features_cat.append(col)
    
    # If target is categorical, use mutual information for categorical features
    elif target_type == 'cat':
        if pd.api.types.is_numeric_dtype(data[target_col]):
            print(f'For target_type "cat", "{target_col}" should be categorical.')
            return None
        
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(data[candidate_cols], data[target_col], discrete_features=True)
        
        # Filter by mutual information threshold
        for col, mi_score in zip(candidate_cols, mi_scores):
            if mi_score >= mi_threshold:
                features_cat.append(col)

    # Return the list of selected categorical features
    return features_cat

# -----------------------------------------------------------------------------------------------------------------------------------

def cv_evaluate(model, X, y, scoring=None, cv=5, return_train_score=False, print_scores=False):
    """
    Cross-validate a model with custom scoring metrics and cross-validation strategy.
    
    Parameters:
    - model: The machine learning model to evaluate.
    - X: Features dataset.
    - y: Target variable.
    - scoring: List of scoring metrics to use. Must contain at least one metric.
    - cv: Cross-validation strategy or number of folds. Default is 5.
    - return_train_score: Whether to return training scores. Default is False.
    - print_scores: If True, print mean scores after cross-validation. Default is False.
    
    Returns:
    - results_dict: Dictionary of mean test scores (and train scores if `return_train_score=True`).
    """
    # Error check for scoring parameter
    if scoring is None or not isinstance(scoring, list) or len(scoring) == 0:
        raise ValueError("The 'scoring' parameter must be a non-empty list with at least one valid metric.")

    # Run cross-validation
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=return_train_score)

    # Calculate mean test scores for each metric
    results_dict = {f'test_{metric}': np.mean(results[f'test_{metric}']) for metric in scoring}
    
    # Optionally include train scores
    if return_train_score:
        train_scores = {f'train_{metric}': np.mean(results[f'train_{metric}']) for metric in scoring}
        results_dict.update(train_scores)
    
    # Print scores if requested
    if print_scores:
        print(f'Mean Cross Validation Scores for {model.__class__.__name__}:\n{"-"*30}')
        for metric, score in results_dict.items():
            print(f'{metric.capitalize()}: {score:.5f}')
            print()
    
    return results_dict

# -----------------------------------------------------------------------------------------------------------------------------------

def fit_test_evaluate(model, X_train, y_train, X_test, y_test, metrics=None, print_report=True, return_scores=False):
    """
    Fit a model, evaluate it on the test set with custom metrics, and print/display results.

    Parameters:
    - model: The machine learning model to evaluate.
    - X_train: Training feature set.
    - y_train: Training labels.
    - X_test: Test feature set.
    - y_test: Test labels.
    - metrics: List of metric names to evaluate. Defaults to common classification/regression metrics.
    - print_report: Whether to print results (scores, classification report, confusion matrix/residual plot). Default is True.
    - return_scores: If True, return a dictionary of computed scores. Default is False.

    Returns:
    - scores_dict (optional): Dictionary of computed scores if return_scores=True.
    """
    # Determine model type and default metrics
    if is_classifier(model):
        model_type = 'classification'
        if metrics is None:
            metrics = ['accuracy', 'f1', 'roc_auc']
    elif is_regressor(model):
        model_type = 'regression'
        if metrics is None:
            metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    else:
        raise ValueError("The model must be either a classifier or a regressor.")
    
    # Fit the model and make predictions
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)

    # Calculate requested metrics
    scores_dict = {}
    for metric in metrics:
        if metric in ['accuracy', 'f1', 'roc_auc', 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
            scorer = get_scorer(metric)
            score = scorer._score_func(y_test, y_preds)
            # Flip sign for negative metrics (like neg_mean_squared_error)
            scores_dict[metric] = -score if 'neg_' in metric else score

    # Print scores if requested
    if print_report:
        print(f'Test Scores ({model_type.capitalize()}):\n{"-"*30}')
        for metric, score in scores_dict.items():
            print(f'{metric.capitalize().replace("_", " ")}: {score:.5f}')
        
        if model_type == 'classification':
            # Classification-specific reporting
            print(f'\nClassification Report:\n{"-"*22}')
            print(classification_report(y_test, y_preds))
            
            print(f'\nConfusion Matrix:\n{"-"*17}')
            ConfusionMatrixDisplay.from_predictions(y_test, y_preds)
            plt.show()
        
        elif model_type == 'regression':
            # Regression-specific visualization
            print(f'\nResidual Plot:\n{"-"*17}')
            residuals = y_test - y_preds
            plt.scatter(y_preds, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.show()

    # Return scores if requested
    if return_scores:
        return scores_dict

# -----------------------------------------------------------------------------------------------------------------------------------