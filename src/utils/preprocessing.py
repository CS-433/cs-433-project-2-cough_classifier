import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def standardize(data, idx_start=0, idx_end=None):
    """
    Standardize columns
    :param data: dataframe
    :type data: pd.DataFrame
    :param idx_start: start index
    :type idx_start: int
    :param idx_end: end index
    :type idx_end: int
    :return: dataframe with standardized columns
    """
    # Standardize the specified columns
    if isinstance(data, np.ndarray):
        return StandardScaler().fit_transform(data)

    data.iloc[:, idx_start:idx_end] = StandardScaler().fit_transform(data.iloc[:, idx_start:idx_end])
    return data


def oversample(X, y):
    """
    Apply SMOTE algorithm to balanced imbalanced dataset
    :param X: feature dataframe
    :type X: pd.DataFrame
    :param y: label dataframe
    :type y: pd.DataFrame
    :return: features and labels with balanced classes
    """
    oversampled = SMOTE(random_state=42)
    X_over, y_over = oversampled.fit_resample(X, y)
    # X_over = pd.DataFrame(X_over, columns=X.columns)
    # y_over = pd.DataFrame(y_over, columns=y.columns)

    return X_over, y_over


def dummy_code(df, columns):
    """
    Dummy code categorical features
    :param df: dataframe
    :type df: pd.DataFrame
    :param columns: columns to dummy code
    :type columns: list of str
    :return: dataframe with dummy coded columns
    """
    df = pd.get_dummies(df, columns=columns)
    # drop reference columns for ['Gender', 'Resp_Condition', 'Symptoms'] to avoid multi-colinearity
    df = df.drop(['Gender_0.5', 'Resp_Condition_0.5', 'Symptoms_0.5'], axis=1)

    return df


def remove_correlated_features(df, threshold, print_features=False):
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if np.any(upper_tri[column] > threshold)]
    if print_features:
        print("Correlated Features: ", to_drop)
    df1 = df.drop(to_drop, axis=1)

    return df1
