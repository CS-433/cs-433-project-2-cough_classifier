import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def classic_preprocessing(X_tr, X_te=None, start=0, stop=-3, thresh=0.95, norm=True, dummy=True, drop_corr=True):
    """
    Do all the preprocessing at once
    :param X_tr: dataframe
    :type X_tr: pd.DataFrame
    :param X_te: test data
    :type X_te: pd.DataFrame
    :param start: start index
    :type start: int
    :param stop: end index
    :type stop: int
    :param thresh: threshold for correlation
    :type X_te: float
    :param norm: do normalisation
    :type norm: boolean
    :param dummy: do dummy coding
    :type dummy: boolean
    :param drop_corr: drop correlated features
    :type drop_corr: boolean
    :return: preprocessed dataframe
    """
    if norm:
        if X_te is not None:
            X_tr, X_te = standardize(X_tr, X_te, idx_start=start, idx_end=stop)
        else:
            X_tr = standardize(X_tr, idx_start=start, idx_end=stop)
    if dummy:
        X_tr = dummy_code(X_tr, columns=['Gender', 'Resp_Condition', 'Symptoms'])
        if X_te is not None:
            X_te = dummy_code(X_te, columns=['Gender', 'Resp_Condition', 'Symptoms'])
    if drop_corr:
        if X_te is not None:
            X_tr, X_te = remove_correlated_features(X_tr, X_te, threshold=thresh)
        else:
            X_tr = remove_correlated_features(X_tr, threshold=thresh)

    if X_te is not None:
        return X_tr, X_te

    return X_tr


def standard_preprocessing(samples, labels, do_standardize=True,
                           do_smote=True, do_dummy_coding=True,
                           categorical_features=('Gender', 'Resp_Condition', 'Symptoms')):
    """
    Do all the standard preprocessing at once
    :param samples: dataframe
    :type samples: pd.DataFrame
    :param labels: test data
    :type labels: pd.DataFrame
    :param do_standardize: do standardisation
    :type do_standardize: boolean
    :param do_smote: do oversampling with SMOTE
    :type do_smote: boolean
    :param do_dummy_coding: do dummy coding
    :type do_dummy_coding: boolean
    :param categorical_features: features to dummy code
    :type categorical_features: list
    :return: preprocessed dataframe
    """
    if do_standardize:
        # standardize all non categorical features
        samples = standardize(samples, None, 0, -len(categorical_features))

    if do_dummy_coding:
        # dummy coding
        samples = dummy_code(samples, columns=[feat for feat in categorical_features if feat in samples.columns])

    if do_smote:
        # smote
        # TODO smote might change dummy coding, i.e., make it continuous
        samples, labels = oversample(samples, labels)

    return samples, labels


def standardize(X_tr, X_te=None, idx_start=0, idx_end=None):
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
    if isinstance(X_tr, np.ndarray):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        if X_te is not None:
            X_te = scaler.transform(X_te)
            return X_tr, X_te
        return X_tr

    scaler = StandardScaler()
    X_tr.iloc[:, idx_start:idx_end] = scaler.fit_transform(X_tr.iloc[:, idx_start:idx_end])
    if X_te is not None:
        X_te.iloc[:, idx_start:idx_end] = scaler.transform(X_te.iloc[:, idx_start:idx_end])
        return X_tr, X_te
    return X_tr


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
    if columns:
        df = pd.get_dummies(df, columns=columns)
        # drop reference columns for ['Gender', 'Resp_Condition', 'Symptoms'] to avoid multi-colinearity
        df = df.drop(['Gender_0.5', 'Resp_Condition_0.5', 'Symptoms_0.5'], axis=1)

    return df


def remove_correlated_features(X_tr, X_te=None, threshold=0.95, verbose=False):
    """
    Remove features with correlation > threshold
    :param X_tr: training data
    :type X_tr: pd.DataFrame
    :param X_te: test data
    :type X_te: pd.DataFrame
    :param threshold: threshold for correlation
    :type X_te: float
    :param verbose: print removed features
    :type verbose: boolean
    :return: dataframe with correlated features removed
    """
    cor_matrix = X_tr.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if np.any(upper_tri[column] > threshold)]
    if verbose:
        print("Correlated Features: ", to_drop)
    X_tr = X_tr.drop(to_drop, axis=1)
    if X_te is not None:
        X_te = X_te.drop(to_drop, axis=1)
        return X_tr, X_te
    else:
        return X_tr
