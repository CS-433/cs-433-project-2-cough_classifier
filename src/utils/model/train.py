import numpy as np
import pandas as pd


def train_test_split(X: pd.DataFrame, y: pd.DataFrame, random_state = 1, fraction = 0.7):
    """
    Split the data set in train and test data
    """
    X_tr = X.loc[X.sample(frac = fraction, random_state=random_state).index.unique()]#.drop(['File_Name'], axis = 1)
    X_te = X.drop(X_tr.index)#.drop(['File_Name'], axis = 1)

    
    y_tr = y.loc[y.sample(frac = fraction, random_state=random_state).index.unique()]#.drop(['File_Name'], axis = 1)
    y_te = y.drop(y_tr.index)#.drop(['File_Name'], axis = 1)
    y_te = y_te.Label
    y_tr = y_tr.Label
    
    return X_tr, y_tr, X_te, y_te

def cross_validation_iter(data: pd.DataFrame, labels: pd.DataFrame, k: int):
    """
    Compute cross validation for a single iteration.
    """
    unique_subjects = np.array(data.index.get_level_values('subject').unique())
    N = len(unique_subjects)
    fold_interval = int(N / k)
    indices = np.random.permutation(N)
    for k_iteration in range(k):
        k_indices = tuple([indices[k_iteration * fold_interval: (k_iteration + 1) * fold_interval]])
        # split the data accordingly into training and validation
        val_subjects = unique_subjects[k_indices]
        tr_mask = np.ones(N, bool)
        tr_mask[k_indices] = False
        tr_subjects = unique_subjects[tr_mask]

        x_tr = data.loc[tr_subjects]
        y_tr = labels.loc[tr_subjects]
        x_val = data.loc[val_subjects]
        y_val = labels.loc[val_subjects]

        yield x_tr, y_tr, x_val, y_val


def cross_validation(data, labels, k, model, metric):
    """
    perform k-fold cross-validation after dividing the training data into k folds
    """
    metric_list = []

    for x_tr, y_tr, x_val, y_val in cross_validation_iter(data, labels, k):
        k_fit = model.fit(x_tr, y_tr)
        y_pred = k_fit.predict(x_val)

        metric_list.append(metric(y_val, y_pred))

    return np.mean(metric_list)
