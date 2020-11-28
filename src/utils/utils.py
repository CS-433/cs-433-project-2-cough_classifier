from sklearn import metrics
import torch
import numpy as np
import shap
import pandas as pd


def binary_acc(y_pred, y_test):
    """
    Calculate accuracy of binary classification labels.

    Args:
        y_pred (list/np.array/torch.tensor): list of predictions
        y_test (list/np.array/torch.tensor): list of correct labels

    Returns:
        acc (float): Accuracy
    """
    # TODO might be wrong
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().numpy()
    else:
        y_pred = np.array(y_pred)

    if torch.is_tensor(y_test):
        y_test = y_test.detach().numpy()
    else:
        y_test = np.array(y_test)

    y_pred_tag = np.array(np.round(torch.sigmoid(torch.tensor(y_pred))))
    correct_results_sum = np.sum([a == b for a, b in zip(y_pred_tag, y_test)])
    acc = correct_results_sum / y_test.shape[0]

    return acc


def area_under_the_curve(y_pred, y_test):
    """
    Calculate area under the curve of binary classification labels.

    Args:
        y_pred (list/np.array/torch.tensor): list of predictions
        y_test (list/np.array/torch.tensor): list of correct labels

    Returns:
        auc (float): Area under the curve
    """
    y_pred = y_pred
    y_test = y_test
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    return metrics.auc(fpr, tpr)


# TODO only accepts model, if all hidden layers have the same dimension????
def get_shap_values(model, X_train, X_test, feature_names,
                    train_sample_size=1000, test_sample_size=300, device="cpu"):
    """
    TODO
    """
    # get training and testing samples
    train_samples = torch.from_numpy(X_train[np.random.choice(X_train.shape[0],
                                                              train_sample_size, replace=False)]).float().to(device)
    test_samples = torch.from_numpy(X_test[np.random.choice(X_test.shape[0],
                                                            test_sample_size, replace=False)]).float().to(device)

    # get the deep explainer
    de = shap.DeepExplainer(model, train_samples)
    # generate the shap values
    shap_values = de.shap_values(test_samples)

    # create a data frame with the absolute mean, std and names of the shap
    # values
    shap_df = pd.DataFrame({
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "stdev_abs_shap": np.std(np.abs(shap_values), axis=0),
        "feature_name": feature_names
    })

    # sort the entries by the mean shap value
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)

    return shap_df
