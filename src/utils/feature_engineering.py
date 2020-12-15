import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from src.utils.model_helpers import roc_w_cross_val


def get_models(model, X, start_idx=1):
    """
    Get a list of models to evaluate
    """
    models = dict()
    for i in range(start_idx, X.shape[1]):
        rfe = RFE(estimator=model, n_features_to_select=i)
        models[str(i)] = Pipeline(steps=[('s', rfe), ('m', model)])
    return models


def evaluate_model(model, X, y):
    """
    Evaluate a model based on its roc auc score using cross validation
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')
    return scores


def RFE_(model, X, y, start_idx=1, plot=False):
    """
    Do Recursive Feature elimination
    """
    # Get the models to evaluate
    models = get_models(model, X, start_idx)
    # Evaluate the models and store results
    results, names, mean_score, std_score = list(), list(), list(), list()
    for name, model_ in models.items():
        scores = evaluate_model(model_, X, y)
        results.append(scores)
        names.append(name)
        mean_score.append(np.mean(scores))
        std_score.append(np.std(scores))
        if int(name) % 10 == 0:
            print('>%s %.3f (%.3f)' % (name, np.mean(scores).dtype(float), np.std(scores).dtype(float)))
    # Write results in pandas df
    results_df = pd.DataFrame(data={"# Features": names, "AUC (mean)": mean_score, "AUC (std)": std_score})

    if plot:
        # Plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()

    return results_df


def train_optimal_features_model(X, y, model, start_idx=1):
    """
    Having extracted the optimal feature model, evaluate it
    """
    # Get optimal amount of features
    RFE_results = RFE_(model, X, y, start_idx)
    n_features = RFE_results["# Features"].iloc[RFE_results["AUC (mean)"].argmax()]
    selector = RFE(model, n_features_to_select=int(n_features), step=1)
    selector = selector.fit(X, y)
    ranks = selector.ranking_
    # Only keep optimal features
    X_opt = X[X.columns[selector.ranking_ == 1]]

    # Train desired model with optimal amount of features
    mean_AUC = roc_w_cross_val(X_opt, y, model)

    return mean_AUC


def get_optimal_features_model(X, y, model, start_idx=1):
    """
    # Get optimal amount of features for a given model
    """
    RFE_results = RFE_(model, X, y, start_idx)
    n_features = RFE_results["# Features"].iloc[RFE_results["AUC (mean)"].argmax()]
    selector = RFE(model, n_features_to_select=int(n_features), step=1)
    selector = selector.fit(X, y)
    ranks = selector.ranking_
    # Only keep optimal features
    X_opt = X[X.columns[selector.ranking_ == 1]]

    return X_opt
