import itertools
from collections import defaultdict

import pandas as pd

# Standard ML models
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.utils.model_helpers import cross_val_w_oversampling


def hyperparameter_tuning_cv(model, data, labels, cv_k, params,
                             metrics=[f1_score, roc_auc_score, accuracy_score]) -> pd.DataFrame:
    implemented_models = (
        'knn', 'logistic', 'lda', 'svc', 'naive_bayes', 'decision_tree', 'random_forest', 'gradient_boosting')

    assert model in implemented_models, "model does not exist"

    d = defaultdict(list)
    param_grid = ParameterGrid(params)
    for param in param_grid:
        for key, value in param.items():
            d[key].append(value)
        scores_dict = None

        if model == 'knn':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=KNeighborsClassifier(n_neighbors=param["n_neighbors"]),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'logistic':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k, model=LogisticRegression(),
                                                   oversampling=param['oversampling'], metrics=metrics)

        if model == 'lda':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k, model=Lda(),
                                                   oversampling=param['oversampling'],
                                                   metrics=metrics)

        if model == 'svc':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=SVC(kernel=param['kernel'], gamma=param['gamma']),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'naive_bayes':
            m = cross_val_w_oversampling(data, labels, k=cv_k, model=GaussianNB(),
                                         oversampling=param['oversampling'], metrics=metrics)
        if model == 'decision_tree':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=DecisionTreeClassifier(max_depth=param['max_depth']),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'random_forest':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=RandomForestClassifier(max_depth=param['max_depth'],
                                                                                n_estimators=param['n_estimators']),
                                                   oversampling=param['oversampling'], metrics=metrics)
        if model == 'gradient_boosting':
            scores_dict = cross_val_w_oversampling(data, labels, k=cv_k,
                                                   model=GradientBoostingClassifier(n_estimators=param['n_estimator'],
                                                                                    max_depth=param['max_depth']),
                                                   oversampling=param['oversampling'], metrics=metrics)

        for metric_name, score in scores_dict.items():
            d[metric_name].append(score)

    df = pd.DataFrame(data=d)

    return df
