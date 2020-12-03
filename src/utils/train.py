import itertools
from collections import defaultdict

import pandas as pd

# Standard ML models
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


def hyperparameter_tuning_cv(model, data, labels, cv_k, params, metric) -> pd.DataFrame:
    d = defaultdict(list)
    param_grid = ParameterGrid(params)
    for param in param_grid:
        for key, value in param.items():
            d[key].append(value)
        score = None
        if model == 'knn':
            score = cross_val_w_oversampling(data, labels, k=cv_k,
                                             model=KNeighborsClassifier(n_neighbors=param["n_neighbors"]),
                                             oversampling=param['oversampling'], metric=metric)
        if model == 'logistic':
            score = cross_val_w_oversampling(data, labels, k=cv_k, model=LogisticRegression(),
                                             oversampling=param['oversampling'], metric=metric)

        if model == 'lda':
            score = cross_val_w_oversampling(data, labels, k=cv_k, model=Lda(), oversampling=param['oversampling'],
                                             metric=m)

        if model == 'svc':
            score = cross_val_w_oversampling(data, labels, k=cv_k,
                                             model=SVC(kernel=param['kernel'], gamma=param['gamma']),
                                             oversampling=param['oversampling'], metric=metric)
        if model == 'naive_bayes':
            m = cross_val_w_oversampling(data, labels, k=cv_k, model=GaussianNB(),
                                         oversampling=param['oversampling'], metric=metric)
        if model == 'decision_tree':
            score = cross_val_w_oversampling(data, labels, k=cv_k,
                                             model=DecisionTreeClassifier(max_depth=param['max_depth']),
                                             oversampling=param['oversampling'], metric=metric)
        if model == 'random_forest':
            score = cross_val_w_oversampling(data, labels, k=cv_k,
                                             model=RandomForestClassifier(max_depth=param['max_depth'],
                                                                          n_estimators=param['n_estimators']),
                                             oversampling=param['oversampling'], metric=metric)
        if model == 'gradient_boosting':
            score = cross_val_w_oversampling(data, labels, k=cv_k,
                                             model=GradientBoostingClassifier(n_estimators=param['n_estimator'],
                                                                              max_depth=param['max_depth']),
                                             oversampling=param['oversampling'], metric=metric)

        d[metric.__name__].append(score)

    return pd.DataFrame(data=d)
