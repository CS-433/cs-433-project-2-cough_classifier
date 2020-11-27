# Define a function that automatically plots the AUC curve for a given classifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def AUC_all_models(X, y, k = 4):
    
    models = ["LogisticRegression", "SVM", "LDA", "KNN", "GaussianNB", "DecisionTree", "RandomForest", "GradientBoosting"]
    
    m1 = roc_w_cross_val(X, y, LogisticRegression())
    m2 = roc_w_cross_val(X, y, SVC(kernel='linear'))
    m3 = roc_w_cross_val(X, y, LDA())
    m4 = roc_w_cross_val(X, y, KNeighborsClassifier(n_neighbors=16))
    m5 = roc_w_cross_val(X, y, GaussianNB())
    m6 = roc_w_cross_val(X, y, DecisionTreeClassifier(random_state=0))
    m7 = roc_w_cross_val(X, y, RandomForestClassifier(max_depth=7, random_state=0))
    m8 = roc_w_cross_val(X, y, GradientBoostingClassifier(random_state=0))

    results = [m1, m2, m3, m4, m5, m6, m7, m8]

    d = {'Models': models, 'AUC (mean)': results}
    
    return pd.DataFrame(data = d)

def homemade_all_models(X, y, k = 4):
    
    models = ["LogisticRegression", "SVM", "LDA", "KNN", "GaussianNB", "DecisionTree", "RandomForest", "GradientBoosting"]
    
    m1 = cross_validation(X, y, k, LogisticRegression(), metric=roc_auc_score)
    m2 = cross_validation(X, y, k, SVC(kernel='linear'), metric=roc_auc_score)
    m3 = cross_validation(X, y, k, LDA(), metric=roc_auc_score)
    m4 = cross_validation(X, y, k, KNeighborsClassifier(n_neighbors=16), metric=roc_auc_score)
    m5 = cross_validation(X, y, k, GaussianNB(), metric=roc_auc_score)
    m6 = cross_validation(X, y, k, DecisionTreeClassifier(random_state=0), metric=roc_auc_score)
    m7 = cross_validation(X, y, k, RandomForestClassifier(max_depth=7, random_state=0), metric=roc_auc_score)
    m8 = cross_validation(X, y, k, GradientBoostingClassifier(random_state=0), metric=roc_auc_score)

    results = [m1, m2, m3, m4, m5, m6, m7, m8]

    d = {'models': models, 'AUC (mean)': results}
    
    return pd.DataFrame(data = d)
    


def roc_w_cross_val(X, y, classifier, plot = False):
    
    cv = StratifiedKFold(n_splits=6)
    
    X = X.to_numpy()
    y = y.to_numpy()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
   
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    if plot == True:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
        ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
       # ax.legend(loc="lower right")
        ax.legend(bbox_to_anchor=(1,0), loc="lower left")
        plt.show()
    
    return mean_auc