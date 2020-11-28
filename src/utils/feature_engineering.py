# Do recursive feature elimination
from sklearn.feature_selection import RFE
# explore the number of selected features for RFE
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

# get a list of models to evaluate
def get_models(model, X):
    models = dict()
    for i in range(1, X.shape[1]):
        rfe = RFE(estimator=model, n_features_to_select=i)
        models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
    return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')
    return scores
 

def RFE_(model, X, y, plot = True):
    # get the models to evaluate
    models = get_models(model, X)
    # evaluate the models and store results
    results, names, mean_score, std_score = list(), list(), list(), list()
    for name, model_ in models.items():
        scores = evaluate_model(model_, X, y)
        results.append(scores)
        names.append(name)
        mean_score.append(mean(scores))
        std_score.append(std(scores))
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # write results in pandas df 
    results_df = pd.DataFrame(data = {"# Features": names, "AUC (mean)": mean_score, "AUC (std)": std_score})
    
    if plot == True:
        # plot model performance for comparison
        pyplot.boxplot(results, labels=names, showmeans=True)
        pyplot.show()
    
    return results_df

def train_optimal_features_model(X,y,model):
    # get optimal amount of features
    RFE_results = RFE_(model, X, y)
    n_features = RFE_results["# Features"].iloc[RFE_results["AUC (mean)"].argmax()]
    selector = RFE(model, n_features_to_select=int(n_features), step=1)
    selector = selector.fit(X, y)
    
    # Only keep optimal features
    X_opt = X[X.columns[selector.ranking_ == 1]]
    
    # Train desired model with optimal amount of features
    mean_AUC = roc_w_cross_val(X_opt, y, model)
    
    return mean_AUC