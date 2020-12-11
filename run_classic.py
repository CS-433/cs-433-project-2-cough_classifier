from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as Lda
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.utils.get_data import import_data
from src.utils.preprocessing import classic_preprocessing
from src.utils.train import train_predict
from src.utils.utils import create_csv_submission

# BEST MODEL PARAMETERS
BEST_PARAMS_NO_METADATA = {
    'coarse': {'model': KNeighborsClassifier(n_neighbors=1), 'oversampling': True},
    'fine': {'model': LogisticRegression(max_iter=10000), 'oversampling': True},
    'no': {'model': Lda(), 'oversampling': True}
}

# TODO: fill real params
BEST_PARAMS_WITH_METADATA = {
    'coarse': {'model': Lda(), 'oversampling': True},
    'fine': {'model': Lda(), 'oversampling': True},
    'no': {'model': Lda(), 'oversampling': True}
}

DATA_PATH = "./data"
SUBMISSION_PATH = "./data/test/predictions"

if __name__ == "__main__":

    # for segm_type, param in BEST_PARAMS_WITH_METADATA.items():
    #     X_tr, y_tr = import_data(DATA_PATH, segmentation_type=segm_type,
    #                              drop_user_features=False,
    #                              drop_expert=True)
    #     X_te = import_data(DATA_PATH, segmentation_type=segm_type,
    #                        drop_user_features=False,
    #                        drop_expert=True,
    #                        is_test=True)
    #
    #     X_tr, X_te = classic_preprocessing(X_tr, X_te)
    #
    #     y_pred = train_predict(param['model'], X_tr, y_tr, X_te, param=param)
    #     create_csv_submission(y_pred, segm_type=segm_type, submission_path=SUBMISSION_PATH, user_features=True)

    for segm_type, param in BEST_PARAMS_NO_METADATA.items():
        X_tr, y_tr = import_data(DATA_PATH, segmentation_type=segm_type,
                                 drop_user_features=True,
                                 drop_expert=True)
        X_te = import_data(DATA_PATH, segmentation_type=segm_type,
                           drop_user_features=True,
                           drop_expert=True,
                           is_test=True)

        X_tr, X_te = classic_preprocessing(X_tr, X_te, dummy=False)

        y_pred = train_predict(param['model'], X_tr, y_tr, X_te, param=param)
        create_csv_submission(y_pred, segm_type=segm_type, submission_path=SUBMISSION_PATH, user_features=False)
