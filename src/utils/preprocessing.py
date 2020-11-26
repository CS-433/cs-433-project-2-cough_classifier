import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def import_data(path, segmentation_type, is_user_features=True):
    """
    Import data
    :param path: path of data
    :type path: str
    :param segmentation_type: 'no', 'coarse', or 'fine'
    :type segmentation_type: str
    :param is_user_features: specify if user features should be dropped
    :return: dataframes containing features and labels
    """
    df_features = pd.read_csv(f'{path}/features_{segmentation_type}_segmentation.csv', index_col=0)
    df_labels = pd.read_csv(f'{path}/labels_{segmentation_type}_segmentation.csv', index_col=0)

    if segmentation_type in ('fine', 'coarse'):
        df_features = create_multi_index(df_features)
        df_labels = create_multi_index(df_labels)
    else:
        df_features.set_index("File_Name")
        df_features.rename(index={'File_Name': 'subject'})
        df_labels.set_index("File_Name")
        df_labels.rename(index={'File_Name': 'subject'})
        
        df_features.drop(["File_Name"], axis = 1)
        df_labels.drop(["File_Name"], axis = 1)

    df_features.drop(['Expert'], axis=1, errors='ignore', inplace=True)

    if not is_user_features:
        df_features.drop(FEATURES['METADATA'], axis=1, errors='ignore', inplace=True)

    return df_features, df_labels


def create_multi_index(data):
    data["subject"] = data["File_Name"].apply(lambda r: r.split("_")[0])
    data["file_id"] = data["File_Name"].apply(lambda r: r.split("_")[1])
    data.set_index(['subject', 'file_id'], inplace=True)
    data.drop(["File_Name"], axis=1, errors='ignore', inplace=True)

    return data


def standardize(df, idx_start=0, idx_end=-1):
    """
    Standardize columns
    :param df: dataframe
    :type path: pandas
    :param idx_start: start index
    :type idx_start: int
    :param idx_end: end index
    :type idx_end: int
    :return: dataframe with standardized columns
    """
    # -1 for idx_end indicates slicing until the last element, 
    # so manually replace -1 by the length of the dataframes columns's
    if idx_end == -1:
        idx_end = len(df.columns)
    # Standardize the specified columns
    df.iloc[:, idx_start:idx_end] = StandardScaler().fit_transform(df.iloc[:, idx_start:idx_end])
    
    return dfsrc/utils/model/train.py


def oversample(X,y):
    """
    Apply SMOTE algorithm to balanced imbalanced dataset
    :param X: feature dataframe
    :type path: pandas
    :param y: label dataframe
    :type columns: pandas
    :return: features and labels with balanced classes
    """
    oversample = SMOTE(random_state=42)
    X_over, y_over = oversample.fit_resample(X, y)
    X_over = pd.DataFrame(X_over, columns=X.columns)
    y_over = pd.DataFrame(y_over, columns=y.columns)
    
    return X_over, y_over
    


def dummy_code(df, columns):
    """
    Dummy code categorical features
    :param df: dataframe
    :type path: pandas
    :param columns: columns to dummy code
    :type columns: list of str
    :return: dataframe with dummy coded columns
    """
    df = pd.get_dummies(df, columns = columns)
    # drop reference columns for ['Gender', 'Resp_Condition', 'Symptoms'] to avoid multicollinearity
    df = df.drop(['Gender_0.5', 'Resp_Condition_0.5', 'Symptoms_0.5'], axis = 1)
    
    return df


