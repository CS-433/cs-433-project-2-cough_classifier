import pandas as pd
from sklearn.preprocessing import StandardScaler


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

    drop_features = ['Expert']
    df_features.drop(drop_features, axis=1, errors='ignore', inplace=True)

    if not is_user_features:
        user_features = ['Age', 'Gender', 'Resp_Condition', 'Symptoms']
        df_features.drop(user_features, axis=1, errors='ignore', inplace=True)

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
    
    return df


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


# TODO Xavi did the same? Replace by his version
def train_test(X, y, fraction = 0.8, random_state = 1, segmentation = True):
    """
    Split the data set in train and test data
    """
    # If there are several instances of the same subject, we don't want it to be in the training and testing set
    if segmentation == True:
        X[['Subject', 'Cough']] = X['File_Name'].str.split("_", expand = True)
        y[['Subject', 'Cough']] = y['File_Name'].str.split("_", expand = True)
    
        # index for Subject
        X = X.set_index(['Subject'])
        y = y.set_index(['Subject'])
    
    train_X = X.loc[X.sample(frac = fraction, random_state=random_state).index.unique()].drop(['File_Name'], axis = 1)
    test_X = X.drop(train_X.index).drop(['File_Name'], axis = 1)
    
    train_y = y.loc[y.sample(frac = fraction, random_state=random_state).index.unique()].drop(['File_Name'], axis = 1)
    test_y = y.drop(train_y.index).drop(['File_Name'], axis = 1)
    
    return train_X, test_X, train_y, test_y

