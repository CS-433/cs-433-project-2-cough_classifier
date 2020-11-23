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


def standardize(df, col_idx_start, col_idx_end):
    #features = list(df.columns)
    df.iloc[:, col_idx_start_col_idx_end:col_idx_start_col_idx_end] = StandardScaler().fit_transform(df.iloc[:, col_idx_start:, col_idx_end])
    
    return df
