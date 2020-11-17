import pandas as pd
from sklearn.preprocessing import StandardScaler


def import_data(path, segmentation_type):
    """
    Import data
    :param path: path of data
    :type path: str
    :param segmentation_type: 'no', 'coarse', or 'fine'
    :type segmentation_type: str
    :return: dataframes containing features and labels
    """
    df_features = pd.read_csv(f'{path}/features_{segmentation_type}_segmentation.csv', index_col=0)
    df_labels = pd.read_csv(f'{path}/labels_{segmentation_type}_segmentation.csv', index_col=0)
    df_features = df_features.set_index("File_Name")
    df_labels = df_labels.set_index("File_Name")

    return df_features, df_labels


def create_multi_index(data, labels):
    data["subject"] = data.index.map(lambda r: r.split("_")[0])
    data["file_id"] = data.index.map(lambda r: r.split("_")[1])
    data.set_index(['subject', 'file_id'], inplace=True)
    
    labels["subject"] = labels.index.map(lambda r: r.split("_")[0])
    labels["file_id"] = labels.index.map(lambda r: r.split("_")[1])
    labels.set_index(['subject', 'file_id'], inplace=True)
    
    return data, labels


def standardize(df):
    features = list(df.columns)
    df[features] = StandardScaler().fit_transform(df[features])

    return df
