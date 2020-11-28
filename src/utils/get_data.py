import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.config import FEATURES

FILES = {
    "fine_segmentation_files": {"samples": "features_fine_segmentation.csv", "labels": "labels_fine_segmentation.csv"},
    "coarse_segmentation_files": {"samples": "features_coarse_segmentation.csv",
                                  "labels": "labels_coarse_segmentation.csv"},
    "no_segmentation_files": {"samples": "features_no_segmentation.csv", "labels": "labels_no_segmentation.csv"}
}


def import_data(path, segmentation_type, is_user_features=True, return_type='pd'):
    """
    Import data
    :param path: path of data
    :type path: str
    :param segmentation_type: 'no', 'coarse', or 'fine'
    :type segmentation_type: str
    :param is_user_features: specify if user features should be dropped
    :param return_type: 'pd', 'np'
    :type return_type: str
    :return: dataframes containing features and labels
    """

    if segmentation_type not in ["fine", "coarse", "no"]:
        raise Exception

    if return_type not in ['pd', 'np']:
        raise Exception

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

        df_features = df_features.drop(["File_Name"], axis=1)
        df_labels = df_labels.drop(["File_Name"], axis=1)

    df_features.drop(['Expert'], axis=1, errors='ignore', inplace=True)

    if not is_user_features:
        df_features.drop(FEATURES['METADATA'], axis=1, errors='ignore', inplace=True)

    if return_type == 'pd':
        return df_features, df_labels

    subject_indices = get_subjects_indices(df_features.index.get_level_values('subject'))

    return df_features.values, df_labels.values, subject_indices, list(df_features.columns)


def create_multi_index(data):
    data["subject"] = data["File_Name"].apply(lambda r: r.split("_")[0])
    data["file_id"] = data["File_Name"].apply(lambda r: r.split("_")[1])
    data.set_index(['subject', 'file_id'], inplace=True)
    data.drop(["File_Name"], axis=1, errors='ignore', inplace=True)

    return data


class CoughDataset(Dataset):
    """
    Custom torch dataset, used in order to make use of torch data loaders.
    """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def get_subjects_indices(subject_names):
    """
    Given a list of subject names, this function will assign numeric values,
    corresponding to the group of each data sample.

    Args:
        subject_names (list): list of file names
    Returns:
        (list): list of group values
    """
    # get a unique list of names
    unique_subject_names_dict = {s: index for index, s in enumerate(np.unique(subject_names))}

    # return the corresponding group index
    # TODO: find out why not sorted
    return [unique_subject_names_dict[s] for s in subject_names]


def get_data_loader(X, y, batch_size=1):
    """
    Returns a data loader for some dataset.

    Args:
        X (np.array): Samples
        y (np.array): Labels
        batch_size (int): batch size used in this data loader
    Returns:
        (torch.DataLoader): complete data loader
    """
    # create pytorch dataset
    dataset = CoughDataset(torch.FloatTensor(X),
                           torch.FloatTensor(y))

    # get pytorch data loaders
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True)

    return data_loader
