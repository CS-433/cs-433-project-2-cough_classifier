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
        df_features["subject"] = df_features["File_Name"]
        df_features.set_index("subject", inplace=True)
        df_labels["subject"] = df_labels["File_Name"]
        df_labels.set_index("subject", inplace=True)
        df_features = df_features.drop(["File_Name"], axis=1)
        df_labels = df_labels.drop(["File_Name"], axis=1)

    #df_features.drop(['Expert'], axis=1, errors='ignore', inplace=True)

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
def expert_models(X, y, oversampling = True):
    
    # Split the data according to which expert labeled it
    merged = X.merge(y, left_index=True, right_index = True)
    
    X_exp_1 = merged[merged['Expert'] == 1].iloc[:,:-1].drop(columns = ['Expert'], axis = 1)
    y_exp_1 = merged[merged['Expert'] == 1].iloc[:,-1]
    
    X_exp_2 = merged[merged['Expert'] == 2].iloc[:,:-1].drop(columns = ['Expert'], axis = 1)
    y_exp_2 = merged[merged['Expert'] == 2].iloc[:,-1]
    
    X_exp_3 = merged[merged['Expert'] == 3].iloc[:,:-1] .drop(columns = ['Expert'], axis = 1)
    y_exp_3 = merged[merged['Expert'] == 3].iloc[:,-1]
    
    # All expert groups are about 1000 samples big
    #print(len(X_exp_1), len(X_exp_2), len(X_exp_3))
    
    # Train all models for all experts
    exp_1 = AUC_all_models(X_exp_1, y_exp_1, k=6, oversampling=oversampling)
    exp_2 = AUC_all_models(X_exp_2, y_exp_2, k=6, oversampling=oversampling)
    exp_3 = AUC_all_models(X_exp_3, y_exp_3, k=6, oversampling=oversampling)
    
    # Gather the results in a df
    exp_1 = exp_1.rename(columns={'AUC (mean)': "Exp_1_AUC"})
    exp_2 = exp_2.rename(columns={'AUC (mean)': "Exp_2_AUC"})
    exp_3 = exp_3.rename(columns={'AUC (mean)': "Exp_3_AUC"})

    results = pd.concat([exp_1, exp_2["Exp_2_AUC"], exp_3["Exp_3_AUC"]], axis=1, sort=False)
    
    return results

def split_experts(X, y):
    
    merged = X.merge(y, left_index=True, right_index = True)
    
    X_exp_1 = merged[merged['Expert'] == 1].iloc[:,:-1]
    y_exp_1 = merged[merged['Expert'] == 1].iloc[:,-1]
    
    X_exp_2 = merged[merged['Expert'] == 2].iloc[:,:-1]
    y_exp_2 = merged[merged['Expert'] == 2].iloc[:,-1]
    
    X_exp_3 = merged[merged['Expert'] == 3].iloc[:,:-1] 
    y_exp_3 = merged[merged['Expert'] == 3].iloc[:,-1]

    return X_exp_1, y_exp_1, X_exp_2, y_exp_2, X_exp_3, y_exp_3


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
