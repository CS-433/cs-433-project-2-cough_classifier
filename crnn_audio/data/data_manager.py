import os

import torch
import torch.utils.data as tdata
from sklearn.model_selection import train_test_split

from src.utils.get_data import import_data
from crnn_audio.data.data_sets import FolderDataset
from crnn_audio.utils.util import load_image, load_audio


class CSVDataManager(object):

    def __init__(self, config):
        load_formats = {
            'image': load_image,
            'audio': load_audio
        }

        assert config['format'] in load_formats, "Pass valid data format"

        self.dir_path = config['path']
        self.loader_params = config['loader']

        self.load_func = load_formats[config['format']]

        M_PATH = '../data'
        _, self.metadata_df = import_data(M_PATH, is_user_features=False, segmentation_type='no', return_type='pd')
        self.metadata_df['cough_type'] = self.metadata_df.apply(lambda row: 'wet' if row['Label'] == 1 else 'dry',
                                                                axis=1)
        self.classes = self._get_classes(
            self.metadata_df[['cough_type', 'Label']])
        self.data_splits = self._10kfold_split(self.metadata_df)

    def _get_classes(self, df):
        c_col = df.columns[0]
        idx_col = df.columns[1]
        return df.drop_duplicates().sort_values(idx_col)[c_col].unique()

    def _10kfold_split(self, df):
        ret = {"train": [], "val": []}
        # df_train = self.metadata_df.iloc[0:int(0.9 * len(self.metadata_df))]
        df_tr, df_val = train_test_split(df, test_size=0.1, random_state=42)
        for row in df_tr[['Label', 'cough_type']].iterrows():
            fname = os.path.join(self.dir_path, 'wav_data', f'{row[0]}.wav')
            ret["train"].append(
                {'path': fname, 'class': row[1]['cough_type'], 'class_idx': row[1]['Label']})
        for row in df_val[['Label', 'cough_type']].iterrows():
            fname = os.path.join(self.dir_path, 'wav_data', f'{row[0]}.wav')
            ret["val"].append(
                {'path': fname, 'class': row[1]['cough_type'], 'class_idx': row[1]['Label']})
        return ret

    def get_loader(self, name, transfs):
        assert name in self.data_splits
        dataset = FolderDataset(
            self.data_splits[name], load_func=self.load_func, transforms=transfs)

        return tdata.DataLoader(dataset=dataset, **self.loader_params, collate_fn=self.pad_seq)

    def pad_seq(self, batch):
        # sort_ind should point to length
        sort_ind = 0
        sorted_batch = sorted(
            batch, key=lambda x: x[0].size(sort_ind), reverse=True)
        seqs, srs, labels = zip(*sorted_batch)

        lengths, srs, labels = map(
            torch.LongTensor, [[x.size(sort_ind) for x in seqs], srs, labels])

        # seqs_pad -> (batch, time, channel)
        seqs_pad = torch.nn.utils.rnn.pad_sequence(
            seqs, batch_first=True, padding_value=0)
        # seqs_pad = seqs_pad_t.transpose(0, 1)
        return seqs_pad, lengths, srs, labels


if __name__ == '__main__':
    pass
