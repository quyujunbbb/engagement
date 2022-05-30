import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch.utils.data import Dataset


def print_session_info():
    session_info = pd.read_csv('data/session_info.csv',
                               usecols=['clip_num', 'p_num'],
                               dtype=int,
                               comment='#')
    count = session_info['clip_num'].sum()
    print(f'{len(session_info)} sessions, {count} clips')


class IMIBHEH(Dataset):

    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
        print(f'initialize features {self.features.size()}, labels {self.labels.size()}')

    def __getitem__(self, index):
        feature = self.features[index, :, :, :, :]
        label = self.labels[index]
        return (feature, label)

    def __len__(self):
        return len(self.labels)


def random_split(roi_path):
    """random roi_features based on the sessions. 9+9+9+8+8=43 sessions.
    Args:
        roi_path (string): the folder path that stores roi_features
    Returns:
        train_files (IMIBHEH) : training data
        test_files  (IMIBHEH) : test data
    """
    roi_files = natsorted(os.listdir(roi_path))
    random.shuffle(roi_files)
    print(roi_files)

    split_1 = roi_files[:9]
    split_2 = roi_files[9:18]
    split_3 = roi_files[18:27]
    split_4 = roi_files[27:35]
    split_5 = roi_files[35:]

    train_files = split_1 + split_2 + split_3 + split_4
    test_files = split_5

    return train_files, test_files


def preprocess(files, labels):
    roi_path = 'features/roi_features/'

    session_info = np.zeros((len(files), 2)).astype(int)
    session_info = pd.DataFrame(session_info, columns=['clip_num', 'p_num'])

    for i, file in enumerate(files):
        session_name = file.split('.')[0]

        features_in = torch.tensor(np.load(roi_path + file))
        clip_num = features_in.size()[0]
        p_num = features_in.size()[1]
        # print(f'processing {session_name}: {clip_num} clips, {p_num} people')
        session_info.iloc[i, :] = [clip_num, p_num]

        features_in = features_in.view((-1, 4, 1024, 7, 7))
        if i == 0:
            features_out = features_in
        else:
            features_out = torch.cat((features_out, features_in), 0)

        label_in = labels[labels['session'] == session_name]
        label_in = label_in.iloc[:, -1].reset_index(drop=True)
        label_temp = pd.DataFrame(np.repeat(label_in.values, p_num, axis=0))
        label_temp = label_temp.reset_index(drop=True)
        label_temp = torch.tensor(label_temp.values)
        if i == 0:
            label_out = label_temp
        else:
            label_out = torch.cat((label_out, label_temp), 0)

    features_out = torch.transpose(features_out, 1, 2)
    # print(f'train set feature size {features_out.size()}')  # [clip_num x p_num, C, T, W, H]
    # print(f'train set label size {label_out.size()}')
    # print(session_info)
    data = IMIBHEH(features_out, label_out)

    return data


def roi_ls(mode):
    """Generate full-size dataset using RoI features based on sessions.
    """
    # train_files, test_files = random_split(roi_path)
    train_files = ['20210105_14.npy', '20210309_01.npy', '20210105_12.npy',
                   '20210309_05.npy', '20201222_10.npy', '20210105_05.npy',
                   '20201222_11.npy', '20210309_09.npy', '20201222_12.npy',
                   '20210105_13.npy', '20201222_03.npy', '20210105_11.npy',
                   '20201222_08.npy', '20201222_06.npy', '20210309_06.npy',
                   '20201229_03.npy', '20210105_09.npy', '20201229_02.npy',
                   '20201222_04.npy', '20201222_09.npy', '20210105_04.npy',
                   '20210105_02.npy', '20201222_13.npy', '20201229_04.npy',
                   '20210309_07.npy', '20201222_07.npy', '20210309_04.npy',
                   '20210105_10.npy', '20201222_14.npy', '20201222_02.npy',
                   '20210105_07.npy', '20201229_01.npy', '20210309_02.npy',
                   '20210105_08.npy', '20210105_01.npy']
    test_files = ['20201229_05.npy', '20210105_06.npy', '20201222_05.npy',
                  '20210309_08.npy', '20210309_03.npy', '20201222_01.npy',
                  '20201229_06.npy', '20210105_03.npy']

    label_path = 'data/annotations/labels.csv'
    labels = pd.read_csv(label_path)
    if mode == 'Reg':
        labels = labels[['session', 'overall']]
    elif mode == 'Class':
        labels = labels[['session', 'class']]

    train_data = preprocess(train_files, labels)
    test_data = preprocess(test_files, labels)

    return train_data, test_data


def roi_l(mode):
    """Generate full-size dataset using RoI features.
    """
    data_path = "features/roi_features.pt"
    data = torch.load(data_path)
    data = torch.transpose(data, 1, 2)  # [12474, 1024, 4, 7, 7]
    # mean = data.mean()
    # std = data.std()
    # data = (data - mean) / std

    session_info = pd.read_csv('data/session_info.csv', usecols=['clip_num', 'p_num'])
    # print(session_info)
    session_info_new = session_info.loc[session_info.index.repeat(
        session_info.c_num)]
    session_info_new = session_info_new.reset_index(drop=True)
    # print(session_info_new)

    label_path = "data/annotations/labels.csv"
    if mode == 'Reg':
        labels = pd.read_csv(label_path)['overall']
    elif mode == 'Class':
        labels = pd.read_csv(label_path)['class']
    # print(labels)
    labels_new = labels.loc[labels.index.repeat(session_info_new.p_num)]
    labels_new = labels_new.reset_index(drop=True)
    labels_new = torch.tensor(labels_new.values)
    # print(labels_new)
    # print(f'label size {labels_new.size()}')

    data = IMIBHEH(data, labels_new)

    return data


def r3d_l(mode):
    """Generate full-size dataset using R3D features.
    """
    data_path = "features/r3d_features.pt"
    data = torch.load(data_path)  # [3140, 1024, 4, 14, 14]

    label_path = "data/annotations/labels.csv"
    if mode == 'Reg':
        labels = pd.read_csv(label_path)['overall']
    elif mode == 'Class':
        labels = pd.read_csv(label_path)['class']
    labels = torch.tensor(labels.values)

    # print(labels)
    # print(f'label size {labels.size()}')

    data = IMIBHEH(data, labels)

    return data


if __name__ == '__main__':
    print_session_info()
    # train_data, test_data = roi_l_session(mode='Reg')
