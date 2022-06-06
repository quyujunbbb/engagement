import os
import random

import numpy as np
import pandas as pd
from natsort import natsorted
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def print_session_info():
    session_info = pd.read_csv('data/session_info.csv',
                               usecols=['clip_num', 'p_num'],
                               dtype=int,
                               comment='#')
    count = session_info['clip_num'].sum()
    print(f'{len(session_info)} sessions, {count} clips')


def get_fold_list(roi_path, fold):
    list = []
    for session in fold:
        print(session)
        files = natsorted(os.listdir(f'{roi_path}{session}'))
        for file in files:
            file_path = f'{session}/{file}'
            list.append(file_path)
    print(f'{len(list)} samples in fold')

    return list


def generate_data_list(roi_path):
    """random roi_features based on the sessions. 9+9+9+8+8=43 sessions."""
    # sessions = pd.read_csv('data/session_info.csv', usecols=['session'], comment='#')
    # sessions = sessions.values.reshape(-1, ).tolist()
    # sessions = random.sample(sessions, len(sessions))
    # print(sessions)

    sessions = [
        '20210105_12', '20210105_03', '20201222_09', '20210105_07',
        '20210105_09', '20210105_08', '20210309_07', '20201222_06',
        '20210105_13', '20201229_02', '20210309_09', '20210309_02',
        '20210105_02', '20210105_01', '20201222_13', '20201222_04',
        '20201229_05', '20201222_01', '20210309_08', '20210105_05',
        '20210309_06', '20201222_12', '20210309_03', '20210105_10',
        '20201222_14', '20201222_08', '20201222_11', '20201229_01',
        '20210309_01', '20210105_14', '20210105_04', '20210105_11',
        '20201222_02', '20201222_10', '20210105_06', '20201229_03',
        '20201222_05', '20210309_04', '20210309_05', '20201229_06',
        '20201229_04', '20201222_07', '20201222_03'
    ]
    fold1 = sessions[:9]
    fold2 = sessions[9:18]
    fold3 = sessions[18:27]
    fold4 = sessions[27:35]
    fold5 = sessions[35:]

    list1 = get_fold_list(roi_path, fold1)
    list2 = get_fold_list(roi_path, fold2)
    list3 = get_fold_list(roi_path, fold3)
    list4 = get_fold_list(roi_path, fold4)
    list5 = get_fold_list(roi_path, fold5)

    print(len(list1), len(list2), len(list3), len(list4), len(list5))
    data_list = np.array([list1, list2, list3, list4, list5], dtype=object)
    np.save('data/data_list', data_list)


class BHEH(Dataset):

    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __getitem__(self, index):
        file = self.files[index]
        x = np.load(f'features/roi_features_new/{file}').astype('float32')
        y = self.labels[file]
        print(np.shape(x), np.shape(y))
        return x, y

    def __len__(self):
        return len(self.files)


def create_label(files, labels):
    labels = labels[['session', 'clip', 'overall']]
    label_out = {}

    for file in files:
        session = file.split('.')[0].split('/')[0]
        clip = int(file.split('.')[0].split('/')[1])
        # print(f'session: {session}, clip: {clip}')

        temp = labels[labels['session'] == session]
        label_out[file] = temp[temp['clip'] == clip+1]['overall'].values[0]
        # print(label_out)

    return label_out


def create_bheh(train_files, test_files):
    label_path = f'data/annotations/labels_new.csv'
    labels = pd.read_csv(label_path)
    train_label = create_label(train_files, labels)
    test_label = create_label(test_files, labels)

    train_data = BHEH(train_files, train_label)
    test_data = BHEH(test_files, test_label)

    return train_data, test_data


if __name__ == '__main__':
    roi_path = 'features/roi_features_new/'
    # print_session_info()
    # generate_data_list(roi_path)

    fold_num = 5
    data_list = np.load('data/data_list.npy', allow_pickle=True)

    for fold in range(fold_num):
        test_files = data_list[fold]
        train_list = np.delete(data_list, fold, axis=0)
        train_files = [item for row in train_list for item in row]
        print(f'fold {fold}: train_num={len(train_files)}, test_num={len(test_files)}')

        train_data, test_data = create_bheh(train_files, test_files)
        train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)

        for x_batch, y_batch in train_loader:
            print(x_batch.size(), y_batch.size())
            break
