import os
import random

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


def generate_data_list():
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

    list1, list2, list3, list4, list5 = [], [], [], [], []
    for body_data_file in body_data_files:
        session, _, user, _ = body_data_file.split('.')[0].split('_')
        session_user = session + '_' + user
        if   session_user in fold1: list1.append(body_data_file)
        elif session_user in fold2: list2.append(body_data_file)
        elif session_user in fold3: list3.append(body_data_file)
        elif session_user in fold4: list4.append(body_data_file)
        elif session_user in fold5: list5.append(body_data_file)
    print(len(list1), len(list2), len(list3), len(list4), len(list5))
    self_body_data_list = np.array([list1, list2, list3, list4, list5], dtype=object)
    np.save('data/data_list/self_body', self_body_data_list)


if __name__ == '__main__':
    # print_session_info()
    generate_data_list()