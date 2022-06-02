import os

import numpy as np
import pandas as pd
from natsort import natsorted


def generate_clip_label_new(input_path, output_path):
    print('generate_clip_label_new:')

    session_info = pd.read_csv('data/session_info_new.csv')

    dates = ['20201222', '20201229', '20210105', '20210309']
    session_num = [14, 6, 14, 9]
    for date_idx, date in enumerate(dates):
        input_label_path = input_path + date + '.csv'
        output_label_path = output_path + date + '.csv'

        # ----------------------------------------------------------------------
        # process begin and end in input_label (convert to second)
        input_label = pd.read_csv(input_label_path)
        input_label['begin'] = pd.to_datetime(input_label['begin'], format='%H:%M:%S')
        input_label['end'] = pd.to_datetime(input_label['end'], format='%H:%M:%S')
        input_label['begin'] = (input_label['begin'].dt.hour+input_label['begin'].dt.minute)*60 + input_label['begin'].dt.second
        input_label['end'] = (input_label['end'].dt.hour+input_label['end'].dt.minute)*60 + input_label['end'].dt.second
        print(input_label[:14])
        print(f'input label shape {input_label.shape}')
        print()

        # ----------------------------------------------------------------------
        # check timestampes are correct
        for i in range(len(input_label)-1):
            try:
                assert input_label.iloc[i, 4] == input_label.iloc[i+1, 3]
            except:
                if input_label.iloc[i+1, 3] == 0:
                    pass
                else:
                    print(f'line {i} has problem: {input_label.iloc[i, 4]}, {input_label.iloc[i+1, 3]}')

        # ----------------------------------------------------------------------
        # process sessions and clip number, initialize output
        sessions = [f'{date}_{session_idx+1:02d}' for session_idx in range(session_num[date_idx])]
        print(sessions)
        clip_num = [session_info[session_info['session']==s]['clip_num'].values[0] for s in sessions]
        print(clip_num)
        output_label = np.zeros([np.sum(clip_num), 6], dtype=object)
        print(f'output label shape {output_label.shape}')
        print()

        # ----------------------------------------------------------------------
        # generate clip-level labels for affective
        print('affective:')
        in_aff_label = input_label[input_label.iloc[:,2]==1]
        idx = 0
        for session_idx in range(len(sessions)):
            aff_session_label = in_aff_label[in_aff_label.iloc[:,0]==sessions[session_idx]]
            # print(aff_session_label)
            # print(aff_session_label.shape)
            # print()

            for clip_idx in range(clip_num[session_idx]):
                # print(f'session {sessions[session_idx]}, clip {clip_idx}')
                clip_begin = clip_idx * 32 / 15
                clip_end = clip_begin + 32 / 15
                # print(np.round(clip_begin, 1), np.round(clip_end, 1))
                # print()

                for row_idx in range(len(aff_session_label)):
                    row = aff_session_label.iloc[row_idx,:].values
                    # print(row)
                    if row[3] <= clip_begin < row[4]:
                        clip_begin_range = row[3:]
                    if row[3] <= clip_end < row[4]:
                        clip_end_range = row[3:]
                if clip_begin_range[-1] == clip_end_range[-1]:
                    output_label[idx,:3] = [sessions[session_idx], int(clip_idx+1), np.around((clip_begin_range[-1] - 1) / 4, 4)]
                    idx += 1
                else:
                    l1 = clip_begin_range[1] - clip_begin
                    l2 = clip_end - clip_end_range[0]
                    score = (l1 * clip_begin_range[-1] + l2 * clip_end_range[-1]) / (l1 + l2)
                    output_label[idx,:3] = [sessions[session_idx], int(clip_idx+1), np.around((score - 1) / 4, 4)]
                    idx += 1

        # ----------------------------------------------------------------------
        # generate clip-level labels for visual
        print('visual:')
        in_vis_label = input_label[input_label.iloc[:,2]==2]
        idx = 0
        for session_idx in range(len(sessions)):
            vis_session_label = in_vis_label[in_vis_label.iloc[:,0]==sessions[session_idx]]

            for clip_idx in range(clip_num[session_idx]):
                clip_begin = clip_idx * 32 / 15
                clip_end = clip_begin + 32 / 15

                for row_idx in range(len(vis_session_label)):
                    row = vis_session_label.iloc[row_idx,:].values
                    if row[3]<=clip_begin<row[4]:
                        clip_begin_range = row[3:]
                    if row[3]<=clip_end<row[4]:
                        clip_end_range = row[3:]
                if clip_begin_range[-1] == clip_end_range[-1]:
                    output_label[idx,3] = np.around((clip_begin_range[-1] - 1) / 4, 4)
                    idx += 1
                else:
                    l1 = clip_begin_range[1] - clip_begin
                    l2 = clip_end - clip_end_range[0]
                    score = (l1 * clip_begin_range[-1] + l2 * clip_end_range[-1]) / (l1 + l2)
                    output_label[idx,3] = np.around((score - 1) / 4, 4)
                    idx += 1

        # ----------------------------------------------------------------------
        # generate clip-level labels for visual
        print('behavior:')
        in_beh_label = input_label[input_label.iloc[:,2]==3]
        idx = 0
        for session_idx in range(len(sessions)):
            beh_session_label = in_beh_label[in_beh_label.iloc[:,0]==sessions[session_idx]]

            for clip_idx in range(clip_num[session_idx]):
                clip_begin = clip_idx * 32 / 15
                clip_end = clip_begin + 32 / 15

                for row_idx in range(len(beh_session_label)):
                    row = beh_session_label.iloc[row_idx,:].values
                    if row[3]<=clip_begin<row[4]:
                        clip_begin_range = row[3:]
                    if row[3]<=clip_end<row[4]:
                        clip_end_range = row[3:]
                if clip_begin_range[-1] == clip_end_range[-1]:
                    output_label[idx,4] = np.around((clip_begin_range[-1] - 1) / 4, 4)
                    idx += 1
                else:
                    l1 = clip_begin_range[1] - clip_begin
                    l2 = clip_end - clip_end_range[0]
                    score = (l1 * clip_begin_range[-1] + l2 * clip_end_range[-1]) / (l1 + l2)
                    output_label[idx,4] = np.around((score - 1) / 4, 4)
                    idx += 1

        cols = ['session', 'clip', 'affective', 'visual', 'behavior', 'overall']
        output_label = pd.DataFrame(output_label, columns=cols)
        output_label['overall'] = pd.to_numeric((output_label['affective'] + output_label['visual'] + output_label['behavior']) / 3).round(4)

        output_label.to_csv(output_label_path, index=False)

def concate_label(input_path, output_path):
    cols = ['session', 'clip', 'affective', 'visual', 'behavior', 'overall']
    output_label = pd.DataFrame(columns=cols)
    processed_label_files = natsorted(os.listdir(input_path))
    for file in processed_label_files:
        if file.endswith('.csv'):
            label = pd.read_csv(input_path + file)
            print(len(label))
            output_label = pd.concat([output_label, label], axis=0)
    output_label.to_csv(output_path + 'labels_new.csv', index=False)


if __name__ == '__main__':
    start_end_path = 'data/annotations/start_end/'
    processed_path = 'data/annotations/processed_new/'
    concate_path = 'data/annotations/'

    generate_clip_label_new(start_end_path, processed_path)
    concate_label(processed_path, concate_path)
