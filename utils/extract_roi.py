import os
import time

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torchvision.ops import roi_align


class roi_feature_extractor():

    def box_txt2csv_addclip(box_txt_path, box_csv_path):
        """convert txt to csv and add clip index."""
        print('box_txt2csv_addclip:')
        box_files = natsorted(os.listdir(box_txt_path))
        for box_file in box_files:
            session_name = box_file.split('.')[0]
            box_file_path = os.path.join(box_txt_path, box_file)
            boxes = np.loadtxt(box_file_path, delimiter=',')
            cols = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']
            boxes = pd.DataFrame(boxes, columns=cols, dtype=int)

            frame_num = boxes['frame'].max()
            assert (frame_num % 4 == 0)
            boxes['clip'] = (boxes['frame'] - 1) // 4
            cols = ['frame', 'clip', 'id', 'x1', 'y1', 'x2', 'y2']
            boxes = boxes[cols]

            boxes.to_csv(f'{box_csv_path}{session_name}.csv', index=False)

    def extract(box_file_path, r3d_folder_path, roi_folder_path):
        """Extract RoI features from R3D features and bounding boxes."""
        boxes = pd.read_csv(box_file_path)
        print(f'boxes shape: {boxes.shape}')
        # print(boxes.head())

        r3d_files = natsorted(os.listdir(r3d_folder_path))
        clip_num = len(r3d_files)
        print(f'clip_num: {clip_num}')

        for r3d_file in r3d_files:
            clip_index = int(r3d_file.split('.')[0])

            # ------------------------------------------------------------------
            # process boxes
            boxes_clip = boxes[boxes['clip'] == clip_index]
            boxes_clip = boxes_clip.sort_values(by=['frame', 'id']).reset_index(drop=True)
            # print(boxes_clip)

            p_num = len(boxes_clip['id'].value_counts())
            # print(f'p_num: {p_num}')

            # boxes_clip_new = np.zeros([4, p_num, 5]) # frame, p, box
            boxes_clip_new = np.zeros([4, 4, 5]) # [frame=4, p=4, box]

            for box_index in range(len(boxes_clip)):
                frame = boxes_clip['frame'].iloc[box_index] % 4 - 1
                id = boxes_clip['id'].iloc[box_index]
                # print(f'frame: {frame}, id: {id}')
                box = boxes_clip.iloc[box_index, 3:]
                boxes_clip_new[frame, id, 1:] = box.values
            # print(boxes_clip_new)
            # boxes_clip_new[:, :, 0] = np.reshape(range(4 * p_num), (4, p_num))
            boxes_clip_new[:, :, 0] = np.reshape(range(16), (4, 4))

            boxes_clip_new = torch.from_numpy(boxes_clip_new).to(torch.float)
            boxes_clip_new = boxes_clip_new.reshape(-1, 5)  # (4, p_num, 5) --> (4xp_num, 5)
            # print(boxes_clip_new)
            # print(f'boxes_clip_new shape: {boxes_clip_new.shape}')

            # ------------------------------------------------------------------
            # process r3d features
            r3d = np.load(os.path.join(r3d_folder_path, r3d_file))
            r3d = torch.from_numpy(r3d).permute(1, 0, 2, 3)

            r3d_new = torch.zeros(len(boxes_clip_new), 1024, 14, 14)
            for frame in range(4):
                for p in range(p_num):
                    r3d_new[frame*p_num + p, :, :, :] = r3d[frame, :, :, :]
            # print(f'r3d_new shape: {r3d_new.shape}')

            roi = roi_align(r3d_new, boxes_clip_new, (7, 7), spatial_scale=14/1080)
            np.save(f'{roi_folder_path}/{clip_index}', roi.numpy())


if __name__ == "__main__":
    print("Extract RoI features")
    # output size: [p_num, 4, 1024, 7, 7]

    r3d_path = 'features/r3d_features/'
    box_txt_path = 'features/bytetrack/boxes/'
    box_csv_path = 'features/boxes/'
    roi_path = 'features/roi_features/'
    os.makedirs(roi_path, exist_ok=True)

    # convert txt to csv and add clip index
    # os.makedirs(box_csv_path, exist_ok=True)
    # roi_feature_extractor.box_txt2csv_addclip(box_txt_path, box_csv_path)

    # extract RoI features
    r3d_folders = natsorted(os.listdir(r3d_path))
    box_files = natsorted(os.listdir(box_csv_path))
    for box_file, r3d_folder in zip(box_files, r3d_folders):
        print(f'processing session {r3d_folder}')
        box_file_path = os.path.join(box_csv_path, box_file)
        r3d_folder_path = os.path.join(r3d_path, r3d_folder)
        roi_folder_path = os.path.join(roi_path, r3d_folder)
        os.makedirs(roi_folder_path, exist_ok=True)
        # print(box_file_path)
        # print(r3d_folder_path)
        # print(roi_folder_path)

        roi_feature_extractor.extract(box_file_path, r3d_folder_path, roi_folder_path)
