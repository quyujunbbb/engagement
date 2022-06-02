import time
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import roi_align
from natsort import natsorted


def load_features(path):
    print(f'Loading R3D features ...')
    # [clip_num, 1024, 4, 14, 14] --> [clip_num, 4, 1024, 14, 14]

    features_original = np.load(path)
    features = features_original.transpose([0, 2, 1, 3, 4])
    clip_num = len(features)
    frame_num = features.shape[0] * features.shape[1]

    return features, clip_num, frame_num


def load_boxes(path, clip_num, frame_num):
    print(f'Loading bounding boxes ...')
    # [total_boxes_num, 6] --> [clip_num, 4, p_num, 5]
    # column = [frame_num, target_id, x1, y1, x2, y2]

    boxes_original = np.loadtxt(path, delimiter=',')
    target_id = boxes_original[:, 1]
    thre = int(frame_num * 0.2)

    # -----------------------------------------------------------
    # filter target id that appear less than 20%
    id_frequency = np.array(np.unique(target_id, return_counts=True)).T
    id_delete = []
    for i in id_frequency:
        if i[1] < thre: id_delete.append(i[0])
    # print(id_delete)
    boxes_tmp = []
    for idx, box in enumerate(boxes_original):
        if box[1] not in id_delete:
            boxes_tmp.append(boxes_original[idx])
    # -----------------------------------------------------------

    boxes = np.array(boxes_tmp)
    target_id = boxes[:, 1]
    id_frequency = np.array(np.unique(target_id, return_counts=True)).T
    # print(id_frequency)
    p_num = len(id_frequency)

    boxes_new = np.zeros([clip_num, 4, p_num, 5])
    for box in boxes:
        clip = int((box[0] - 1) / 4)
        frame = int((box[0] - 1) % 4)

        id = box[1]
        for p in range(p_num):
            if id == id_frequency[p, 0]:
                boxes_new[clip, frame, p, 1:] = box[2:]

    for box in boxes_new:
        list = np.reshape(range(4 * p_num), (4, p_num))
        box[:, :, 0] = list

    return boxes_new, p_num


def preprocessing(features, boxes, clip_num, p_num):
    """Reshape features and bounding boxes.
    Args:
        features : [clip_num, 4, 1024, 14, 14]
        boxes    : [clip_num, 4, p_num, 5]

    Returns:
        features_new : [clip_num, 4xp_num, 1024, 14, 14]
        boxes_new    : [clip_num, 4xp_num, 5]
    """
    print(f'Preprocessing feature maps and bounding boxes ...')
    N = 4 * p_num

    features_new = np.zeros([clip_num, N, 1024, 14, 14])
    for clip in range(clip_num):
        for i in range(4):
            for p in range(p_num):
                features_new[clip, i * p_num + p, :, :, :] = features[clip, i, :, :, :]

    boxes_new = np.reshape(boxes, (clip_num, N, 5))

    return features_new, boxes_new


def extract_roi_features(features, boxes, clip_num, p_num):
    print(f'Extracting RoI Align features...')
    # [clip_num, p_num, 4, 1024, 7, 7]

    features = torch.from_numpy(features)
    boxes = torch.from_numpy(boxes)

    for clip in range(clip_num):
        roi_features_clip = roi_align(features[clip],
                                      boxes[clip], (7, 7),
                                      spatial_scale=0.0625)
        if clip == 0:
            roi_features = roi_features_clip
        else:
            roi_features = torch.cat([roi_features, roi_features_clip], dim=0)
    roi_features = roi_features.view(clip_num, -1, 1024, 7, 7)
    roi_features = roi_features.view(clip_num, 4, p_num, 1024, 7, 7)
    roi_features = torch.transpose(roi_features, 1, 2)
    # roi_features = torch.mean(roi_features, 2)

    return roi_features


def save_to_pt(roi_feature_path):
    """Save RoI features into a single .pt file.
    Args:
        roi_feature_path (string): folder for RoI features of every session.
    """
    feature_files = natsorted(
        [str(f) for f in Path(roi_feature_path).glob('**/*.npy')])

    startime = time.time()
    for i, file in enumerate(feature_files):
        features = torch.tensor(np.load(file))  # (clip_num, p_num, T, C, W, H)

        features = features.view((-1, 4, 1024, 7, 7))
        if i == 0:
            data = features
        else:
            data = torch.cat((data, features), 0)

    torch.save(data, "features/roi_features.pt")

    print(f'output data size {data.size()}')
    # c, 12474 = (clip_num x p_num) for every session
    print(f'Done in {time.time() - startime:.3f}s')


if __name__ == "__main__":
    print("Extract RoI features")
    # output size: [clip_num, p_num, 4, 1024, 7, 7]

    r3d_path = 'features/r3d_features_new/'
    box_path = 'features/bytetrack_new/boxes/'
    roi_path = 'features/roi_features_new/'
    os.makedirs(roi_path, exist_ok=True)

    r3d_folders = natsorted(os.listdir(r3d_path))
    box_files = natsorted(os.listdir(box_path))

    for r3d_folder, box_file in zip(r3d_folders, box_files):
        r3d_folder_path = os.path.join(r3d_path, r3d_folder)
        print(r3d_folder_path)
        box_file_path = os.path.join(box_path, box_file)
        print(box_file_path)
        roi_folder_path = os.path.join(roi_path, r3d_folder)
        print(roi_folder_path)
        os.makedirs(roi_folder_path, exist_ok=True)

        features, clip_num, frame_num = load_features(r3d_folder_path)
        # boxes, p_num = load_boxes(box_file_path, clip_num, frame_num)
        # features_new, boxes_new = preprocessing(features, boxes, clip_num, p_num)
        # roi_features = extract_roi_features(features_new, boxes_new, clip_num, p_num)

        # np.save(os.path.join(roi_folder_path, 'roi_features.npy'), roi_features)
        # save_to_pt(roi_folder_path)
    # for i in range(len(r3d_files)):
    #     r3d_file_name = r3d_files[i].split("/")[-1].split(".")[0]
    #     box_file_name = box_files[i].split("/")[-1].split(".")[0]
    #     assert r3d_file_name == box_file_name

    #     print(f'Processing {r3d_file_name} ...')

    #     startime = time.time()

    #     features, clip_num, frame_num = load_features(r3d_files[i])
    #     print(f'  + feature size {features.shape}')
    #     # [clip_num, 4, 1024, 14, 14]

    #     boxes, p_num = load_boxes(box_files[i], clip_num, frame_num)
    #     print(f'  + bounding box size {boxes.shape}')
    #     # [clip_num, 4, p_num, 5]

    #     features, boxes = preprocessing(features, boxes, clip_num, p_num)
    #     print(f'  + feature size {features.shape}')
    #     # [clip_num, 4xp_num, 1024, 14, 14]
    #     print(f'  + bounding box size {boxes.shape}')
    #     # [clip_num, 4xp_num, 5]

    #     features = extract_roi_features(features, boxes, clip_num, p_num)
    #     features = features.cpu().numpy()
    #     print(f'  + roi align feature size {features.shape}')
    #     # [clip_num, p_num, 4, 1024, 7, 7]

    #     np.save(roi_feature_path + "/" + r3d_file_name, features)
    #     print(f'  + Done in {time.time() - startime:.3f}s')
