import time
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


def load_bbxes(path, clip_num, frame_num):
    print(f'Loading bounding boxes ...')
    # [total_boxes_num, 6] --> [clip_num, 4, p_num, 5]
    # column = [frame_num, target_id, x1, y1, x2, y2]

    bbxes_original = np.loadtxt(path, delimiter=',')
    target_id = bbxes_original[:, 1]
    thre = int(frame_num * 0.2)

    # -----------------------------------------------------------
    # filter target id that appear less than 20%
    id_frequency = np.array(np.unique(target_id, return_counts=True)).T
    id_delete = []
    for i in id_frequency:
        if i[1] < thre: id_delete.append(i[0])
    # print(id_delete)
    bbxes_tmp = []
    for idx, bbx in enumerate(bbxes_original):
        if bbx[1] not in id_delete:
            bbxes_tmp.append(bbxes_original[idx])
    # -----------------------------------------------------------

    bbxes = np.array(bbxes_tmp)
    target_id = bbxes[:, 1]
    id_frequency = np.array(np.unique(target_id, return_counts=True)).T
    # print(id_frequency)
    p_num = len(id_frequency)

    bbxes_new = np.zeros([clip_num, 4, p_num, 5])
    for bbx in bbxes:
        clip = int((bbx[0] - 1) / 4)
        frame = int((bbx[0] - 1) % 4)

        id = bbx[1]
        for p in range(p_num):
            if id == id_frequency[p, 0]:
                bbxes_new[clip, frame, p, 1:] = bbx[2:]

    for bbx in bbxes_new:
        list = np.reshape(range(4 * p_num), (4, p_num))
        bbx[:, :, 0] = list

    return bbxes_new, p_num


def preprocessing(features, bbxes, clip_num, p_num):
    """Reshape features and bounding boxes.
    Args:
        features : [clip_num, 4, 1024, 14, 14]
        bbxes    : [clip_num, 4, p_num, 5]

    Returns:
        features_new : [clip_num, 4xp_num, 1024, 14, 14]
        bbxes_new    : [clip_num, 4xp_num, 5]
    """
    print(f'Preprocessing feature maps and bounding boxes ...')
    N = 4 * p_num

    features_new = np.zeros([clip_num, N, 1024, 14, 14])
    for clip in range(clip_num):
        for i in range(4):
            for p in range(p_num):
                features_new[clip, i * p_num + p, :, :, :] = features[clip, i, :, :, :]

    bbxes_new = np.reshape(bbxes, (clip_num, N, 5))

    return features_new, bbxes_new


def extract_roi_features(features, bbxes, clip_num, p_num):
    print(f'Extracting RoI Align features...')
    # [clip_num, p_num, 4, 1024, 7, 7]

    features = torch.from_numpy(features)
    bbxes = torch.from_numpy(bbxes)

    for clip in range(clip_num):
        roi_features_clip = roi_align(features[clip],
                                      bbxes[clip], (7, 7),
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
    """Extracted RoI featrues.

    Input:
        r3d_feature_path, bounding_box_path

    Output:
        roi aligned features to "roi_features/xxx.npy"
        size: [clip_num, p_num, 4, 1024, 7, 7]
    """
    print("--------------------------")
    print("Extract RoI Align features")
    print("--------------------------")

    r3d_feature_path = 'features/r3d_features/'
    bounding_box_path = 'features/bytetrack_roi/boxes/'
    roi_feature_path = 'features/roi_features/'
    Path(roi_feature_path).mkdir(parents=True, exist_ok=True)

    r3d_files = natsorted(
        [str(f) for f in Path(r3d_feature_path).glob('**/*.npy')])
    bbx_files = natsorted(
        [str(f) for f in Path(bounding_box_path).glob('**/*.txt')])
    assert len(r3d_files) == len(bbx_files)

    for i in range(len(r3d_files)):
        r3d_file_name = r3d_files[i].split("/")[-1].split(".")[0]
        bbx_file_name = bbx_files[i].split("/")[-1].split(".")[0]
        assert r3d_file_name == bbx_file_name

        print(f'Processing {r3d_file_name} ...')

        startime = time.time()

        features, clip_num, frame_num = load_features(r3d_files[i])
        print(f'  + feature size {features.shape}')
        # [clip_num, 4, 1024, 14, 14]

        bbxes, p_num = load_bbxes(bbx_files[i], clip_num, frame_num)
        print(f'  + bounding box size {bbxes.shape}')
        # [clip_num, 4, p_num, 5]

        features, bbxes = preprocessing(features, bbxes, clip_num, p_num)
        print(f'  + feature size {features.shape}')
        # [clip_num, 4xp_num, 1024, 14, 14]
        print(f'  + bounding box size {bbxes.shape}')
        # [clip_num, 4xp_num, 5]

        features = extract_roi_features(features, bbxes, clip_num, p_num)
        features = features.cpu().numpy()
        print(f'  + roi align feature size {features.shape}')
        # [clip_num, p_num, 4, 1024, 7, 7]

        np.save(roi_feature_path + "/" + r3d_file_name, features)
        print(f'  + Done in {time.time() - startime:.3f}s')

    # save_to_pt(roi_feature_path)
