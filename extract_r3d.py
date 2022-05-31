import os
import time
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable

from models.resnet import i3_res50


def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.ANTIALIAS)
    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1
    assert (data.max() <= 1.0)
    assert (data.min() >= -1.0)
    return data


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]]))
    return batch_data


def forward_batch(i3d, b_data):
    b_data = b_data.transpose([0, 4, 1, 2, 3])
    b_data = torch.from_numpy(b_data)  # [b,c,t,h,w]
    with torch.no_grad():
        b_data = Variable(b_data.cuda()).float()
        inp = {'frames': b_data}
        features = i3d(inp)
    return features.cpu().numpy()


def r3d_features(net, image_folder_path):
    chunk_size = 32

    rgb_files = natsorted([i for i in os.listdir(frames_dir)])
    frame_cnt = len(rgb_files)
    print(f'total frame: {frame_cnt}')
    rgb_files = rgb_files[0:frame_cnt:5]
    frame_cnt = len(rgb_files)
    print(f'sampled frame: {frame_cnt}')

    assert (frame_cnt > chunk_size)
    clipped_length = ((frame_cnt - chunk_size) // frequency) * frequency
    frame_indices = []
    for i in range(clipped_length // frequency + 1):
        frame_indices.append(
            [j for j in range(i * frequency, i * frequency + chunk_size)])
    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]
    batch_num = int(np.ceil(chunk_num / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)
    print(f'batch size: {batch_size}  '
          f'batch number: {batch_num}  '
          f'clip number: {chunk_num}  ')
    # print(f'frame_indices\n{frame_indices}')

    full_features = [[]]
    for batch_id in range(batch_num):
        batch_data = load_rgb_batch(frames_dir, rgb_files,
                                    frame_indices[batch_id])
        batch_data = batch_data[:, :, 16:240, 58:282, :]
        assert (batch_data.shape[-2] == 224)
        assert (batch_data.shape[-3] == 224)
        temp = forward_batch(i3d, batch_data)
        # print(f'feature {np.shape(temp)}')
        full_features[0].append(temp)

    # print(f'full_features size {np.shape(full_features)}')
    full_features = [np.concatenate(i, axis=0) for i in full_features]
    # print(f'full_features size {np.shape(full_features)}')
    # full_features = [np.expand_dims(i, axis=0) for i in full_features]
    # print(f'full_features size {np.shape(full_features)}')
    full_features = np.concatenate(full_features, axis=0)
    # print(f'full_features size {np.shape(full_features)}')
    # full_features = full_features[:, :, :, 0, 0, 0]
    # print(f'full_features size {np.shape(full_features)}')
    # full_features = np.array(full_features).transpose([1, 0, 2])
    # print(f'full_features size {np.shape(full_features)}')

    return full_features


if __name__ == "__main__":
    print("Extract R3D features")

    weight_path = 'pretrained/i3d_r50_kinetics.pth'
    session_folder_path = 'data/images/'
    r3d_feature_path = 'features/r3d_features/'
    os.makedirs(r3d_feature_path, exist_ok=True)

    net = i3_res50(400, weight_path)
    net.cuda()
    net.train(False)

    image_folders = natsorted(os.listdir(session_folder_path))
    for image_folder in image_folders:
        image_folder_path = session_folder_path + image_folder + "/"
        print(image_folder_path)

        startime = time.time()
        features = r3d_features(net, image_folder_path)
        # np.save(r3d_feature_path + "/" + image_folder, features)
        # print(f'extract R3D features {np.shape(features)} '
        #       f'in {time.time() - startime:.3f}s')
        break
