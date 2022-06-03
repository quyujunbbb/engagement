import os
import time

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable

from models.resnet import i3_res50


def load_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img)
    img = img.astype(float)
    img = (img * 2 / 255) - 1
    assert (img.max() <= 1.0)
    assert (img.min() >= -1.0)

    return img


def load_batch(path, files, chunk_size, frame_indices):
    batch_img = np.zeros((1, chunk_size, 224, 224, 3))
    for i in range(chunk_size):
        img_path = path + files[frame_indices[i]]
        batch_img[0, i, :, :, :] = load_image(img_path)

    return batch_img


def forward_batch(net, batch_img):
    batch_img = batch_img.transpose([0, 4, 1, 2, 3])  # (1, 3, 32, 224, 224)
    batch_img = torch.from_numpy(batch_img)  # [bs, c, t, h, w]
    with torch.no_grad():
        batch_img = Variable(batch_img.cuda()).float()
        inp = {'frames': batch_img}
        features = net(inp)

    return features.cpu().numpy()


def r3d_features(net, image_folder_path, r3d_output_path):
    image_files = natsorted(os.listdir(image_folder_path))

    total_frame = len(image_files)
    clip_num = int(total_frame / 32)
    print(f'total frame: {total_frame}, clip num: {clip_num}')
    chunk_size = 32

    for clip in range(clip_num):
        start_frame = clip * chunk_size
        end_frame = start_frame + chunk_size
        frame_indices = np.arange(start_frame, end_frame)
        batch_data = load_batch(image_folder_path, image_files, chunk_size,
                                frame_indices)
        features = forward_batch(net, batch_data)  # (1, 1024, 4, 14, 14)
        features = features[0]  # (1024, 4, 14, 14)
        np.save(f'{r3d_output_path}{clip}', features)


if __name__ == "__main__":
    print("Extract R3D features")

    weight_path = 'pretrained/i3d_r50_kinetics.pth'
    session_folder_path = 'data/images_crop/'
    r3d_feature_path = 'features/r3d_features_new/'
    os.makedirs(r3d_feature_path, exist_ok=True)

    net = i3_res50(400, weight_path)
    net.cuda()
    net.train(False)

    image_folders = natsorted(os.listdir(session_folder_path))
    for image_folder in image_folders:
        print(f'processing {image_folder}')
        image_folder_path = session_folder_path + image_folder + "/"
        r3d_output_path = r3d_feature_path + image_folder +'/'
        os.makedirs(r3d_output_path, exist_ok=True)

        startime = time.time()
        r3d_features(net, image_folder_path, r3d_output_path)
        print(f'Done in {time.time() - startime:.3f}s\n')
