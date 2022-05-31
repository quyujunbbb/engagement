import os
import time
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.autograd import Variable

from models.resnet import i3_res50
from extract_r3d import extract_r3d_features, save_to_pt


if __name__ == "__main__":
    pretrained_path = 'pretrained/i3d_r50_kinetics.pth'
    image_folder_path = 'data/demo/images/'
    r3d_feature_path = 'data/demo/r3d_features/'
    r3d_feature_path_pt = 'data/demo/r3d_features.pt'
    Path(r3d_feature_path).mkdir(parents=True, exist_ok=True)

    frequency = 32
    batch_size = 20
    sample_mode = 'center_crop'

    i3d = i3_res50(400, pretrained_path)
    i3d.cuda()
    i3d.train(False)

    image_folders = natsorted(os.listdir(image_folder_path))
    print(image_folders)
    for image_folder in image_folders:
        image_path = image_folder_path + image_folder + "/"
        print(image_path)

        startime = time.time()
        features = extract_r3d_features(i3d, frequency, image_path, batch_size,
                                        sample_mode)
        np.save(r3d_feature_path + "/" + image_folder, features)
        print(f'extract R3D features {np.shape(features)} '
              f'in {time.time() - startime:.3f}s')

    save_to_pt(r3d_feature_path, r3d_feature_path_pt)  # [3, 1024, 4, 14, 14]
