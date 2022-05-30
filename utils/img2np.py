import  os
import time

import cv2
import numpy as np
from natsort import natsorted


def convert_img_to_numpy(input_path, output_path):
    images_out = []
    images = natsorted([i for i in os.listdir(input_path)])
    totle_image_number = len(images)
    images = images[0:totle_image_number:5]
    sampled_image_number = len(images)
    print(f'total {totle_image_number} images, samples {sampled_image_number} images')
    for image in images:
        temp = cv2.imread(input_path + image)
        temp = temp[:, 420:1500, :]
        images_out.append(temp)
    images_out = np.array(images_out)
    print(images_out.shape)

    np.save(output_path, images_out)


# image_folders = natsorted([i for i in os.listdir(root_image_folder)])
# print(image_folders)

image_folders = ['20210309_01', '20210309_02', '20210309_03', '20210309_04', '20210309_05']
# '20210309_06', '20210309_07', '20210309_08', '20210309_09']

for image_folder in image_folders:
    print(f'processing {image_folder} ...')

    image_path = "data/images/" + image_folder + '/'
    numpy_path = "data/images/" + image_folder
    print(image_path)
    print(numpy_path)

    start_time = time.time()
    convert_img_to_numpy(image_path, numpy_path)
    print(f'done in {time.time() - start_time:.2f}s')
