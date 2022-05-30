import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from retinaface import RetinaFace


def main():
    image_folder = "data/images/20210105_08/"
    face_folder = "data/faces/20210105_08/"
    Path(face_folder).mkdir(parents=True, exist_ok=True)
    images = natsorted(os.listdir(image_folder))

    for image in images:
        image_name = image.split('.')[0]
        image_path = image_folder + '/' + image
        print(f'processing {image_name} ...')

        resp = RetinaFace.detect_faces(image_path)
        faces = RetinaFace.extract_faces(img_path=image_path, align=True)
        idx = 0
        for i, face in enumerate(faces):
            if face.shape[0] * face.shape[1] >= 6000:
                # plt.imshow(face)
                # plt.show()
                cv2.imwrite(face_folder + f'face_{image_name}_{idx}.jpg',
                            cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                idx += 1

    # resp = pd.DataFrame.from_dict(resp)
    # print(resp)
    # print(resp.iloc[:, 0])


if __name__ == "__main__":
    main()
