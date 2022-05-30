import os
import subprocess

import cv2
import numpy as np
from natsort import natsorted

class process_videos():

    def video_info(video_path):
        print('video_info')
        videos = natsorted(os.listdir(video_path))
        for video in videos:
            video_name = video.split('.')[0]
            vidcap = cv2.VideoCapture(video_path + video)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            print(f'{video_name}: {frame_count:5d}frms, {duration:6.1f}s')
            vidcap.release()

    def split_single_video(video_path):
        print('split_single_video')
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', '00:00:22',
            '-to', '00:12:15',
            '-c', 'copy',
            f'{20210309_02}.mp4'
        ]
        print(' '.join(cmd))
        subprocess.run(cmd)

    def split_multiple_videos(video_path):
        print('split_multiple_videos')
        video_list = ['20210309_01', '20210309_02', '20210309_03']
        clp_num = [117, 135, 69]

        for i, video in enumerate(video_list):
            print(f'Extracting images for {video}')

            for clp in range(clp_num[i]):
                clp_begin = clp * 32 / 3
                clp_end = clp_begin + 32 / 3
                cmd = [
                    'ffmpeg',
                    '-i', f'{video_path}/{video}.mp4',
                    '-ss', f'{clp_begin}',
                    '-to', f'{clp_end}',
                    '-c', 'copy',
                    f'{video}_{clp:02}.mp4'
                ]
                print(' '.join(cmd))
                subprocess.run(cmd)

    def video2images(video_folder_path, image_folder_path):
        print('video2images')
        videos = natsorted(os.listdir(video_folder_path))
        for video in videos:
            video_name = video.split('.')[0]
            image_path = image_folder_path + video_name + '/'
            os.makedirs(image_path, exist_ok=True)
            print(f'{video} --> {image_path}')

            vidcap = cv2.VideoCapture(video_folder_path + video)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(f'{image_path}{count}.jpg', image)
                success, image = vidcap.read()
                count += 1


class process_images():

    def image2np(input_path, output_path):
        images_out = []
        images = natsorted(os.listdir(input_path))
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


if __name__ == '__main__':
    video_path = 'data/videos/'
    image_path = 'data/images_new/'
    process_videos.video_info(video_path)
    process_videos.video2images(video_path, image_path)

    image_folders = ['20201222_01', '20201222_02']
    for image_folder in image_folders:
        image_path = "data/images/" + image_folder + '/'
        numpy_path = "data/images/" + image_folder + '/'
        process_images.image2np(image_path, numpy_path)
