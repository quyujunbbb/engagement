import os
import subprocess

import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from requests import session

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
        print('image2np')
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
    
    def crop_images(input_path, output_path, box_path):
        print('crop_images')
        boxes = pd.read_csv(box_path)
        sessions = natsorted(os.listdir(input_path))
        for i, session in enumerate(sessions):
            session_path = input_path + session + '/'
            # print(session_path)
            images = natsorted(os.listdir(session_path))
            img_1 = cv2.imread(session_path + images[200])
            img_2 = cv2.imread(session_path + images[500])
            img_3 = cv2.imread(session_path + images[800])

            box = boxes.iloc[i,:]
            center = box['x1'] + (box['x2'] - box['x1']) / 2
            x_left = int(center - 540)
            x_right = int(center + 540)
            if x_left < 0:
                x_left = 0
                x_right = 1080
            if x_right > 1920:
                x_left = 1920 - 1080
                x_right = 1920
            print(x_left, x_right)
            img_1 = img_1[:, x_left:x_right, :]
            img_2= img_2[:, x_left:x_right, :]
            img_3 = img_3[:, x_left:x_right, :]
            # cv2.imwrite(output_path + session + '_' + images[200], img_1)
            # cv2.imwrite(output_path + session + '_' + images[500], img_2)
            # cv2.imwrite(output_path + session + '_' + images[800], img_3)
    
    def session_info(input_path):
        print('session_info')
        cols=['session', 'total_frame', 'clip_num']
        info = []
        sessions = natsorted(os.listdir(input_path))
        for session in sessions:
            session_path = input_path + session + '/'
            images = natsorted(os.listdir(session_path))
            total_frame = len(images)
            clip_num = int(len(images)/32)
            info.append([session, total_frame, clip_num])
        info = pd.DataFrame(info, columns=cols)
        print(info)
        info.to_csv('data/session_info_new.csv', index=False)


if __name__ == '__main__':
    video_path = 'data/videos/'
    image_path = 'data/images/'
    box_path = 'data/crop_boxes.csv'
    # process_videos.video_info(video_path)
    # process_videos.video2images(video_path, image_path)

    # image2np
    # image_folders = ['20201222_01', '20201222_02']
    # for image_folder in image_folders:
    #     image_path = "data/images/" + image_folder + '/'
    #     numpy_path = "data/images/" + image_folder + '/'
    #     process_images.image2np(image_path, numpy_path)

    # crop_images
    # process_images.crop_images(image_path, 'data/cropped/', box_path)

    # session_info
    process_images.session_info(image_path)
