import os
import subprocess

import cv2
import numpy as np
import pandas as pd
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
    
    def session_info(input_path):
        print('session_info')
        cols=['session', 'total_frame', 'sampled_frame', 'clip_num']
        info = []
        sessions = natsorted(os.listdir(input_path))
        for session in sessions:
            session_path = input_path + session + '/'
            images = natsorted(os.listdir(session_path))
            total_frame = len(images)
            clip_num = int(len(images)/32)
            sampled_frame = clip_num * 32
            info.append([session, total_frame, sampled_frame, clip_num])
        info = pd.DataFrame(info, columns=cols)
        print(info)
        # info.to_csv('data/session_info.csv', index=False)
        print(info.sum())

    def select_images():
        print('select_images')
        input_path = 'images/'
        output_path = 'selected_images/'
        folders = natsorted(os.listdir(input_path))
        for folder in folders:
            input_folder_path = input_path + folder + '/'
            output_folder_path = output_path + folder + '/'
            os.makedirs(output_folder_path, exist_ok=True)
            imgs = natsorted(os.listdir(input_folder_path))
            img1 = cv2.imread(output_folder_path + imgs[100])
            img2 = cv2.imread(output_folder_path + imgs[300])
            img3 = cv2.imread(output_folder_path + imgs[500])
            img4 = cv2.imread(output_folder_path + imgs[700])
            img5 = cv2.imread(output_folder_path + imgs[900])
            cv2.imwrite(output_folder_path + imgs[100], img1)
            cv2.imwrite(output_folder_path + imgs[300], img2)
            cv2.imwrite(output_folder_path + imgs[500], img3)
            cv2.imwrite(output_folder_path + imgs[700], img4)
            cv2.imwrite(output_folder_path + imgs[900], img5)

    def crop_images(input_path, output_root_path, crop_boxes_path):
        print('crop_images:')
        boxes = pd.read_csv(crop_boxes_path)
        sessions = natsorted(os.listdir(input_path))
        sessions = ['20210309_04', '20210309_05', '20210309_06', '20210309_07', '20210309_08', '20210309_09']

        for i, session in enumerate(sessions):
            session_path = input_path + session + '/'
            output_path = output_root_path + session + '/'
            os.makedirs(output_path, exist_ok=True)
            print(f'{i+1}/{len(sessions)}: {session_path} --> {output_path}')

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

            images = natsorted(os.listdir(session_path))
            for image in images:
                img = cv2.imread(session_path + image)
                img = img[:, x_left:x_right, :]
                cv2.imwrite(output_path + image, img)

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


class process_bytetrack():

    def process_bounding_boxes(bounding_box_path):
        print('process_bounding_boxes:')

        box_files = natsorted(os.listdir(bounding_box_path))
        clos = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']

        for box_file in box_files:
            box_file_path = bounding_box_path + box_file
            boxes = pd.read_csv(box_file_path, names=clos)
            print(box_file_path)
            print(boxes['id'].value_counts())
            print(len(boxes))
            print()

            # --------------------------------------------------------------
            # delete id count < thr
            # frame_num = boxes['frame'].max()
            # thr = int(frame_num * 0.1)
            # print(f'frame_num {frame_num}, thr {thr}')
            # boxes = boxes.groupby('id').filter(lambda x : len(x)>thr).reset_index(drop=True)
            # print(boxes['id'].value_counts())
            # print(len(boxes))
            # print()

            # --------------------------------------------------------------
            # delete rows based on drop_ids
            # drop_ids = []
            # for drop_id in drop_ids:
            #     boxes = boxes.drop(boxes[boxes['id'] == drop_id].index)
            # print(boxes['id'].value_counts())
            # print(len(boxes))
            # print()

            # --------------------------------------------------------------
            # change id
            # boxes['id'][boxes['id'] == 5] = 0
            # boxes['id'][boxes['id'] == 2] = 1
            # boxes['id'][boxes['id'] == 10] = 2
            # print(boxes['id'].value_counts())
            # print(len(boxes))
            # print()
            # boxes.to_csv(box_file_path, index=False, header=False)


if __name__ == '__main__':
    video_path = 'data/videos/'
    image_path = 'data/images/'
    cropped_image_path = 'data/images_crop/'
    crop_boxes_path = 'data/crop_boxes.csv'
    bounding_box_path = 'features/bytetrack/boxes/'

    # --------------------------------------------------------------------------
    # process_videos.video_info(video_path)
    # process_videos.video2images(video_path, image_path)

    # --------------------------------------------------------------------------
    process_images.select_images(image_path, cropped_image_path)

    # image2np
    # image_folders = ['20201222_01', '20201222_02']
    # for image_folder in image_folders:
    #     image_path = "data/images/" + image_folder + '/'
    #     numpy_path = "data/images/" + image_folder + '/'
    #     process_images.image2np(image_path, numpy_path)

    # crop_images
    # process_images.crop_images(image_path, cropped_image_path, crop_boxes_path)
    # img = cv2.imread('data/images_crop/20201222_01/xxxxx.jpg')
    # print(img.shape)
    # img = img[355:1106, 237:440, :]  # [y1:y2, x1:x2, :]
    # cv2.imwrite('xxxxx.jpg', img)

    # session_info
    # process_images.session_info(image_path)

    # --------------------------------------------------------------------------
    # bounding box
    # process_bytetrack.process_bounding_boxes(bounding_box_path)
