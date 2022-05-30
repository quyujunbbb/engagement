import os
import subprocess

import cv2
from loguru import logger
from natsort import natsorted


logger.add('video-info.log')


def get_video_info(video_path):
    print('Video Info')
    videos = natsorted(os.listdir(video_path))

    for video in videos:
        vidcap = cv2.VideoCapture(video_path + video)

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        minutes = int(duration / 60)
        seconds = duration % 60
        logger.info(f'| {video.split(".")[0]} | {frame_count:5d} | {duration:7.2f} |')

        vidcap.release()


def split_single_video(video_path):
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ss", "00:00:22",
        "-to", "00:12:15",
        "-c", "copy",
        f"{20210309_02}.mp4"
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)


def split_multiple_videos(video_path):
    video_list = ["20210309_01", "20210309_02", "20210309_03", "20210309_04",
                  "20210309_05", "20210309_06", "20210309_07", "20210309_08",
                  "20210309_09"]
    clp_num = [117, 135, 69, 216, 41, 110, 125, 39, 121]

    for i, video in enumerate(video_list):
        print(f'Extracting images for {video}')

        for clp in range(clp_num[i]):
            clp_begin = clp * 32 / 3
            clp_end = clp_begin + 32 / 3
            cmd = [
                "ffmpeg",
                "-i", f"{video_path}/{video}.mp4",
                "-ss", f"{clp_begin}",
                "-to", f"{clp_end}",
                "-c", "copy",
                f"{video}_{clp:02}.mp4"
            ]
            print(" ".join(cmd))
            subprocess.run(cmd)


if __name__ == "__main__":
    video_path = "data/videos/"
    get_video_info(video_path)

    # split_single_video(video_path + "20210309_02.mp4")
    # split_multiple_videos(video_path)
