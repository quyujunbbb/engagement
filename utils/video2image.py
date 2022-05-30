import shutil
import time
from pathlib import Path

import ffmpeg
from natsort import natsorted


if __name__ == "__main__":
    print("Extract images for videos")

    video_folder_path = 'data/videos/'
    image_folder_path = 'data/images/'
    videos = natsorted(
        [str(f) for f in Path(video_folder_path).glob('**/*.mp4')])
    print(videos)

    for video in videos:
        video_name = video.split("/")[-1].split(".")[0]
        image_path = image_folder_path + video_name + "/"
        print(image_path)
        startime = time.time()
        print(f'Extracting images for {video_name}')
        Path(image_path).mkdir(parents=True, exist_ok=True)
        ffmpeg.input(video).output('{}%d.jpg'.format(image_path),
                                   start_number=0).global_args(
                                       '-loglevel', 'quiet').run()
        # shutil.rmtree(image_path)
        print(f'Done in {time.time() - startime:.3f}s')
