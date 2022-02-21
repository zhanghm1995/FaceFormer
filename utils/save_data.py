'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-20 23:44:25
Email: haimingzhang@link.cuhk.edu.cn
Description: Some useful functions to save data
'''
import os.path as osp
import tempfile
import cv2
import subprocess
from tqdm import tqdm


def save_video(image, output_video_fname, image_size=512, audio_fname=None):
    print("================== Start create the video =================================")
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=osp.dirname(output_video_fname))
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (image_size, image_size), True)
    for idx in tqdm(range(len(image))):
        writer.write(image[idx][..., ::-1])
    writer.release()

    print("================== Generate the final video with audio signal =====================")
    if audio_fname is not None:
        cmd = f'ffmpeg -y -i {audio_fname} -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video_fname}'
    else:
        cmd = f'ffmpeg -y -i {tmp_video_file.name} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {output_video_fname}'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Save video done!")