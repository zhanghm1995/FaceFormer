'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-20 14:28:58
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the VOCA dataset visualization
'''

import os
import os.path as osp
import torch
import tempfile
from tqdm import tqdm
import cv2
import numpy as np
import subprocess
from dataset import get_dataset
from utils.pytorch3d_renderer import FaceRenderer
from scipy.io import wavfile


# Set our device:
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


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


def visualize_voca_GT(config):
    face_renderer = FaceRenderer(device, rendered_image_size=512)

    batcher = get_dataset(config)
    data_list = batcher.get_training_sequences_in_order(2)
    print(len(data_list))

    for idx, data_dict in enumerate(data_list):
        face_vertex = torch.from_numpy(data_dict['face_vertices']).to(dtype=torch.float32)
        print(face_vertex.shape)

        rendered_image = []
        for i in tqdm(range(0, len(face_vertex), 100)):
            sub_face_vertex = face_vertex[i:i+100].to(device)
            image_array = face_renderer(sub_face_vertex)
            rendered_image.append(image_array)

        rendered_image = np.concatenate(rendered_image, axis=0)
        print(rendered_image.shape)

        output_video_fname = osp.join("./output", f"{idx}.mp4")
        print(output_video_fname)

        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(output_video_fname))
        wavfile.write(tmp_audio_file.name, 22000, data_dict['raw_audio'])

        save_video(rendered_image, output_video_fname, image_size=512, audio_fname=tmp_audio_file.name)


def main():
    from omegaconf import OmegaConf

    config = OmegaConf.load('./config/config.yaml')
    visualize_voca_GT(config['dataset'])


if __name__ == "__main__":
    main()