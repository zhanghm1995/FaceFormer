'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-02 19:50:17
Email: haimingzhang@link.cuhk.edu.cn
Description: Inference scripts to get demo
'''

import argparse
from omegaconf import OmegaConf
import numpy as np
import os
import cv2
import os.path as osp
from utils.post_process import merge_face_contour_only
from tqdm import tqdm
import librosa
import subprocess
import tempfile
from utils.face_detector import detect_face_from_one_image


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/config_2d_3d_fusion_gan.yaml', help='the config_file')
    parser.add_argument('--video_path', type=str, default="/home/haimingzhang/Research/Programming/cv-fighter/facial_preprocessed/obama_weekly_25fps/obama_weekly_023.mp4", help='input video path')
    parser.add_argument('--driven_audio_path', type=str, default="./data/audio_samples/slogan_english_16k.wav", help='the driven audio path')
    parser.add_argument('--output_dir', type=str, default="./", help='log location')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')

    args = parser.parse_args()
    config = OmegaConf.load(args.cfg)
    config.update(vars(args)) # override the configuration using the value in args

    return config


def get_frames_from_video(video_path, num_need_frames=-1):
    """Read all frames from a video file

    Args:
        video_path (str): video file path

    Returns:
        list: including all images in OpenCV BGR format with HxWxC size
    """
    video_stream = cv2.VideoCapture(video_path)

    frames = []
    if num_need_frames < 0:
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
    else:
        num_count = 0
        while num_count < num_need_frames:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
            num_count += 1

    return frames


def get_model(cfg):
    pass

def crop_face(input_image, face_coord, crop_squared=True):
    img_H, img_W = input_image.shape[:2]

    face_H = face_coord[3] - face_coord[1]
    # face_coord[3] += face_H / 10
    # face_coord[1] -= face_H / 5.0

    face_coord[3] += face_H / 10
    face_coord[1] -= face_H / 8.0

    face_coord[3] = np.clip(face_coord[3], 0, img_H)
    face_coord[1] = np.clip(face_coord[1], 0, img_H)

    x1, y1, x2, y2 = face_coord

    if crop_squared:
        orig_H, orig_W = y2 - y1, x2 - x1
        size_diff = abs(orig_H - orig_W)

        H_padding, W_padding = 0, 0
        if orig_H > orig_W:
            W_padding = size_diff // 2
        else:
            H_padding = size_diff // 2
        
        ## Get a squared face
        x1, x2 = x1 - W_padding, x2 + W_padding
        y1, y2 = y1 - H_padding, y2 + H_padding
    
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    return input_image[y1:y2, x1:x2], (x1, y1, x2, y2)


def get_face_coords(input_image, resize_factor=1):
    if resize_factor > 1:
        image_src = cv2.resize(input_image, (input_image.shape[1]//resize_factor, input_image.shape[0]//resize_factor))
    else:
        image_src = input_image
    
    face_coord = detect_face_from_one_image(image_src)

    if face_coord is None:
       return None
    
    if resize_factor > 1:
        face_coord = tuple(resize_factor * x for x in face_coord)

    face_coord = np.clip(face_coord, 0, None)
    return face_coord


def detect_face(frames, resize_factor=4):
    face_image, face_coords = [], []

    for i, frame in tqdm(enumerate(frames)):
        face_coord = get_face_coords(frame, resize_factor=resize_factor)

        if face_coord is None:
            print(f"We cannot detect any face in frame {i}!")
            has_problem = True
            break

        face, coords = crop_face(frame, face_coord, crop_squared=True)

        face_image.append(cv2.resize(face, (224, 224)))
        face_coords.append(coords)

    return face_image, face_coords 

def read_audio(audio_path):
    real_audio_path = None

    if audio_path.endswith(".wav"):
        real_audio_path = audio_path
    elif audio_path.endswith((".mp4", ".avi")):
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav')

        command = f"ffmpeg -y -i {audio_path} -acodec pcm_s16le -f wav -ac 1 -ar 16000 {tmp_audio_file.name}"
        subprocess.call(command, shell=True, stdout=subprocess.DEVNULL)
        real_audio_path = tmp_audio_file.name

    audio_data =  librosa.core.load(real_audio_path, sr=16000)[0]
    return audio_data


def main(cfg):
    ## 1) Load the audio
    driven_audio_data = read_audio(audio_path=cfg.driven_audio_path)
    
    driven_audio_data = driven_audio_data[:64000]
    
    ## 2) Load the video
    all_frames = get_frames_from_video(cfg.video_path, num_need_frames=100)
    
    ## 3) Detect the face
    face_image, face_coords = detect_face(all_frames)
    
    ## 4) Forward the network



if __name__ == "__main__":
    cfg = parse_config()

    main(cfg)