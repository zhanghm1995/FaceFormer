'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-04 19:22:49
Email: haimingzhang@link.cuhk.edu.cn
Description: Use to load dataset for testing
'''

import cv2
import os
import os.path as osp
from glob import glob
import torch
from torch import Tensor
from torch.utils.data import Dataset
import librosa
import numpy as np
from typing import List
import torchvision.transforms as transforms


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


class Face2D3DTestDataset(Dataset):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.audio_path = config['audio_path']
        self.video_path = config['video_path']
        self.face_3d_params_path = config['face_3d_params_path']

        self.fetch_length = kwargs.get("fetch_length", 75)

        self.target_image_size = (192, 192)

        ## Define the image transformation operations
        transform_list = [transforms.ToTensor()]
        self.image_transforms = transforms.Compose(transform_list)

        self._build_dataset()
        
    def _build_dataset(self):
        ## 1)  Read the audio data
        self.driven_audio_data = librosa.core.load(self.audio_path, sr=16000)[0]

        self.audio_stride = round(16000 * self.fetch_length / 25)
        self.audio_chunks = range(0, len(self.driven_audio_data), self.audio_stride)

        if osp.isdir(self.video_path):
            self.all_images_path = sorted(glob(osp.join(self.video_path, "*.jpg")))
        else:
            raise NotImplementedError
        
        self.frame_chunks = range(0, len(self.all_images_path), self.fetch_length)

        if self.face_3d_params_path is not None:
            self.face_3d_prams = np.load(open(self.face_3d_params_path, 'rb'))['face']

        assert len(self.audio_chunks) <= len(self.frame_chunks)

    def _read_image_sequence(self, image_path_list: List):
        img_list = []
        for img_path in image_path_list:
            img = cv2.resize(cv2.imread(img_path), self.target_image_size)
            img = self.image_transforms(img)
            
            img_list.append(img)
        
        img_seq_tensor = torch.stack(img_list) # to (T, 3, H, W)
        return img_seq_tensor

    def __len__(self):
        return len(self.audio_chunks)
    
    def __getitem__(self, index):
        ## 1) Read the audio
        audio_start_idx = self.audio_chunks[index]

        audio_seq = self.driven_audio_data[audio_start_idx: audio_start_idx + self.audio_stride]

        actual_frame_lenth = int(len(audio_seq) / 16000 * 25)

        frame_start_dix = self.frame_chunks[index]
        image_seq_tensor = self._read_image_sequence(
            self.all_images_path[frame_start_dix: frame_start_dix + actual_frame_lenth])
        
        data_dict = {}
        data_dict['gt_face_image'] = image_seq_tensor
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32))
        if hasattr(self, "face_3d_prams"):
            face_3d_params_tensor = torch.from_numpy(
                self.face_3d_prams[frame_start_dix: frame_start_dix + actual_frame_lenth]).type(torch.float32)
            data_dict['gt_face_3d_params'] = face_3d_params_tensor
        else:
            data_dict['gt_face_3d_params'] = torch.zeros((actual_frame_lenth, 64)).type(torch.float32)

        return data_dict
