'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-21 19:14:47
Email: haimingzhang@link.cuhk.edu.cn
Description: Wav2Lip dataset loader
'''

import os
import os.path as osp
from typing import List
import numpy as np
from glob import glob
import cv2
from torch.utils.data import Dataset
import librosa

def get_all_valid_indices(total_length, fetch_length, stride) -> List:
    idx_list = list(range(0, total_length - fetch_length, stride))
    last_idx = total_length - fetch_length
    idx_list += [last_idx]
    return idx_list


class FaceImageDataset(Dataset):
    """Class to load a list of images and corresponding audio data.
       Currently just load one person all videos (means there are multiple
       image sets belongs to single person waiting for load)

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, data_root, split, **kwargs) -> None:
        super().__init__()

        self.data_root = data_root

        self.all_videos_dir = open(osp.join(data_root, f'{split}.txt')).read().splitlines()

        self.fetch_length = kwargs.get("fetch_length", 100)
        self.video_fps = kwargs.get("video_fps", 25)
        self.audio_sample_rate = kwargs.get("audio_sample_rate", 16000)

        self.build_dataset()        


    def build_dataset(self):
        self.length_token_list = []

        self.all_sliced_indices = [] # list of list

        total_length = 0
        for video_dir in self.all_videos_dir:
            all_images_path = sorted(glob(osp.join(self.data_root, video_dir, "face_image", "*.jpg")))
            num_frames = len(all_images_path)

            valid_indices = get_all_valid_indices(num_frames, self.fetch_length, stride=25)
            self.all_sliced_indices.append(valid_indices)

            total_length += len(valid_indices)
            self.length_token_list.append(total_length)
    
    def _get_data(self, index):
        """Get the seperate index location from the total index

        Args:
            index (int): index in all avaible sequeneces
        
        Returns:
            main_idx (int): index specifying which video
            sub_idx (int): index specifying what the start index in this video
        """
        def fetch_data(length_list, index):
            assert index < length_list[-1]
            temp_idx = np.array(length_list) > index
            list_idx = np.where(temp_idx==True)[0][0]
            sub_idx = index
            if list_idx != 0:
                sub_idx = index - length_list[list_idx - 1]
            return list_idx, sub_idx

        main_idx, sub_idx = fetch_data(self.length_token_list, index)
        return main_idx, sub_idx

    def __len__(self):
        return sum([len(x) for x in self.all_sliced_indices])
    
    def _slice_raw_audio(self, choose_video, sub_idx):
        audio_path = osp.join(self.data_root, choose_video, f"{osp.basename(choose_video)}.wav")
        
        start_idx, end_idx = sub_idx, sub_idx + self.fetch_length
        audio_start_idx = round(start_idx / self.video_fps * self.audio_sample_rate)
        audio_end_idx = round(end_idx / self.video_fps * self.audio_sample_rate)
        
        whole_audio_data = librosa.core.load(audio_path, sr=self.audio_sample_rate)[0]
        fetch_audio_data = whole_audio_data[audio_start_idx:audio_end_idx]
        return fetch_audio_data

    def __getitem__(self, index):
        main_idx, sub_idx = self._get_data(index)
        
        choose_video = self.all_videos_dir[main_idx]
        start_idx = self.all_sliced_indices[main_idx][sub_idx]

        audio_seq = self._slice_raw_audio(choose_video, sub_idx)

        img_list = []
        for idx in range(start_idx, start_idx + self.fetch_length):
            img_path = osp.join(self.data_root, choose_video, "face_image", f"{idx:06d}.jpg")
            img = cv2.imread(img_path)
            img_list.append(img)
        
        img_seq = np.stack(img_list)
        
        data_dict = {}
        data_dict['face_image'] = img_seq
        data_dict['raw_audio'] = audio_seq
        return data_dict


if __name__ == "__main__":
    data_root = "/home/haimingzhang/Research/Face/FACIAL/video_preprocessed/id00001"
    split = "train"
    dataset = FaceImageDataset(data_root, split)
    print(len(dataset))

    data = dataset[180]
    print(data.shape)


