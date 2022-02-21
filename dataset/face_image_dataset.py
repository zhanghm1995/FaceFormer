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
    def __init__(self, data_root, data_infos, split, **kwargs) -> None:
        super().__init__()
        
        self.all_videos_dir = open(osp.join(data_root, f'{split}.txt')).read().splitlines()

        self.fetch_length = 100        


    def build_dataset(self):
        self.length_token_list = []

        self.all_sliced_indices = [] # list of list

        total_length = 0
        for video_dir in self.all_videos_dir:
            all_images_path = sorted(glob(osp.join(video_dir, "face_image", "*.jpg")))
            num_frames = len(all_images_path)

            valid_indices = get_all_valid_indices(num_frames, self.fetch_length, strid=25)
            self.all_sliced_indices.append(valid_indices)

            total_length += len(valid_indices)
            self.length_token_list.append(total_length)
    
    def _get_data(self, index):
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
        return len(self.all_sliced_indices)

    def __getitem__(self, index):
        main_idx, sub_idx = self._get_data(index)
        
        choose_video = self.all_videos_dir[main_idx]
        start_idx = self.all_sliced_indices[main_idx][sub_idx]

        img_list = []
        for idx in range(start_idx, start_idx + self.fetch_length):
            img_path = osp.join(choose_video, "face_image", f"{idx:06d}.jpg")
            img = cv2.imread(img_path)
            img_list.append(img)
        
        img_seq = np.stack(img_seq)
        return img_seq
            


        

