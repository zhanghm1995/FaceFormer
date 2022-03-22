'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 10:17:20
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset to get one-hot vector like official FaceFormer
'''


import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from transformers import Wav2Vec2Processor
from .base_video_dataset import BaseVideoDataset


class Face3DMMOneHotDataset(BaseVideoDataset):
    def __init__(self, split, **kwargs) -> None:
        super(Face3DMMOneHotDataset, self).__init__(split, **kwargs)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        self.one_hot_labels = np.eye(len(self.all_videos_dir))
        
    def _get_mat_vector(self, face_params_dict,
                        keys_list=['id', 'exp', 'tex', 'angle', 'gamma', 'trans']):
        """Get coefficient vector from Deep3DFace_Pytorch results

        Args:
            face_params_dict (dict): face params dictionary loaded by using loadmat function

        Returns:
            np.ndarray: (1, L)
        """

        coeff_list = []
        for key in keys_list:
            coeff_list.append(face_params_dict[key])
        
        coeff_res = np.concatenate(coeff_list, axis=1)
        return coeff_res

    def _get_face_3d_params(self, video_dir, start_idx):
        """Get face 3d params from a video and specified start index

        Args:
            video_dir (str): video name
            start_idx (int): start index

        Returns:
            Tensor: (L, C), L is the fetch length, C is the needed face parameters dimension
        """
        face_3d_params_list = []
        for idx in range(start_idx, start_idx + self.fetch_length):
            face_3d_params_path = osp.join(self.data_root, video_dir, "deep3dface", f"{idx:06d}.mat")
            
            face_3d_params = loadmat(face_3d_params_path) # dict type
            face_3d_params = self._get_mat_vector(face_3d_params, keys_list=["exp"])

            face_3d_params_list.append(torch.from_numpy(face_3d_params).type(torch.float32))
        
        face_3d_params_tensor = torch.concat(face_3d_params_list, dim=0)

        return face_3d_params_tensor

    def __getitem__(self, index):
        main_idx, sub_idx = self._get_data(index)

        one_hot = self.one_hot_labels[main_idx]
        
        choose_video = self.all_videos_dir[main_idx] # choosed video directory name, str type
        start_idx = self.all_sliced_indices[main_idx][sub_idx] # the actual index in this video

        ## Get the GT raw audio vector
        audio_seq = self._slice_raw_audio(choose_video, start_idx) # (M, )
        if audio_seq is None:
            return None
        
        audio_seq = np.squeeze(self.audio_processor(audio_seq, sampling_rate=16000).input_values)

        ## Get the GT image and GT 3D face parameters
        gt_face_3d_params_tensor = self._get_face_3d_params(choose_video, start_idx)

        data_dict = {}
        data_dict['gt_face_3d_params'] = gt_face_3d_params_tensor # (fetch_length, 64)
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32)) #(L, )
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        return data_dict


