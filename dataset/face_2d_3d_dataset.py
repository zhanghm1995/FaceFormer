'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-25 09:01:05
Email: haimingzhang@link.cuhk.edu.cn
Description: Load 2D and 3D face dataset
'''

import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat
from .face_image_dataset import FaceImageDataset

class Face2D3DDataset(FaceImageDataset):
    def __init__(self, data_root, split, **kwargs) -> None:
        super(Face2D3DDataset, self).__init__(data_root, split)

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
            if key == 'trans':
                face_params_dict[key][:, ...] = 0.
            coeff_list.append(face_params_dict[key])
        
        coeff_res = np.concatenate(coeff_list, axis=1)
        return coeff_res

    def _get_face_3d_params(self, video_dir, start_idx):
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
        
        choose_video = self.all_videos_dir[main_idx] # choosed video directory name, str type
        start_idx = self.all_sliced_indices[main_idx][sub_idx]

        audio_seq = self._slice_raw_audio(choose_video, sub_idx)

        ## Get the GT image
        gt_img_seq_tensor = self._read_image_sequence(choose_video, start_idx)

        ## Get the reference image
        ref_img_seq_tensor = self._get_reference_image(
            self.total_frames_list[main_idx], choose_video, start_idx)

        ## Get the 3D face parameters
        face_3d_params_tensor = self._get_face_3d_params(choose_video, start_idx)
        
        data_dict = {}
        data_dict['gt_face_image'] = gt_img_seq_tensor
        data_dict['ref_face_image'] = ref_img_seq_tensor
        data_dict['face_3d_params'] = face_3d_params_tensor
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32))
        return data_dict

