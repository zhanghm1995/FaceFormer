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


class BFMModel(object):
    def __init__(self, BFM_model_path) -> None:
        model = loadmat(BFM_model_path)
        self.mean_shape = model['meanshape'].astype(np.float32) # [3*N,1]
        self.id_base = model['idBase'].astype(np.float32) # [3*N,80]
        self.exp_base = model['exBase'].astype(np.float32) # [3*N,64]
        self.key_point = np.squeeze(model['keypoints']).astype(np.int64) - 1

    def compute_shape(self, id_coeff=None, exp_coeff=None):
        """Compute the complete 3D face shape

        Args:
            id_coeff (np.ndarray): (1, 80)
            exp_coeff (np.ndarray): (1, 64)

        Returns:
            np.ndarray: (B, 3*N)
        """
        if id_coeff is None:
            id_coeff = np.zeros((1, 80)).astype(self.id_base.dtype)
        if exp_coeff is None:
            exp_coeff = np.zeros((1, 64)).astype(self.exp_base.dtype)

        id_info = id_coeff @ self.id_base.T
        exp_info = exp_coeff @ self.exp_base.T
        face_shape = self.mean_shape.reshape([1, -1]) + id_info + exp_info

        return face_shape


class Face3DMMOneHotDataset(BaseVideoDataset):
    def __init__(self, split, **kwargs) -> None:
        super(Face3DMMOneHotDataset, self).__init__(split, **kwargs)
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        # self.one_hot_labels = np.eye(len(self.all_videos_dir))
        self.one_hot_labels = np.eye(8)
        self.facemodel = BFMModel("./data/BFM/BFM_model_front.mat")
        
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
            np.ndarray: (L, C), L is the fetch length, C is the needed face parameters dimension
        """
        face_3d_params_list = []
        for idx in range(start_idx, start_idx + self.fetch_length):
            face_3d_params_path = osp.join(self.data_root, video_dir, "deep3dface", f"{idx:06d}.mat")
            
            face_3d_params = loadmat(face_3d_params_path) # dict type
            face_3d_params = self._get_mat_vector(face_3d_params, keys_list=["exp"])

            face_3d_params_list.append(face_3d_params)
        
        face_3d_params_arr = np.concatenate(face_3d_params_list, axis=0)

        return face_3d_params_arr

    def _get_template(self, choose_video):
        ## Assume the first frame is the template face
        video_path = osp.join(self.data_root, choose_video)

        template_face = np.load(osp.join(video_path, "template_face.npy"))
        id_coeff = np.load(osp.join(video_path, "id_coeff.npy"))
        return template_face, id_coeff
        
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

        ## Get the GT 3D face parameters
        gt_face_3d_params_arr = self._get_face_3d_params(choose_video, start_idx)

        ## Get the template info
        template_face, id_coeff = self._get_template(choose_video)

        ## Get the GT 3D face vertex ()
        gt_face_3d_vertex = self.facemodel.compute_shape(
            id_coeff=id_coeff, exp_coeff=gt_face_3d_params_arr)

        data_dict = {}
        data_dict['gt_face_3d_params'] = torch.from_numpy(gt_face_3d_params_arr.astype(np.float32)) # (fetch_length, 64)
        data_dict['raw_audio'] = torch.tensor(audio_seq.astype(np.float32)) #(L, )
        data_dict['one_hot'] = torch.FloatTensor(one_hot)
        data_dict['template'] = torch.FloatTensor(template_face.reshape((-1))) # (N,)
        data_dict['face_vertex'] = torch.FloatTensor(gt_face_3d_vertex)
        data_dict['video_name']  = choose_video
        return data_dict

