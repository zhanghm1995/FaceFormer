'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-24 20:58:07
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the MMFusionFormer class
'''

import torch
from models.mm_fusion_transformer import MMFusionFormer
from omegaconf import OmegaConf


config = OmegaConf.load('./config/config_2d_3d.yaml')

device = torch.device("cuda")

mm_fusion_former = MMFusionFormer(config, device).to(device)

face_image = torch.randn((2, 100, 6, 96, 96)).to(device)
face_3d_params = torch.randn((2, 100, 64)).to(device)
raw_audio = torch.randn((2, 64000)).to(device)

data_dict = {'raw_audio': raw_audio, 
             'input_image': face_image, 
             'face_3d_params': face_3d_params}

output_dict = mm_fusion_former(data_dict)

for key, value in output_dict.items():
    print(key, value.shape)