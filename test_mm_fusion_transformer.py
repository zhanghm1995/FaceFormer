'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-24 20:58:07
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the MMFusionFormer class
'''

import torch
from models.mm_fusion_transformer import MMFusionFormer, Face3DMMFormer
from omegaconf import OmegaConf


def test_MMFusionFormer():
    config = OmegaConf.load('./config/config_2d_3d.yaml')

    device = torch.device("cuda")

    mm_fusion_former = MMFusionFormer(config, device).to(device)

    face_image = torch.randn((1, 100, 6, 224, 224)).to(device)
    face_3d_params = torch.randn((1, 100, 64)).to(device)
    raw_audio = torch.randn((1, 64000)).to(device)

    data_dict = {'raw_audio': raw_audio, 
                'input_image': face_image, 
                'ref_face_3d_params': face_3d_params}

    output_dict = mm_fusion_former(data_dict)

    for key, value in output_dict.items():
        print(key, value.shape)


def test_Face3DMMFormer():
    config = OmegaConf.load('./config/config_2d_3d.yaml')

    device = torch.device("cuda")

    model = Face3DMMFormer(config).to(device)

    face_3d_params = torch.randn((2, 100, 64)).to(device)
    encoded_audio = torch.randn((100, 2, 128)).to(device)

    data_dict = {'gt_face_3d_params': face_3d_params}

    output_dict = model(data_dict, encoded_audio)

    print(output_dict.shape)


if __name__ == "__main__":
    test_Face3DMMFormer()