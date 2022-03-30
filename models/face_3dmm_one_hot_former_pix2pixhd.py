'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-30 00:34:27
Email: haimingzhang@link.cuhk.edu.cn
Description: The End-to-End one hot 3DMM transformer with Pix2PixHD generator
'''


import torch
import torch.nn as nn
from .face_3dmm_one_hot_former import Face3DMMOneHotFormer


class Face3DMMOneHotFormerPix2PixHD(nn.Module):
    def __init__(self, config, **kwargs):
        super(Face3DMMOneHotFormerPix2PixHD, self).__init__()

        ## Define the 3DMMFormer model
        self.face_3dmm_model = Face3DMMOneHotFormer(config)

        ## Define the Renderer
        self.renderer = None

        ## Define the Pix2PixHD network
        self.pix2pix_model = None


    def forward(self):
        pass
