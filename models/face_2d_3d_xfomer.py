'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 14:55:16
Email: haimingzhang@link.cuhk.edu.cn
Description: Face 2D-3D Cross-Modal transformer
'''

from xml.etree.ElementInclude import include
import torch.nn as nn
from torch.nn import TransformerEncoder, LayerNorm, TransformerEncoderLayer
from .image_token_encoder import ImageTokenEncoder192


class Face2D3DXFormer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        encoder_layer = TransformerEncoderLayer(config['d_model'], config['nhead'], config['d_feed_forward'])
        encoder_norm = LayerNorm(config['d_model'])
        self.encoder = TransformerEncoder(encoder_layer, config['n_layer'], encoder_norm)

        ## Define the 3D Decoder
        self.face_3d_decoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=3, stride=1,padding=1),
        )
        
        self.output_fc = nn.Linear(128, 64)

        ## Define the 2D Decoder
        self.face_2d_decoder = ImageTokenEncoder192(in_ch=3)
    
    def forward(self, x, masked_image):
        x = self.encoder(x, mask=None, src_key_padding_mask=None) # (2S, B, E)

        seq_len = int(x.shape[0] // 2)
        face_3d_embedding = x[:seq_len, ...].permute(1, 2, 0) # to (B, E, S)
        
        face_3d_params = self.output_fc(self.face_3d_decoder(face_3d_embedding).permute(0, 2, 1)) # to (B, S, E)
        
        face_2d_embedding = x[seq_len:, ...] # (S, B, E)
        face_2d_embedding = face_2d_embedding.permute(1, 0, 2) # to (B, S, E)

        face_2d_image = self.face_2d_decoder(face_2d_embedding, masked_image)

        return {'face_3d_params': face_3d_params,
                'face_2d_image': face_2d_image}

