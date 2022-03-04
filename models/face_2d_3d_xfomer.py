'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 14:55:16
Email: haimingzhang@link.cuhk.edu.cn
Description: Face 2D-3D Cross-Modal transformer
'''

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, LayerNorm, TransformerEncoderLayer
from .image_token_encoder import ImageTokenEncoder192


class Face2D3DXFormer(nn.Module):
    """The cross-modal transformer to fuse 2D and 3D face
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        ## Define the transformer encoder
        encoder_layer = TransformerEncoderLayer(config['d_model'], config['nhead'], config['d_feed_forward'])
        encoder_norm = LayerNorm(config['d_model'])
        self.encoder = TransformerEncoder(encoder_layer, config['n_layer'], encoder_norm)

        if self.config['use_3d']:
            ## Define the 3D Decoder
            # self.face_3d_decoder = nn.Sequential(
            #     nn.Conv1d(128, 256, kernel_size=3, stride=1,padding=1),
            #     nn.BatchNorm1d(256),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(256, 256, kernel_size=3, stride=1,padding=1),
            #     nn.BatchNorm1d(256),
            #     nn.Conv1d(256, 128, kernel_size=3, stride=1,padding=1),
            # )

            self.face_3d_decoder = nn.Identity()
        
            self.output_fc = nn.Linear(128, 64)
        else:
            print("[Face2D3DXFormer] Only use the 2D information")

        ## Define the 2D Decoder
        self.face_2d_decoder = ImageTokenEncoder192(in_ch=3)
    
    def forward_2d_only(self, input, masked_image: Tensor):
        face_2d_embedding = self.encoder(input, mask=None, src_key_padding_mask=None) # (2S, B, E)

        face_2d_embedding = face_2d_embedding.permute(1, 0, 2) # to (B, S, E)

        face_2d_image = self.face_2d_decoder(face_2d_embedding, masked_image)
        return {'face_2d_image': face_2d_image}

    def forward(self, x: Tensor, masked_image: Tensor):
        """Forward the network

        Args:
            x (Tensor): (T, B, E)
            masked_image (Tensor): (B, T, 3, H, W)

        Returns:
            _type_: _description_
        """
        if self.config['use_3d'] and not self.config.test_mode:
            attention_mask = None

            if self.config.use_3d_mask:
                fusion_seq_len = x.shape[0]

                seq_len = int(fusion_seq_len // 2)
                ## add the mask matrix like SAT to mask the 3D information
                attention_mask = torch.zeros((fusion_seq_len, fusion_seq_len)).to(x)

                attention_mask[-seq_len:, :seq_len] = 1.0
                attention_mask = attention_mask * -10000.0

            x = self.encoder(x, mask=attention_mask, src_key_padding_mask=None) # (2S, B, E)

            seq_len = int(x.shape[0] // 2)
            face_3d_embedding = x[:seq_len, ...].permute(1, 2, 0) # to (B, E, S)
            
            face_3d_params = self.output_fc(self.face_3d_decoder(face_3d_embedding).permute(0, 2, 1)) # to (B, S, E)
            
            face_2d_embedding = x[seq_len:, ...] # (S, B, E)
            face_2d_embedding = face_2d_embedding.permute(1, 0, 2) # to (B, S, E)

            face_2d_image = self.face_2d_decoder(face_2d_embedding, masked_image)

            return {'face_3d_params': face_3d_params,
                    'face_2d_image': face_2d_image}
        else:
            output = self.forward_2d_only(x, masked_image)
            return output

