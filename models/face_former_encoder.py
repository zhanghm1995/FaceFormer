'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-22 08:49:37
Email: haimingzhang@link.cuhk.edu.cn
Description: The FaceFormer encoder modified from Wav2Vec 2.0.
'''

import math
import torch
import torch.nn as nn
import torchaudio


class FaceFormerEncoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2_model = bundle.get_model().to(device)

        ## Frozen the TCN
        for param in self.wav2vec2_model.feature_extractor.parameters():
            param.requires_grad = False

        encoder_out_channels, output_channels = 768, 128
        self.final_projection = nn.Linear(encoder_out_channels, output_channels)

        self.scale_factor = 60 / 16000.0

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.final_projection.weight, -initrange, initrange)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, waveforms, lengths=None):
        ## Extract the audio features
        x, lengths = self.wav2vec2_model.feature_extractor(waveforms, lengths)

        output_seq_len = int(waveforms.size(1) * self.scale_factor)
        
        ## Add the linear interpolation TODO
        x = torch.permute(x, (0, 2, 1))
        x = nn.functional.interpolate(x, size=[output_seq_len])
        x = torch.permute(x, (0, 2, 1))

        ## Forward the transformer encoder
        x = self.wav2vec2_model.encoder(x, lengths)

        ## Add the linear projection
        x = self.final_projection(x)
        return x