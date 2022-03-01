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
    """The Wav2Vec 2.0 FaceFormerEncoder to process the audio data

    Args:
        nn (_type_): _description_
    """
    def __init__(self, device, video_fps=60) -> None:
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2_model = bundle.get_model().to(device)

        ## Fix the TCN
        for param in self.wav2vec2_model.feature_extractor.parameters():
            param.requires_grad = False

        encoder_out_channels, output_channels = 768, 128
        self.final_projection = nn.Linear(encoder_out_channels, output_channels)

        self.scale_factor = video_fps / 16000.0

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


class Wav2Vec2Encoder(nn.Module):
    """The Wav2Vec 2.0 FaceFormerEncoder to process the audio data

    Args:
        nn (_type_): _description_
    """
    def __init__(self, device, video_fps=60) -> None:
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2_model = bundle.get_model().to(device)

        ## Fix the TCN
        for param in self.wav2vec2_model.feature_extractor.parameters():
            param.requires_grad = False

        encoder_out_channels, output_channels = 768, 128
        self.final_projection = nn.Linear(encoder_out_channels, output_channels)

        self.scale_factor = video_fps / 16000.0

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.final_projection.weight, -initrange, initrange)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, waveforms, lengths=None):
        ## Extract the audio features
        x, lengths = self.wav2vec2_model.feature_extractor(waveforms, lengths)

        output_seq_len = int(waveforms.size(1) * self.scale_factor)
        
        ## Add the linear interpolation
        x = torch.permute(x, (0, 2, 1))
        x = nn.functional.interpolate(x, size=[output_seq_len])
        x = torch.permute(x, (0, 2, 1))

        ## Forward the transformer encoder
        x = self.wav2vec2_model.encoder(x, lengths)

        ## Add the linear projection
        x = self.final_projection(x)
        return x

class FaceFormerEncoderMixedAudio(nn.Module):
    """The Wav2Vec 2.0 FaceFormerEncoder to process the mixed audio

    Args:
        nn (_type_): _description_
    """
    def __init__(self, device, video_fps=60) -> None:
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2_model = bundle.get_model().to(device)

        ## Fix the TCN
        for param in self.wav2vec2_model.feature_extractor.parameters():
            param.requires_grad = False

        encoder_out_channels, output_channels = 768, 128
        self.final_projection = nn.Linear(encoder_out_channels * 2, output_channels)

        self.scale_factor = video_fps / 16000.0

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.final_projection.weight, -initrange, initrange)
        nn.init.zeros_(self.final_projection.bias)

    def forward(self, waveforms, lengths=None):
        """Forward the network

        Args:
            waveforms (Tensor): (B, S)
            lengths (Tensor, optional): (B, ), represent valid audio length in each sequence. Defaults to None.

        Returns:
            Tensor: (B, Sx, 128)
        """
        ## Extract the audio features
        x, lengths = self.wav2vec2_model.feature_extractor(waveforms, lengths)

        output_seq_len = int(waveforms.size(1) * self.scale_factor)
        
        ## Add the linear interpolation TODO
        x = torch.permute(x, (0, 2, 1))
        x = nn.functional.interpolate(x, size=[output_seq_len])
        x = torch.permute(x, (0, 2, 1))

        ## Forward the transformer encoder
        x = self.wav2vec2_model.encoder(x, lengths)
        
        ## Concanate the tensor in feature dimension
        B = int(x.shape[0] / 2)
        x = torch.split(x, B, dim=0)
        x = torch.cat(x, dim=-1)

        ## Add the linear projection
        x = self.final_projection(x)
        return x