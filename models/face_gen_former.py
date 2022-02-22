'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-21 22:10:00
Email: haimingzhang@link.cuhk.edu.cn
Description: Face generation transformer
'''

import math
from turtle import forward
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Callable
from torch.nn import TransformerDecoder, LayerNorm, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_
import torchaudio
import torchaudio.models.wav2vec2 as ta_wav2vec2
from .image_token_encoder import ImageTokenEncoder
from .pos_encoder import PositionalEncoding
from .face_former_encoder import FaceFormerEncoder


class FaceGenFormerDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        config = config['face_gen_former_decoder']
        ## Build the decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config['d_model'], nhead=config['n_head'], dim_feedforward=config['d_feed_forward'],
            batch_first=False)
        decoder_norm = LayerNorm(config['d_model'])
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config['n_layer'], norm=decoder_norm)

        self.pos_encoder = PositionalEncoding(config['d_model'])

        ## The final output decoder
        self.output_decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5023*3),
            nn.Sigmoid()
        )

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        # 1) positional encoding
        tgt_embedding = self.pos_encoder(tgt)
        
        # 2) transformer
        decoder_output = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        # 3) motion decoder
        output = self.output_decoder(decoder_output)
        return output



class FaceGenFormer(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()

        self.encoder = FaceFormerEncoder(device)

        self.decoder = FaceGenFormerDecoder(config)

        self.image_token_encoder_decoder = ImageTokenEncoder()
    
    def encode(self, x: Tensor, lengths=None, sample_rate=16000):
        """_summary_

        Args:
            x (Tensor): (B, Sx)

        Returns:
            Tensor: (Sx, B, E)
        """
        if sample_rate != 16000:
            ## resample the audio sample rate
            x = torchaudio.functional.resample(x, 22000, 16000)
            if lengths is not None:
                # rescale the lengths
                lengths = (lengths * 16000 / 22000.0).to(lengths.dtype)
        
        enc_output = self.encoder(x, lengths)
        return enc_output.permute(1, 0, 2)

    def _generate_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def decode(self, y: Tensor, encoded_x: Tensor,
               tgt_lengths=None, shift_target_tright=True) -> Tensor:
        """_summary_

        Args:
            y (Tensor): (B, Sy, C)
            encoded_x (Tensor): (Sx, B, E)

        Returns:
            Tensor: (Sy, B, E)
        """
        ## facial motion target decoder
        y = y.permute(1, 0, 2) # to (Sy, B, ...)
        
        if shift_target_tright:
            y = self._generate_shifted_target(y)

        tgt_mask = self._generate_subsequent_mask(len(y)).to(y.device) # (Sy, B, C)

        tgt_key_padding_mask = self._generate_key_mapping_mask(y, tgt_lengths)
        output = self.decoder(y, encoded_x, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)

        return output, tgt_key_padding_mask

    def _gen_image(self, input: Tensor):
        output_image = self.image_token_encoder_decoder.decode(input)
        return output_image
    
    def _get_image_tokens(self, input: Tensor):
        """Encoding input image sequences to embedding image tokens

        Args:
            input (Tensor): (B, Sy, 3, H, W)

        Returns:
            Tensor: (B, Sy, C)
        """
        image_tokens = self.image_token_encoder_decoder.encode(input)
        return image_tokens
        
    def forward(self, data_dict):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, 16000)
        encoded_x = self.encode(audio_seq, lengths=data_dict['raw_audio_lengths'])

        ## 2) Get image tokens
        image_tokens = self._get_image_tokens(data_dict['face_image'])

        ## 3) Image Transformer Decoder
        image_tokens = image_tokens
        output, output_mask = self.decode(image_tokens, encoded_x,
                                          trg_lengths=data_dict['tgt_lengths']) # output: (Sy, B, C)
                                          
        ## 4) Image Generation
        output = torch.permute(output, (1, 0, 2)) # to (B, Sy, C)

        image_output = self._gen_image(output) # to (B, Sy, 3, H, W)
        return image_output
