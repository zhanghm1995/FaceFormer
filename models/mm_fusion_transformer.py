'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-24 19:13:47
Email: haimingzhang@link.cuhk.edu.cn
Description: Multi-Modal fusion transformer implementation to complete the talking face generation task
'''

from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import os
import os.path as osp
import torchaudio
from typing import Optional, Any, Union, Callable
from torch.nn import TransformerDecoder, LayerNorm, TransformerDecoderLayer
from .face_former_encoder import FaceFormerEncoder
from .image_token_encoder import ImageTokenEncoder
from .pos_encoder import PositionalEncoding


class MMFusionDecoder(nn.Module):
    """Multi-Modal fucion decoder class to fusion the 2D and 3D input tokens and audio
       features.
    """
    def __init__(self, config) -> None:
        super().__init__()

        config = config['mm_fusion_decoder']
        assert config is not None

        ## Build the decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config['d_model'], nhead=config['n_head'], dim_feedforward=config['d_feed_forward'],
            batch_first=False)
        decoder_norm = LayerNorm(config['d_model'])
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config['n_layer'], norm=decoder_norm)

        ## Change to 512-d
        self.output_fc = nn.Linear(config['d_model'], 512)

    def forward(self, tgt_embedding: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        
        decoder_output = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.output_fc(decoder_output)
        return output


class Face3DMMFormer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        d_input_3d_params = config['d_input_3d_params']
        d_embedding_model = config['d_model']

        self.input_encoder = nn.Linear(d_input_3d_params, d_embedding_model)

        self.output_encoder = nn.Linear(d_embedding_model, d_input_3d_params)

        self.pos_encoder = PositionalEncoding(d_embedding_model)

    def forward(self):
        pass

    def encode_embedding(self, input_seq: Tensor, add_positional_encoding=True):
        """Encode the raw face 3DMM sequence parameters into embeddings

        Args:
            input_seq (Tensor): (B, Sy, C)
            add_positional_encoding (bool, optional): whether adding positional encoding operation. Defaults to True.

        Returns:
            Tensor: (B, Sy, E)
        """
        ## Get the embeddings
        embedding = self.input_encoder(input_seq)
        
        ## Add the positional encoding
        if add_positional_encoding:
            embedding = self.pos_encoder(embedding)
        return embedding
    
    def decode_embedding(self, input_seq):
        ## Get the embeddings
        output = self.output_encoder(input_seq)
        
        return output
        

class MMFusionFormer(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()

        ## Define the audio encoder
        self.audio_encoder = FaceFormerEncoder(device, video_fps=25)

        ## Define the 2D image generation model
        self.image_token_encoder_decoder = ImageTokenEncoder(in_ch=6)
        
        ## Define the target 2D image tokens sequence encoder
        self.image_token_sequence_encoder = nn.Sequential(
            nn.Linear(512, config['d_model']),
            PositionalEncoding(config['d_model'])
        )

        ## Define the 3D information embedding
        self.face_3d_param_model = Face3DMMFormer(config)

        ## Define the tranformer decoder
        self.mm_fusion_decoder = MMFusionDecoder(config)
    
    def encode_audio(self, x: Tensor, lengths=None, sample_rate=16000):
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

    def encode_target(self, input_dict):
        """Encode the multi modal target and combine them together in sequence dimension

        Args:
            input_dict (dict): dictionary contains 2D, 3D data

        Returns:
            Tensor: (B, S, E)
        """
        assert isinstance(input_dict, dict)

        ## 1) Get the 2D image tokens embedding
        image_tokens = self.image_token_encoder_decoder.encode(input_dict['input_image'])
        image_tokens = self.image_token_sequence_encoder(image_tokens) ## Add positional information

        ## 2) Get the 3D informations embedding
        face_3d_param_embedding = self.face_3d_param_model.encode_embedding(input_dict['face_3d_params'])

        ## 3) Combine the 2D-3D embeddings together
        assert image_tokens.size[2] == face_3d_param_embedding.size[2]
        mm_embedding = torch.concat([image_tokens, face_3d_param_embedding], dim=1)
        return mm_embedding

    def decode(self, y: Tensor, encoded_x: Tensor, tgt_lengths=None,
               shift_target_tright=True):
        """Transformer Decoder to complete the target self attention and encoded_x cross attention

        Args:
            y (Tensor): (B, Sy, E)
            encoded_x (Tensor): (Sx, B, E)
            tgt_lengths (_type_, optional): valid target length (B, ). Defaults to None.
            shift_target_tright (bool, optional): whether shifted the target right. Defaults to True.
        """
        ## Permute the dimensions
        y = y.permute(1, 0, 2) # to (Sy, B, ...)
        
        if shift_target_tright:
            y = self._generate_shifted_target(y)

        tgt_mask = self._generate_subsequent_mask(len(y)).to(y.device) # (Sy, B, C)

        tgt_key_padding_mask = self._generate_key_mapping_mask(y, tgt_lengths)
        output = self.mm_fusion_decoder(y, encoded_x, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask)

        return output, tgt_key_padding_mask

    def forward(self, data_dict):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        tgt_multi_modal_embedding = self.encode_target(data_dict)

        ## 3) MM Transformer Decoder
        output, output_mask = self.decode(tgt_multi_modal_embedding, encoded_x,
                                          tgt_lengths=None) # output: (Sy, B, C)
                                          
        ## Forward 2D, 3D decoder seperately
        image_2d_seq_len = data_dict['input_image'].shape[1]

        image_2d_feat = output[:image_2d_seq_len, ...] # (Sy, B, C)
        face_3d_feat = output[image_2d_seq_len:, ...]  # (Sy, B, C)

        ## Generate the output image
        image_2d_feat = torch.permute(image_2d_feat, (1, 0, 2)) # to (B, Sy, C)
        output_image = self.image_token_encoder_decoder.decode(image_2d_feat) # to (B, T, 3, H, W)

        ## Generate the 3DMM parameters
        output_3d_params = self.face_3d_param_model.decode_embedding(face_3d_feat)
        
        output_dict = defaultdict(lambda: None)
        output_dict['face_image'] = output_image
        output_dict['face_3d_params'] = output_3d_params

        return output_dict

    def _generate_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_key_mapping_mask(self, trg, lengths):
        """_summary_

        Args:
            trg (Tensor): (Sy, B, C)
            lengths (Tensor): (B, )

        Returns:
            Tensor: (B, Sy)
        """
        if lengths is None:
            return None
        max_len, batch_size , _ = trg.shape
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
        return mask
    
    def _generate_shifted_target(self, target: Tensor):
        """_summary_

        Args:
            target (Tensor): (Sy, B, C)

        Returns:
            _type_: shifted target with a inserted start token
        """
        ret = torch.zeros_like(target)
        ret[1:, ...] = target[:-1, ...]
        return ret
    
    