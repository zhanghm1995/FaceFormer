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
from typing import Optional, Any, Union, Callable, Dict
from torch.nn import TransformerDecoder, LayerNorm, TransformerDecoderLayer
from .face_former_encoder import FaceFormerEncoder, FaceFormerEncoderMixedAudio
from .image_token_encoder import ImageTokenEncoder, ImageTokenEncoder224
from .pos_encoder import PositionalEncoding
from .transformer_utils import (generate_shifted_target, 
                                generate_subsequent_mask,
                                generate_key_mapping_mask)


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


class Face3DMMDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        config = config['face_3dmm_former']
        assert config is not None

        d_input_3d_params = config['d_input_3d_params']
        d_embedding_model = config['d_model']

        self.input_encoder = nn.Linear(d_input_3d_params, d_embedding_model)

        # self.output_encoder = nn.Sequential(
        #     nn.Linear(512, d_embedding_model),
        #     nn.ReLU(),
        #     nn.Linear(d_embedding_model, d_input_3d_params))

        self.output_encoder = nn.Linear(d_embedding_model, d_input_3d_params)

        self.face_3d_feat_layer_norm = LayerNorm(config['d_model'])

        self.pos_encoder = PositionalEncoding(d_embedding_model)

        ## Define the one-hot subject index
        index = 0
        self.subject_idx_one_hot = torch.zeros((1, 8), dtype=torch.float32)
        self.subject_idx_one_hot[:, index] = 1

        self.style_embedding = nn.Linear(8, d_embedding_model) # use to embedding the one-hot style

        ## Build the transformer layer
        decoder_layer = TransformerDecoderLayer(
            d_model=config['d_model'], nhead=config['n_head'], dim_feedforward=config['d_feed_forward'],
            batch_first=False)
        decoder_norm = LayerNorm(config['d_model'])
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config['n_layer'], norm=decoder_norm)

    def _decode(self, y: Tensor, encoded_x: Tensor,
               tgt_lengths=None, need_tgt_mask=True) -> Tensor:
        """_summary_

        Args:
            y (Tensor): (Sy, B, E)
            encoded_x (Tensor): (Sx, B, E)

        Returns:
            Tensor: (Sy, B, E)
        """
        if need_tgt_mask:
            tgt_mask = generate_subsequent_mask(len(y)).to(y.device) # (Sy, B, C)
        else:
            tgt_mask = None

        tgt_key_padding_mask = generate_key_mapping_mask(y, tgt_lengths)
        output = self.decoder(y, encoded_x, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)

        return output, tgt_key_padding_mask

    def forward(self, y: Tensor, encoded_x: Tensor,
                shift_target_right=True,
                need_tgt_mask=True):
        """Forward the decoder

        Args:
            y (Tensor): (B, Sy, C)
            encoded_x (Tensor): (B, Sx, C)

        Returns:
            Tensor: (Sy, B, E)
        """

        y = y.permute(1, 0, 2) # to (Sy, B, ...)

        ## 1) Shifting the target
        if shift_target_right:
            y = generate_shifted_target(y)

        ## 2) Encoding the input to get the embedding
        embedding = self.encode_embedding(y, 
                                          apply_layer_norm=False, 
                                          apply_style_encoding=False,
                                          add_positional_encoding=True) # in (Sy, B, E)
        
        ## 2) Transformer attention
        output, output_mask = self._decode(embedding, encoded_x,
                                           tgt_lengths=None,
                                           need_tgt_mask=need_tgt_mask)

        output = self.decode_embedding(output) # (Sy, B, C)
        return output

    def encode_embedding(self, input_seq: Tensor, 
                         apply_layer_norm=True, 
                         apply_style_encoding=True,
                         add_positional_encoding=True):
        """Encode the raw face 3DMM sequence parameters into embeddings

        Args:
            input_seq (Tensor): (Sy, B, C)
            add_positional_encoding (bool, optional): whether adding positional encoding operation. Defaults to True.

        Returns:
            Tensor: (Sy, B, E)
        """
        ## Get the embeddings
        embedding = self.input_encoder(input_seq)

        if apply_style_encoding:
            T, B = input_seq.shape[:2]
            subject_idx_tensor = self.subject_idx_one_hot.repeat((T, B, 1)).to(embedding)
            embedding += self.style_embedding(subject_idx_tensor)

        ## Apply the Layer Norm
        if apply_layer_norm:
            embedding = self.face_3d_feat_layer_norm(embedding)
            
        ## Add the positional encoding
        if add_positional_encoding:
            embedding = self.pos_encoder(embedding)
        return embedding
    
    def decode_embedding(self, input_seq):
        ## Get the embeddings
        output = self.output_encoder(input_seq)
        
        return output


class Face3DMMFormer(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()

        ## Define the audio encoder
        self.audio_encoder = FaceFormerEncoder(device, video_fps=25)

        ## Define the 3D information embedding
        self.face_3d_param_model = Face3DMMDecoder(config)

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
        
        enc_output = self.audio_encoder(x, lengths)
        return enc_output.permute(1, 0, 2)

    def forward(self, data_dict: Dict, shift_target_right=True):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        face_3d_params = data_dict['gt_face_3d_params'] ## target sequence
        output = self.face_3d_param_model(face_3d_params, encoded_x,
                                          shift_target_right=shift_target_right)

        output = output.permute(1, 0, 2) # to (B, Sy, C)

        return {'face_3d_params': output}

    def forward_autoregressive(self, data_dict: Dict, shift_target_right=True):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, seq_len, 64)).to(encoded_x.device) # in (B, Sy, C)

        for seq_idx in range(1, seq_len):
            y = output[:, :seq_idx]
            dec_output = self.face_3d_param_model(y, encoded_x, 
                                                  shift_target_right=False) # in (Sy, B, C)
            output[:, seq_idx] = dec_output[-1:, ...]
        return {'face_3d_params': output}

    def inference(self, data_dict):
        ## audio source encoder
        audio_seq = data_dict['raw_audio']
        encoded_x = self.encode_audio(audio_seq, lengths=None)

        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, seq_len, 64)).to(encoded_x.device)

        for seq_idx in range(1, seq_len):
            y = output[:, :seq_idx]
            dec_output = self.face_3d_param_model(y, encoded_x, 
                                                 shift_target_right=False,
                                                 need_tgt_mask=False) # in (Sy, B, C)
            output[:, seq_idx] = dec_output[-1:, ...]
        return {'face_3d_params': output}

    def inference_new(self, data_dict): # TODO
        ## audio source encoder
        audio_seq = data_dict['raw_audio']
        encoded_x = self.encode_audio(audio_seq, lengths=None)

        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, 1, 64)).to(encoded_x.device)

        for _ in range(seq_len):
            dec_output = self.face_3d_param_model(output, encoded_x, 
                                                  shift_target_right=False,
                                                  need_tgt_mask=False) # in (Sy, B, C)
            
            dec_output = dec_output.permute(1, 0, 2)
            output = torch.concat([output, dec_output[:, -1:, :]], dim=1)
        
        output = output[:, 1:, :]
        return {'face_3d_params': dec_output}


class Face3DMMFormerMixedAudio(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()

        ## Define the audio encoder
        self.audio_encoder = FaceFormerEncoderMixedAudio(device, video_fps=25)

        ## Define the 3D information embedding
        self.face_3d_param_model = Face3DMMDecoder(config)

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
        
        enc_output = self.audio_encoder(x, lengths)
        return enc_output.permute(1, 0, 2)

    def forward(self, data_dict: Dict):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        ref_audio_seq = data_dict['ref_raw_audio']
        
        mixed_audio_seq = torch.cat([audio_seq, ref_audio_seq], dim=0)

        encoded_x = self.encode_audio(mixed_audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        face_3d_params = data_dict['ref_face_3d_params'] ## target sequence
        output = self.face_3d_param_model(face_3d_params, encoded_x,
                                          shift_target_right=False,
                                          need_tgt_mask=False)

        output = output.permute(1, 0, 2) # to (B, Sy, C)

        return {'face_3d_params': output}

    def forward_autoregressive(self, data_dict: Dict, shift_target_right=True):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, seq_len, 64)).to(encoded_x.device) # in (B, Sy, C)

        for seq_idx in range(1, seq_len):
            y = output[:, :seq_idx]
            dec_output = self.face_3d_param_model(y, encoded_x, 
                                                  shift_target_right=False) # in (Sy, B, C)
            output[:, seq_idx] = dec_output[-1:, ...]
        return {'face_3d_params': output}

    def inference(self, data_dict):
        ## audio source encoder
        audio_seq = data_dict['raw_audio']
        encoded_x = self.encode_audio(audio_seq, lengths=None)

        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, seq_len, 64)).to(encoded_x.device)

        for seq_idx in range(1, seq_len):
            y = output[:, :seq_idx]
            dec_output = self.face_3d_param_model(y, encoded_x, 
                                                 shift_target_right=False,
                                                 need_tgt_mask=False) # in (Sy, B, C)
            output[:, seq_idx] = dec_output[-1:, ...]
        return {'face_3d_params': output}

    def inference_new(self, data_dict): # TODO
        ## audio source encoder
        audio_seq = data_dict['raw_audio']
        encoded_x = self.encode_audio(audio_seq, lengths=None)

        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, 1, 64)).to(encoded_x.device)

        for _ in range(seq_len):
            dec_output = self.face_3d_param_model(output, encoded_x, 
                                                  shift_target_right=False,
                                                  need_tgt_mask=False) # in (Sy, B, C)
            
            dec_output = dec_output.permute(1, 0, 2)
            output = torch.concat([output, dec_output[:, -1:, :]], dim=1)
        
        output = output[:, 1:, :]
        return {'face_3d_params': dec_output}


class MMFusionFormer(nn.Module):
    def __init__(self, config, device) -> None:
        super().__init__()

        ## Define the audio encoder
        self.audio_encoder = FaceFormerEncoder(device, video_fps=25)

        ## Define the 2D image generation model
        self.image_token_encoder_decoder = ImageTokenEncoder224(in_ch=6)
        
        ## Define the target 2D image tokens sequence encoder
        self.image_token_sequence_encoder = nn.Sequential(
            nn.Linear(512, config['d_model']),
            LayerNorm(config['d_model']),
            PositionalEncoding(config['d_model'])
        )

        ## Define the 3D information embedding
        self.face_3d_param_model = Face3DMMDecoder(config)

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
        
        enc_output = self.audio_encoder(x, lengths)
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
        face_3d_param_embedding = self.face_3d_param_model.encode_embedding(input_dict['ref_face_3d_params'])

        ## 3) Combine the 2D-3D embeddings together along the sequence dimension
        assert image_tokens.shape[2] == face_3d_param_embedding.shape[2]
        mm_embedding = torch.concat([image_tokens, face_3d_param_embedding], dim=1)
        return mm_embedding

    def decode(self, y: Tensor, encoded_x: Tensor, tgt_lengths=None,
               shift_target_right=True):
        """Transformer Decoder to complete the target self attention and encoded_x cross attention

        Args:
            y (Tensor): (B, Sy, E)
            encoded_x (Tensor): (Sx, B, E)
            tgt_lengths (_type_, optional): valid target length (B, ). Defaults to None.
            shift_target_tright (bool, optional): whether shifted the target right. Defaults to True.
        """
        ## Permute the dimensions
        y = y.permute(1, 0, 2) # to (Sy, B, ...)
        
        if shift_target_right:
            y = generate_shifted_target(y)

        tgt_mask = generate_subsequent_mask(len(y)).to(y.device) # (Sy, B, C)

        tgt_key_padding_mask = generate_key_mapping_mask(y, tgt_lengths)
        output = self.mm_fusion_decoder(y, encoded_x, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask)

        return output, tgt_key_padding_mask

    def forward(self, data_dict: Dict, shift_target_right=True):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        tgt_multi_modal_embedding = self.encode_target(data_dict)

        ## 3) MM Transformer Decoder
        output, output_mask = self.decode(tgt_multi_modal_embedding, encoded_x,
                                          tgt_lengths=None, shift_target_right=shift_target_right) # output: (Sy, B, C)
                                          
        ## Forward 2D, 3D decoder seperately
        image_2d_seq_len = data_dict['input_image'].shape[1]

        image_2d_feat = output[:image_2d_seq_len, ...] # (Sy, B, C)
        face_3d_feat = output[image_2d_seq_len:, ...]  # (Sy, B, C)

        ## Generate the output image
        image_2d_feat = torch.permute(image_2d_feat, (1, 0, 2)) # to (B, Sy, C)
        output_image = self.image_token_encoder_decoder.decode(image_2d_feat) # to (B, T, 3, H, W)

        ## Generate the 3DMM parameters
        output_3d_params = self.face_3d_param_model.decode_embedding(face_3d_feat)
        output_3d_params = torch.permute(output_3d_params, (1, 0, 2)) # to (B, Sy, C)
        
        output_dict = defaultdict(lambda: None)
        output_dict['face_image'] = output_image
        output_dict['face_3d_params'] = output_3d_params

        return output_dict