'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-11 11:47:32
Email: haimingzhang@link.cuhk.edu.cn
Description: The wav2vec 2.0 implementation
'''

import math
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Any, Union, Callable
from torch.nn import TransformerDecoder, LayerNorm, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_
import torchaudio
import torchaudio.models.wav2vec2 as ta_wav2vec2
from models.transformer import Decoder


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, _ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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


class FaceFormerDecoder(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()

        motion_embedding_dim = config['motion_embedding_dim']
        num_training_subjects = config['dataset']['num_training_subjects']
        style_embedding_dim = config['style_embedding_dim']
        final_channels = config['final_channels']

        self.motion_encoder = nn.Linear(final_channels, motion_embedding_dim)
        self.style_embedding = nn.Linear(num_training_subjects, style_embedding_dim)

        config = config['faceformer_decoder']
        ## Build the decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config['d_model'], nhead=config['n_head'], dim_feedforward=config['d_feed_forward'],
            batch_first=False)
        decoder_norm = LayerNorm(config['d_model'])
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config['n_layer'], norm=decoder_norm)

        self.pos_encoder = PositionalEncoding(config['d_model'])

        ## The final motion decoder
        self.motion_decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 5023*3),
            nn.Sigmoid()
        )

        self.init_weights()

    def forward(self, tgt: Tensor, speaker_id: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        tgt_embedding = self.motion_encoder(tgt) + self.style_embedding(speaker_id)

        # 1) positional encoding
        tgt_embedding = self.pos_encoder(tgt_embedding)
        
        # 2) transformer
        decoder_output = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        # 3) motion decoder
        output = self.motion_decoder(decoder_output)
        return output

    def init_weights(self):
        r"""Initiate parameters in the transformer model."""

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         xavier_uniform_(p)

        initrange = 0.1
        # nn.init.uniform_(self.motion_decoder.weight, -initrange, initrange)
        # self.motion_decoder.bias.data.zero_()
        # self.motion_decoder.weight.data.uniform_(-initrange, initrange)
        

class FaceFormerV2(nn.Module):
    """FaceFormer implemention by using Pytorch Transformer decoder"""

    def __init__(self, config, device):
        super().__init__()
        
        self.encoder = FaceFormerEncoder(device)
        
        ## Define the Decoder TODO
        self.decoder = FaceFormerDecoder(config)

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

    def encode(self, x: Tensor, lengths=None):
        """_summary_

        Args:
            x (Tensor): (B, Sx)

        Returns:
            Tensor: (Sx, B, E)
        """
        ## resample the audio sample rate
        x = torchaudio.functional.resample(x, 22000, 16000)
        if lengths is not None:
            # rescale the lengths
            lengths = (lengths * 16000 / 22000.0).to(lengths.dtype)
        enc_output = self.encoder(x, lengths)
        return enc_output.permute(1, 0, 2)

    def decode(self, y: Tensor, speaker_id: Tensor, encoded_x: Tensor,
               trg_lengths=None, shift_target_tright=True) -> Tensor:
        """_summary_

        Args:
            y (Tensor): (B, Sy, C)
            speaker_id (Tensor): (B, Sy, num_subject)
            encoded_x (Tensor): (Sx, B, E)

        Returns:
            Tensor: (Sy, B, E)
        """
        ## facial motion target decoder
        y = y.permute(1, 0, 2) # to (Sy, B, ...)
        
        if shift_target_tright:
            y = self._generate_shifted_target(y)

        speaker_id = speaker_id.permute(1, 0, 2)

        trg_mask = self._generate_subsequent_mask(len(y)).to(y.device) # (Sy, B, C)

        tgt_key_padding_mask = self._generate_key_mapping_mask(y, trg_lengths)
        output = self.decoder(y, speaker_id, encoded_x, tgt_mask=trg_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)

        return output, tgt_key_padding_mask

    def forward(self, data_dict):
        ## audio source encoder
        audio_seq = data_dict['raw_audio'] # (B, 22000)
        encoded_x = self.encode(audio_seq, lengths=data_dict['raw_audio_lengths'])

        ## facial motion target decoder
        face_seq = data_dict['target_face_motion'] # (B, Sy, C)
        speaker_id = data_dict['subject_idx'] # (B, Sy, 8)
        
        output, output_mask = self.decode(face_seq, speaker_id, encoded_x,
                                          trg_lengths=data_dict['face_vertices_lengths']) # output: (Sy, B, C)

        output = torch.permute(output, (1, 0, 2)) # to (B, Sy, C)

        output_mask = ~output_mask[..., None]
        return torch.mul(output, output_mask.type(torch.float)) 

    def inference(self, data_dict):
        ## audio source encoder
        audio_seq = data_dict['raw_audio']
        encoded_x = self.encode(audio_seq, lengths=None)

        speaker_id = data_dict['subject_idx'] # (B, Sy, 8)

        seq_len, batch_size = encoded_x.shape[:2]
        
        output = torch.zeros((batch_size, seq_len, 15069)).to(encoded_x.device)

        for seq_idx in range(1, seq_len):
            y = output[:, :seq_idx]
            dec_output, _ = self.decode(y, speaker_id[:, :seq_idx], encoded_x, shift_target_tright=False) # in (Sy, B, C)
            output[:, seq_idx] = dec_output[-1:, ...]
        return output
    
    def inference_whole_sequence(self, data_dict, net_sequence_length, device):
        """Inference a whole sequence, we slice the sequence inside this function

        Args:
            data_dict (dict): numpy array

        Returns:
            _type_: _description_
        """
        def infer(raw_audio, subject_idx, start_token):
            ## audio source encoder
            encoded_x = self.encode(raw_audio)

            seq_len, batch_size = encoded_x.shape[:2]
        
            output = torch.zeros((batch_size, seq_len, 15069)).to(encoded_x.device)
            if start_token is not None:
                output[:, 0:1, :] = start_token

            ## Forward transformer autoregressively
            for seq_idx in range(1, seq_len):
                y = output[:, :seq_idx]
                dec_output = self.decode(y, subject_idx[:, :seq_idx], encoded_x, shift_target_tright=False) # in (Sy, B, C)
                output[:, seq_idx] = dec_output[-1:, ...]
            return output

        whole_seq_len = data_dict['target_face_motion'].shape[1]
        
        whole_infer_output = torch.zeros_like(data_dict['target_face_motion']) # (B, seq_len, 15609)

        start_token = torch.zeros((1, 1, 15069)).to(device)
        
        for i in range(0, whole_seq_len - net_sequence_length - 1, net_sequence_length - 1):
            start_idx, end_idx = i, i + net_sequence_length
            audio_start_idx = round(start_idx / 60.0 * 22000)
            audio_end_idx = round(end_idx / 60.0 * 22000)

            curr_raw_audio = data_dict['raw_audio'][:, audio_start_idx:audio_end_idx] # (1, Sx)
            curr_subject_idx = data_dict['subject_idx']

            output = infer(curr_raw_audio, curr_subject_idx, start_token)
            whole_infer_output[:, i:i+net_sequence_length, :] = output[0, :, :]

            # use last output in last forward
            start_token = output[:, -2:-1, :]

            if i + net_sequence_length - 1 >= whole_seq_len - net_sequence_length - 1:
                j = whole_seq_len - net_sequence_length
                start_idx, end_idx = j, j + net_sequence_length
                audio_start_idx = round(start_idx / 60.0 * 22000)
                audio_end_idx = round(end_idx / 60.0 * 22000)
                curr_raw_audio = data_dict['raw_audio'][:, audio_start_idx:audio_end_idx] # (1, Sx)

                start_token = output[:, j-i:j-i+1, :]
                output = infer(curr_raw_audio, curr_subject_idx, start_token)
                whole_infer_output[:, j:j+net_sequence_length, :] = output[0, :, :]
            
        return whole_infer_output


class FaceFormer(nn.Module):
    """Transformer implementation for FaceFormer framework"""

    def __init__(self, config, final_channels, device):
        super().__init__()
        
        self.encoder = FaceFormerEncoder(device)
        
        ## Define the Decoder TODO
        self.decoder = Decoder(
            n_trg_vocab=60, n_position=200,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=1, n_head=8, d_k=64, d_v=64,
            pad_idx=None, dropout=0.1, scale_emb=True)

        self.motion_decoder = nn.Linear(512, final_channels)

        motion_embedding_dim = config['motion_embedding_dim']
        num_training_subjects = config['dataset']['num_training_subjects']
        style_embedding_dim = config['style_embedding_dim']

        self.motion_encoder = nn.Linear(final_channels, motion_embedding_dim)
        self.style_embedding = nn.Linear(num_training_subjects, style_embedding_dim)

    def forward(self, data_dict):
        audio_seq = data_dict['raw_audio']
        audio_seq = torchaudio.functional.resample(audio_seq, 22000, 16000)

        face_seq = data_dict['target_face_motion']
        batch_size, seq_len = face_seq.shape[:2]

        face_seq = torch.reshape(face_seq, (batch_size, seq_len, -1))

        speaker_id = data_dict['subject_idx']

        trg_mask = get_subsequent_mask(face_seq)

        audio_seq = {"waveforms": audio_seq}

        enc_output = self.encoder(audio_seq)

        ## process decoder input
        # face_embed = self.motion_encoder(face_seq) + self.style_embedding(speaker_id)
        face_embed = self.motion_encoder(face_seq)
        
        dec_output = self.decoder(face_embed, trg_mask, enc_output, src_mask=None)

        seq_output = self.motion_decoder(dec_output)

        return torch.reshape(seq_output, (batch_size, seq_len, -1, 3))