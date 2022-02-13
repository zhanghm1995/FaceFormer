'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-11 11:47:32
Email: haimingzhang@link.cuhk.edu.cn
Description: The wav2vec 2.0 implementation
'''

import torch
import torch.nn as nn
import torchaudio
import torchaudio.models.wav2vec2 as ta_wav2vec2
from models.transformer import Decoder


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Wav2Vec2Encoder(ta_wav2vec2.Wav2Vec2Model):
    def __init__(self, device) -> None:
        bundle = torchaudio.pipelines.WAV2VEC2_BASE

        self.model = bundle.get_model().to(device)

        input_channels, output_channels = -1, -1
        self.interpolation = nn.Linear(input_channels, output_channels)

        encoder_out_channels, output_channels = 768, 512
        self.final_projection = nn.Linear(encoder_out_channels, output_channels)
        

    def forward(self, data_dict):
        ## Move to GPU
        
        waveforms = data_dict['waveforms']
        lengths = data_dict['lengths']
        
        ## Extract the audio features
        x, lengths = self.feature_extractor(waveforms, lengths)

        ## Forward the transformer encoder
        x = self.encoder(x, lengths)

        ## Add the linear projection
        x = self.final_projection(x)
        return x, lengths


class FaceFormerEncoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec2_model = bundle.get_model().to(device)

        input_channels, output_channels = 768, 512
        self.interpolation = nn.Linear(input_channels, output_channels)

        encoder_out_channels, output_channels = 768, 512
        self.final_projection = nn.Linear(encoder_out_channels, output_channels)

        self.scale_factor = 60 / 16000.0
        
    def forward(self, data_dict):
        waveforms = data_dict['waveforms']
        lengths = None
        
        ## Extract the audio features
        x, lengths = self.wav2vec2_model.feature_extractor(waveforms, lengths)

        output_seq_len = int(waveforms.size(1) * self.scale_factor)
        
        ## Add the linear interpolation
        x = torch.permute(x, (0, 2, 1))
        x = nn.functional.interpolate(x, size=[output_seq_len])
        x = torch.permute(x, (0, 2, 1))

        # x = self.interpolation(x)

        ## Forward the transformer encoder
        x = self.wav2vec2_model.encoder(x, lengths)

        ## Add the linear projection
        x = self.final_projection(x)
        return x


def one_hot(x):
    """Get the one hot matrix

    Args:
        x (Tensor): Bxseq_len dimension

    Returns:
        [type]: Bxseq_lenx8
    """
    x = x.unsqueeze(-1)
    condition = torch.zeros(x.shape[0], x.shape[1], 8).scatter_(2, x.type(torch.LongTensor), 1)
    return condition


class FaceFormer(nn.Module):
    """Transformer implementation for FaceFormer framework"""

    def __init__(self, config, final_channels) -> None:
        self.encoder = FaceFormerEncoder()

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.motion_decoder = nn.Linear(d_model, final_channels, bias=False)

        num_training_subjects = config['num_training_subjects']
        style_embedding_dim = config['style_embedding_dim']

        self.motion_encoder = nn.Linear(final_channels, motion_embedding_dim)
        self.style_embedding = nn.Linear(num_training_subjects, style_embedding_dim, bias=False)


    def forward(self, audio_seq, face_seq, speaker_id):
        trg_mask = get_pad_mask(face_seq, self.trg_pad_idx) & get_subsequent_mask(face_seq)

        audio_seq = {"waveforms": audio_seq}

        enc_output = self.encoder(audio_seq)

        ## decoder process
        face_embed = self.motion_encoder(face_seq) + self.style_embedding(speaker_id)
        
        dec_output = self.decoder(face_embed, trg_mask, enc_output)

        seq_output = self.motion_decoder(dec_output)

        return seq_output