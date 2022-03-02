'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 00:18:02
Email: haimingzhang@link.cuhk.edu.cn
Description: Face 3DMM transformer in PL framework
'''

from easydict import EasyDict
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F 
from typing import Dict
import pytorch_lightning as pl
import os.path as osp
import os
import numpy as np
from .face_former_encoder import Wav2Vec2Encoder

class Face3DMMFormer(pl.LightningModule):
    """Generate the 3DMM parameters by using Transformer Encoder only
    """
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        ## Define the audio encoder
        self.audio_encoder = Wav2Vec2Encoder(self.device, video_fps=25)

        self.conv42 = nn.Conv1d(192, 256, kernel_size=3, stride=1,padding=1)
        self.bn42 = nn.BatchNorm1d(256)

        self.conv52 = nn.Conv1d(256, 256, kernel_size=3, stride=1,padding=1)
        self.bn52 = nn.BatchNorm1d(256)

        self.conv62 = nn.Conv1d(256, 128, kernel_size=3, stride=1,padding=1)

        self.output_fc = nn.Linear(128, 64)

    def configure_optimizers(self):
        if self.config.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=self.config.wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd,
                                         betas=(0.5, 0.999), eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, data_dict: Dict):
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Encoding the target
        face_3d_params = data_dict['gt_face_3d_params'] ## target sequence
        input_3d_params = face_3d_params[:, :1, :] # Only use the first frame (B, 1, 64)
        input_3d_params = input_3d_params.permute(1, 0, 2).repeat(encoded_x.shape[0], 1, 1)

        out = torch.cat([encoded_x, input_3d_params], dim=-1) # (S, B, C)
        out = out.permute(1, 2, 0) # to (B, C, S)

        out = F.leaky_relu(self.bn42(self.conv42(out)))
        out = F.leaky_relu(self.bn52(self.conv52(out)))
        out = self.conv62(out).permute(0,2,1) # (B, S, C)
        
        output = self.output_fc(out)

        return {'face_3d_params': output}

    def training_step(self, batch, batch_idx):
        ## 1) Forward the network
        model_output = self(batch)

        ## 2) Calculate the loss
        loss = self.compute_loss(batch, model_output)
        
        return loss

    def test_step(self, batch, batch_idx):
        from scipy.io import wavfile

        ## 1) Forward the network
        model_output = self(batch)

        ## 2) Visualization
        ## Save the 3DMM parameters to npz file
        face_params = model_output['face_3d_params'][0].cpu().numpy()
        save_dir = osp.join(self.config['checkpoint_dir'], "vis")
        os.makedirs(save_dir, exist_ok=True)
        np.savez(osp.join(save_dir, f"{batch_idx:03d}.npz"), face=face_params)

        ## Save audio
        audio_data = batch['raw_audio'][0].cpu().numpy()
        wavfile.write(osp.join(save_dir, f"{batch_idx:03d}.wav"), 16000, audio_data)

    def compute_loss(self, data_dict, model_output):
        pred_params = model_output['face_3d_params']
        tgt_params = data_dict['gt_face_3d_params']

        motionlogits = pred_params[:, 1:, :] - pred_params[:, :-1, :]
        tgt_motion = tgt_params[:, 1:, :] - tgt_params[:, :-1, :]

        loss_s = 10 * (F.smooth_l1_loss(pred_params[:, :1, :], tgt_params[:, :1, :]))
        lossg_e = 20 * F.smooth_l1_loss(pred_params[:, :, :], tgt_params[:, :, :])
        lossg_em = 200 * F.smooth_l1_loss(motionlogits[:,:,:], tgt_motion[:,:,:])

        loss = loss_s + lossg_e + lossg_em
        
        self.log('loss_s', loss_s, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lossg_e', lossg_e, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lossg_em', lossg_em, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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