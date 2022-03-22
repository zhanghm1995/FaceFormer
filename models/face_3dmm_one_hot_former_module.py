'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 11:04:16
Email: haimingzhang@link.cuhk.edu.cn
Description: PL module to train model
'''

import torch
import numpy as np
import os
import os.path as osp
import torch.nn as nn
from scipy.io import wavfile
from torch.nn import functional as F 
import pytorch_lightning as pl
from .face_3dmm_one_hot_former import Face3DMMOneHotFormer


class Face3DMMOneHotFormerModule(pl.LightningModule):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.config = config

        self.save_hyperparameters()

        ## Define the model
        self.model = Face3DMMOneHotFormer(config['Face3DMMFormer'])

        self.criterion = nn.MSELoss()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                     lr=self.config.lr, 
                                     weight_decay=self.config.wd,
                                     betas=(0.9, 0.999), 
                                     eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        audio = batch['raw_audio']
        template = torch.zeros((audio.shape[0], 64)).to(audio)
        vertice = batch['gt_face_3d_params']
        one_hot = batch['one_hot']

        loss = self.model(
            audio, template, vertice, one_hot, self.criterion, teacher_forcing=False)
        
        ## Calcuate the loss
        self.log('train/total_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    # def validation_step(self, batch, batch_idx):
    #     audio = batch['raw_audio']
    #     template = torch.zeros((audio.shape[0], 64)).to(audio)
    #     vertice = batch['gt_face_3d_params']
    #     one_hot = batch['one_hot']

    #     loss = self.model(
    #         audio, template, vertice, one_hot, self.criterion, teacher_forcing=False)
        
    #     ## Calcuate the loss
    #     self.log('val/total_loss', loss, on_epoch=True, prog_bar=True)
    #     return loss

    def compute_loss(self, data_dict, model_output):
        loss_dict = {}
        
        ## 3D loss
        pred_params = model_output['face_3d_params']
        tgt_params = data_dict['gt_face_3d_params']

        loss_3dmm = 20 * F.smooth_l1_loss(pred_params[:, :, :], tgt_params[:, :, :])

        loss_dict['loss_3dmm'] = loss_3dmm
        return loss_dict

    def test_step(self, batch, batch_idx):
        audio = batch['raw_audio']
        template = torch.zeros((audio.shape[0], 64)).to(audio)
        vertice = batch['gt_face_3d_params']
        one_hot = batch['one_hot']
        
        model_output = self.model.predict(audio, template, one_hot)
        model_output = model_output.squeeze().detach().cpu().numpy() # (seq_len, 64)
        
        ## Save the results
        save_dir = osp.join(self.logger.log_dir, "vis")
        os.makedirs(save_dir, exist_ok=True)
        np.savez(osp.join(save_dir, f"{batch_idx:03d}.npz"), face=model_output)

        ## Save audio
        audio_data = audio[0].cpu().numpy()
        wavfile.write(osp.join(save_dir, f"{batch_idx:03d}.wav"), 16000, audio_data)
            