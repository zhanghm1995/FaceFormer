'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-24 21:13:40
Email: haimingzhang@link.cuhk.edu.cn
Description: The PL module to train the Face2D3DFusionFormer network
'''

import torch
import numpy as np
import os
import os.path as osp
import torch.nn as nn
from scipy.io import wavfile
from torch.nn import functional as F 
import pytorch_lightning as pl
from .face_2d_3d_fusion_former import Face2D3DFusionFormer


class Face2D3DFusionFormerModule(pl.LightningModule):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.config = config

        self.save_hyperparameters()

        ## Define the model
        self.model = Face2D3DFusionFormer(config['Face2D3DFusionFormer'])

        self.criterion = nn.MSELoss()

        if self.config.use_mouth_mask:
            binary_mouth_mask = np.load("./data/big_mouth_mask.npy")
            mouth_mask = np.ones(35709)
            mouth_mask[binary_mouth_mask] = 1.8
            self.mouth_mask_weight = torch.from_numpy(np.expand_dims(mouth_mask, 0)) # (1, 35709)
    
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
        model_output = self.model(
            batch, teacher_forcing=self.config.teacher_forcing)

        ## Calcuate the loss
        loss_dict = self.compute_loss(batch, model_output)
        loss = loss_dict['loss']

        batch_size = batch['raw_audio'].shape[0]

        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
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

    def test_step(self, batch, batch_idx):
        ## We do testing like official FaceFormer to conditioned on different one_hot
        audio = batch['raw_audio']
        template = batch['template']
        vertice = batch['face_vertex']
        one_hot = batch['one_hot']
        video_name = batch['video_name'][0]
        
        model_output = self.model.predict(audio, template, one_hot)
        model_output = model_output.squeeze().detach().cpu().numpy() # (seq_len, 64)
        
        ## Save the results
        save_dir = osp.join(self.logger.log_dir, "vis")
        os.makedirs(save_dir, exist_ok=True)
        # np.savez(osp.join(save_dir, f"{batch_idx:03d}.npz"), face=model_output)
        np.save(osp.join(save_dir, f"{video_name}_{batch_idx:03d}.npy"), model_output) # save face vertex

        ## Save audio
        audio_data = audio[0].cpu().numpy()
        wavfile.write(osp.join(save_dir, f"{video_name}_{batch_idx:03d}.wav"), 16000, audio_data)

    def compute_loss(self, batch, pred_dict):
        pred_face_vertex = pred_dict['pred_face_vertex']
        pred_face_image = pred_dict['pred_face_image']

        loss_dict = {}
        ## Compute the 3D loss
        tgt_face_vertex = batch['face_vertex']
        if self.config.use_mouth_mask:
            batch_size, seq_len = pred_face_vertex.shape[:2]
            ## If consider mouth region weight
            vertice_out = pred_face_vertex.reshape((batch_size, seq_len, -1, 3))
            vertice = tgt_face_vertex.reshape((batch_size, seq_len, -1, 3))

            loss_3d = torch.sum((vertice_out - vertice)**2, dim=-1) * self.mouth_mask_weight[None, ...].to(vertice)
            loss_3d = torch.mean(loss_3d)
        else:
            loss_3d = self.criterion(vertice_out, vertice) # (batch_size, seq_len, V*3)
            loss_3d = torch.mean(loss_3d)
        loss_dict['loss_3d'] = loss_3d

        ## Compute the 2D loss
        tgt_face_image = batch['gt_face_image']
        loss_2d = F.l1_loss(pred_face_image, tgt_face_image)
        loss_dict['loss_2d'] = loss_2d

        loss = loss_3d + loss_2d

        loss_dict['loss'] = loss

        return loss_dict
            