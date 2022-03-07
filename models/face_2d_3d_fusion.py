'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 14:29:16
Email: haimingzhang@link.cuhk.edu.cn
Description: 2D-3D Fusion transformer in Pytorch-Lightning
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
from .resnet_embedding import ResNetEmbedding
from .face_2d_3d_xfomer import Face2D3DXFormer
from utils.save_data import save_image_array_to_video, save_video

class Face2D3DFusion(pl.LightningModule):
    """Generate the face image in by fusion 2D-3D information"""
    
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        self.save_hyperparameters()

        ## Define the audio encoder
        self.audio_encoder = Wav2Vec2Encoder(self.device, video_fps=25)

        ## Define the 2D image encoder
        self.resnet_encoder = ResNetEmbedding()

        self.face_3d_layer_norm = nn.LayerNorm(128)
        self.fc_3d = nn.Linear(192, 128)

        self.face_2d_layer_norm = nn.LayerNorm(128)
        self.fc_2d = nn.Linear(256, 128)

        ## Define the 2D-3D Cross-modal Transformer
        self.face_2d_3d_xformer = Face2D3DXFormer(self.config['Face2D3DXFormer'])

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
        """Forward the network

        Args:
            data_dict (Dict): optional keys: raw_audio, gt_face_image, gt_face_3d_params

        Returns:
            _type_: _description_
        """
        ## 1) Audio encoder
        audio_seq = data_dict['raw_audio'] # (B, L)
        encoded_x = self.encode_audio(audio_seq, lengths=None) # (Sx, B, E)

        ## 2) Get the template image embedding
        gt_face_image = data_dict['gt_face_image']
        image_template = gt_face_image[:, 0, :, :, :] # (B, 3, 224, 224)
        
        image_template_embedding = self.resnet_encoder(image_template) # (B, 128)

        ## 3) Combine the 2D template image and audio features
        # to (S, B, 128)
        image_template_embedding = image_template_embedding[:, None, :].repeat(1, encoded_x.shape[0], 1).permute(1, 0, 2)
        audio_face_image_embedding = torch.cat([encoded_x, image_template_embedding], dim=-1) # to (S, B, 256)
        audio_face_image_embedding = self.face_2d_layer_norm(self.fc_2d(audio_face_image_embedding))

        ## 4) Combine the 3D and audio features
        if self.config.use_3d:
            face_3d_params = data_dict['gt_face_3d_params'] ## target sequence
            fac3_3d_template = face_3d_params[:, :1, :] # Only use the first frame (B, 1, 64)
            fac3_3d_template = fac3_3d_template.permute(1, 0, 2).repeat(encoded_x.shape[0], 1, 1) # (S, B, 64)

            audio_face_3d_embedding = torch.cat([encoded_x, fac3_3d_template], dim=-1) # (S, B, C)
            audio_face_3d_embedding = self.fc_3d(audio_face_3d_embedding)
            audio_face_3d_embedding = self.face_3d_layer_norm(audio_face_3d_embedding)

            fusion_embedding = torch.concat([audio_face_3d_embedding, audio_face_image_embedding], dim=0) # (2S, B, E)
        else:
            fusion_embedding = audio_face_image_embedding

        ## 4) Decoder
        ## Build the masked image
        # masked_image = data_dict['gt_face_image'].clone().detach()
        # masked_image[:, :, :, masked_image.shape[3]//2:] = 0.
        ## Padding the reference face image
        # masked_image = torch.concat([masked_image, data_dict['ref_face_image']], dim=2)

        model_output_dict = self.face_2d_3d_xformer(fusion_embedding, data_dict['input_image'])

        return model_output_dict

    def training_step(self, batch, batch_idx):
        ## 1) Forward the network
        model_output = self(batch)

        ## 2) Calculate the loss
        loss_dict = self.compute_loss(batch, model_output)
        
        total_loss = 0.0
        for value in loss_dict.values():
            total_loss += value

        self.log('train/total_recon_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/loss_s', loss_dict['loss_s'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/lossg_e', loss_dict['lossg_e'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lossg_em', loss_dict['lossg_em'], on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/loss_2d_l1', loss_dict['loss_2d_l1'], on_step=True, on_epoch=True, prog_bar=True)

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        ## 1) Forward the network
        model_output = self(batch)

        ## 2) Calculate the loss
        loss_dict = self.compute_loss(batch, model_output)

        total_loss = 0.0
        for value in loss_dict.values():
            total_loss += value

        self.log('val/total_recon_loss', total_loss)
        self.log('val/loss_s', loss_dict['loss_s'])
        self.log('val/lossg_e', loss_dict['lossg_e'])
        self.log('val/lossg_em', loss_dict['lossg_em'])
        self.log('val/loss_2d_l1', loss_dict['loss_2d_l1'])

        if batch_idx == 0:
            return (model_output, batch)

    def validation_epoch_end(self, outputs):
        model_output, batch = outputs[0]

         ## 3) Save the video
        save_image_array_to_video(model_output['face_2d_image'],
                                  osp.join(self.logger.log_dir, "vis", f"epoch_{self.current_epoch:03d}"),
                                  audio_array=batch['raw_audio'])

    def test_step(self, batch, batch_idx):
        ## 1) Forward the network
        model_output = self(batch)

        ## 2) Visualization
        save_image_array_to_video(model_output['face_2d_image'],
                                  osp.join(self.config['checkpoint_dir'], "vis"),
                                  audio_array=batch['raw_audio'],
                                  name=batch_idx)

    def compute_loss(self, data_dict, model_output):
        loss_dict = {}
        if self.config.use_3d:
            ## 3D loss
            pred_params = model_output['face_3d_params']
            tgt_params = data_dict['gt_face_3d_params']

            motionlogits = pred_params[:, 1:, :] - pred_params[:, :-1, :]
            tgt_motion = tgt_params[:, 1:, :] - tgt_params[:, :-1, :]

            loss_s = 10 * (F.smooth_l1_loss(pred_params[:, :1, :], tgt_params[:, :1, :]))
            lossg_e = 20 * F.smooth_l1_loss(pred_params[:, :, :], tgt_params[:, :, :])
            lossg_em = 200 * F.smooth_l1_loss(motionlogits[:,:,:], tgt_motion[:,:,:])

            loss_dict['loss_s'] = loss_s
            loss_dict['lossg_e'] = lossg_e
            loss_dict['lossg_em'] = lossg_em
        
        ## 2D loss
        pred_face_image = model_output['face_2d_image']
        tgt_face_image = data_dict['gt_face_image']
        loss_2d = F.l1_loss(pred_face_image, tgt_face_image) * 200.0

        loss_dict['loss_2d_l1'] = loss_2d
        
        return loss_dict

    def encode_audio(self, x: Tensor, lengths=None, sample_rate=16000):
        """_summary_

        Args:
            x (Tensor): (B, Sx)

        Returns:
            Tensor: (Sx, B, E)
        """
        if sample_rate != 16000:
            import torchaudio
            ## resample the audio sample rate
            x = torchaudio.functional.resample(x, 22000, 16000)
            if lengths is not None:
                # rescale the lengths
                lengths = (lengths * 16000 / 22000.0).to(lengths.dtype)
        
        enc_output = self.audio_encoder(x, lengths)
        return enc_output.permute(1, 0, 2)
