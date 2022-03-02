'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 14:29:16
Email: haimingzhang@link.cuhk.edu.cn
Description: 2D-3D Fusion transformer with GAN scheme
'''

from re import template
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
import torchaudio
from .face_former_encoder import Wav2Vec2Encoder
from .resnet_embedding import ResNetEmbedding
from .face_2d_3d_xfomer import Face2D3DXFormer
from utils.save_data import save_image_array_to_video, save_video
from .discriminators.img_gen_disc import Feature2Face_D as PatchGANDiscriminator
from .face_2d_3d_fusion import Face2D3DFusion
from utils.loss import GANLoss, VGGLoss, compute_feature_matching_loss


class Face2D3DFusionGAN(pl.LightningModule):
    """Generate the face image in by fusion 2D-3D information"""
    
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        if config is None:
            config = EasyDict(kwargs)
        self.config = config

        ## Define the Generator
        self.generator = Face2D3DFusion(config)
        
        if config.pretrained_gan_checkpoint is not None:
            print("Load pretrained GAN generator...")
            self.generator.load_from_checkpoint(config.pretrained_gan_checkpoint, config=config)

        ## Define the Discriminator
        self.discriminator = PatchGANDiscriminator(self.config['PatchGANDiscriminator'])

        self.criterionGAN = GANLoss()
        self.criterionVGG = VGGLoss()

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.config.lr, 
                                       weight_decay=self.config.wd, betas=(0.5, 0.999), eps=1e-06)

        scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=self.config.lr_decay_step,
                                                      gamma=self.config.lr_decay_rate)

        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))

        return ({"optimizer": optimizer_g, "lr_scheduler": scheduler_g},
                {'optimizer': optimizer_d})

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            ## ============= Train the Generator ============== ##
            ## 1) Forward the network
            model_output = self.generator(batch)

            ## 2) Calculate the loss
            loss_dict = self.compute_loss(batch, model_output)

            total_loss = 0.0
            for value in loss_dict.values():
                total_loss += value

            loss_3d = loss_dict['loss_s'] + loss_dict['lossg_e'] + loss_dict['lossg_em']

            self.log('total_recon_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_s', loss_dict['loss_s'], on_step=True, on_epoch=True, prog_bar=False)
            self.log('lossg_e', loss_dict['lossg_e'], on_step=True, on_epoch=True, prog_bar=False)
            self.log('lossg_em', loss_dict['lossg_em'], on_step=True, on_epoch=True, prog_bar=False)
            self.log('loss_3d', loss_3d, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_2d_l1', loss_dict['loss_2d_l1'], on_step=True, on_epoch=True, prog_bar=True)

            g = model_output['face_2d_image']
            gt = batch['gt_face_image']

            ## Add GAN related loss
            pred_real = self.discriminator(gt)
            pred_fake = self.discriminator(g)
            loss_G_GAN = self.criterionGAN(pred_fake, True)

            loss_vgg = self.criterionVGG(g, gt) * self.config.lambda_feat # Perceptual loss
            
            loss_FM = compute_feature_matching_loss(pred_fake, pred_real, self.config['PatchGANDiscriminator'])
            
            total_loss += loss_G_GAN + loss_vgg + loss_FM

            self.log("total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('loss_G_GAN', loss_G_GAN, on_step=True, on_epoch=True, prog_bar=False)
            self.log('loss_vgg', loss_vgg, on_step=True, on_epoch=True, prog_bar=False)
            self.log('loss_FM', loss_FM, on_step=True, on_epoch=True, prog_bar=False)

            return total_loss
        
        if optimizer_idx == 1:
            ## ============= Train the Generator ============== ##
            ## 1) Forward the network
            pred_real = self.discriminator(batch['gt_face_image'])
            pred_fake = self.discriminator(self.generator(batch)['face_2d_image'].detach())

            loss_D_real = self.criterionGAN(pred_real, True) * 2
            loss_D_fake = self.criterionGAN(pred_fake, False)
            
            total_loss_D = (loss_D_real + loss_D_fake) * 0.5

            self.log('total_loss_D', total_loss_D, on_step=True, on_epoch=True, prog_bar=False)
            return total_loss_D
    
    def validation_step(self, batch, batch_idx):
        ## 1) Forward the network
        model_output = self.generator(batch)

        ## 2) Calculate the loss
        loss = self.compute_loss(batch, model_output)

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
                                  audio_array=batch['raw_audio'])

    def compute_loss(self, data_dict, model_output):
        ## 3D loss
        pred_params = model_output['face_3d_params']
        tgt_params = data_dict['gt_face_3d_params']

        motionlogits = pred_params[:, 1:, :] - pred_params[:, :-1, :]
        tgt_motion = tgt_params[:, 1:, :] - tgt_params[:, :-1, :]

        loss_s = 10 * (F.smooth_l1_loss(pred_params[:, :1, :], tgt_params[:, :1, :]))
        lossg_e = 20 * F.smooth_l1_loss(pred_params[:, :, :], tgt_params[:, :, :])
        lossg_em = 200 * F.smooth_l1_loss(motionlogits[:,:,:], tgt_motion[:,:,:])

        ## 2D loss
        pred_face_image = model_output['face_2d_image']
        tgt_face_image = data_dict['gt_face_image']
        loss_2d = F.l1_loss(pred_face_image, tgt_face_image) * 100.0
        
        loss_dict = {'loss_s': loss_s, 'lossg_e': lossg_e, 'lossg_em': lossg_em,
                     'loss_2d_l1': loss_2d}
        return loss_dict

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


