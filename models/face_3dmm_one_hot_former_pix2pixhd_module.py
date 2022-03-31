'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-22 11:04:16
Email: haimingzhang@link.cuhk.edu.cn
Description: PL module to train model
'''

from collections import OrderedDict
import torch
import numpy as np
import os
import os.path as osp
import torch.nn as nn
from scipy.io import wavfile
from torch.nn import functional as F 
import pytorch_lightning as pl
from PIL import Image
from .face_3dmm_one_hot_former import Face3DMMOneHotFormer
from .nn import define_G, define_D
from .losses import photo_loss, VGGLoss, GANLoss
from .bfm import ParametricFaceModel
from .nvdiffrast_utils import MeshRenderer
from utils.save_data import save_image_array_to_video


class Face3DMMOneHotFormerPix2PixHDModule(pl.LightningModule):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()

        self.config = config

        self.save_hyperparameters()

        ## Define the model
        self.face3dmmformer_model = Face3DMMOneHotFormer(config['Face3DMMFormer'])

        self.mouth_mask_weight = self.face3dmmformer_model.mouth_mask_weight

        ## Define the Generator
        self.face_generator = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

        ## Define the Discriminator
        self.netD = define_D(6, 64, 3, norm="instance", use_sigmoid=False, num_D=2,
                             getIntermFeat=False)

        opt = self.config['FaceRendererParameters']

        self.facemodel = ParametricFaceModel(
            bfm_folder=opt.bfm_folder, camera_distance=opt.camera_d, focal=opt.focal, center=opt.center,
            is_train=True
        )

        fov = 2 * np.arctan(opt.center / opt.focal) * 180 / np.pi
        self.face_renderer = MeshRenderer(
            rasterize_fov=fov, znear=opt.z_near, zfar=opt.z_far, rasterize_size=int(2 * opt.center)
        )

        ## Define criterions
        self.criterionPhoto = photo_loss
        self.criterionVGG = VGGLoss()
        self.criterionGAN = GANLoss()

        self.criterion = nn.MSELoss()
    
    def configure_optimizers(self):
        # optimizer_face_3dmm = torch.optim.Adam(filter(lambda p: p.requires_grad, self.face3dmmformer_model.parameters()), 
        #                                        lr=1e-5)
        
        optimizer_G = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.face3dmmformer_model.parameters()), 'lr': 1e-4},
             {'params': filter(lambda p: p.requires_grad, self.face_generator.parameters())}], 
             lr=self.config.lr, 
             weight_decay=self.config.wd,
             betas=(0.9, 0.999), 
             eps=1e-06)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=self.config.lr_decay_step,
                                                    gamma=self.config.lr_decay_rate)

        optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.config.lr, betas=(0.9, 0.999))
        
        return ({"optimizer": optimizer_G, "lr_scheduler": scheduler},
                {'optimizer': optimizer_D})

    def forward(self, batch):
        ## Forward the Face3DMMFormer
        pred_expression = self.face3dmmformer_model(
            batch, teacher_forcing=False, return_loss=False) # (B, S, 64)
        
        face_coeffs = batch['gt_face_origin_3d_params'] # (B, S, 257)
        # face_coeffs[:, :, 80:144] = pred_expression
        
        face_coeffs = face_coeffs.reshape((-1, 257)) # (B*S, 257)
        pred_expression = pred_expression.reshape((-1, 64))

        self.facemodel.to(self.device)
        ## Forward the renderer
        self.pred_shape, self.pred_vertex, self.pred_tex, self.pred_color, self.pred_lm = \
            self.facemodel.compute_for_render(face_coeffs, pred_exp=pred_expression)
        self.pred_mask, _, self.pred_face = self.face_renderer(
            self.pred_vertex, self.facemodel.face_buf, feat=self.pred_color)
        
        ## Forward the Generator network
        face_2d_img = self.face_generator(self.pred_face)
        
        return {'generated_face': face_2d_img}
    
    def generator_step(self, batch):
        ## 1) Forward the network
        self.model_output = self(batch)

        ## ============= Train the Generator ============== ##
        ## 2) Compute the loss
        loss_dict = OrderedDict()
        ## 3D face loss
        vertice = batch['face_vertex'] # GT
        batch_size, seq_len = vertice.shape[:2]

        ## If consider mouth region weight
        vertice_out = self.pred_shape.reshape((batch_size, seq_len, -1, 3))
        vertice = vertice.reshape((batch_size, seq_len, -1, 3))

        loss_3d = torch.sum((vertice_out - vertice)**2, dim=-1) * self.mouth_mask_weight[None, ...].to(vertice)
        loss_3d = torch.mean(loss_3d)
        loss_dict['loss_3d'] = loss_3d

        ## Get the images
        input_image = self.pred_face
        fake_image = self.model_output['generated_face']
        real_image = batch['gt_masked_face_image'].reshape(fake_image.shape)

        ## photo loss
        # face_mask = self.pred_mask
        # if self.config.use_crop_face:
        #     face_mask, _, _ = self.renderer(self.pred_vertex, self.facemodel.front_face_buf)

        # face_mask = face_mask.detach()
        # loss_photo = self.criterionPhoto(input_image, real_image, face_mask)
        loss_photo = F.l1_loss(input_image, real_image)
        loss_dict['loss_photo'] = self.config.w_color * loss_photo

        ## GAN loss
        pred_fake = self.netD.forward(torch.cat((input_image, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_dict['loss_G_GAN'] = loss_G_GAN

        ## VGG loss
        loss_G_VGG = self.criterionVGG(fake_image, real_image)
        loss_dict['loss_G_VGG'] = self.config.w_G_VGG * loss_G_VGG
        
        ## FM loss

        loss_total = loss_dict['loss_3d'] + loss_dict['loss_photo'] + loss_dict['loss_G_GAN'] + loss_dict['loss_G_VGG']
        self.log_dict(loss_dict, prog_bar=True)
        return loss_total

    def discriminator_step(self, batch):
        ## ============= Train the Discriminator ============== ##
        self.model_output = self.forward(batch)
        
        fake_image = self.model_output['generated_face']
        input_image = self.pred_face
        real_image = batch['gt_masked_face_image'].reshape(fake_image.shape)

        pred_real = self.netD.forward(torch.cat((input_image, real_image.detach()), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)

        pred_fake = self.netD.forward(torch.cat((input_image, fake_image.detach()), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        self.log('Disc/loss_D', loss_D, on_step=True, on_epoch=True)
        self.log('Disc/loss_D_real', loss_D_real, on_step=True, on_epoch=True)
        self.log('Disc/loss_D_fake', loss_D_fake, on_step=True, on_epoch=True)
        return loss_D

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            loss = self.generator_step(batch)
        
        if optimizer_idx == 1:
            loss = self.discriminator_step(batch)
        
        if batch_idx % 1 == 0:
            generate_image = self.model_output['generated_face'][None, ...].detach()
            save_dir = osp.join(self.logger.log_dir, "vis", f"epoch_{self.current_epoch:03d}")
            save_image_array_to_video(generate_image, 
                                      save_dir, 
                                      name=batch_idx)
            
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
        # vertice = batch['face_vertex']
        video_name = batch['video_name'][0]
        
        model_output = self.model.predict(batch)
        model_output = model_output.squeeze().detach().cpu().numpy() # (seq_len, 64)
        
        ## Save the results
        save_dir = osp.join(self.logger.log_dir, "vis")
        os.makedirs(save_dir, exist_ok=True)
        # np.savez(osp.join(save_dir, f"{batch_idx:03d}.npz"), face=model_output)
        np.save(osp.join(save_dir, f"{video_name}_{batch_idx:03d}.npy"), model_output) # save face vertex

        ## Save audio
        audio_data = audio[0].cpu().numpy()
        wavfile.write(osp.join(save_dir, f"{video_name}_{batch_idx:03d}.wav"), 16000, audio_data)
            