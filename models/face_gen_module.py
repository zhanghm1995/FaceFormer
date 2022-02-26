'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-25 21:25:35
Email: haimingzhang@link.cuhk.edu.cn
Description: Face generation Module class with different GAN related losses and Discriminators
'''

import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from .mm_fusion_transformer import MMFusionFormer
from .discriminators.img_gen_disc import Feature2Face_D
from utils import loss
from utils.model_serializer import ModelSerializer


class FaceGenModule(object):
    def __init__(self, opt, device) -> None:
        self.isTrain = opt.isTrain
        self.device = device
        
        self.opt = opt

        ## Define the models
        self.net_G = MMFusionFormer(self.opt, device).to(self.device)

        ## Define the criterions
        if self.isTrain:
            self.net_D = Feature2Face_D(self.opt['discriminator']).to(self.device)

            ## Define the optimizers
            net_params_G = [p for p in self.net_G.parameters() if p.requires_grad]
            self.optimizer_G = torch.optim.Adam(net_params_G, lr=1e-4)

            net_params_D = [p for p in self.net_D.parameters() if p.requires_grad]
            self.optimizer_D = torch.optim.Adam(net_params_D, lr=1e-4)

            self.loss_names_G = ['L1', 'VGG', 'loss_G_GAN', 'loss_G_FM', 'loss_face_3d'] # Generator loss
            self.loss_names_D = ['D_real', 'D_fake']                     # Discriminator loss

            self.criterionGAN = loss.GANLoss('ls').to(device)
            self.criterionL1 = nn.L1Loss().to(device)
            self.criterionVGG = loss.VGGLoss().to(device)
            self.criterionL2 = nn.MSELoss().to(self.device)

        ## Init weights
        self._init_weights()


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, data_dict):
        ## 1) Set model to train mode
        self.net_G.train()
        self.net_D.train()

        ## 2) Forward the network
        model_output = self.net_G(data_dict, shift_target_right=False) # Generator results
        return model_output

    def validate(self, data_dict, autoregressive=False, return_loss=False):
        ## 1) Set model to train mode
        self.net_G.eval()

        ## 2) Forward the network
        model_output = self.net_G(data_dict, shift_target_right=False) # Generator results

        if not return_loss:
            return model_output

        ## 3) Calculate the loss
        # GAN loss
        tgt_image = data_dict['gt_face_image']
        tgt_face_3d_params = data_dict['gt_face_3d_params']

        fake_pred = model_output['face_image']
        pred_face_3d_params = model_output['face_3d_params']

        pred_real = self.net_D(tgt_image)
        pred_fake = self.net_D(fake_pred)

        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1, vgg, style loss
        loss_l1 = self.criterionL1(fake_pred, tgt_image) * self.opt.lambda_L1

        loss_vgg = self.criterionVGG(fake_pred, tgt_image, style=False)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat 
        # loss_style = torch.mean(loss_style) * self.opt.lambda_feat 

        # feature matching loss
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real)

        ## Face 3DMM parameters loss
        loss_face_3d_params = self.criterionL2(pred_face_3d_params, tgt_face_3d_params) * 10.0
        
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_FM + loss_face_3d_params

        val_loss_dict = {'total_loss_G': loss_G,
                         'loss_G_GAN': loss_G_GAN,
                         'loss_l1': loss_l1,
                         'loss_vgg': loss_vgg,
                         'loss_FM': loss_FM,
                         'loss_face_3d_params': loss_face_3d_params}

        return model_output, val_loss_dict

    def inference(self, data_dict):
        pass

    def optimize_parameters(self, data_dict, model_pred):
        assert isinstance(data_dict, Dict)

        ## ============== Update Discrimator ==================== ##
        self.set_requires_grad(self.net_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D(data_dict, model_pred)                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        ## ============== Update Generator ==================== ##
        self.set_requires_grad(self.net_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(data_dict, model_pred)                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def compute_FeatureMatching_loss(self, pred_fake, pred_real):  # feature matching loss 有利于稳定GAN训练的损失
        # GAN feature matching loss
        loss_FM = torch.zeros(1).cuda()
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        D_weights = 1.0 / self.opt.num_D
        for i in range(min(len(pred_fake), self.opt.num_D)):
            for j in range(len(pred_fake[i])):
                loss_FM += D_weights * feat_weights * \
                        self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        return loss_FM

    def backward_D(self, input_data_dict, model_pred):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        pred_real = self.net_D(input_data_dict['gt_face_image'])
        pred_fake = self.net_D(model_pred['face_image'].detach())

        loss_D_real = self.criterionGAN(pred_real, True) * 2
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5 
        
        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))   
        
        self.loss_D.backward()

    def backward_G(self, input_data_dict, model_pred):
        """Calculate GAN and other loss for the generator"""
        # GAN loss
        gt_image = input_data_dict['gt_face_image']
        gt_face_3d_params = input_data_dict['gt_face_3d_params']

        ## Model prediction
        fake_pred = model_pred['face_image']
        pred_face_3d_params = model_pred['face_3d_params']

        pred_real = self.net_D(gt_image)
        pred_fake = self.net_D(fake_pred)

        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1, vgg, style loss
        loss_l1 = self.criterionL1(fake_pred, gt_image) * self.opt.lambda_L1

        loss_vgg = self.criterionVGG(fake_pred, gt_image, style=False)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat 
        # loss_style = torch.mean(loss_style) * self.opt.lambda_feat 

        # feature matching loss
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real)

        ## Face 3DMM parameters loss
        loss_face_3d_params = self.criterionL2(pred_face_3d_params, gt_face_3d_params) * 10.0
        
        # combine loss and calculate gradients
        self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_FM + loss_face_3d_params
        self.loss_G.backward()
        
        ## Combine the discriminator and generator losses together
        self.loss_dict = {**self.loss_dict, **dict(zip(self.loss_names_G, 
                                                       [loss_l1, loss_vgg, loss_G_GAN, loss_FM, loss_face_3d_params]))}

    def _init_weights(self):
        pass


if __name__ == "__main__":
    print("hello")
