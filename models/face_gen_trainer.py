'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-25 21:25:35
Email: haimingzhang@link.cuhk.edu.cn
Description: Face generation trainer class with different GAN related losses and Discriminators
'''

import torch
import torch.nn as nn
import numpy as np

from ..utils import loss


class FaceGenTrainer(object):
    def __init__(self, opt, device) -> None:
        self.isTrain = opt.isTrain
        
        ## Define the models
        self.net_G = None
        self.net_D = None

        ## Define the optimizers
        self.optimizer_G = None
        self.optimizer_D = None

        ## Define the criterions
        self.criterionGAN = loss.GANLoss(opt.gan_mode, tensor=self.Tensor).to(device)
        self.criterionL1 = nn.L1Loss().to(device)


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


    def forward(self, data_dict):
        self.input_image
        self.tgt_image = data_dict['gt_face_image']

        self.fake_pred = self.net_G(data_dict['face_image'])


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
        pred_real = self.net_D(real_AB)
        pred_fake = self.net_D(fake_AB.detach())
        loss_D_real = self.criterionGAN(pred_real, True) * 2
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5 
        
        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))   
        
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and other loss for the generator"""
        # GAN loss
        real_AB = torch.cat((self.input_feature_maps, self.tgt_image), dim=1)
        fake_AB = torch.cat((self.input_feature_maps, self.fake_pred), dim=1)
        pred_real = self.Feature2Face_D(real_AB)
        pred_fake = self.Feature2Face_D(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # L1, vgg, style loss
        loss_l1 = self.criterionL1(self.fake_pred, self.tgt_image) * self.opt.lambda_L1
#        loss_maskL1 = self.criterionMaskL1(self.fake_pred, self.tgt_image, self.facial_mask * self.opt.lambda_mask)
        loss_vgg, loss_style = self.criterionVGG(self.fake_pred, self.tgt_image, style=True)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat 
        loss_style = torch.mean(loss_style) * self.opt.lambda_feat 
        # feature matching loss
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real)
        
        # combine loss and calculate gradients
        self.loss_G = loss_G_GAN + loss_l1 + loss_vgg + loss_style + loss_FM #+ loss_maskL1
        self.loss_G.backward()
        
        self.loss_dict = {**self.loss_dict, **dict(zip(self.loss_names_G, [loss_l1, loss_vgg, loss_style, loss_G_GAN, loss_FM]))}


    def optimize_parameters(self, pred_):
        ## ============== Update Discrimator ==================== ##
        self.set_requires_grad(self.net_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D() # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        ## ============== Update Generator ==================== ##
        self.set_requires_grad(self.net_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights   

    def _init_weights(self):
        pass


if __name__ == "__main__":
    print("hello")
