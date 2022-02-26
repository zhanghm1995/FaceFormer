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

    def train(self, data_dict):
        ## 1) Set model to train mode
        self.net_G.train()
        self.net_D.train()

        ## 2) Forward the network
        self.forward(data_dict)

        ## 3) Optimize the network
        self.optimize_parameters()

        ## 4) Output
        return self.loss_dict

    def validate(self, data_dict, autoregressive=False):
        ## 1) Set model to train mode
        self.net_G.val()

        ## 2) Forward the network
        self.forward(data_dict)

        return self.loss_dict

    def inference(self, data_dict):
        pass

    def prepare_data(self, data_dict):
        ## Build the input
        masked_gt_image = data_dict['gt_face_image'].clone().detach() # (B, T, 3, H, W)
        masked_gt_image[:, :, :, masked_gt_image.shape[3]//2:] = 0.
        data_dict['input_image'] = torch.concat([masked_gt_image, data_dict['ref_face_image']], dim=2)

        ## Move to GPU
        for key, value in data_dict.items():
            data_dict[key] = value.to(self.device)
        return data_dict


    def forward(self, data_dict):
        self.data_dict = self.prepare_data(data_dict)

        self.tgt_image = data_dict['gt_face_image']
        self.tgt_face_3d_params = data_dict['face_3d_params']

        ## Forward the network
        self.model_output = self.net_G(self.data_dict) # Generator results
        
        ## Get the output
        self.fake_pred = self.model_output['face_image'] # Generator predicted face image in (B, T, 3, H, W)
        self.pred_face_3d_params = self.model_output['face_3d_params']

    def optimize_parameters(self):
        ## ============== Update Discrimator ==================== ##
        self.set_requires_grad(self.net_D, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        ## ============== Update Generator ==================== ##
        self.set_requires_grad(self.net_D, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
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

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # GAN loss
        pred_real = self.net_D(self.tgt_image)
        pred_fake = self.net_D(self.fake_pred.detach())
        loss_D_real = self.criterionGAN(pred_real, True) * 2
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5 
        
        self.loss_dict = dict(zip(self.loss_names_D, [loss_D_real, loss_D_fake]))   
        
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and other loss for the generator"""
        # GAN loss
        pred_real = self.net_D(self.tgt_image)
        pred_fake = self.net_D(self.fake_pred)

        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1, vgg, style loss
        loss_l1 = self.criterionL1(self.fake_pred, self.tgt_image) * self.opt.lambda_L1

        loss_vgg = self.criterionVGG(self.fake_pred, self.tgt_image, style=False)
        loss_vgg = torch.mean(loss_vgg) * self.opt.lambda_feat 
        # loss_style = torch.mean(loss_style) * self.opt.lambda_feat 

        # feature matching loss
        loss_FM = self.compute_FeatureMatching_loss(pred_fake, pred_real)

        ## Face 3DMM parameters loss
        loss_face_3d_params = self.criterionL2(self.pred_face_3d_params, self.tgt_face_3d_params)
        
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
