'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-21 11:21:27
Email: haimingzhang@link.cuhk.edu.cn
Description: Some customized defined loss functions
'''

import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from .basic_models import Vgg19


class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        weight = -torch.log(1 - target + 1e-5)
        weighted_diff = torch.multiply((pred - target), weight)
        
        return torch.mean(torch.pow(weighted_diff, 2))


def cosine_loss(pred, target):
    ## Normalize the data
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    cosine_score = nn.functional.cosine_similarity(pred, target, dim=-1)
    
    cosine_loss = 1 - cosine_score # convert to [0, 2]
    loss = torch.mean(cosine_loss).to(pred)
    return loss


def pixel_wise_loss(pred, target, mask=None, weight=1.0):
    """Calculate the Pixel-Wise loss and if given mask matrix, we
       consider the mask region with different weight

    Args:
        pred (Tensor): (B, C, H, W)
        target (Tensor): (B, C, H, W)
        mask (Tensor, optional): (B, 1, H, W). Defaults to None.
        weight (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: loss value
    """
    l1_dist = F.l1_loss(pred, target, reduction="none")

    if mask is None:
        return torch.mean(l1_dist)
    
    loss1 = weight * l1_dist * mask
    loss2 = l1_dist * (1 - mask)
    
    loss = torch.mean(loss1 + loss2)
    return loss
    

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = torch.Tensor(input.size()).type_as(input).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = torch.Tensor(input.size()).type_as(input).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input: Tensor, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def compute_feature_matching_loss(pred_fake, pred_real, opt):
    feat_weights = 4.0 / (opt.n_layers_D + 1)
    D_weights = 1.0 / opt.num_D
    
    loss_G_GAN_Feat = 0.0

    for i in range(min(len(pred_fake), opt.num_D)):
        for j in range(len(pred_fake[i])):
            loss_G_GAN_Feat += D_weights * feat_weights * \
                F.l1_loss(pred_fake[i][j], pred_real[i][j].detach()) * opt.lambda_feat
    return loss_G_GAN_Feat

class GANLoss_origin(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss_origin, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None        
        gpu_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).cuda(gpu_id).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)                
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, x, y):
        Gx = gram_matrix(x)
        Gy = gram_matrix(y)
        return F.mse_loss(Gx, Gy) * 30000000


class VGGLoss(nn.Module):
    """Adapted from Pix2PixHD
    """
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):
        B, T, C, H, W = x.shape
        x = x.reshape((-1, C, H, W)) # to (B*T, C, H, W)
        y = y.reshape((-1, C, H, W))

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


class VGGLossOrigin(nn.Module):
    """Perceputal Loss by using VGG network

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model=None):
        super(VGGLoss, self).__init__()
        if model is None:
            self.vgg = Vgg19()
        else:
            self.vgg = model

        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.style_criterion = StyleLoss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y, style=False):
        B, T, C, H, W = x.shape
        x = x.reshape((-1, C, H, W)) # to (B*T, C, H, W)
        y = y.reshape((-1, C, H, W))
        
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        if style:
            # return both perceptual loss and style loss.
            style_loss = 0
            for i in range(len(x_vgg)):
                this_loss = (self.weights[i] *
                             self.criterion(x_vgg[i], y_vgg[i].detach()))
                this_style_loss = (self.style_weights[i] *
                                   self.style_criterion(x_vgg[i], y_vgg[i].detach()))
                loss += this_loss
                style_loss += this_style_loss
            return loss, style_loss

        for i in range(len(x_vgg)):
            this_loss = (self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach()))
            loss += this_loss
        return loss


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

