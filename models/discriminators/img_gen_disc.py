'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-23 10:18:24
Email: haimingzhang@link.cuhk.edu.cn
Description: Image generation commonly used discriminators
'''

import torch
import torch.nn as nn
import numpy as np

class MultiscaleDiscriminator(nn.Module):
    #MultiscaleDiscriminator(23 + 3, opt.ndf, opt.n_layers_D, opt.num_D, not opt.no_ganFeat)
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3, getIntermFeat=True):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        ndf_max = 64
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, min(ndf_max, ndf*(2**(num_D-1-i))), n_layers, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]            
            for i in range(len(model)):
                result.append(model[i](result[-1]))            
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))                                
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)                    
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))    # 2
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm2d(nf), 
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)            


    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)      


from torch.cuda.amp import autocast as autocast

class Feature2Face_D(nn.Module):
    def __init__(self, opt):
        super(Feature2Face_D, self).__init__()
        # initialize
        self.opt = opt

        self.Tensor = torch.cuda.FloatTensor

        self.output_nc = opt.output_nc        

        ##################### define networks
        self.netD = MultiscaleDiscriminator(3, opt.ndf, opt.n_layers_D, opt.num_D, not opt.no_ganFeat)  ###PatchGAN判别器
                    
        print('---------- Discriminator networks initialized -------------') 
       
    #@autocast()    
    def forward(self, input):
        if self.opt.fp16:   #####默认情况下没有用这个
            with autocast():
                pred = self.netD(input)
        else:
            input = torch.cat([input[:, :, i] for i in range(input.size(2))], dim=0)         #####
            #注：生成器的输出是  torch.Size([6, 3, 5, 192, 192])   6是批大小
            #生成的图片是6个窗口，每个窗口5张图片，所以 把它变成30张图片,即 30，3 ，192，192，再输入到判别器当中
            pred = self.netD(input)
        return pred