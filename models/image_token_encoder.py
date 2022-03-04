'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-02-21 21:36:41
Email: haimingzhang@link.cuhk.edu.cn
Description: Encode a sequence of images into a embedding tokens for Transformer input
'''

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class ImageTokenEncoder(nn.Module):
    def __init__(self, image_size=96, in_ch=3):
        super(ImageTokenEncoder, self).__init__()

        self.image_size = image_size

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(in_ch, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96
        
        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 

    def encode(self, input: Tensor):
        """Encode the batched input image sequences into tokens

        Args:
            input (Tensor): (B, T, 3, H, W)

        Returns:
            (Tensor): (B, T, C)
        """
        # input (B, T, C, H, W)
        B, T, C, H, W = input.shape
        input = input.reshape((-1, C, H, W))

        self.feats = []
        x = input
        for f in self.face_encoder_blocks:
            x = f(x)
            self.feats.append(x)
        
        ## Convert to (B, T, C)
        output = self.feats[-1] # 4-d tensor
        output = output.reshape(B, T, -1)
        return output
    
    def decode(self, input: Tensor):
        """Decode the embedding to whole image

        Args:
            input (Tensor): (B, T, C)

        Raises:
            e: _description_

        Returns:
            Tensor: (B, T, 3, H, W)
        """
        B, T, C = input.shape

        input = input.reshape(-1, C, 1, 1)
        x = input
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, self.feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            self.feats.pop()
        
        x = self.output_block(x) # (B, C, H, W)

        output = x.reshape((B, T, 3, self.image_size, self.image_size))
        return output

    def forward(self, audio_sequences, face_sequences):
        # face_sequences = (B, C, T, 96, 96)

        face_encoder_blocks = self.face_encoder_blocks
        face_decoder_blocks = self.face_decoder_blocks

        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0) # 80x6x96x96

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            feats.pop()
        
        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs


class ImageTokenEncoder224(nn.Module):
    """Image tokenization encoder for input (224, 224) image
    """
    def __init__(self, image_size=224, in_ch=6):
        super(ImageTokenEncoder224, self).__init__()

        self.image_size = image_size

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(in_ch, 16, kernel_size=7, stride=1, padding=3)), # 224,224

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 112,112
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 56,56
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 28,28
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 14,14
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=2, padding=1),     # 7,7
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=0),     # 3, 3
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=0, output_padding=0),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),
            ]) 

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 
        
        self.audio_fc = nn.Linear(128, 512)

    def encode(self, input: Tensor):
        """Encode the batched input image sequences into tokens

        Args:
            input (Tensor): (B, T, 3, H, W)

        Returns:
            (Tensor): (B, T, C)
        """
        # input (B, T, C, H, W)
        B, T, C, H, W = input.shape
        input = input.reshape((-1, C, H, W))

        self.feats = []
        x = input
        for f in self.face_encoder_blocks:
            x = f(x)
            self.feats.append(x)
        
        ## Convert to (B, T, C)
        output = self.feats[-1] # 4-d tensor
        output = output.reshape(B, T, -1)
        return output
    
    def decode(self, input: Tensor):
        """Decode the embedding to whole image

        Args:
            input (Tensor): (B, T, C)

        Raises:
            e: _description_

        Returns:
            Tensor: (B, T, 3, H, W)
        """
        B, T, C = input.shape

        input = input.reshape(-1, C, 1, 1)
        x = input
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, self.feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            self.feats.pop()
        
        x = self.output_block(x) # (B, C, H, W)

        output = x.reshape((B, T, 3, self.image_size, self.image_size))
        return output

    def forward(self, audio_embedding, face_sequences):
        """Generate the image by using audio embedding and masked face sequences

        Args:
            audio_embedding (_type_): (B, S, E)
            face_sequences (Tensor): (B, T, 3, 224, 224)

        Raises:
            e: _description_
        """
        B, T, C, H, W = face_sequences.shape
        face_sequences = face_sequences.reshape((-1, C, H, W))

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        
        x = self.audio_fc(audio_embedding)[..., None, None]
        B, T, C = x.shape[:3]

        x = x.reshape(-1, C, 1, 1)
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            feats.pop()
        
        x = self.output_block(x)
        output = x.reshape((B, T, 3, self.image_size, self.image_size))
        return output


class ImageTokenEncoder192(nn.Module):
    """Image tokenization encoder for input (192, 192) image
    """
    def __init__(self, image_size=192, in_ch=6):
        super(ImageTokenEncoder192, self).__init__()

        self.image_size = image_size

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(in_ch, 16, kernel_size=15, stride=1, padding=7)), # 96*2,96*2      # kernel_size和 padding变大了

            nn.Sequential(Conv2d(16, 32, kernel_size=11, stride=2, padding=5), # 48*2,48*2      # kernel_size和 padding变大了
                        Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=7, stride=2, padding=3),    # 24*2,24*2      # kernel_size和 padding变大了
                        Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12*2,12*2
                        Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6*2,6*2
                        Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                        Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3*2,3*2
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),


            ####################下面为新增加的1个下采样快######################
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3,3                # 新增加的卷积层
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),    #新增加的卷积层
            # ######################上面为新增加的1个下采样快######################


            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
                        Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])    #1*1

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),              #  audio_embedding

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3    #(N-1)*S-2P+K    audio_embedding +
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            ####################下面为新增加的1个 上采样快######################
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),       #新增加的卷积层
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),                    #新增加的卷积层
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12        #新增加的卷积层
            ####################上面为新增加的1个 上采样快######################


            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 96, 96

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 192,192    
        
        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 
        
        self.audio_fc = nn.Linear(128, 512)

    def forward(self, audio_embedding, face_sequences):
        """Generate the image by using audio embedding and masked face sequences

        Args:
            audio_embedding (_type_): (B, S, E)
            face_sequences (Tensor): (B, T, 3, 224, 224)

        Raises:
            e: _description_
        """
        B, T, C, H, W = face_sequences.shape
        face_sequences = face_sequences.reshape((-1, C, H, W))

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)
        
        x = self.audio_fc(audio_embedding)[..., None, None]
        B, T, C = x.shape[:3]

        x = x.reshape(-1, C, 1, 1)
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                raise e
            
            feats.pop()
        
        x = self.output_block(x)
        output = x.reshape((B, T, 3, self.image_size, self.image_size))
        return output


if __name__ == "__main__":
    image_token_encoder = ImageTokenEncoder()

    input = torch.randn(2, 100, 6, 96, 96)
    output = image_token_encoder.encode(input)
    print(output.shape)

    output_image = image_token_encoder.decode(output)
    print(output_image.shape)