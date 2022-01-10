#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Generator_gpt_LN_encoder copy.py
# Created Date: Saturday October 9th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 11th October 2021 5:22:22 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from components.ResBlock import ResBlock
from components.DeConv   import DeConv

class ImageLN(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.layer = nn.LayerNorm(dim)
    def forward(self, x):
        y = self.layer(x.permute(0,2,3,1)).permute(0,3,1,2)
        return y

class Generator(nn.Module):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__()

        chn         = kwargs["g_conv_dim"]
        k_size      = kwargs["g_kernel_size"]
        res_num     = kwargs["res_num"]
        class_num   = kwargs["n_class"]
        window_size = kwargs["window_size"]
        image_size  = kwargs["image_size"]

        padding_size = int((k_size -1)/2)
        
        self.resblock_list = []
        embed_dim       = 96
        window_size     = 8
        num_heads       = 8
        mlp_ratio       = 2.
        norm_layer      = nn.LayerNorm
        qk_scale        = None
        qkv_bias        = True
        self.patch_norm = True
        self.lnnorm     = norm_layer(embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size=k_size, stride=1, padding=1, bias= False),
            nn.InstanceNorm2d(chn),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size=k_size, stride=2, padding=1,bias =False), # 
            nn.InstanceNorm2d(chn * 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn*2, out_channels = embed_dim, kernel_size=k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(embed_dim),
            nn.LeakyReLU(),
        )

        # self.encoder2 = nn.Sequential(
            
        #     nn.Conv2d(in_channels = chn*4  , out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
        #     ImageLN(chn * 8),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
        #     ImageLN(chn * 8),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
        #     ImageLN(chn * 8),
        #     nn.LeakyReLU()
        # )
        self.decoder = nn.Sequential(
            # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.LeakyReLU(),
            # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.LeakyReLU(),
            # DeConv(in_channels = chn * 8, out_channels = chn *4, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            # nn.LeakyReLU(),
            DeConv(in_channels = embed_dim, out_channels = chn * 2 , kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
            nn.InstanceNorm2d(chn * 2),
            nn.LeakyReLU(),
            DeConv(in_channels = chn *2, out_channels = chn, kernel_size=k_size),
            nn.InstanceNorm2d(chn),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn, out_channels =3, kernel_size=k_size, stride=1, padding=1,bias =True)
        )


    #     self.__weights_init__()

    # def __weights_init__(self):
    #     for layer in self.encoder:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    #     for layer in self.encoder2:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        x2 = self.encoder(input)
        out = self.decoder(x2)
        return out
    
if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 1024
    width = 1024
    model = Generator()
    print(model)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)