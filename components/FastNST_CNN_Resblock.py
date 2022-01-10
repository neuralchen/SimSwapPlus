#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Generator_gpt_LN_encoder copy.py
# Created Date: Saturday October 9th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 19th October 2021 7:35:08 pm
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

        padding_size = int((k_size -1)/2)
        
        self.resblock_list = []

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size=k_size, stride=1, padding=1, bias= False),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size=k_size, stride=2, padding=1,bias =False), # 
            nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn*2, out_channels = chn*4, kernel_size=k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn*4  , out_channels = chn * 4, kernel_size=k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            nn.LeakyReLU(),
        )
        for _ in range(res_num):
            self.resblock_list += [ResBlock(chn * 4,k_size),]
        self.resblocks = nn.Sequential(*self.resblock_list) 
        self.decoder = nn.Sequential(
            # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.LeakyReLU(),
            # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.LeakyReLU(),
            DeConv(in_channels = chn * 4, out_channels = chn *2, kernel_size=k_size),
            nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
            nn.LeakyReLU(),
            DeConv(in_channels = chn * 2, out_channels = chn * 2 , kernel_size=k_size),
            nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
            nn.ReLU(),
            DeConv(in_channels = chn *2, out_channels = chn, kernel_size=k_size),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.ReLU(),
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
        x2 = self.resblocks(x2)
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