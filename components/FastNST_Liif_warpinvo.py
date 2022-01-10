#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: FastNST_Liif.py
# Created Date: Thursday October 14th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 19th October 2021 8:47:28 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from components.ResBlock import ResBlock
from components.DeConv   import DeConv
from components.Liif_invo import LIIF


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
        batch_size  = kwargs["batch_size"]
        # mlp_in_dim  = kwargs["mlp_in_dim"]
        # mlp_out_dim = kwargs["mlp_out_dim"]


        padding_size = int((k_size -1)/2)
        
        self.resblock_list = []
        embed_dim       = 96
        norm_layer      = nn.LayerNorm

        self.img_token = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size=k_size, stride=1, padding=1, bias= False),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size=k_size, stride=2, padding=1,bias =False), # 
            nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels = chn*2, out_channels = chn*4, kernel_size=k_size, stride=2, padding=1,bias =False),
            # nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels = chn*4  , out_channels = chn * 4, kernel_size=k_size, stride=2, padding=1,bias =False),
            # nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            # nn.LeakyReLU(),
        )
        image_size     = image_size // 2
        self.downsample1 = LIIF(chn * 2, chn * 4)
        self.downsample1.gen_coord((batch_size, \
                    chn,image_size,image_size),(image_size//2,image_size//2))
        image_size      = image_size // 2
        self.downsample2 = LIIF(chn * 4, chn * 4)
        self.downsample2.gen_coord((batch_size, \
                    chn,image_size,image_size),(image_size//2,image_size//2))


        for _ in range(res_num):
            self.resblock_list += [ResBlock(chn * 4,k_size),]
        self.resblocks = nn.Sequential(*self.resblock_list)
        # self.decoder = nn.Sequential(
        #     # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
        #     # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
        #     # nn.LeakyReLU(),
        #     # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
        #     # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
        #     # nn.LeakyReLU(),
        #     DeConv(in_channels = chn * 4, out_channels = chn *2, kernel_size=k_size),
        #     nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
        #     nn.LeakyReLU(),
        #     # DeConv(in_channels = chn * 2, out_channels = chn, kernel_size=k_size),
        #     # # nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
        #     # nn.InstanceNorm2d(chn, affine=True, momentum=0),
        #     # nn.LeakyReLU()
        #     # DeConv(in_channels = chn *2, out_channels = chn, kernel_size=k_size),
        #     # nn.InstanceNorm2d(chn),
        #     # nn.LeakyReLU(),
        #     # nn.Conv2d(in_channels = chn, out_channels =3, kernel_size=k_size, stride=1, padding=1,bias =True)
        # )
        image_size     = image_size // 2
        self.upsample1 = LIIF(chn*4, chn * 4)
        self.upsample1.gen_coord((batch_size, \
                    chn,image_size,image_size),(image_size*2,image_size*2))
        image_size     = image_size * 2
        self.upsample2 = LIIF(chn*4, chn * 2)
        self.upsample2.gen_coord((batch_size, \
                    chn,image_size,image_size),(image_size*2,image_size*2))
        # image_size     = image_size * 2
        # self.upsample2 = LIIF(chn, chn)
        # self.upsample2.gen_coord((batch_size, \
        #             chn,image_size,image_size),(image_size*2,image_size*2))
        self.decoder = nn.Sequential(
            DeConv(in_channels = chn * 2, out_channels = chn, kernel_size=k_size),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn, out_channels =3, kernel_size=k_size, stride=1, padding=1,bias =True)
        )            
        # self.out_conv = nn.Conv2d(in_channels = chn, out_channels =3, kernel_size=k_size, stride=1, padding=1,bias =True)
    #     self.__weights_init__()

    # def __weights_init__(self):
    #     for layer in self.encoder:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    #     for layer in self.encoder2:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        out = self.img_token(input)
        out = self.downsample1(out)
        out = self.downsample2(out)
        out = self.resblocks(out)
        
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.decoder(out)
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