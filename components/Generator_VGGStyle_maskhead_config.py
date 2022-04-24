#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Generator_Invobn_config1.py
# Created Date: Saturday February 26th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 6th April 2022 12:55:51 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import os

import torch
from torch import nn
import torch.nn.functional as F
import math


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=512,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.equal_var = math.sqrt(2)
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / self.equal_var
        return out

class AdainUpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=512,
                 actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.actv = actv
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.norm = AdaIN(style_dim, dim_out)

    def forward(self, x, s):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.norm(x, s)
        x = self.actv(x)
        return x


class Generator(nn.Module):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__()

        id_dim      = kwargs["id_dim"]
        k_size      = kwargs["g_kernel_size"]
        res_num     = kwargs["res_num"]
        in_channel  = kwargs["in_channel"]
        up_mode     = kwargs["up_mode"]
        norm        = kwargs["norm"]
        
        aggregator  = kwargs["aggregator"]
        res_mode    = kwargs["res_mode"]

        padding_size= int((k_size -1)/2)
        padding_type= 'reflect'
        
        
        activation = nn.LeakyReLU(0.2)
        # activation = nn.ReLU()

        self.from_rgb = nn.Conv2d(3, in_channel, 1, 1, 0)
        # self.first_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, bias=False), # 256
                                nn.BatchNorm2d(in_channel), activation)
                                
        self.down2 = nn.Sequential(nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=2, padding=1, bias=False), # 128
                                nn.BatchNorm2d(in_channel*2), activation)

        self.down3 = nn.Sequential(nn.Conv2d(in_channel*2, in_channel*4, kernel_size=3, stride=2, padding=1, bias=False), # 64
                                nn.BatchNorm2d(in_channel*4), activation)

        self.down4 = nn.Sequential(nn.Conv2d(in_channel*4, in_channel*8, kernel_size=3, stride=2, padding=1, bias=False), # 32
                                nn.BatchNorm2d(in_channel*8), activation)
        
        self.down5 = nn.Sequential(nn.Conv2d(in_channel*8, in_channel*8, kernel_size=3, stride=2, padding=1, bias=False), # 32
                                nn.BatchNorm2d(in_channel*8), activation)

        # self.down6 = ResBlk(in_channel*8, in_channel*8, normalize=True, downsample=True)# 8
        self.maskhead = nn.Sequential(
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel*8, in_channel, kernel_size=3, stride=1, padding=1,bias=False),
                                nn.BatchNorm2d(in_channel), # 32
                                activation,
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel, in_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(in_channel//2), # 64
                                activation,
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel//2, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.Sigmoid()
        )
        
        ### resnet blocks
        BN = []
        for i in range(res_num):
            BN += [
                AdainResBlk(in_channel*8, in_channel*8, style_dim=id_dim, upsample=False)]
        self.BottleNeck = nn.Sequential(*BN)

        # self.up6 = AdainResBlk(in_channel*8, in_channel*8, style_dim=id_dim, upsample=True) # 16
        
        self.up5 = AdainUpBlock(in_channel*8, in_channel*8, style_dim=id_dim) # 32

        self.up4 = AdainUpBlock(in_channel*8, in_channel*4, style_dim=id_dim) # 64

        self.up3 = AdainUpBlock(in_channel*4, in_channel*2, style_dim=id_dim) # 128
        
        self.up2 = AdainUpBlock(in_channel*2, in_channel, style_dim=id_dim)

        self.up1 = AdainUpBlock(in_channel, in_channel, style_dim=id_dim)
        # ResUpBlk(in_channel, in_channel, normalize="in") # 512


        self.to_rgb = nn.Sequential(nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channel, 3, kernel_size=3, padding=0))


    #     self.__weights_init__()

    # def __weights_init__(self):
    #     for layer in self.encoder:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    #     for layer in self.encoder2:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    def forward(self, img, id):
        res = self.from_rgb(img)
        res = self.down1(res)
        skip = self.down2(res)
        res = self.down3(skip)
        res = self.down4(res)
        res = self.down5(res)
        mask= self.maskhead(res)
        for i in range(len(self.BottleNeck)):
            res = self.BottleNeck[i](res, id)
        res = self.up5(res,id)
        res = self.up4(res,id)
        res = self.up3(res,id)
        res = (1-mask) * skip + mask * res
        res = self.up2(res,id) #  + skip
        res = self.up1(res,id)
        res = self.to_rgb(res)
        return res, mask