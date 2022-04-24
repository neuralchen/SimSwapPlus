#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Generator_Invobn_config1.py
# Created Date: Saturday February 26th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 19th April 2022 7:03:46 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import os

import torch
from torch import nn
import torch.nn.functional as F
import math



class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize="in", downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.equal_var  = math.sqrt(2)
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize.lower() == "in":
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        elif self.normalize.lower() == "bn":
            self.norm1 = nn.BatchNorm2d(dim_in)
            self.norm2 = nn.BatchNorm2d(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x /self.equal_var   # unit variance

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

class ResUpBlk(nn.Module):
    def __init__(self, dim_in, dim_out,actv=nn.LeakyReLU(0.2),normalize="in"):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.learned_sc = dim_in != dim_out
        self.equal_var = math.sqrt(2)
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        if self.normalize.lower() == "in":
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_out, affine=True)
        elif self.normalize.lower() == "bn":
            self.norm1 = nn.BatchNorm2d(dim_in)
            self.norm2 = nn.BatchNorm2d(dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.actv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        out = self._residual(x)
        out = (out + self._shortcut(x)) / self.equal_var
        return out

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
        norm        = kwargs["norm"].lower()
        
        aggregator  = kwargs["aggregator"]
        res_mode    = kwargs["res_mode"]

        padding_size= int((k_size -1)/2)
        padding_type= 'reflect'

        if norm.lower() == "in":
            norm_out = nn.InstanceNorm2d(in_channel, affine=True)
            norm_mask= nn.InstanceNorm2d
        elif norm.lower() == "bn":
            norm_out = nn.BatchNorm2d(in_channel)
            norm_mask = nn.BatchNorm2d
        
        
        activation = nn.LeakyReLU(0.2)
        # activation = nn.ReLU()

        self.from_rgb = nn.Conv2d(3, in_channel, 1, 1, 0)
        # self.first_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(64), activation)
        ### downsample
        self.down1 = ResBlk(in_channel, in_channel, normalize=norm, downsample=True)# 128

        self.down2 = ResBlk(in_channel, in_channel*2, normalize=norm, downsample=True)# 64
                                
        self.down3 = ResBlk(in_channel*2, in_channel*4,normalize=norm, downsample=True)# 32

        self.down4 = ResBlk(in_channel*4, in_channel*8, normalize=norm, downsample=True)# 16

        # self.down6 = ResBlk(in_channel*8, in_channel*8, normalize=True, downsample=True)# 8
        
        
        ### resnet blocks
        BN = []
        for i in range(res_num):
            BN += [
                AdainResBlk(in_channel*8, in_channel*8, style_dim=id_dim, upsample=False)]
        self.BottleNeck = nn.Sequential(*BN)

        # self.up6 = AdainResBlk(in_channel*8, in_channel*8, style_dim=id_dim, upsample=True) # 16
        
        # self.up5 = AdainResBlk(in_channel*8, in_channel*8, style_dim=id_dim, upsample=True) # 1

        self.up4 = AdainResBlk(in_channel*8, in_channel*4, style_dim=id_dim, upsample=True) # 32

        self.up3 = AdainResBlk(in_channel*4, in_channel*2, style_dim=id_dim, upsample=True) # 64

        # self.maskhead = nn.Sequential(
        #                         nn.Conv2d(in_channel*2, in_channel//2, kernel_size=3, stride=1, padding=1, bias=False),
        #                         norm_mask, # 64
        #                         activation,
        #                         nn.Conv2d(in_channel//2, 1, kernel_size=3, stride=1, padding=1),
        #                         nn.Sigmoid())
        self.maskhead_lr = nn.Sequential(
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel*8, in_channel, kernel_size=3, stride=1, padding=1,bias=False),
                                norm_mask(in_channel, affine=True), # 32
                                activation,
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel, in_channel//4, kernel_size=3, stride=1, padding=1, bias=False),
                                norm_mask(in_channel//4, affine=True), # 64
                                activation
        )
        self.maskhead_hr = nn.Sequential(
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel//4, in_channel//16, kernel_size=3, stride=1, padding=1,bias=False),
                                norm_mask(in_channel//16, affine=True), # 128
                                activation,
                                nn.UpsamplingNearest2d(scale_factor = 2),
                                nn.Conv2d(in_channel//16, 1, kernel_size=3, stride=1, padding=1),
                                nn.Sigmoid() # 256
        )
        self.maskhead_out = nn.Sequential(nn.Conv2d(in_channel//4, 1, kernel_size=1, stride=1),
                                nn.Sigmoid())
        
        # self.up2 = AdainResBlk(in_channel*2, in_channel, style_dim=id_dim, upsample=True)
        # self.up2 = AdainResBlk(in_channel*2, in_channel, style_dim=id_dim, upsample=True)
        self.up2 = ResUpBlk(in_channel*2, in_channel, normalize=norm)

        # self.up1 = AdainResBlk(in_channel, in_channel, style_dim=id_dim, upsample=True)
        self.up1 = ResUpBlk(in_channel, in_channel, normalize=norm)
        # ResUpBlk(in_channel, in_channel, normalize="in") # 512

        
        
        self.to_rgb = nn.Sequential(
            norm_out,
            activation,
            nn.Conv2d(in_channel, 3, 3, 1, 1))

        # self.last_layer = nn.Sequential(nn.ReflectionPad2d(3),
        #             nn.Conv2d(64, 3, kernel_size=7, padding=0))


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
        mask_feat= self.maskhead_lr(res)
        
        for i in range(len(self.BottleNeck)):
            res = self.BottleNeck[i](res, id)
        res = self.up4(res,id)
        res = self.up3(res,id)
        mask_lr= self.maskhead_out(mask_feat)
        # res = (1-mask) * self.sigma(skip) + mask * res
        res = (1-mask_lr) * skip + mask_lr * res
        res = self.up2(res) #  + skip
        res = self.up1(res)
        res = self.to_rgb(res)
        mask_hr=self.maskhead_hr(mask_feat)
        res = (1-mask_hr) * img + mask_hr * res
        return res, mask_lr, mask_hr