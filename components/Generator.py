#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Generator_gpt_LN_encoder copy.py
# Created Date: Saturday October 9th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 26th October 2021 3:25:47 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import torch
from torch import nn
from ResBlock_Adain import ResBlock_Adain

from functools import partial

class Generator(nn.Module):
    def __init__(
                self,
                **kwargs
                ):
        super(Generator, self).__init__()

        input_nc    = kwargs["g_conv_dim"]
        output_nc   = kwargs["g_kernel_size"]
        latent_size = kwargs["latent_size"]
        n_blocks    = kwargs["resblock_num"]
        norm_name   = kwargs["norm_name"]
        padding_type= kwargs["reflect"]

        if norm_name == "bn":
            norm_layer = partial(nn.BatchNorm2d, affine = True, track_running_stats=True)
        elif norm_name == "in":
            norm_name = nn.InstanceNorm2d
        
        assert (n_blocks >= 0)
        activation = nn.ReLU(True)

        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
        self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                       norm_layer(512), activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, id):
        x = input  # 3*224*224
        res = self.first_layer(x)
        res = self.down1(res)
        res = self.down2(res)
        res = self.down4(res)
        res = self.down3(res)

        for i in range(len(self.BottleNeck)):
            res = self.BottleNeck[i](res, id)

        res = self.up4(res)
        res = self.up3(res)
        res = self.up2(res)
        res = self.up1(res)
        res = self.last_layer(res)
        return res
    
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