#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: arcface_decoder.py
# Created Date: Saturday January 29th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 29th January 2022 2:55:39 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__()

        activation = nn.ReLU(True)
        
        self.fc = nn.Linear(512, 7*7*512)

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
        
        self.last_layer = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1))
    def forward(self, input):
        x = input  # 
        x = self.fc(x)
        x = x.view(x.size(0),512,7,7)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.last_layer(x)
        
        return x