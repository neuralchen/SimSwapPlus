#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DeConv copy.py
# Created Date: Tuesday July 20th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 19th February 2022 6:16:08 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################
from tokenize import group
from torch import nn
import math

class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampl_scale = 2, padding="zero", up_mode = "bilinear"):
        super().__init__()
        if up_mode.lower() == "bilinear":
            self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampl_scale)
        elif up_mode.lower() == "nearest":
            self.upsampling = nn.UpsamplingNearest2d(scale_factor=upsampl_scale)
        b = 1
        gamma = 2
        k_size = int(abs(math.log(out_channels,2)+b)/gamma)
        k_size = k_size if k_size % 2 else k_size+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        
        padding_size = int((kernel_size -1)/2)
        self.conv1x1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= 1)
        self.conv    = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding_size, bias=False, groups=out_channels)
            # nn.init.xavier_uniform_(self.conv.weight)
    #     self.__weights_init__()

    # def __weights_init__(self):
    #     nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        h   = self.conv1x1(input)
        h   = self.upsampling(h)
        y   = self.avg_pool(h)
        y   = self.se(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y   = self.sigmoid(y)
        
        h   = self.conv(h)
        return h * y.expand_as(h)