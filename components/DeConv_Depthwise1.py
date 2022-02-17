#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DeConv copy.py
# Created Date: Tuesday July 20th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 16th February 2022 1:42:49 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################
from audioop import bias
from tokenize import group
from torch import nn

class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampl_scale = 2, padding="zero"):
        super().__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampl_scale)
        padding_size = int((kernel_size -1)/2)
        self.conv1x1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= 1, bias = False)
        self.conv    = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding_size, groups=in_channels)
            # nn.init.xavier_uniform_(self.conv.weight)
    #     self.__weights_init__()

    # def __weights_init__(self):
    #     nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        h   = self.upsampling(input)
        h   = self.conv(h)
        h   = self.conv1x1(h)
        return h