#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DeConv copy.py
# Created Date: Tuesday July 20th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 26th February 2022 5:55:46 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

from torch import nn
from components.misc.Involution_ECA import involution


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampl_scale = 2, padding="zero", up_mode = "bilinear"):
        super().__init__()
        if up_mode.lower() == "bilinear":
            self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampl_scale)
        elif up_mode.lower() == "nearest":
            self.upsampling = nn.UpsamplingNearest2d(scale_factor=upsampl_scale)
        # self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampl_scale)
        padding_size = int((kernel_size -1)/2)
        self.conv1x1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= 1)
        # self.same_padding   = nn.ReflectionPad2d(padding_size)
        if padding.lower() == "reflect":
            
            self.conv = involution(out_channels,kernel_size,1)
            # self.conv = nn.Sequential(
            #     nn.ReflectionPad2d(padding_size),
            #     nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size= kernel_size, bias= False))
            # for layer in self.conv:
            #     if isinstance(layer,nn.Conv2d):
            #         nn.init.xavier_uniform_(layer.weight)
        elif padding.lower() == "zero":
            self.conv = involution(out_channels,kernel_size,1)
            # nn.init.xavier_uniform_(self.conv.weight)
    #     self.__weights_init__()

    # def __weights_init__(self):
    #     nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        h   = self.conv1x1(input)
        h   = self.upsampling(h)
        h   = self.conv(h)
        return h