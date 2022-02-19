#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DeConv.py
# Created Date: Tuesday July 20th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 19th February 2022 5:35:38 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


from torch import nn

class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, upsampl_scale = 2, padding="zero"):
        super().__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampl_scale)
        padding_size = int((kernel_size -1)/2)
        if padding.lower() == "reflect":
            self.conv = nn.Sequential(
                nn.ReflectionPad2d(padding_size),
                nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels, kernel_size= kernel_size, bias= False))
            # for layer in self.conv:
            #     if isinstance(layer,nn.Conv2d):
            #         nn.init.xavier_uniform_(layer.weight)
        elif padding.lower() == "zero":
            self.conv = nn.Conv2d(in_channels = in_channels, padding = 1,
                            out_channels = out_channels, kernel_size= kernel_size, bias= False)
            # nn.init.xavier_uniform_(self.conv.weight)
    #     self.__weights_init__()

    # def __weights_init__(self):
    #     nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        h   = self.upsampling(input)
        h   = self.conv(h)
        return h