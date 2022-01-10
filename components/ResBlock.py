#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: ResBlock.py
# Created Date: Monday July 5th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 5th July 2021 12:18:18 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, k_size = 3, stride=1):
        super().__init__()
        padding_size = int((k_size -1)/2)
        self.block = nn.Sequential(
                nn.ReflectionPad2d(padding_size),
                nn.Conv2d(in_channels = in_channel , out_channels = in_channel , kernel_size= k_size, stride=stride, bias= False),
                nn.InstanceNorm2d(in_channel, affine=True, momentum=0),
                nn.ReflectionPad2d(padding_size),
                nn.Conv2d(in_channels = in_channel , out_channels = in_channel , kernel_size= k_size, stride=stride, bias= False),
                nn.InstanceNorm2d(in_channel, affine=True, momentum=0)
            )
        self.__weights_init__()

    def __weights_init__(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input):
        res = input
        h   = self.block(input)
        out = h + res
        return out
