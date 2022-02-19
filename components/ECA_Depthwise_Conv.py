#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: DeConv copy.py
# Created Date: Tuesday July 20th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 19th February 2022 6:15:53 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

from torch import nn
import math

class ECADW(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 2, padding="zero"):
        super().__init__()
        b = 1
        gamma = 2
        k_size = int(abs(math.log(in_channels,2)+b)/gamma)
        k_size = k_size if k_size % 2 else k_size+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        
        padding_size = int((kernel_size -1)/2)
        self.conv    = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                            padding=padding_size, bias=False, groups=in_channels, stride=stride)
            # nn.init.xavier_uniform_(self.conv.weight)
    #     self.__weights_init__()

    # def __weights_init__(self):
    #     nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input):
        y   = self.avg_pool(input)
        y   = self.se(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y   = self.sigmoid(y)
        h   = self.conv(input)
        return h * y.expand_as(h)