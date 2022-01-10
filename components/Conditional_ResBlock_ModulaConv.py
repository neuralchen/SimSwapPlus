#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_ResBlock_v2.py
# Created Date: Tuesday June 29th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 29th June 2021 3:59:44 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \ASMegaGAN\components\Conditional_ResBlock_v2.py
###   @Author: Ziang Liu
###   @Date: 2021-06-28 21:30:17
###   @LastEditors: Ziang Liu
###   @LastEditTime: 2021-06-28 21:46:24
###   @Copyright (C) 2021 SJTU. All rights reserved.
###################################################################
import torch
from torch import nn
import torch.nn.functional as F
# from ops.Conditional_BN import Conditional_BN
# from components.Adain import Adain

class Conv2DMod(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_channels
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel, kernel)))
        self.eps = eps

        padding_size = int((kernel -1)/2)
        self.same_padding  = nn.ReplicationPad2d(padding_size)
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        x = self.same_padding(x)
        x = F.conv2d(x, weights, groups=b)
        
        x = x.reshape(-1, self.filters, h, w)
        return x
        
class Conditional_ResBlock(nn.Module):
    def __init__(self, in_channel, k_size = 3, n_class = 2, stride=1):
        super().__init__()
        
        self.embed1 = nn.Embedding(n_class, in_channel)
        self.embed2 = nn.Embedding(n_class, in_channel)
        self.conv1  = Conv2DMod(in_channels = in_channel , out_channels = in_channel, kernel= k_size, stride=stride)
        self.conv2  = Conv2DMod(in_channels = in_channel , out_channels = in_channel, kernel= k_size, stride=stride)

    def forward(self, input, condition):
        res = input
        style1 = self.embed1(condition)
        h   = self.conv1(res, style1)
        style2 = self.embed2(condition)
        h   = self.conv2(h, style2)
        out = h + res
        return out