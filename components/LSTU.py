#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Generator.py
# Created Date: Sunday January 16th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 28th March 2022 11:47:55 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import math
import  torch
from    torch import nn

import torch.nn.functional as F

# class LSTU(nn.Module):
#     def __init__(
#                 self,
#                 in_channel,
#                 out_channel,
#                 latent_channel,
#                 scale = 4
#                 ):
#         super().__init__()
#         sig              = nn.Sigmoid()
#         self.relu        = nn.ReLU(True)

#         self.up_sample   = nn.Sequential(nn.Conv2d(latent_channel, out_channel/4, kernel_size=3, stride=1, padding=1, bias=False),
#                                 nn.BatchNorm2d(out_channel/4),
#                                 self.relu,
#                                 nn.Conv2d(latent_channel/4, out_channel, kernel_size=3, stride=1, padding=1),
#                                 )

#         self.forget_gate = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
#                                 nn.BatchNorm2d(out_channel), sig)
        
#         self.reset_gate  = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
#                                 nn.BatchNorm2d(out_channel), sig)
        
#         self.conv11      = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=True))
    
#     def forward(self, encoder_in, bottleneck_in):
#         h_hat_l_1   = self.up_sample(bottleneck_in)  # upsample and make `channel` identical to `out_channel`
#         h_bar_l     = self.conv11(h_hat_l_1)
#         f_l         = self.forget_gate(h_hat_l_1)
#         r_l         = self.reset_gate (h_hat_l_1)
#         h_hat_l     = (1-f_l)*h_bar_l + f_l* encoder_in
#         x_hat_l     = r_l* self.relu(h_hat_l) + (1-r_l)* h_hat_l_1
#         return x_hat_l


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

class LSTU(nn.Module):
    def __init__(
                self,
                in_channel,
                norm
                ):
        super().__init__()
        self.sig   = nn.Sigmoid()

        self.mask_head  = ResBlk(in_channel, 1, normalize=norm)
        # self.forget_gate = ResBlk(in_channel,in_channel, normalize=norm)
    
    def forward(self, encoder_in, decoder_in):
        mask    = self.sig(self.mask_head(decoder_in))  # upsample and make `channel` identical to `out_channel`
        # enc_feat= self.forget_gate(encoder_in)
        out     = (1-mask)*encoder_in + mask * decoder_in
        return out, mask