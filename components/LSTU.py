#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Generator.py
# Created Date: Sunday January 16th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 13th February 2022 2:03:21 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import  torch
from    torch import nn


class LSTU(nn.Module):
    def __init__(
                self,
                in_channel,
                out_channel,
                latent_channel,
                scale = 4
                ):
        super().__init__()
        sig              = nn.Sigmoid()
        self.relu        = nn.ReLU(True)

        self.up_sample   = nn.Sequential(nn.ConvTranspose2d(latent_channel, out_channel, kernel_size=4, stride=scale, padding=0, bias=False),
                                nn.BatchNorm2d(out_channel), sig)

        self.forget_gate = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(out_channel), sig)
        
        self.reset_gate  = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(out_channel), sig)
        
        self.conv11      = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=True))
    
    def forward(self, encoder_in, bottleneck_in):
        h_hat_l_1   = self.up_sample(bottleneck_in)  # upsample and make `channel` identical to `out_channel`
        h_bar_l     = self.conv11(h_hat_l_1)
        f_l         = self.forget_gate(h_hat_l_1)
        r_l         = self.reset_gate (h_hat_l_1)
        h_hat_l     = (1-f_l)*h_bar_l + f_l* encoder_in
        x_hat_l     = r_l* self.relu(h_hat_l) + (1-r_l)* h_hat_l_1
        return x_hat_l