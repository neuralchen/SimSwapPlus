#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: SliceWassersteinDistance.py
# Created Date: Tuesday October 12th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th October 2021 3:11:23 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import torch

from torch import nn
import torch.nn.functional as F


class SWD(nn.Module):
    """ Slicing layer: computes projections and returns sorted vector """
    def __init__(self, channel, direction_num=16):
        super().__init__()
        # Number of directions
        self.direc_num  = direction_num
        self.channel    = channel
        self.seed = nn.Parameter(torch.normal(mean=0.0, std=torch.ones(self.direc_num, self.channel)),requires_grad=False)

    def update(self):
        """ Update random directions """
        # Generate random directions
        self.seed.normal_()
        # norm            = self.directions.norm(dim=-1,keepdim=True)
        self.directions = F.normalize(self.seed)
        
        # Normalize directions
        # self.directions = self.directions/norm
        # print("self.directions shape:", self.directions.shape)
        # print("self.directions:", self.directions)

    def forward(self, input):
        """ Implementation of figure 2 """
        input       = input.flatten(-2)
        sliced      = self.directions @ input
        sliced, _   = sliced.sort()
        
        return sliced

if __name__ == "__main__":
    wocao = torch.ones((4,3,5,5))
    slice = SWD(wocao.shape[1])
    slice.update()
    wocao_slice = slice(wocao)
    print(wocao_slice.shape)
    print(wocao_slice)