#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: ImagenetNorm.py
# Created Date: Friday January 21st 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 21st January 2022 10:41:50 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import torch.nn as nn
import numpy as np
import torch
class ImagenetNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(ImagenetNorm, self).__init__()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.mean = torch.from_numpy(self.mean).float().cuda()
        self.mean = self.mean.view([1, 3, 1, 1])

        self.std = np.array([0.229, 0.224, 0.225])
        self.std = torch.from_numpy(self.std).float().cuda()
        self.std = self.std.view([1, 3, 1, 1])

    def forward(self, x):
        mean = self.mean.expand([1, 3, x.shape[2], x.shape[3]])
        std = self.std.expand([1, 3, x.shape[2], x.shape[3]])

        x = (x - mean) / std

        return x