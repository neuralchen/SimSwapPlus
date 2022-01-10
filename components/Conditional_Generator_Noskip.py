
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Generator_tanh.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 6th July 2021 1:16:46 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F

from components.ResBlock import ResBlock
from components.DeConv   import DeConv
from components.Conditional_ResBlock_ModulaConv import Conditional_ResBlock

class Generator(nn.Module):
    def __init__(
                self,
                chn=32,
                k_size=3,
                res_num = 5,
                class_num = 3,
                **kwargs):
        super().__init__()
        padding_size = int((k_size -1)/2)
        self.resblock_list = []
        self.n_class    = class_num
        self.encoder1 = nn.Sequential(
            # nn.InstanceNorm2d(3, affine=True),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size= k_size, stride=1, padding=1, bias= False),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size= k_size, stride=2, padding=1,bias =False), # 
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn*2, out_channels = chn * 4, kernel_size= k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn*4  , out_channels = chn * 8, kernel_size= k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            # # nn.ReflectionPad2d(padding_size),
            nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size= k_size, stride=2, padding=1,bias =False),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU()
        )


        res_size = chn * 8
        for _ in range(res_num-1):
            self.resblock_list += [ResBlock(res_size,k_size),]
        self.resblocks = nn.Sequential(*self.resblock_list)
        self.conditional_res = Conditional_ResBlock(res_size, k_size, class_num)
        self.decoder1 = nn.Sequential(
            DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size= k_size),
            nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            DeConv(in_channels = chn * 8, out_channels = chn *4, kernel_size= k_size),
            nn.InstanceNorm2d(chn *4, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            DeConv(in_channels = chn * 4, out_channels = chn * 2 , kernel_size= k_size),
            nn.InstanceNorm2d(chn*2, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            DeConv(in_channels = chn *2, out_channels = chn, kernel_size= k_size),
            nn.InstanceNorm2d(chn, affine=True, momentum=0),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = chn, out_channels =3, kernel_size= k_size, stride=1, padding=1,bias =True)
            # nn.Tanh()
        )

        self.__weights_init__()

    def __weights_init__(self):
        for layer in self.encoder1:
            if isinstance(layer,nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)

        # for layer in self.encoder2:
        #     if isinstance(layer,nn.Conv2d):
        #         nn.init.xavier_uniform_(layer.weight)

    def forward(self, input, condition=None, get_feature = False):
        feature = self.encoder1(input)
        if get_feature:
            return feature
        out = self.conditional_res(feature, condition)
        out = self.resblocks(out)
        # n, _,h,w = out.size()
        # attr = condition.view((n, self.n_class, 1, 1)).expand((n, self.n_class, h, w))
        # out = torch.cat([out, attr], dim=1)
        out = self.decoder1(out)
        return out,feature