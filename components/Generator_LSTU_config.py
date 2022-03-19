#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Generator_Invobn_config1.py
# Created Date: Saturday February 26th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 27th February 2022 7:50:18 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import torch
from torch import nn
from LSTU import LSTU

# from components.DeConv_Invo import DeConv
class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out

class Generator(nn.Module):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__()

        id_dim      = kwargs["id_dim"]
        k_size      = kwargs["g_kernel_size"]
        res_num     = kwargs["res_num"]
        in_channel  = kwargs["in_channel"]
        up_mode     = kwargs["up_mode"]
        
        aggregator  = kwargs["aggregator"]
        res_mode    = aggregator

        padding_size= int((k_size -1)/2)
        padding_type= 'reflect'
        
        activation = nn.ReLU(True)
        from components.DeConv_Depthwise import DeConv

        # self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
                                # nn.BatchNorm2d(64), activation)
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(1), 
                                nn.Conv2d(3, in_channel, kernel_size=3, padding=0, bias=False),
                                nn.BatchNorm2d(in_channel),
                                activation)
        # self.first_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(64), activation)
        ### downsample
        self.down1 = nn.Sequential(
                                nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                                nn.Conv2d(in_channel, in_channel*2, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channel*2),
                                activation)
                                
        self.down2 = nn.Sequential(
                                nn.Conv2d(in_channel*2, in_channel*2, kernel_size=3, padding=1, groups=in_channel*2),
                                nn.Conv2d(in_channel*2, in_channel*4, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channel*4),
                                activation)

        self.lstu  = LSTU(in_channel*4,in_channel*4,in_channel*8,4)

        self.down3 = nn.Sequential(
                                nn.Conv2d(in_channel*4, in_channel*4, kernel_size=3, padding=1, groups=in_channel*4),
                                nn.Conv2d(in_channel*4, in_channel*8, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channel*8),
                                activation)

        self.down4 = nn.Sequential(
                                nn.Conv2d(in_channel*8, in_channel*8, kernel_size=3, padding=1, groups=in_channel*8),
                                nn.Conv2d(in_channel*8, in_channel*8, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channel*8),
                                activation)
        
        

        ### resnet blocks
        BN = []
        for i in range(res_num):
            BN += [
                ResnetBlock_Adain(in_channel*8, latent_size=id_dim, 
                        padding_type=padding_type, activation=activation, res_mode=res_mode)]
        self.BottleNeck = nn.Sequential(*BN)

        self.up4 = nn.Sequential(
            DeConv(in_channel*8,in_channel*8,3,up_mode=up_mode),
            nn.BatchNorm2d(in_channel*8),
            activation
        )
        
        self.up3 = nn.Sequential(
            DeConv(in_channel*8,in_channel*4,3,up_mode=up_mode),
            nn.BatchNorm2d(in_channel*4),
            activation
        )
        
        self.up2 = nn.Sequential(
            DeConv(in_channel*4,in_channel*2,3,up_mode=up_mode),
            nn.BatchNorm2d(in_channel*2),
            activation
        )

        self.up1 = nn.Sequential(
            DeConv(in_channel*2,in_channel,3,up_mode=up_mode),
            nn.BatchNorm2d(in_channel),
            activation
        )
        # self.last_layer = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1))
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channel, 3, kernel_size=3, padding=0))
        # self.last_layer = nn.Sequential(nn.ReflectionPad2d(3),
        #             nn.Conv2d(64, 3, kernel_size=7, padding=0))


    #     self.__weights_init__()

    # def __weights_init__(self):
    #     for layer in self.encoder:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    #     for layer in self.encoder2:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    def forward(self, img, id):
        res = self.first_layer(img)
        res = self.down1(res)
        res1 = self.down2(res)
        res = self.down3(res1)
        res = self.down4(res)

        for i in range(len(self.BottleNeck)):
            res = self.BottleNeck[i](res, id)

        res = self.up4(res)
        res = self.up3(res)
        skip = self.lstu(res1)
        res = self.up2(res + skip)
        res = self.up1(res)
        res = self.last_layer(res)

        return res