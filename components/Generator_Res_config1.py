#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Generator_Invobn_config1.py
# Created Date: Saturday February 26th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 24th March 2022 11:24:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import torch
from torch import nn
from components.LSTU import LSTU

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
    def __init__(self, 
                dim,
                latent_size,
                activation=nn.LeakyReLU(0.2),
                res_mode="depthwise"):
        super(ResnetBlock_Adain, self).__init__()

        conv1 = []
        self.in1 = InstanceNorm()
        self.in2 = InstanceNorm()
        if res_mode.lower() == "conv":
            
            conv1 += [activation, 
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)]

        elif res_mode.lower() == "depthwise":
            conv1 += [activation,
                nn.Conv2d(dim, dim, kernel_size=3, padding=1,groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1)]

        elif res_mode.lower() == "depthwise_eca":
            conv1 += [activation,
                nn.Conv2d(dim, dim, kernel_size=3, padding=1,groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1)]

        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)

        conv2 = []
        if res_mode.lower() == "conv":
            conv2 += [activation, 
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)]

        elif res_mode.lower() == "depthwise":
            conv2 += [activation,
                nn.Conv2d(dim, dim, kernel_size=3, padding=1,groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1)]

        elif res_mode.lower() == "depthwise_eca":
            conv2 += [activation,
                nn.Conv2d(dim, dim, kernel_size=3, padding=1,groups=dim, bias=False),
                nn.Conv2d(dim, dim, kernel_size=1)]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.in1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.conv1(y)
        y = self.in2(y)
        y = self.style2(y, dlatents_in_slice)
        y = self.conv2(y)

        out = x + y
        return out

class ResUpSampleBlock(nn.Module):
    def __init__(self, 
                in_dim,
                out_dim,
                latent_size,
                activation=nn.LeakyReLU(0.2),
                res_mode="depthwise"):
        super(ResUpSampleBlock, self).__init__()
        conv1 = []
        self.in1 = InstanceNorm()
        self.in2 = InstanceNorm()
        if res_mode.lower() == "conv":
            
            conv1 += [activation, 
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)]

        elif res_mode.lower() == "depthwise":
            conv1 += [activation,
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1,groups=in_dim, bias=False),
                nn.Conv2d(in_dim, out_dim, kernel_size=1)]

        elif res_mode.lower() == "depthwise_eca":
            conv1 += [activation,
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1,groups=in_dim, bias=False),
                nn.Conv2d(in_dim, out_dim, kernel_size=1)]

        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, in_dim)

        conv2 = []
        if res_mode.lower() == "conv":
            conv2 += [activation, 
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)]

        elif res_mode.lower() == "depthwise":
            conv2 += [activation,
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,groups=out_dim, bias=False),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)]

        elif res_mode.lower() == "depthwise_eca":
            conv2 += [activation,
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,groups=out_dim, bias=False),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, out_dim)
         
        self.reshape1_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.resampling = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x, dlatents_in_slice):
        y = self.in1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.conv1(y)
        y = self.resampling(y)
        y = self.in2(y)
        y = self.style2(y, dlatents_in_slice)
        y = self.conv2(y)
        res = self.reshape1_1(x)
        res = self.resampling(res)
        out = res + y
        return out


class ResDownSampleBlock(nn.Module):
    def __init__(self, 
                in_dim,
                out_dim,
                activation=nn.LeakyReLU(0.2),
                res_mode="depthwise"):
        super(ResDownSampleBlock, self).__init__()
        conv1 = []
        if res_mode.lower() == "conv":
            
            conv1 += [
                nn.BatchNorm2d(in_dim),
                activation, 
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False)]

        elif res_mode.lower() == "depthwise":
            conv1 += [
                nn.BatchNorm2d(in_dim),
                activation,
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1,groups=in_dim, bias=False),
                nn.Conv2d(in_dim, in_dim, kernel_size=1)]

        elif res_mode.lower() == "depthwise_eca":
            conv1 += [
                nn.BatchNorm2d(in_dim),
                activation,
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1,groups=in_dim, bias=False),
                nn.Conv2d(in_dim, in_dim, kernel_size=1)]

        self.conv1 = nn.Sequential(*conv1)

        conv2 = []
        if res_mode.lower() == "conv":
            conv2 += [
                nn.BatchNorm2d(in_dim),
                activation, 
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)]

        elif res_mode.lower() == "depthwise":
            conv2 += [
                nn.BatchNorm2d(in_dim),
                activation,
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1,groups=out_dim, bias=False),
                nn.Conv2d(in_dim, out_dim, kernel_size=1)]

        elif res_mode.lower() == "depthwise_eca":
            conv2 += [
                nn.BatchNorm2d(in_dim),
                activation,
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1,groups=out_dim, bias=False),
                nn.Conv2d(in_dim, out_dim, kernel_size=1)]
        self.conv2 = nn.Sequential(*conv2)
         
        self.reshape1_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.resampling = nn.AvgPool2d(3,2,1)


    def forward(self, x):
        y = self.conv1(x)
        y = self.resampling(y)
        y = self.conv2(y)
        res = self.reshape1_1(x)
        res = self.resampling(res)
        out = res + y
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
        res_mode    = kwargs["res_mode"]

        padding_size= int((k_size -1)/2)
        padding_type= 'reflect'
        
        # activation = nn.LeakyReLU(0.2)
        activation = nn.ReLU()

        # self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
                                # nn.BatchNorm2d(64), activation)
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(1), 
                                nn.Conv2d(3, in_channel, kernel_size=3, stride=2, padding=0, bias=False),
                                nn.BatchNorm2d(in_channel),
                                activation) # 256
        # self.first_layer = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(64), activation)
        ### downsample
        self.down1 = ResDownSampleBlock(in_channel, in_channel*2, activation=activation, res_mode=res_mode)  # 128
        # nn.Sequential(
        #                         nn.Conv2d(in_channel, in_channel*2, stride=2, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(in_channel*2),
        #                         activation) # 128
                                
        self.down2 = ResDownSampleBlock(in_channel*2, in_channel*4, activation=activation, res_mode=res_mode) # 64
        # nn.Sequential(
        #                         nn.Conv2d(in_channel*2, in_channel*4, stride=2, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(in_channel*4),
        #                         activation) # 64

        # self.lstu  = LSTU(in_channel*4,in_channel*4,in_channel*8,4)

        self.down3 = ResDownSampleBlock(in_channel*4, in_channel*8, activation=activation, res_mode=res_mode) # 32

        self.down4 = ResDownSampleBlock(in_channel*8, in_channel*8, activation=activation, res_mode=res_mode) # 16
        # nn.Sequential(
        #                         nn.Conv2d(in_channel*4, in_channel*8, stride=2, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(in_channel*8),
        #                         activation) # 32

        # self.down4 = nn.Sequential(
        #                         nn.Conv2d(in_channel*8, in_channel*8, stride=2, kernel_size=3, padding=1, bias=False),
        #                         nn.BatchNorm2d(in_channel*8),
        #                         activation)
        
        

        ### resnet blocks
        BN = []
        for i in range(res_num):
            BN += [
                ResnetBlock_Adain(in_channel*8, latent_size=id_dim, activation=activation, res_mode=res_mode)]
        self.BottleNeck = nn.Sequential(*BN)
        
        self.up5 = ResUpSampleBlock(in_channel*8, in_channel*8, id_dim, activation=activation, res_mode=res_mode) # 32

        self.up4 = ResUpSampleBlock(in_channel*8, in_channel*4, id_dim, activation=activation, res_mode=res_mode) # 64
        # nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(in_channel*8, in_channel*8, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channel*8),
        #     activation
        # )
        
        self.up3 = ResUpSampleBlock(in_channel*4, in_channel*2, id_dim, activation=activation, res_mode=res_mode) # 128
        # nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(in_channel*8, in_channel*4, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channel*4),
        #     activation
        # )
        
        self.up2 = ResUpSampleBlock(in_channel*2, in_channel, id_dim, activation=activation, res_mode=res_mode) # 256
        # nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(in_channel*4, in_channel*2, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channel*2),
        #     activation
        # )

        self.up1 = ResUpSampleBlock(in_channel, in_channel , id_dim, activation=activation, res_mode=res_mode) # 512
        # nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.Conv2d(in_channel*2, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(in_channel),
        #     activation
        # )
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
        res = self.down2(res)
        res = self.down3(res)
        for i in range(len(self.BottleNeck)):
            res = self.BottleNeck[i](res, id)
        res = self.up4(res,id)
        res = self.up3(res,id)
        res = self.up2(res,id) #  + skip
        res = self.up1(res,id)
        res = self.last_layer(res)

        return res