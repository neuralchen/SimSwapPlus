#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Conditional_Generator_gpt_LN_encoder copy.py
# Created Date: Saturday October 9th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 26th October 2021 3:25:47 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from components.DeConv   import DeConv
from components.network_swin import SwinTransformerBlock, PatchEmbed, PatchUnEmbed

class ImageLN(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.layer = nn.LayerNorm(dim)
    def forward(self, x):
        y = self.layer(x.permute(0,2,3,1)).permute(0,3,1,2)
        return y

class Generator(nn.Module):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__()

        chn         = kwargs["g_conv_dim"]
        k_size      = kwargs["g_kernel_size"]
        res_num     = kwargs["res_num"]
        class_num   = kwargs["n_class"]
        window_size = kwargs["window_size"]
        image_size  = kwargs["image_size"]

        padding_size = int((k_size -1)/2)
        
        self.resblock_list = []
        embed_dim       = 96
        window_size     = 8
        num_heads       = 8
        mlp_ratio       = 2.
        norm_layer      = nn.LayerNorm
        qk_scale        = None
        qkv_bias        = True
        self.patch_norm = True
        self.lnnorm     = norm_layer(embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = chn , kernel_size=k_size, stride=1, padding=1, bias= False),
            ImageLN(chn),
            nn.ReLU(),
            nn.Conv2d(in_channels = chn , out_channels = chn*2, kernel_size=k_size, stride=2, padding=1,bias =False), # 
            ImageLN(chn * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = chn*2, out_channels = embed_dim, kernel_size=k_size, stride=2, padding=1,bias =False),
            ImageLN(embed_dim),
            nn.ReLU(),
        )

        # self.encoder2 = nn.Sequential(
            
        #     nn.Conv2d(in_channels = chn*4  , out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
        #     ImageLN(chn * 8),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
        #     ImageLN(chn * 8),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size, stride=2, padding=1,bias =False),
        #     ImageLN(chn * 8),
        #     nn.LeakyReLU()
        # )

        self.fea_size = (image_size//4, image_size//4)
        # self.conditional_GPT = GPT_Spatial(2, res_dim, res_num, class_num)

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=embed_dim, input_resolution=self.fea_size,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0.1,
                                 norm_layer=norm_layer)
            for i in range(res_num)])

        self.decoder = nn.Sequential(
            # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.LeakyReLU(),
            # DeConv(in_channels = chn * 8, out_channels = chn * 8, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 8, affine=True, momentum=0),
            # nn.LeakyReLU(),
            # DeConv(in_channels = chn * 8, out_channels = chn *4, kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 4, affine=True, momentum=0),
            # nn.LeakyReLU(),
            DeConv(in_channels = embed_dim, out_channels = chn * 2 , kernel_size=k_size),
            # nn.InstanceNorm2d(chn * 2, affine=True, momentum=0),
            ImageLN(chn * 2),
            nn.ReLU(),
            DeConv(in_channels = chn *2, out_channels = chn, kernel_size=k_size),
            ImageLN(chn),
            nn.ReLU(),
            nn.Conv2d(in_channels = chn, out_channels =3, kernel_size=k_size, stride=1, padding=1,bias =True)
        )

        self.patch_embed = PatchEmbed(
            img_size=self.fea_size[0], patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=self.fea_size[0], patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

    #     self.__weights_init__()

    # def __weights_init__(self):
    #     for layer in self.encoder:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    #     for layer in self.encoder2:
    #         if isinstance(layer,nn.Conv2d):
    #             nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        x2 = self.encoder(input)
        x2 = self.patch_embed(x2)
        for blk in self.blocks:
            x2 = blk(x2,self.fea_size)
        x2 = self.lnnorm(x2)
        x2 = self.patch_unembed(x2,self.fea_size)
        out = self.decoder(x2)
        return out
    
if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 1024
    width = 1024
    model = Generator()
    print(model)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)