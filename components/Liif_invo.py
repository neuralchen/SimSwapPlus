#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Liif.py
# Created Date: Monday October 18th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 19th October 2021 8:25:18 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from   components.Involution import involution


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        print("i: %d, n: %d"%(i,n))
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class LIIF(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        imnet_in_dim = in_dim
        # imnet_in_dim += 2 # attach coord
        # imnet_in_dim += 2

        self.conv1x1 = nn.Conv2d(in_channels = imnet_in_dim, out_channels = out_dim, kernel_size= 1)
        # self.same_padding   = nn.ReflectionPad2d(padding_size)
            
        # self.conv = involution(out_dim,5,1)
        self.imnet   = nn.Sequential( \
            # nn.Conv2d(in_channels = imnet_in_dim, out_channels = out_dim, kernel_size= 3,padding=1),
            involution(out_dim,5,1),
            nn.InstanceNorm2d(out_dim, affine=True, momentum=0),
            nn.LeakyReLU(),
            # nn.Conv2d(in_channels = out_dim, out_channels = out_dim, kernel_size= 3,padding=1),
            # nn.InstanceNorm2d(out_dim),
            # nn.LeakyReLU(),
            )
    
    def gen_coord(self, in_shape, output_size):

        self.vx_lst = [-1, 1]
        self.vy_lst = [-1, 1]
        eps_shift = 1e-6
        self.image_size=output_size

        # field radius (global: [-1, 1])
        rx = 2 / in_shape[-2] / 2
        ry = 2 / in_shape[-1] / 2

        self.coord = make_coord(output_size,flatten=False) \
                    .expand(in_shape[0],output_size[0],output_size[1],2)
        
        # cell = torch.ones_like(coord)
        # cell[:, :, 0] *= 2 / coord.shape[-2]
        # cell[:, :, 1] *= 2 / coord.shape[-1]

        # feat_coord = make_coord(in_shape[-2:], flatten=False) \
        #     .permute(2, 0, 1) \
        #     .unsqueeze(0).expand(in_shape[0], 2, *in_shape[-2:])
        
        # areas = []

        # self.rel_coord  = torch.zeros((2,2,in_shape[0],output_size[0]*output_size[1],2))
        # self.rel_cell   = torch.zeros((2,2,in_shape[0],output_size[0]*output_size[1],2))
        # self.coord_     = torch.zeros((2,2,in_shape[0],output_size[0]*output_size[1],2))
        # for vx in self.vx_lst:
        #     for vy in self.vy_lst:
        #         self.coord_[(vx+1)//2,(vy+1)//2,:, :, :] = coord.clone()
        #         self.coord_[(vx+1)//2,(vy+1)//2,:, :, 0] += vx * rx + eps_shift
        #         self.coord_[(vx+1)//2,(vy+1)//2,:, :, 1] += vy * ry + eps_shift
        #         self.coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        #         q_coord = F.grid_sample(
        #             feat_coord, self.coord_[(vx+1)//2,(vy+1)//2,:, :, :].flip(-1).unsqueeze(1),
        #             mode='nearest', align_corners=False)[:, :, 0, :] \
        #             .permute(0, 2, 1)
        #         self.rel_coord[(vx+1)//2,(vy+1)//2,:, :, :] = coord - q_coord
        #         self.rel_coord[(vx+1)//2,(vy+1)//2,:, :, 0] *= in_shape[-2]
        #         self.rel_coord[(vx+1)//2,(vy+1)//2,:, :, 1] *= in_shape[-1]

        #         self.rel_cell[(vx+1)//2,(vy+1)//2,:, :, :] = cell.clone()
        #         self.rel_cell[(vx+1)//2,(vy+1)//2,:, :, 0] *= in_shape[-2]
        #         self.rel_cell[(vx+1)//2,(vy+1)//2,:, :, 1] *= in_shape[-1]
        #         area = torch.abs(self.rel_coord[(vx+1)//2,(vy+1)//2,:, :, 0] * self.rel_coord[(vx+1)//2,(vy+1)//2,:, :, 1])
        #         areas.append(area + 1e-9)
        # tot_area = torch.stack(areas).sum(dim=0)
        # t = areas[0]; areas[0] = areas[3]; areas[3] = t
        # t = areas[1]; areas[1] = areas[2]; areas[2] = t
        # self.area_weights = []
        # for item in areas:
        #     self.area_weights.append((item / tot_area).unsqueeze(-1).cuda())
        
        # self.rel_coord  = self.rel_coord.cuda()
        # self.rel_cell   = self.rel_cell.cuda()
        # self.coord_     = self.coord_.cuda()
        self.coord     = self.coord.cuda()


    def forward(self, feat):
        # B K*K*Cin H W
        # feat = F.unfold(feat, 3, padding=1).view(
        #     feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # preds = []
        # for vx in [0,1]:
        #     for vy in [0,1]:
        # print("feat shape: ", feat.shape)
        # print("coor shape: ", self.coord.shape)
        q_feat = self.conv1x1(feat)
        q_feat = F.grid_sample(
            q_feat, self.coord,
            mode='bilinear', align_corners=False)
        out = self.imnet(q_feat)
        # inp = torch.cat([q_feat, self.rel_coord[vx,vy,:,:,:], self.rel_cell[vx,vy,:,:,:]], dim=-1)

        #         bs, q = self.coord_[0,0,:,:,:].shape[:2]
        #         pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
        #         # print("pred shape: ",pred.shape)
        #         preds.append(pred)
        # ret = 0
        # for pred, area in zip(preds, self.area_weights):
        #     ret = ret + pred * area
        # print("warp output shape: ",out.shape)
        
        return out