#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: filter.py
# Created Date: Wednesday April 13th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 13th April 2022 3:49:23 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
from   PIL import Image
from   torchvision import transforms

class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

if __name__ == "__main__":
    transformer_Arcface = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    imagenet_std    = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)

    img  = "G:/swap_data/ID/2.jpg"
    attr = cv2.imread(img)
    attr = Image.fromarray(cv2.cvtColor(attr,cv2.COLOR_BGR2RGB))
    attr = transformer_Arcface(attr).unsqueeze(0)
    results = HighPass(0.5,torch.device("cpu"))(attr)

    results     = results * imagenet_std + imagenet_mean
    results     = results.cpu().permute(0,2,3,1)[0,...]
    results     = results.numpy()
    results     = np.clip(results,0.0,1.0) * 255
    results     = cv2.cvtColor(results,cv2.COLOR_RGB2BGR)
    cv2.imwrite("filter_results2.png",results)
