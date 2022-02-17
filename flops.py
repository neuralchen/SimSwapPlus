#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: flops.py
# Created Date: Sunday February 13th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 17th February 2022 2:32:48 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import os

import torch
from thop import profile
from thop import clever_format



if __name__ == '__main__':

   # script      = "Generator_modulation_up"
    script      = "Generator_modulation_depthwise_config"
    # script      = "Generator_ori_config"
    class_name  = "Generator"
    arcface_ckpt= "arcface_ckpt/arcface_checkpoint.tar"
    model_config={
        "id_dim": 512,
        "g_kernel_size": 3,
        "in_channel":16,
        "res_num": 9,
        # "up_mode": "nearest",
        "up_mode": "bilinear",
        "res_mode": "depthwise"
    }


    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    print("GPU used : ", os.environ['CUDA_VISIBLE_DEVICES'])

    gscript_name     = "components." + script
    
    
    package  = __import__(gscript_name, fromlist=True)
    gen_class= getattr(package, class_name)
    gen      = gen_class(**model_config)
    model    = gen.cuda().eval().requires_grad_(False)
    arcface1 = torch.load(arcface_ckpt, map_location=torch.device("cpu"))
    arcface  = arcface1['model'].module
    arcface  = arcface.cuda()
    arcface.eval().requires_grad_(False)

    attr_img    = torch.rand((1,3,512,512)).cuda()
    id_img      = torch.rand((1,3,112,112)).cuda()
    id_latent   = torch.rand((1,512)).cuda()

    macs, params = profile(model, inputs=(attr_img, id_latent))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)