#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: speed_test.py
# Created Date: Thursday February 10th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 15th February 2022 12:54:56 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################
import os
import time

import torch



if __name__ == '__main__':

    script      = "Generator_modulation_depthwise_config"
    class_name  = "Generator"
    arcface_ckpt= "arcface_ckpt/arcface_checkpoint.tar"
    model_config={
        "id_dim": 512,
        "g_kernel_size": 3,
        "in_channel":16,
        "res_num": 9
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


    id_img      = torch.rand((4,3,112,112)).cuda()
    id_latent   = torch.rand((4,512)).cuda()
    # cv2.imwrite(os.path.join("./swap_results", "id_%s.png"%(id_basename)),id_img_align_crop[0]

    attr        = torch.rand((4,3,512,512)).cuda()
    
    import datetime
    start_time  = time.time()
    for i in range(100):
        with torch.no_grad():

            id_latent   = arcface(id_img)

            results     = model(attr, id_latent)
            elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))
    information="Elapsed [{}]".format(elapsed)
    print(information)