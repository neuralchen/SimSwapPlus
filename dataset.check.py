#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: dataset.check.py
# Created Date: Sunday April 3rd 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 3rd April 2022 2:57:48 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import os
import glob
from   utilities.json_config import readConfig, writeConfig

# dataset = "G:/VGGFace2-HQ/VGGface2_None_norm_512_true_bygfpgan"
# mask_dir= "G:/VGGFace2-HQ/VGGface2_HQ_original_aligned_mask"

savePath = "./vggface2hq_failed.txt"
env_config = readConfig('env/env.json')
env_config = env_config["path"]
dataset = env_config["dataset_paths"]["vggface2_hq"]["images"]
mask_dir = env_config["dataset_paths"]["vggface2_hq"]["masks"]

temp_path   = os.path.join(dataset,'*/')
pathes      = glob.glob(temp_path)
for dir_item in pathes:
    join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
    print("processing %s"%dir_item,end='\r')
    dir_path = os.path.dirname(join_path[1])
    dir_name = os.path.join(mask_dir, os.path.basename(dir_path))
    # print(dir_name)
    temp_list = []
    for item in join_path:
        img_name    = os.path.basename(item)
        img_name, _ = os.path.splitext(img_name)
        mask_name   = os.path.join(dir_name, img_name + ".png")
        if not os.path.exists(mask_name):
            print(mask_name)