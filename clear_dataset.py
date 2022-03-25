#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: clear_dataset.py
# Created Date: Thursday March 24th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 24th March 2022 3:32:40 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import os
import json
from   utilities.json_config import readConfig, writeConfig

if __name__ == "__main__":
    savePath = "./vggface2hq_failed.txt"
    env_config = readConfig('env/env.json')
    env_config = env_config["path"]
    dataset_root = env_config["dataset_paths"]["vggface2_hq"]

    with open(savePath,'r') as logf:
        for line in logf:
            img_path = os.path.join(dataset_root,line[:-1]).replace("\\","/")
            try:
                os.rename(img_path,img_path+".deleted")
            except: pass