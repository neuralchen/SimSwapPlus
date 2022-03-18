#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test_arcface.py
# Created Date: Thursday March 17th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 17th March 2022 12:34:57 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################
import  torch

if __name__ == "__main__":
    arcface1        = torch.load("./arcface_ckpt/arcface_checkpoint.tar", map_location=torch.device("cpu"))
    print(arcface1)
    arcface         = arcface1['model'].module