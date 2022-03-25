#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: translation_list2json.py
# Created Date: Thursday March 24th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 24th March 2022 3:20:06 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import json

if __name__ == "__main__":
    savePath = "./vggface2hq_failed.txt"
    log_txt  = "./vggface2hq_failed.json"
    images = {}

    with open(savePath,'r') as logf:
        for line in logf:
            cells = line.split("/")
            if images.__contains__(cells[0]):
                images[cells[0]] += [cells[1]]
            else:
                images[cells[0]] = [cells[1]]
    with open(log_txt, 'w') as cf:
        configjson  = json.dumps(images, indent=4)
        cf.writelines(configjson)