#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: annotation.py
# Created Date: Saturday February 26th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 27th February 2022 11:03:58 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import cv2
import os
import glob
import json

import argparse

keytable={
    "left":2424832,
    "right":2555904,
    "up":2490368,
    "down":2621440,
    "esc":27,
    "space":32
}

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--image_dir', type=str, default="G:\\VGGFace2-HQ\\VGGface2_None_norm_512_true_bygfpgan")
    parser.add_argument('--savetxt', type=str, default="./check_list.txt")
    parser.add_argument('--winWidth', type=int, default=512)
    parser.add_argument('--winHeight', type=int, default=512)
    return parser.parse_args()

if __name__ == "__main__":
    config = getParameters()
    savePath = config.savetxt
    
    log_txt = "./breakpoint.json"
    

    temp_path   = os.path.join(config.image_dir,'*/')
    pathes      = glob.glob(temp_path)
    dataset = []
    for dir_item in pathes:
        join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
        print("processing %s"%dir_item,end='\r')
        temp_list = []
        for item in join_path:
            temp_list.append(item)
        dataset.append(temp_list)
    cv2.namedWindow("Annatation",0)
    cv2.resizeWindow("Annatation", config.winWidth, config.winHeight)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        with open(log_txt,'r') as cf:
            breakpoint_json = cf.read()
            breakpoint_json = json.loads(breakpoint_json)
            if isinstance(breakpoint_json,str):
                breakpoint_json = json.loads(breakpoint_json)
    except:
        breakpoint_json = {"breakpoint":[0,0]}
    save_point = breakpoint_json["breakpoint"]
    
    
    total_dir = len(dataset)
    dir_pointer     = save_point[0]
    indir_pointer   = save_point[1]
    
    
    while(1):
        try:
            img = cv2.imread(os.path.join(config.image_dir,dataset[dir_pointer][indir_pointer]))
            imgshow = cv2.putText(img, "[%d]/[%d]-[%d]/[%d]"%(dir_pointer+1,total_dir,
                                indir_pointer+1,len(dataset[dir_pointer])), (0, 30), font, 0.4, (255, 255, 255), 1)
            imgshow = cv2.putText(imgshow, "%s"%(dataset[dir_pointer][indir_pointer][-19:]), (0, 60), font, 0.4, (255, 255, 255), 1)
            cv2.imshow('Annatation',imgshow)
            waitkey_num = cv2.waitKeyEx(20)
            # if waitkey_num != -1:
            #     print(waitkey_num)
            if waitkey_num == keytable["left"]:
                # print("Left")
                indir_pointer -= 1
                if indir_pointer<0:
                    indir_pointer = 0
                
            if waitkey_num == keytable["right"]:
                # print("Right")
                indir_pointer += 1
                if indir_pointer>= len(dataset[dir_pointer]):
                    indir_pointer = len(dataset[dir_pointer])-1
            
            if waitkey_num == keytable["up"]:
                # print("Left")
                dir_pointer -= 1
                indir_pointer = 0
                if dir_pointer<0:
                    dir_pointer = 0
                
            if waitkey_num == keytable["down"]:
                # print("Right")
                dir_pointer += 1
                indir_pointer = 0
                if dir_pointer>= total_dir:
                    dir_pointer = total_dir-1

            if waitkey_num == keytable["space"]:
                image_name_cur = dataset[dir_pointer][indir_pointer][-19:]
                print("Save image name %s"%image_name_cur)
                with open(savePath,'a+') as logf:
                    logf.writelines("%s\n"%(image_name_cur))
            if waitkey_num == keytable["esc"]:
                breakpoint_json = {"breakpoint":[dir_pointer,indir_pointer]}
                with open(log_txt, 'w') as cf:
                    configjson  = json.dumps(breakpoint_json, indent=4)
                    cf.writelines(configjson)
                break
        except KeyboardInterrupt:
            breakpoint_json = {"breakpoint":[dir_pointer,indir_pointer]}
            with open(log_txt, 'w') as cf:
                configjson  = json.dumps(breakpoint_json, indent=4)
                cf.writelines(configjson)
            break
    cv2.destroyAllWindows()