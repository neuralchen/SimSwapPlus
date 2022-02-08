#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: face_crop.py
# Created Date: Tuesday February 1st 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 2nd February 2022 11:17:04 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import os
import cv2
import sys
import glob
import argparse
from   tqdm import tqdm

from pathlib import Path

from insightface_func.face_detect_crop_multi import Face_detect_crop


def getParameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--save_path', type=str, default="./output/",
                                            help="The root path for saving cropped images")
    parser.add_argument('-v', '--video', type=str, default="G:\\4K\\05.mp4",
                                            help="The path for input video")                                        
    parser.add_argument('-c', '--crop_size', type=int, default=512,
                                            help="expected image resolution")
    parser.add_argument('-s', '--min_scale', type=float, default=0.7,
                                            help="tolerance range for the size of the captured face image")
    parser.add_argument('-m', '--mode', type=str, default="none", 
                                            choices=['ffhq', 'none'],help="none:VGG crop, ffhq:FFHQ crop")
    parser.add_argument('-f', '--format', type=str, default="png", 
                                            choices=['jpg', 'png'],help="target file format")
    parser.add_argument('-i', '--interval', type=int, default=20,
                                            help="number of frames interval")
    parser.add_argument('-b', '--blur', type=float, default=10.0,
                                            help="blur degree")
    return parser.parse_args()

def main(config):
    mode        = config.mode
    crop_size   = config.crop_size
    video       = config.video
    tg_path     = config.save_path
    tg_format   = config.format
    min_scale   = config.min_scale
    blur_t      = config.blur
    interval    = config.interval
    font        = cv2.FONT_HERSHEY_SIMPLEX
    detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
    detect.prepare(ctx_id = 0, det_thresh=0.6,\
                    det_size=(640,640),mode = mode,crop_size=crop_size,ratio=min_scale)
    video_path = os.path.basename(video)
    video_basename = os.path.splitext(video_path)[0]
    save_path = os.path.join(tg_path,video_basename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    # for frame_index in tqdm(range(0,frame_count,interval)):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret==True:
            
            img_align_crop = detect.get(frame)
            if img_align_crop:
                img_align_crop = img_align_crop[0]
                sub_index = 0
                for face_i in img_align_crop:
                    imageVar = cv2.Laplacian(face_i, cv2.CV_64F).var()
                    f_path =os.path.join(save_path, str(frame_index).zfill(6)+"_%d.%s"%(sub_index,tg_format))
                    if imageVar < blur_t:
                        print("Over blurry image!")
                        continue
                    # face_i = cv2.putText(face_i, '%.1f'%imageVar,(50, 50), font, 0.8, (15, 9, 255), 2)
                    cv2.imwrite(f_path,face_i)
                    sub_index += 1
            # else:
            #     print("Detect no face!")
            frame_index += 1
    cap.release()

if __name__ == "__main__":
    config = getParameters()
    main(config)