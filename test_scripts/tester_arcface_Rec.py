#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_commonn.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 29th January 2022 12:41:01 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################



import os
import cv2
import time
import glob

import torch
import torch.nn.functional as F
from   torchvision import transforms
from   torchvision.utils  import save_image

import numpy as np
from   PIL import Image

from insightface_func.face_detect_crop_single import Face_detect_crop

class Tester(object):
    def __init__(self, config, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter

        self.transformer_Arcface = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.imagenet_std    = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.imagenet_mean   = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        if self.config["cuda"] >=0:
            self.imagenet_std    = self.imagenet_std .cuda()
            self.imagenet_mean   = self.imagenet_mean.cuda()
       
    
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        model_config    = self.config["model_configs"]
        gscript_name    = self.config["com_base"] + model_config["g_model"]["script"]
        class_name      = model_config["g_model"]["class_name"]
        package         = __import__(gscript_name, fromlist=True)
        gen_class       = getattr(package, class_name)
        self.network    = gen_class(**model_config["g_model"]["module_params"])

        # TODO replace below lines to define the model framework        
        self.network = gen_class(**model_config["g_model"]["module_params"])
        self.network = self.network.eval()
        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        self.reporter.writeModel(self.network.__str__())

        arcface1        = torch.load(self.arcface_ckpt, map_location=torch.device("cpu"))
        self.arcface    = arcface1['model'].module
        self.arcface.eval()
        self.arcface.requires_grad_(False)
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
            self.arcface = self.arcface.cuda()
            
        model_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
        self.network.load_state_dict(torch.load(model_path))
        print('loaded trained backbone model step {}...!'.format(self.config["checkpoint_step"]))

    def test(self):
        
        save_dir    = self.config["test_samples_path"]
        ckp_step    = self.config["checkpoint_step"]
        version     = self.config["version"]
        attr_files  = self.config["attr_files"]
        self.arcface_ckpt= self.config["arcface_ckpt"]
        imgs_list = []
        if os.path.isdir(attr_files):
            print("Input a dir....")
            imgs = glob.glob(os.path.join(attr_files,"**"), recursive=True)
            for item in imgs:
                imgs_list.append(item)
            print(imgs_list)
        else:
            print("Input an image....")
            imgs_list.append(attr_files)
        img_num = len(imgs_list)

                            
        # models
        self.__init_framework__()

        mode        = None
        self.detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.detect.prepare(ctx_id = 0, det_thresh=0.6, det_size=(640,640),mode = mode)
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        index = 0
        with torch.no_grad():
            for img in imgs_list[1:]:
                print(img)
                attr_img_ori= cv2.imread(img)
                # try:
                #     attr_img_align_crop, mat = self.detect.get(attr_img_ori,512)
                # except:
                #     print("No face detected!")
                #     continue
                # attr_img_align_crop_pil = Image.fromarray(cv2.cvtColor(attr_img_align_crop[0],cv2.COLOR_BGR2RGB))
                attr_img_align_crop_pil = Image.fromarray(cv2.cvtColor(attr_img_ori,cv2.COLOR_BGR2RGB))
                attr_img    = self.transformer_Arcface(attr_img_align_crop_pil).unsqueeze(0).cuda()

                attr_img_arc= F.interpolate(attr_img,size=(112,112), mode='bicubic')
                attr_id     = self.arcface(attr_img_arc)
                results     = self.network(attr_id)

                results     = results * self.imagenet_std + self.imagenet_mean
                results     = results.clamp_(0, 1)
                attr        = attr_img_arc  * self.imagenet_std + self.imagenet_mean
                results     = torch.concat((attr, results), dim=2)
                if index == 0:
                    final_img   = results
                else:
                    final_img   = torch.concat((final_img, results), dim=0)
                index += 1
            save_filename = os.path.join(save_dir, "ckp_%s_v_%s.png"%(ckp_step, version))
            mark = 0
            while(True):
                if os.path.exists(save_filename):
                    save_filename = os.path.join(save_dir, "ckp_%s_v_%s_%d.png"%(ckp_step, version,mark))
                    mark += 1
                else:
                    break
            save_image(final_img, save_filename, nrow=img_num//8)
                                            
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))