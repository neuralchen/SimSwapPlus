#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_commonn.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 24th March 2022 12:40:35 pm
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
        self.imagenet_std    = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3,1,1)
        self.imagenet_mean   = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3,1,1)
       
    
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
        # for name in self.network.state_dict():
        #     print(name)
        self.features = {}
        mapping_layers = [
            "first_layer",
            "down4",
            "BottleNeck.2"
        ]

        

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        self.reporter.writeModel(self.network.__str__())

        arcface1        = torch.load(self.arcface_ckpt, map_location=torch.device("cpu"))
        self.arcface    = arcface1['model'].module
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        model_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
        self.network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print('loaded trained backbone model step {}...!'.format(self.config["checkpoint_step"]))
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
            self.arcface = self.arcface.cuda()
            
        

    def test(self):
        
        save_dir    = self.config["test_samples_path"]
        ckp_step    = self.config["checkpoint_step"]
        version     = self.config["version"]
        id_imgs     = self.config["id_imgs"]
        crop_mode   = self.config["crop_mode"]
        attr_files  = self.config["attr_files"]
        specified_save_path = self.config["specified_save_path"]
        self.arcface_ckpt= self.config["arcface_ckpt"]
        imgs_list = []

        self.reporter.writeInfo("Version %s"%version)

        if os.path.isdir(specified_save_path):
            print("Input a legal specified save path!")
            save_dir = specified_save_path

        if os.path.isdir(attr_files):
            print("Input a dir....")
            imgs = glob.glob(os.path.join(attr_files,"**"), recursive=True)
            for item in imgs:
                imgs_list.append(item)
            print(imgs_list)
        else:
            print("Input an image....")
            imgs_list.append(attr_files)
        id_basename = os.path.basename(id_imgs)
        id_basename = os.path.splitext(os.path.basename(id_imgs))[0]
                            
        # models
        self.__init_framework__()
        
        id_img                  = cv2.imread(id_imgs)
        id_img_align_crop_pil   = Image.fromarray(cv2.cvtColor(id_img,cv2.COLOR_BGR2RGB)) 
        id_img                  = self.transformer_Arcface(id_img_align_crop_pil)
        id_img                  = id_img.unsqueeze(0).cuda()

        #create latent id
        id_img      = F.interpolate(id_img,size=(112,112), mode='bicubic')
        latend_id   = self.arcface(id_img)
        latend_id   = F.normalize(latend_id, p=2, dim=1)
        cos_loss    = torch.nn.CosineSimilarity()
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        cos_dict = {}
        average_cos = 0
        with torch.no_grad():
            for img in imgs_list:
                print(img)
                attr_img_ori= cv2.imread(img)
                attr_img_align_crop_pil = Image.fromarray(cv2.cvtColor(attr_img_ori,cv2.COLOR_BGR2RGB))
                attr_img    = self.transformer_Arcface(attr_img_align_crop_pil).unsqueeze(0).cuda()

                attr_img_arc = F.interpolate(attr_img,size=(112,112), mode='bicubic')
                # cv2.imwrite(os.path.join("./swap_results", "id_%s.png"%(id_basename)),id_img_align_crop[0])
                attr_id   = self.arcface(attr_img_arc)
                attr_id   = F.normalize(attr_id, p=2, dim=1)
                cos_dis   = 1 -  cos_loss(latend_id, attr_id)
                results     = self.network(attr_img, latend_id)

                results_arc = F.interpolate(results,size=(112,112), mode='bicubic')
                results_arc   = self.arcface(results_arc)
                results_arc   = F.normalize(results_arc, p=2, dim=1)
                results_cos_dis     = 1 -  cos_loss(latend_id, results_arc)
                average_cos += results_cos_dis

                results     = results * self.imagenet_std + self.imagenet_mean
                results     = results.cpu().permute(0,2,3,1)[0,...]
                results     = results.numpy()
                results     = np.clip(results,0.0,1.0) * 255
                results     = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)
                
                final_img   = results.astype(np.uint8)
                attr_basename = os.path.splitext(os.path.basename(img))[0]
                save_filename = os.path.join(save_dir, 
                                    "id_%s--attr_%s_ckp_%s_v_%s.png"%(id_basename,
                                        attr_basename,ckp_step,version))
                
                cv2.imwrite(save_filename, final_img)
        average_cos /= len(imgs_list)                                    
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))
        print("Average cosin similarity between ID and results [{}]".format(average_cos.item()))
        self.reporter.writeInfo("Average cosin similarity between ID and results [{}]".format(average_cos.item()))