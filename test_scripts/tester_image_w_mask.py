#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_commonn.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 23rd April 2022 10:05:22 pm
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
        
        if self.config["preprocess"]:
            print("Employ GFPGAN to upsampling detected face images!")
            from face_enhancer.gfpgan import GFPGANer
            version = '1.2'
            if version == '1':
                arch = 'original'
                channel_multiplier = 1
                model_name = 'GFPGANv1'
            elif version == '1.2':
                arch = 'clean'
                channel_multiplier = 2
                model_name = 'GFPGANCleanv1-NoCE-C2'
            elif version == '1.3':
                arch = 'clean'
                channel_multiplier = 2
                model_name = 'GFPGANv1.3'

            # determine model paths
            model_path = os.path.join('./face_enhancer/experiments/pretrained_models', model_name + '.pth')
            if not os.path.isfile(model_path):
                model_path = os.path.join('./face_enhancer/realesrgan/weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                raise ValueError(f'Model {model_name} does not exist.')

            self.restorer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=None)
        
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

        mode        = crop_mode.lower()
        if mode == "vggface":
            mode = "none"
        self.detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.detect.prepare(ctx_id = 0, det_thresh=0.6, det_size=(640,640),mode = mode)
        
        id_img                  = cv2.imread(id_imgs)
        id_img_align_crop, _    = self.detect.get(id_img,512)
        id_img_align_crop_pil   = Image.fromarray(cv2.cvtColor(id_img_align_crop[0],cv2.COLOR_BGR2RGB)) 
        id_img                  = self.transformer_Arcface(id_img_align_crop_pil)
        id_img                  = id_img.unsqueeze(0).cuda()

        #create latent id
        id_img      = F.interpolate(id_img,size=(112,112), mode='bicubic')
        latend_id   = self.arcface(id_img)
        latend_id   = F.normalize(latend_id, p=2, dim=1)
        cos_loss    = torch.nn.CosineSimilarity()
        font        = cv2.FONT_HERSHEY_SIMPLEX 
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
                try:
                    attr_img_align_crop, mat = self.detect.get(attr_img_ori,512)
                except:
                    continue
                restored_face       = attr_img_align_crop[0]
                if self.config["preprocess"]:
                    _, _, restored_face = self.restorer.enhance(
                    restored_face, has_aligned=False, only_center_face=True, paste_back=True)
                attr_img_align_crop_pil = Image.fromarray(cv2.cvtColor(restored_face,cv2.COLOR_BGR2RGB))
                attr_img    = self.transformer_Arcface(attr_img_align_crop_pil).unsqueeze(0).cuda()

                attr_img_arc = F.interpolate(attr_img,size=(112,112), mode='bicubic')
                # cv2.imwrite(os.path.join("./swap_results", "id_%s.png"%(id_basename)),id_img_align_crop[0])
                attr_id   = self.arcface(attr_img_arc)
                attr_id   = F.normalize(attr_id, p=2, dim=1)
                cos_dis   = 1 -  cos_loss(latend_id, attr_id)

                mat         = mat[0]
                pred     = self.network(attr_img, latend_id)
                results  = pred[0]

                results_arc = F.interpolate(results,size=(112,112), mode='bicubic')
                results_arc   = self.arcface(results_arc)
                results_arc   = F.normalize(results_arc, p=2, dim=1)
                results_cos_dis     = 1 -  cos_loss(latend_id, results_arc)
                average_cos += results_cos_dis

                results     = results * self.imagenet_std + self.imagenet_mean
                results     = results.cpu().permute(0,2,3,1)[0,...]
                results     = results.numpy()
                results     = np.clip(results,0.0,1.0)
                img_white   = np.full((512,512), 255, dtype=float)
                
                # inverse the Affine transformation matrix
                mat_rev = np.zeros([2,3])
                div1 = mat[0][0]*mat[1][1]-mat[0][1]*mat[1][0]
                mat_rev[0][0] = mat[1][1]/div1
                mat_rev[0][1] = -mat[0][1]/div1
                mat_rev[0][2] = -(mat[0][2]*mat[1][1]-mat[0][1]*mat[1][2])/div1
                div2 = mat[0][1]*mat[1][0]-mat[0][0]*mat[1][1]
                mat_rev[1][0] = mat[1][0]/div2
                mat_rev[1][1] = -mat[0][0]/div2
                mat_rev[1][2] = -(mat[0][2]*mat[1][0]-mat[0][0]*mat[1][2])/div2

                orisize = (attr_img_ori.shape[1], attr_img_ori.shape[0])

                target_image = cv2.warpAffine(results, mat_rev, orisize)

                img_white = cv2.warpAffine(img_white, mat_rev, orisize)


                img_white[img_white>20] =255

                img_mask = img_white

                kernel = np.ones((40,40),np.uint8)
                img_mask = cv2.erode(img_mask,kernel,iterations = 1)
                kernel_size = (20, 20)
                blur_size = tuple(2*i+1 for i in kernel_size)
                img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

                img_mask /= 255

                img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])

                target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255

                img1 = np.array(attr_img_ori, dtype=np.float)
                img1 = img_mask * target_image + (1-img_mask) * img1
                final_img = img1.astype(np.uint8)
                attr_basename = os.path.splitext(os.path.basename(img))[0]
                if self.config["record_metric"]:
                    final_img = cv2.putText(final_img, 'id dis=%.4f'%results_cos_dis, (50, 50), font, 0.8, (15, 9, 255), 2)
                    final_img = cv2.putText(final_img, 'id--attr dis=%.4f'%cos_dis, (50, 80), font, 0.8, (15, 9, 255), 2)
                if self.config["preprocess"]:
                    save_filename = os.path.join(save_dir, 
                                        "id_%s--attr_%s_ckp_%s_v_%s_gfpgan.png"%(id_basename,
                                            attr_basename,ckp_step,version))
                else:
                    save_filename = os.path.join(save_dir, 
                                        "id_%s--attr_%s_ckp_%s_v_%s.png"%(id_basename,
                                            attr_basename,ckp_step,version))
                cv2.imwrite(save_filename, final_img)

                if self.config["save_mask"]:
                    num = 0

                    for mask in pred[1:]:

                        mask     = mask.cpu().permute(0,2,3,1)[0,...]
                        mask     = mask.numpy()
                        mask     = (mask - np.min(mask))/np.max(mask)
                        mask     = np.clip(mask,0.0,1.0) * 255

                        if self.config["preprocess"]:
                            save_filename = os.path.join(save_dir, 
                                            "id_%s--attr_%s_ckp_%s_v_%s_mask%d_gfpgan.png"%(id_basename,
                                                attr_basename,ckp_step,version,num))
                        else:
                            save_filename = os.path.join(save_dir, 
                                            "id_%s--attr_%s_ckp_%s_v_%s_mask%d.png"%(id_basename,
                                                attr_basename,ckp_step,version,num))

                        
                        cv2.imwrite(save_filename,mask)
                        num += 1
                
        average_cos /= len(imgs_list)                                    
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))
        print("Average cosin similarity between ID and results [{}]".format(average_cos.item()))
        self.reporter.writeInfo("Average cosin similarity between ID and results [{}]".format(average_cos.item()))