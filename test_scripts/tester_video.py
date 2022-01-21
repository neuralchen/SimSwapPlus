#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_commonn.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 21st January 2022 11:06:37 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################



import os
import cv2
import time
import shutil

import torch
import torch.nn.functional as F
from torchvision import transforms

from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import numpy as np
from tqdm import tqdm
from PIL import Image
import glob

from utilities.ImagenetNorm import ImagenetNorm
from parsing_model.model import BiSeNet
from insightface_func.face_detect_crop_single import Face_detect_crop
from utilities.reverse2original import reverse2wholeimage

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
    
    def cv2totensor(self, cv2_img):
        """
        cv2_img: an image read by cv2, H*W*C
        return: an 1*C*H*W tensor
        """
        cv2_img = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
        cv2_img = torch.from_numpy(cv2_img)
        cv2_img = cv2_img.permute(2,0,1).cuda()
        temp    = cv2_img / 255.0
        temp    -= self.imagenet_mean
        temp    /= self.imagenet_std
        return temp.unsqueeze(0)

    def video_swap(
        self,
        video_path,
        id_vetor,
        save_path,
        temp_results_dir='./temp_results',
        crop_size=512,
        use_mask =False
    ):

        video_forcheck = VideoFileClip(video_path)
        if video_forcheck.audio is None:
            no_audio = True
        else:
            no_audio = False

        del video_forcheck

        if not no_audio:
            video_audio_clip = AudioFileClip(video_path)

        video = cv2.VideoCapture(video_path)
        ret = True
        frame_index = 0

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps = video.get(cv2.CAP_PROP_FPS)
        if  os.path.exists(temp_results_dir):
                shutil.rmtree(temp_results_dir)
        spNorm =ImagenetNorm()
        if use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None

        # while ret:
        for frame_index in tqdm(range(frame_count)): 
            ret, frame = video.read()
            if  ret:
                detect_results = self.detect.get(frame,crop_size)

                if detect_results is not None:
                    # print(frame_index)
                    if not os.path.exists(temp_results_dir):
                            os.mkdir(temp_results_dir)
                    frame_align_crop_list = detect_results[0]
                    frame_mat_list = detect_results[1]
                    swap_result_list = []
                    frame_align_crop_tenor_list = []
                    for frame_align_crop in frame_align_crop_list:
                        frame_align_crop_tenor = self.cv2totensor(frame_align_crop)
                        swap_result = self.network(frame_align_crop_tenor, id_vetor)[0]
                        swap_result = swap_result* self.imagenet_std + self.imagenet_mean
                        swap_result = torch.clip(swap_result,0.0,1.0)
                        cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                        swap_result_list.append(swap_result)
                        frame_align_crop_tenor_list.append(frame_align_crop_tenor)
                    reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame,\
                        os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),pasring_model =net,use_mask=use_mask, norm = spNorm)

                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
            else:
                break

        video.release()

        # image_filename_list = []
        path = os.path.join(temp_results_dir,'*.jpg')
        image_filenames = sorted(glob.glob(path))

        clips = ImageSequenceClip(image_filenames,fps = fps)

        if not no_audio:
            clips = clips.set_audio(video_audio_clip)
        basename = os.path.basename(video_path)
        basename = os.path.splitext(basename)[0]
        save_filename = os.path.join(save_path, basename+".mp4")
        index = 0
        while(True):
            if os.path.exists(save_filename):
                save_filename = os.path.join(save_path, basename+"_%d.mp4"%index)
                index += 1
            else:
                break
        clips.write_videofile(save_filename,audio_codec='aac')
            
    
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
        # loader1 = torch.load(self.config["ckp_name"]["generator_name"])
        # print(loader1.key())
        # pathwocao = "H:\\Multi Scale Kernel Prediction Networks\\Mobile_Oriented_KPN\\train_logs\\repsr_pixel_0\\checkpoints\\epoch%d_RepSR_Plain.pth"%self.config["checkpoint_epoch"]
        model_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
        self.network.load_state_dict(torch.load(model_path))
        # self.network.load_state_dict(torch.load(pathwocao))
        print('loaded trained backbone model step {}...!'.format(self.config["checkpoint_step"]))

    def test(self):
        
        # save_result = self.config["saveTestResult"]
        save_dir    = self.config["test_samples_path"]
        ckp_step    = self.config["checkpoint_step"]
        version     = self.config["version"]
        id_imgs     = self.config["id_imgs"]
        attr_files  = self.config["attr_files"]
        self.arcface_ckpt= self.config["arcface_ckpt"]
                            
        # models
        self.__init_framework__()

        

        mode        = None
        self.detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
        self.detect.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode = mode)
        
        id_img                  = cv2.imread(id_imgs)
        id_img_align_crop, _    = self.detect.get(id_img,512)
        id_img_align_crop_pil   = Image.fromarray(cv2.cvtColor(id_img_align_crop[0],cv2.COLOR_BGR2RGB)) 
        id_img                  = self.transformer_Arcface(id_img_align_crop_pil)
        id_img                  = id_img.unsqueeze(0).cuda()

        #create latent id
        id_img      = F.interpolate(id_img,size=(112,112), mode='bicubic')
        latend_id   = self.arcface(id_img)
        latend_id   = F.normalize(latend_id, p=2, dim=1)
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        with torch.no_grad():
            self.video_swap(attr_files, latend_id, save_dir, temp_results_dir="./.temples",\
                use_mask=False,crop_size=512)
                                            
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))