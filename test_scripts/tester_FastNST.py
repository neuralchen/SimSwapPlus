#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_commonn.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th October 2021 8:22:37 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################



import os
import cv2
import time

import torch
from utilities.utilities import tensor2img

# from utilities.Reporter import Reporter
from tqdm import tqdm

class Tester(object):
    def __init__(self, config, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter

        #============build evaluation dataloader==============#
        print("Prepare the test dataloader...")
        dlModulename    = config["test_dataloader"]
        package         = __import__("data_tools.test_dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'TestDataset')
        dataloader      = dataloaderClass(config["test_data_path"],
                                        1,
                                        ["png","jpg"])
        self.test_loader= dataloader

        self.test_iter  = len(dataloader)
        # if len(dataloader)%config["batch_size"]>0:
        #     self.test_iter+=1
        
    
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        model_config    = self.config["model_configs"]
        script_name     = self.config["com_base"] + model_config["g_model"]["script"]
        class_name      = model_config["g_model"]["class_name"]
        package         = __import__(script_name, fromlist=True)
        network_class   = getattr(package, class_name)

        # TODO replace below lines to define the model framework        
        self.network = network_class(**model_config["g_model"]["module_params"])
        
        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        self.reporter.writeModel(self.network.__str__())
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
        
        model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["checkpoint_epoch"],
                                        self.config["checkpoint_names"]["generator_name"]))
    
        self.network.load_state_dict(torch.load(model_path))
        # self.network.load_state_dict(torch.load(pathwocao))
        print('loaded trained backbone model epoch {}...!'.format(self.config["project_checkpoints"]))

    def test(self):
        
        # save_result = self.config["saveTestResult"]
        save_dir    = self.config["test_samples_path"]
        ckp_epoch   = self.config["checkpoint_epoch"]
        version     = self.config["version"]
        batch_size  = self.config["batch_size"]
        win_size    = self.config["model_configs"]["g_model"]["module_params"]["window_size"]
                            
        # models
        self.__init_framework__()

        total = len(self.test_loader)
        print("total:", total)
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        with torch.no_grad():
            for _ in tqdm(range(total)):
                contents, img_names = self.test_loader()
                B, C, H, W  = contents.shape
                crop_h      = H - H%32
                crop_w      = W - W%32
                crop_s      = min(crop_h, crop_w)
                contents    = contents[:,:,(H//2 - crop_s//2):(crop_s//2 + H//2),
                                        (W//2 - crop_s//2):(crop_s//2 + W//2)]
                if self.config["cuda"] >=0:
                    contents = contents.cuda()
                res    = self.network(contents, (crop_s, crop_s))
                print("res shape:", res.shape)
                res    = tensor2img(res.cpu())
                temp_img = res[0,:,:,:]
                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                print(save_dir)
                print(img_names[0])
                cv2.imwrite(os.path.join(save_dir,'{}_version_{}_step{}.png'.format(
                                    img_names[0], version, ckp_epoch)),temp_img)
                                            
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))