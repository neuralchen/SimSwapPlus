#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_commonn.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 4th July 2021 11:32:14 am
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
                                        config["batch_size"],
                                        ["png","jpg"])
        self.test_loader= dataloader

        self.test_iter  = len(dataloader)//config["batch_size"]
        if len(dataloader)%config["batch_size"]>0:
            self.test_iter+=1
        
    
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        script_name     = "components."+self.config["module_script_name"]
        class_name      = self.config["class_name"]
        package         = __import__(script_name, fromlist=True)
        network_class   = getattr(package, class_name)
        n_class         = len(self.config["selectedStyleDir"])

        # TODO replace below lines to define the model framework        
        self.network = network_class(self.config["GConvDim"],
                                    self.config["GKS"],
                                    self.config["resNum"],
                                    n_class
                                    #**self.config["module_params"]
                                )
        
        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        self.reporter.writeModel(self.network.__str__())
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
        # loader1 = torch.load(self.config["ckp_name"]["generator_name"])
        # print(loader1.key())
        # pathwocao = "H:\\Multi Scale Kernel Prediction Networks\\Mobile_Oriented_KPN\\train_logs\\repsr_pixel_0\\checkpoints\\epoch%d_RepSR_Plain.pth"%self.config["checkpoint_epoch"]
        self.network.load_state_dict(torch.load(self.config["ckp_name"]["generator_name"])["g_model"])
        # self.network.load_state_dict(torch.load(pathwocao))
        print('loaded trained backbone model epoch {}...!'.format(self.config["checkpoint_epoch"]))

    def test(self):
        
        # save_result = self.config["saveTestResult"]
        save_dir    = self.config["test_samples_path"]
        ckp_epoch   = self.config["checkpoint_epoch"]
        version     = self.config["version"]
        batch_size  = self.config["batch_size"]
        style_names = self.config["selectedStyleDir"]
        n_class     = len(style_names)
                            
        # models
        self.__init_framework__()

        condition_labels = torch.ones((n_class, batch_size, 1)).long()
        for i in range(n_class):
            condition_labels[i,:,:] = condition_labels[i,:,:]*i
        if self.config["cuda"] >=0:
            condition_labels = condition_labels.cuda()
        total = len(self.test_loader)
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        with torch.no_grad():
            for _ in tqdm(range(total//batch_size)):
                contents, img_names = self.test_loader()
                for i in range(n_class):
                    if self.config["cuda"] >=0:
                        contents = contents.cuda()
                    res, _ = self.network(contents, condition_labels[i, 0, :])
                    res    = tensor2img(res.cpu())
                    for t in range(batch_size):
                        temp_img = res[t,:,:,:]
                        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(save_dir,'{}_version_{}_step{}_style_{}.png'.format(
                                            img_names[t], version, ckp_epoch, style_names[i])),temp_img)
                                            
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))