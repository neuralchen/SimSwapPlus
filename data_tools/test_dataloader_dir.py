#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: eval_dataloader_DIV2K.py
# Created Date: Tuesday January 12th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th October 2021 8:29:51 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import cv2
import glob
import torch

class TestDataset:
    def __init__(   self,
                    path,
                    batch_size  = 16,
                    subffix=['png','jpg']):
        """Initialize and preprocess the setX dataset."""
        self.path      = path
        
        self.subffix        = subffix
        self.dataset        = []
        self.pointer        = 0
        self.batch_size     = batch_size
        self.__preprocess__()
        self.num_images = len(self.dataset)
    
    def __preprocess__(self):
        """Preprocess the SetX dataset."""
            
        print("processing content images...")
        for i_suf in self.subffix:
            temp_path   = os.path.join(self.path,'*.%s'%(i_suf))
            images      = glob.glob(temp_path)
            for item in images:
                file_name   = os.path.basename(item)
                file_name   = os.path.splitext(file_name)
                file_name   = file_name[0]
                # lr_name     = os.path.join(set5lr_path, file_name)
                self.dataset.append([item,file_name])
        # self.dataset = images
        print('Finished preprocessing the content dataset, total image number: %d...'%len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer>=self.num_images:
            self.pointer = 0
            a = "The end of the story!"
            raise StopIteration(print(a))
        elif (self.pointer+self.batch_size) > self.num_images:
            end = self.num_images
        else:
            end = self.pointer+self.batch_size
        for i in range(self.pointer, end):
            filename    = self.dataset[i][0]
            hr_img      = cv2.imread(filename)
            hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
            hr_img      = hr_img.transpose((2,0,1))#.astype(np.float)
            
            hr_img      = torch.from_numpy(hr_img)
            hr_img      = hr_img/255.0
            hr_img      = 2 * (hr_img - 0.5)
            if (i-self.pointer) == 0:
                hr_ls   = hr_img.unsqueeze(0)
                nm_ls   = [self.dataset[i][1],]
            else:
                hr_ls   = torch.cat((hr_ls,hr_img.unsqueeze(0)),0)
                nm_ls   += [self.dataset[i][1],]
        self.pointer = end
        return hr_ls, nm_ls
    
    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'