#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: id_cos.py
# Created Date: Friday March 25th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 29th March 2022 11:58:30 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################
import cv2
from   PIL import Image

import torch
import torch.nn.functional as F
from   torchvision import transforms
from insightface_func.face_detect_crop_single import Face_detect_crop

from    arcface_torch.backbones.iresnet import iresnet100

if __name__ == "__main__":
    imagenet_std    = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3,1,1)
    imagenet_mean   = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3,1,1)
    transformer_Arcface = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    arcface_ckpt = "./arcface_ckpt/arcface_checkpoint.tar"
    arcface1     = torch.load(arcface_ckpt, map_location=torch.device("cpu"))
    arcface      = arcface1['model'].module
    arcface.eval()

    root1 = "G:/VGGFace2-HQ/VGGface2_ffhq_align_256_9_28_512_bygfpgan/n000002/"
    root2 = "G:/VGGFace2-HQ/VGGface2_None_norm_512_true_bygfpgan/n000002/"

    # arcface_ckpt = "./arcface_torch/checkpoints/backbone.pth" # backbone.pth glint360k_cosface_r100_fp16_backbone.pth
    # arcface         = iresnet100(pretrained=False, fp16=False)
    # arcface.load_state_dict(torch.load(arcface_ckpt, map_location='cpu'))
    # arcface.eval()

    # id1 = "G:/swap_data/ID/hinton.jpg"
    # id2 = "G:/hififace-master/hififace-master/assets/inference_samples/hififace/img-172.jpg"
    id1 = root2 + "0003_01.jpg"
    id2 = root2 + "0036_01.jpg"

    mode        = "none"
    cos_loss    = torch.nn.CosineSimilarity()
    # detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
    # detect.prepare(ctx_id = 0, det_thresh=0.6, det_size=(640,640),mode = mode)
    id_img                  = cv2.imread(id1)
    # id_img_align_crop, _    = detect.get(id_img,256)
    # cv2.imwrite("id1_crop.png",id_img_align_crop[0])
    # id_img_align_crop_pil   = Image.fromarray(cv2.cvtColor(id_img_align_crop[0],cv2.COLOR_BGR2RGB))
    id_img_align_crop_pil   = Image.fromarray(cv2.cvtColor(id_img,cv2.COLOR_BGR2RGB))
    id_img                  = transformer_Arcface(id_img_align_crop_pil)
    id_img                  = id_img.unsqueeze(0)
    id_img                  = F.interpolate(id_img,size=(112,112), mode='bicubic')
    # id_img                  = (id_img-0.5)*2.0
    latend_id               = arcface(id_img)
    latend_id               = F.normalize(latend_id, p=2, dim=1)

    id_img2                  = cv2.imread(id2)
    # id_img_align_crop2, _    = detect.get(id_img2,256)
    # cv2.imwrite("id2_crop.png",id_img_align_crop2[0])
    # id_img_align_crop_pil2   = Image.fromarray(cv2.cvtColor(id_img_align_crop2[0],cv2.COLOR_BGR2RGB)) 
    id_img_align_crop_pil2   = Image.fromarray(cv2.cvtColor(id_img2,cv2.COLOR_BGR2RGB))
    id_img2                  = transformer_Arcface(id_img_align_crop_pil2)
    id_img2                  = id_img2.unsqueeze(0)
    id_img2                  = F.interpolate(id_img2,size=(112,112), mode='bicubic')
    # id_img2                  = (id_img2-0.5)*2.0
    latend_id2               = arcface(id_img2)
    latend_id2               = F.normalize(latend_id2, p=2, dim=1)
    
    cos_dis   = 1 -  cos_loss(latend_id, latend_id2)
    print("cosine similarity:", cos_dis.item())