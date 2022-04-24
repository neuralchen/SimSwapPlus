import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils import utils
from PIL import Image
from tqdm import tqdm
import torch
import time 
import numpy as np
import cv2
import glob
from torchvision.transforms import transforms

if __name__ == '__main__':
    opt = TestOptions()
    opt = opt.parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.load_pretrain_models()

    netP = model.netP
    model.eval()

    to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    image_dir = "G:/VGGFace2-HQ/VGGface2_None_norm_512_true_bygfpgan/"
    output_dir = "G:/VGGFace2-HQ/VGGface2_HQ_original_aligned_mask"

    temp_path   = os.path.join(image_dir,'*/')
    pathes      = glob.glob(temp_path)
    dataset = []
    for dir_item in pathes:
        join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
        print("processing %s"%dir_item,end='\r')
        temp_list = []
        for item in join_path:
            temp_list.append(item)
        dataset.append(temp_list)

    # ------------------------ restore ------------------------
    for i_dir in dataset:
        path = os.path.dirname(i_dir[0])
        dir_name = os.path.join(output_dir, os.path.basename(path))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        for img_path in i_dir:
            hr_img = Image.open(img_path).convert('RGB')
            inp    = to_tensor(hr_img).unsqueeze(0)
            with torch.no_grad():
                parse_map, _ = netP(inp)
                parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            ref_parse_img = utils.color_parse_map(parse_map_sm)
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            save_face_name = f'{basename}.png'
            # print(save_face_name)
            save_path = os.path.join(dir_name, save_face_name)
            # os.makedirs(opt.save_masks_dir, exist_ok=True)
            img = cv2.cvtColor(ref_parse_img[0],cv2.COLOR_RGB2GRAY)
            cv2.imwrite(save_path,img)

    # for i, data in tqdm(enumerate(dataset), total=len(dataset)//opt.batch_size):
    #     inp = data['LR']
    #     with torch.no_grad():
    #         parse_map, _ = netP(inp)
    #         parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
    #     img_path = data['LR_paths']     # get image paths
    #     ref_parse_img = utils.color_parse_map(parse_map_sm)
    #     for i in range(len(img_path)):
    #         img_name = os.path.basename(img_path[i])
    #         basename, ext = os.path.splitext(img_name)
    #         save_face_name = f'{basename}.png'
    #         # print(save_face_name)
    #         save_path = os.path.join(opt.save_masks_dir, save_face_name)
    #         os.makedirs(opt.save_masks_dir, exist_ok=True)
    #         img = cv2.cvtColor(ref_parse_img[i],cv2.COLOR_RGB2GRAY)
    #         cv2.imwrite(save_path,img)
            # save_img = Image.fromarray(ref_parse_img[i])
            # save_img.save(save_path)


       
 
