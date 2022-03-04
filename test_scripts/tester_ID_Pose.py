#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_ID_Pose.py
# Created Date: Friday March 4th 2022
# Author: Liu Naiyuan
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 5th March 2022 1:00:29 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import  os
import  cv2
import  time
import  glob
from    tqdm import tqdm

import  torch
import  torch.nn.functional as F
from    torchvision import transforms
from    torch.utils import data

import  numpy as np

import  PIL
from    PIL import Image


class TotalDataset(data.Dataset):
    """Dataset class for the vggface dataset with precalulated face landmarks."""

    def __init__(self,image_dir,content_transform):
        self.image_dir= image_dir
        self.content_transform= content_transform
        self.dataset = []
        self.preprocess()
        self.num_images = len(self.dataset)
    
    def preprocess(self):
        """Preprocess the Face++ original frames."""
        filenames = sorted(glob.glob(os.path.join(self.image_dir, '*'), recursive=False))
        # self.total_num = len(lines)
        for filename in  filenames:
            self.dataset.append(filename)

        print('Finished preprocessing the Face++ original frames dataset...')

            
    def __getitem__(self, index):
        """Return two src domain images and two dst domain images."""
        src_filename = self.dataset[index]

        split_tmp = src_filename.split('/')

        save_filename = split_tmp[-1]

        src_image1           = self.content_transform(Image.open(src_filename))

        return src_image1, save_filename


    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

def getLoader(c_image_dir, batch_size=16):
    """Build and return a data loader."""
    num_workers     = 8

    c_transforms    = []
    
    c_transforms.append(transforms.ToTensor())
    c_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    c_transforms = transforms.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir, c_transforms)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=False,shuffle=False,num_workers=num_workers,pin_memory=True)
    return content_data_loader, len(content_dataset)


class Tester(object):
    def __init__(self, config, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter
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
        self.network    = gen_class(**model_config["g_model"]["module_params"])
        self.network    = self.network.eval()
        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        self.reporter.writeModel(self.network.__str__())

        arcface1        = torch.load(self.arcface_ckpt, map_location=torch.device("cpu"))
        self.arcface    = arcface1['model'].module
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        model_path      = os.path.join(self.config["project_checkpoints"],
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
        version     = self.config["version"]
        batch_size  = self.config["batch_size"]
        specified_save_path = self.config["specified_save_path"]
        self.arcface_ckpt= self.config["arcface_ckpt"]

        self.reporter.writeInfo("Version %s"%version)

        if os.path.isdir(specified_save_path):
            print("Input a legal specified save path!")
            save_dir = specified_save_path
            save_dir = os.path.join(save_dir,"v_%s_step_%d"%(version,self.config["checkpoint_step"]))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        source_loader, dataet_len = getLoader(
                self.config["env_config"]["dataset_paths"]["id_pose_source_root"], batch_size=batch_size)
        target_loader, dataet_len = getLoader(
                self.config["env_config"]["dataset_paths"]["id_pose_source_root"], batch_size=batch_size)
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
                            
        # models
        self.__init_framework__()
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        with torch.no_grad():
            for profile_batch, filename_batch in tqdm(source_iter):
                profile_batch           = profile_batch.cuda()
                profile_id_downsample   = F.interpolate(profile_batch, (112,112), mode='bicubic')
                profile_latent_id       = self.arcface(profile_id_downsample)
                profile_latent_id       = F.normalize(profile_latent_id, p=2, dim=1)
                if init_batch ==True:
                    wholeid_batch = profile_latent_id.cpu()
                    init_batch = False
                else:
                    wholeid_batch = torch.cat([wholeid_batch,profile_latent_id.cpu()],dim=0)

            target_source_pair_dict = np.load(
                    self.config["env_config"]["dataset_paths"]["pairs_dict"] ,allow_pickle=True).item()

            for target_batch, filename_batch in tqdm(target_iter):
                target_index_list = []
                init_id_batch = True

                for filename_tmp in filename_batch:
                    source_index = int(filename_tmp.split('_')[0])
                    target_index = target_source_pair_dict[source_index]
                    target_index_list.append(target_index)
                    if init_id_batch:
                        batch_id = wholeid_batch[target_index][None].cuda()
                        init_id_batch = False
                    else:
                        batch_id = torch.cat([batch_id, wholeid_batch[target_index][None].cuda()],dim = 0)

                img_fakes = self.network(target_batch.cuda(), batch_id)

                for img_fake, target_index_tmp,filename_tmp in zip(img_fakes, target_index_list,filename_batch):
                    filename_tmp_split = filename_tmp.split('_')
                    final_filename = filename_tmp_split[0] + '_' +str(target_index_tmp) + '_' + filename_tmp_split[-1]
                    save_path = os.path.join(save_dir,final_filename)
                    img_fake     = img_fake * self.imagenet_std + self.imagenet_mean
                    img_fake     = img_fake.numpy().transpose(1,2,0)
                    img_fake     = np.clip(img_fake,0.0,1.0) * 255
                    PIL.Image.fromarray(img_fake).save(save_path,quality=100)
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))