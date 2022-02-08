#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: data_loader_VGGFace2HQ copy.py
# Created Date: Sunday February 6th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 8th February 2022 1:24:27 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
# from StyleResize import StyleResize

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

class data_prefetcher():
    def __init__(self, loader, cur_gpu):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream(device=cur_gpu)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda(device=cur_gpu).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda(device=cur_gpu).view(1,3,1,1)
        self.cur_gpu = cur_gpu
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        # self.num_images = loader.__len__()
        self.preload()

    def preload(self):
        # try:
        self.src_image1, self.src_image2 = next(self.dataiter)
        # except StopIteration:
        #     self.dataiter = iter(self.loader)
        #     self.src_image1, self.src_image2 = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.src_image1  = self.src_image1.cuda(device= self.cur_gpu, non_blocking=True)
            self.src_image1  = self.src_image1.sub_(self.mean).div_(self.std)
            self.src_image2  = self.src_image2.cuda(device= self.cur_gpu, non_blocking=True)
            self.src_image2  = self.src_image2.sub_(self.mean).div_(self.std)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
    def next(self):
        torch.cuda.current_stream(device= self.cur_gpu,).wait_stream(self.stream)
        src_image1  = self.src_image1
        src_image2  = self.src_image2
        self.preload()
        return src_image1, src_image2
    
    # def __len__(self):
    #     """Return the number of images."""
    #     return self.num_images

class VGGFace2HQDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                    image_dir,
                    img_transform,
                    subffix='jpg',
                    random_seed=1234):
        """Initialize and preprocess the VGGFace2 HQ dataset."""
        self.image_dir      = image_dir
        self.img_transform  = img_transform   
        self.subffix        = subffix
        self.dataset        = []
        self.random_seed    = random_seed
        self.preprocess()
        self.num_images = len(self.dataset)

    def preprocess(self):
        """Preprocess the VGGFace2 HQ dataset."""
        print("processing VGGFace2 HQ dataset images...")

        temp_path   = os.path.join(self.image_dir,'*/')
        pathes      = glob.glob(temp_path)
        self.dataset = []
        for dir_item in pathes:
            join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
            print("processing %s"%dir_item,end='\r')
            temp_list = []
            for item in join_path:
                temp_list.append(item)
            self.dataset.append(temp_list)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        print('Finished preprocessing the VGGFace2 HQ dataset, total dirs number: %d...'%len(self.dataset))
             
    def __getitem__(self, index):
        """Return two src domain images and two dst domain images."""
        dir_tmp1        = self.dataset[index]
        dir_tmp1_len    = len(dir_tmp1)

        filename1   = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        filename2   = dir_tmp1[random.randint(0,dir_tmp1_len-1)]
        image1      = self.img_transform(Image.open(filename1))
        image2      = self.img_transform(Image.open(filename2))
        return image1, image2
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

def GetLoader(  dataset_roots,
                rank,
                num_gpus,
                batch_size=16,
                **kwargs
                ):
    """Build and return a data loader."""
        
    data_root       = dataset_roots
    random_seed     = kwargs["random_seed"]
    num_workers     = kwargs["dataloader_workers"]
    
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    content_dataset = VGGFace2HQDataset(
                            data_root, 
                            c_transforms,
                            "jpg",
                            random_seed)
    device = torch.device('cuda', rank)
    # sampler = InfiniteSampler(dataset=content_dataset, rank=rank, num_replicas=num_gpus, seed=random_seed)
    # content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
    #                 drop_last=False,shuffle=False,num_workers=num_workers,pin_memory=True, sampler=sampler)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=False,shuffle=True,num_workers=num_workers,pin_memory=True)
    # prefetcher = data_prefetcher(content_data_loader,device)
    return content_data_loader

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

if __name__ == "__main__":
    from torchvision.utils import save_image
    style_class  = ["vangogh","picasso","samuel"]
    categories_names = \
        ['a/abbey', 'a/arch', 'a/amphitheater', 'a/aqueduct', 'a/arena/rodeo', 'a/athletic_field/outdoor',
         'b/badlands', 'b/balcony/exterior', 'b/bamboo_forest', 'b/barn', 'b/barndoor', 'b/baseball_field',
         'b/basilica', 'b/bayou', 'b/beach', 'b/beach_house', 'b/beer_garden', 'b/boardwalk', 'b/boathouse',
         'b/botanical_garden', 'b/bullring', 'b/butte', 'c/cabin/outdoor', 'c/campsite', 'c/campus',
         'c/canal/natural', 'c/canal/urban', 'c/canyon', 'c/castle', 'c/church/outdoor', 'c/chalet',
         'c/cliff', 'c/coast', 'c/corn_field', 'c/corral', 'c/cottage', 'c/courtyard', 'c/crevasse',
         'd/dam', 'd/desert/vegetation', 'd/desert_road', 'd/doorway/outdoor', 'f/farm', 'f/fairway',
         'f/field/cultivated', 'f/field/wild', 'f/field_road', 'f/fishpond', 'f/florist_shop/indoor',
         'f/forest/broadleaf', 'f/forest_path', 'f/forest_road', 'f/formal_garden', 'g/gazebo/exterior',
         'g/glacier', 'g/golf_course', 'g/greenhouse/indoor', 'g/greenhouse/outdoor', 'g/grotto', 'g/gorge',
         'h/hayfield', 'h/herb_garden', 'h/hot_spring', 'h/house', 'h/hunting_lodge/outdoor', 'i/ice_floe',
         'i/ice_shelf', 'i/iceberg', 'i/inn/outdoor', 'i/islet', 'j/japanese_garden', 'k/kasbah',
         'k/kennel/outdoor', 'l/lagoon', 'l/lake/natural', 'l/lawn', 'l/library/outdoor', 'l/lighthouse',
         'm/mansion', 'm/marsh', 'm/mausoleum', 'm/moat/water', 'm/mosque/outdoor', 'm/mountain',
         'm/mountain_path', 'm/mountain_snowy', 'o/oast_house', 'o/ocean', 'o/orchard', 'p/park',
         'p/pasture', 'p/pavilion', 'p/picnic_area', 'p/pier', 'p/pond', 'r/raft', 'r/railroad_track',
         'r/rainforest', 'r/rice_paddy', 'r/river', 'r/rock_arch', 'r/roof_garden', 'r/rope_bridge',
         'r/ruin', 's/schoolhouse', 's/sky', 's/snowfield', 's/swamp', 's/swimming_hole',
         's/synagogue/outdoor', 't/temple/asia', 't/topiary_garden', 't/tree_farm', 't/tree_house',
         'u/underwater/ocean_deep', 'u/utility_room', 'v/valley', 'v/vegetable_garden', 'v/viaduct',
         'v/village', 'v/vineyard', 'v/volcano', 'w/waterfall', 'w/watering_hole', 'w/wave',
         'w/wheat_field', 'z/zen_garden', 'a/alcove', 'a/apartment-building/outdoor', 'a/artists_loft',
         'b/building_facade', 'c/cemetery']

    s_datapath      = "D:\\F_Disk\\data_set\\Art_Data\\data_art_backup"
    c_datapath      = "D:\\Downloads\\data_large"
    savepath        = "D:\\PatchFace\\PleaseWork\\multi-style-gan\\StyleTransfer\\dataloader_test"
    
    imsize          = 512
    s_datasetloader= getLoader(s_datapath,c_datapath, 
                style_class, categories_names,
                crop_size=imsize, batch_size=16, num_workers=4)
    wocao           = iter(s_datasetloader)
    for i in range(500):
        print("new batch")
        s_image,c_image,label     = next(wocao)
        print(label)
        # print(label)
        # saved_image1 = torch.cat([denorm(image.data),denorm(hahh.data)],3)
        # save_image(denorm(image), "%s\\%d-label-%d.jpg"%(savepath,i), nrow=1, padding=1)
    pass
    # import cv2
    # import os
    # for dir_item in categories_names:
    #     join_path = Path(contentdatapath,dir_item)
    #     if join_path.exists():
    #         print("processing %s"%dir_item,end='\r')
    #         images = join_path.glob('*.%s'%("jpg"))
    #         for item in images:
    #             temp_path = str(item)
    #             # temp = cv2.imread(temp_path)
    #             temp = Image.open(temp_path)
    #             if temp.layers<3:
    #                 print("remove broken image...")
    #                 print("image name:%s"%temp_path)
    #                 del temp
    #                 os.remove(item)