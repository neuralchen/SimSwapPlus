#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: tester_ID_Pose.py
# Created Date: Friday March 4th 2022
# Author: Liu Naiyuan
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 4th March 2022 5:33:47 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import  os
import  cv2
import  time
import  glob
from    tqdm import tqdm

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
from    torchvision import transforms
from    torch.utils import data

import  numpy as np

import  PIL
from    PIL import Image





class TotalDataset(data.Dataset):
    """Dataset class for the vggface dataset with precalulated face landmarks."""

    def __init__(self,image_dir,content_transform, img_size=224):
        self.image_dir= image_dir
        self.content_transform= content_transform
        self.img_size = img_size
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

def getLoader_sourceface(c_image_dir, 
                img_size=224, batch_size=16, num_workers=8):
    """Build and return a data loader."""
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    c_transforms = T.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir, c_transforms, 224)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=False,shuffle=False,num_workers=num_workers,pin_memory=True)
    return content_data_loader, len(content_dataset)


def getLoader_targetface(c_image_dir, 
                img_size=224, batch_size=16, num_workers=8):
    """Build and return a data loader."""
    c_transforms = []
    
    c_transforms.append(transforms.ToTensor())
    # c_transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    c_transforms = transforms.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir, c_transforms, 224)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=False,shuffle=False,num_workers=num_workers,pin_memory=True)
    return content_data_loader, len(content_dataset)

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
        ckp_step    = self.config["checkpoint_step"]
        version     = self.config["version"]
        id_imgs     = self.config["id_imgs"]
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

        source_loader, dataet_len = getLoader_sourceface(
                self.config["env_config"]["dataset_paths"]["id_pose_source_root"], batch_size=opt.batchSize)
        target_loader, dataet_len = getLoader_targetface(
                self.config["env_config"]["dataset_paths"]["id_pose_source_root"], batch_size=opt.batchSize)
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
                            
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
                profile_latent_id       = model.netArc(profile_id_downsample)
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
                img_fakes = model(None, target_batch.cuda(), batch_id, None, True)

                for img_fake, target_index_tmp,filename_tmp in zip(img_fakes, target_index_list,filename_batch):
                    filename_tmp_split = filename_tmp.split('_')
                    final_filename = filename_tmp_split[0] + '_' +str(target_index_tmp) + '_' + filename_tmp_split[-1]
                    save_path = os.path.join(simswap_eval_save_image_path,final_filename)
                    save_image = postprocess(img_fake.cpu().numpy().transpose(1,2,0))
                    PIL.Image.fromarray(save_image).save(save_path,quality=95)

            for img in imgs_list:
                print(img)
                attr_img_ori= cv2.imread(img)
                attr_img_align_crop_pil = Image.fromarray(cv2.cvtColor(attr_img_align_crop[0],cv2.COLOR_BGR2RGB))
                attr_img    = self.transformer_Arcface(attr_img_align_crop_pil).unsqueeze(0).cuda()

                attr_img_arc = F.interpolate(attr_img,size=(112,112), mode='bicubic')
                # cv2.imwrite(os.path.join("./swap_results", "id_%s.png"%(id_basename)),id_img_align_crop[0])
                attr_id   = self.arcface(attr_img_arc)
                attr_id   = F.normalize(attr_id, p=2, dim=1)

                results     = self.network(attr_img, latend_id)


                results     = results * self.imagenet_std + self.imagenet_mean
                results     = results.cpu().permute(0,2,3,1)[0,...]
                results     = results.numpy()
                results     = np.clip(results,0.0,1.0)
                final_img = img1.astype(np.uint8)
                attr_basename = os.path.splitext(os.path.basename(img))[0]
                final_img = cv2.putText(final_img, 'id dis=%.4f'%results_cos_dis, (50, 50), font, 0.8, (15, 9, 255), 2)
                final_img = cv2.putText(final_img, 'id--attr dis=%.4f'%cos_dis, (50, 80), font, 0.8, (15, 9, 255), 2)
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




if __name__ == '__main__':
    opt = TestOptions().parse()

    with torch.no_grad():
    
        source_loader, dataet_len = getLoader_sourceface('/home/gdp/harddisk/Data2/Faceswap/FaceForensics++_image_hififacestyle_source_Nonearcstyle', batch_size=opt.batchSize)
        target_loader, dataet_len = getLoader_targetface('/home/gdp/harddisk/Data2/Faceswap/FaceForensics++_image_target_even10_pro_withmat_Nonearcstyle_256', batch_size=opt.batchSize)

        simswap_eval_save_image_path = opt.output_path
        criterion = nn.L1Loss()
        if not os.path.exists(simswap_eval_save_image_path):
            os.makedirs(simswap_eval_save_image_path)
        torch.nn.Module.dump_patches = True
        model = create_model(opt)
        model.eval()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        init_batch = True
        for profile_batch, filename_batch in tqdm(source_iter):
            # src_batch, filename_batch = data_iter.next()
            profile_batch = profile_batch.cuda()
            profile_id_downsample = F.interpolate(profile_batch, (112,112))
            profile_latent_id = model.netArc(profile_id_downsample)
            profile_latent_id = F.normalize(profile_latent_id, p=2, dim=1)
            if init_batch ==True:
                wholeid_batch = profile_latent_id.cpu()
                init_batch = False
            else:
                wholeid_batch = torch.cat([wholeid_batch,profile_latent_id.cpu()],dim=0)
        print(wholeid_batch.shape)
        # np.save("simswap_wholeid_batch.npy", wholeid_batch.detach().cpu().numpy())

        target_source_pair_dict = np.load('/home/gdp/harddisk/Data2/Faceswap/npy_file/target_source_pair.npy' ,allow_pickle=True).item()

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
            img_fakes = model(None, target_batch.cuda(), batch_id, None, True)

            for img_fake, target_index_tmp,filename_tmp in zip(img_fakes, target_index_list,filename_batch):
                filename_tmp_split = filename_tmp.split('_')
                final_filename = filename_tmp_split[0] + '_' +str(target_index_tmp) + '_' + filename_tmp_split[-1]
                save_path = os.path.join(simswap_eval_save_image_path,final_filename)
                save_image = postprocess(img_fake.cpu().numpy().transpose(1,2,0))
                PIL.Image.fromarray(save_image).save(save_path,quality=95)