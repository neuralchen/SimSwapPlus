#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_naiv512.py
# Created Date: Sunday January 9th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 29th January 2022 3:54:06 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import  os
import  time
import  random
import  shutil
from cv2 import sqrt

import  numpy as np

import  torch
import  torch.nn.functional as F
from    torchvision.utils  import save_image

from    train_scripts.trainer_base import TrainerBase

class Trainer(TrainerBase):

    def __init__(self, 
                config, 
                reporter):
        super(Trainer, self).__init__(config, reporter)

        import inspect
        print("Current training script -----------> %s"%inspect.getfile(inspect.currentframe()))
        
        self.img_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.img_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    
    # TODO modify this function to build your models
    def init_framework(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]

        model_config    = self.config["model_configs"]
        
        if self.config["phase"] == "train":
            gscript_name     = "components." + model_config["g_model"]["script"]

            file1       = os.path.join("components", model_config["g_model"]["script"]+".py")
            tgtfile1    = os.path.join(self.config["project_scripts"], model_config["g_model"]["script"]+".py")
            shutil.copyfile(file1,tgtfile1)
            
        elif self.config["phase"] == "finetune":
            gscript_name     = self.config["com_base"] + model_config["g_model"]["script"]
        
        class_name      = model_config["g_model"]["class_name"]
        package         = __import__(gscript_name, fromlist=True)
        gen_class       = getattr(package, class_name)
        self.gen        = gen_class(**model_config["g_model"]["module_params"])
        
        # print and recorde model structure
        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(self.gen.__str__())

        
        # print and recorde model structure
        arcface1        = torch.load(self.arcface_ckpt, map_location=torch.device("cpu"))
        self.arcface    = arcface1['model'].module
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.gen    = self.gen.cuda()
            self.arcface= self.arcface.cuda()
        
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
            self.gen.load_state_dict(torch.load(model_path))

            
            print('loaded trained backbone model step {}...!'.format(self.config["project_checkpoints"]))
    
    # TODO modify this function to configurate the optimizer of your pipeline
    def __setup_optimizers__(self):
        g_train_opt     = self.config['g_optim_config']

        g_optim_params  = []
        for k, v in self.gen.named_parameters():
            if v.requires_grad:
                g_optim_params.append(v)
            else:
                self.reporter.writeInfo(f'Params {k} will not be optimized.')
                print(f'Params {k} will not be optimized.')

        optim_type = self.config['optim_type']
        
        if optim_type == 'Adam':
            self.g_optimizer = torch.optim.Adam(g_optim_params,**g_train_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        # self.optimizers.append(self.optimizer_g)
        if self.config["phase"] == "finetune":
            opt_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_optim_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["optimizer_names"]["generator_name"]))
            self.g_optimizer.load_state_dict(torch.load(opt_path))

            
            print('loaded trained optimizer step {}...!'.format(self.config["project_checkpoints"]))
    

    # TODO modify this function to evaluate your model
    # Evaluate the checkpoint
    def __evaluation__(self,
            step = 0,
            **kwargs
            ):
        src_image1  = kwargs["src1"]
        self.gen.eval()
        with torch.no_grad():
            id_vector_src1  = self.arcface(src_image1)
            img_fake        = self.gen(id_vector_src1).cpu()
            img_fake        = img_fake * self.img_std
            img_fake        = img_fake + self.img_mean
            img_fake        = img_fake.clamp_(0, 1)
            print("Save test data")
            save_image(img_fake,
                os.path.join(self.sample_dir, 'step_'+str(step+1)+'.jpg'),
                    nrow=8)

    
                

    def train(self):

        ckpt_dir    = self.config["project_checkpoints"]
        log_freq    = self.config["log_step"]
        model_freq  = self.config["model_save_step"]
        sample_freq = self.config["sample_step"]
        total_step  = self.config["total_step"]
        random_seed = self.config["dataset_params"]["random_seed"]

        self.batch_size  = self.config["batch_size"]
        self.sample_dir  = self.config["project_samples"]
        self.arcface_ckpt= self.config["arcface_ckpt"]
        
        
        super().train()
        
        #===============build losses===================#
        # TODO replace below lines to build your losses
        # MSE_loss    = torch.nn.MSELoss()
        l1_loss     = torch.nn.L1Loss()

            
        start_time  = time.time()

        # Caculate the epoch number
        print("Total step = %d"%total_step)
        random.seed(random_seed)
        randindex = [i for i in range(self.batch_size)]
        random.shuffle(randindex)
        import datetime
        for step in range(self.start, total_step):
            self.gen.train()
            src_image1      = self.train_loader.next()
            
            latent_id       = self.arcface(src_image1)
            img_fake        = self.gen(latent_id.detach())
            loss            = l1_loss(img_fake, src_image1)

            self.g_optimizer.zero_grad()
            loss.backward()
            self.g_optimizer.step()
            
            # Print out log info
            if (step + 1) % log_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                
                epochinformation="[{}], Elapsed [{}], Step [{}/{}], Reconstruction: {:.4f}". \
                        format(self.config["version"], elapsed, step, total_step, loss.item())
                print(epochinformation)
                self.reporter.writeInfo(epochinformation)

                if self.config["logger"] == "tensorboard":
                    self.logger.add_scalar('Rec_loss', loss.item(), step)
                elif self.config["logger"] == "wandb":
                    self.logger.log({"Rec_loss": loss.item()}, step = step)
            
            if (step + 1) % sample_freq == 0:
                self.__evaluation__(
                    step = step,
                    **{
                    "src1": src_image1
                })
                    
                        
                
            #===============adjust learning rate============#
            # if (epoch + 1) in self.config["lr_decay_step"] and self.config["lr_decay_enable"]:
            #     print("Learning rate decay")
            #     for p in self.optimizer.param_groups:
            #         p['lr'] *= self.config["lr_decay"]
            #         print("Current learning rate is %f"%p['lr'])

            #===============save checkpoints================#
            if (step+1) % model_freq==0:
                
                torch.save(self.gen.state_dict(),
                        os.path.join(ckpt_dir, 'step{}_{}.pth'.format(step + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))

                torch.save(self.g_optimizer.state_dict(),
                        os.path.join(ckpt_dir, 'step{}_optim_{}'.format(step + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))

                print("Save step %d model checkpoint!"%(step+1))
                torch.cuda.empty_cache()

                self.__evaluation__(
                    step = step,
                    **{
                    "src1": src_image1
                })