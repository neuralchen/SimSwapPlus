#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_naiv512.py
# Created Date: Sunday January 9th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 22nd January 2022 12:45:09 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import  os
import  time
import  random

import  numpy as np

import  torch
import  torch.nn.functional as F
from    utilities.plot import plot_batch

from    train_scripts.trainer_base import TrainerBase

class Trainer(TrainerBase):

    def __init__(self, 
                config, 
                reporter):
        super(Trainer, self).__init__(config, reporter)
        
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
            dscript_name     = "components." + model_config["d_model"]["script"]
            
        elif self.config["phase"] == "finetune":
            gscript_name     = self.config["com_base"] + model_config["g_model"]["script"]
            dscript_name     = self.config["com_base"] + model_config["d_model"]["script"]
        
        class_name      = model_config["g_model"]["class_name"]
        package         = __import__(gscript_name, fromlist=True)
        gen_class       = getattr(package, class_name)
        self.gen        = gen_class(**model_config["g_model"]["module_params"])
        
        # print and recorde model structure
        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(self.gen.__str__())

        class_name      = model_config["d_model"]["class_name"]
        package         = __import__(dscript_name, fromlist=True)
        dis_class       = getattr(package, class_name)
        self.dis        = dis_class(**model_config["d_model"]["module_params"])
        self.dis.feature_network.requires_grad_(False)
        
        # print and recorde model structure
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(self.dis.__str__())
        arcface1        = torch.load(self.arcface_ckpt, map_location=torch.device("cpu"))
        self.arcface    = arcface1['model'].module
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.gen    = self.gen.cuda()
            self.dis    = self.dis.cuda()
            self.arcface= self.arcface.cuda()
        
        self.arcface.eval()
        self.arcface.requires_grad_(False)

        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
            self.gen.load_state_dict(torch.load(model_path))

            model_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["discriminator_name"]))
            self.dis.load_state_dict(torch.load(model_path))
            
            print('loaded trained backbone model step {}...!'.format(self.config["project_checkpoints"]))
    
    # TODO modify this function to configurate the optimizer of your pipeline
    def __setup_optimizers__(self):
        g_train_opt     = self.config['g_optim_config']
        d_train_opt     = self.config['d_optim_config']

        g_optim_params  = []
        d_optim_params  = []
        for k, v in self.gen.named_parameters():
            if v.requires_grad:
                g_optim_params.append(v)
            else:
                self.reporter.writeInfo(f'Params {k} will not be optimized.')
                print(f'Params {k} will not be optimized.')
        
        for k, v in self.dis.named_parameters():
            if v.requires_grad:
                d_optim_params.append(v)
            else:
                self.reporter.writeInfo(f'Params {k} will not be optimized.')
                print(f'Params {k} will not be optimized.')

        optim_type = self.config['optim_type']
        
        if optim_type == 'Adam':
            self.g_optimizer = torch.optim.Adam(g_optim_params,**g_train_opt)
            self.d_optimizer = torch.optim.Adam(d_optim_params,**d_train_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        # self.optimizers.append(self.optimizer_g)
        if self.config["phase"] == "finetune":
            opt_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_optim_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["optimizer_names"]["generator_name"]))
            self.g_optimizer.load_state_dict(torch.load(opt_path))

            opt_path = os.path.join(self.config["project_checkpoints"],
                                        "step%d_optim_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["optimizer_names"]["discriminator_name"]))
            self.d_optimizer.load_state_dict(torch.load(opt_path))
            
            print('loaded trained optimizer step {}...!'.format(self.config["project_checkpoints"]))
    

    # TODO modify this function to evaluate your model
    # Evaluate the checkpoint
    def __evaluation__(self,
            step = 0,
            **kwargs
            ):
        src_image1  = kwargs["src1"]
        src_image2  = kwargs["src2"]
        batch_size  = self.batch_size
        self.gen.eval()
        with torch.no_grad():
            imgs        = []
            zero_img    = (torch.zeros_like(src_image1[0,...]))
            imgs.append(zero_img.cpu().numpy())
            save_img    = ((src_image1.cpu())* self.img_std + self.img_mean).numpy()
            for r in range(batch_size):
                imgs.append(save_img[r,...])
            arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
            id_vector_src1  = self.arcface(arcface_112)
            id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)

            for i in range(batch_size):
                
                imgs.append(save_img[i,...])
                image_infer = src_image1[i, ...].repeat(batch_size, 1, 1, 1)
                img_fake    = self.gen(image_infer, id_vector_src1).cpu()
                
                img_fake    = img_fake * self.img_std
                img_fake    = img_fake + self.img_mean
                img_fake    = img_fake.numpy()
                for j in range(batch_size):
                    imgs.append(img_fake[j,...])
            print("Save test data")
            imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
            plot_batch(imgs, os.path.join(self.sample_dir, 'step_'+str(step+1)+'.jpg'))

    
                

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
        
        
        # prep_weights= self.config["layersWeight"]
        id_w        = self.config["id_weight"]
        rec_w       = self.config["reconstruct_weight"]
        feat_w      = self.config["feature_match_weight"]
        
        
        
        super().train()
        
        #===============build losses===================#
        # TODO replace below lines to build your losses
        # MSE_loss    = torch.nn.MSELoss()
        l1_loss     = torch.nn.L1Loss()
        cos_loss    = torch.nn.CosineSimilarity()

            
        start_time  = time.time()

        # Caculate the epoch number
        print("Total step = %d"%total_step)
        random.seed(random_seed)
        randindex = [i for i in range(self.batch_size)]
        random.shuffle(randindex)
        import datetime
        for step in range(self.start, total_step):
            self.gen.train()
            self.dis.train()
            for interval in range(2):
                random.shuffle(randindex)
                src_image1, src_image2  = self.train_loader.next()
                
                if step%2 == 0:
                    img_id = src_image2
                else:
                    img_id = src_image2[randindex]

                img_id_112      = F.interpolate(img_id,size=(112,112), mode='bicubic')
                latent_id       = self.arcface(img_id_112)
                latent_id       = F.normalize(latent_id, p=2, dim=1)
                if interval:
                    
                    img_fake        = self.gen(src_image1, latent_id)
                    gen_logits,_    = self.dis(img_fake.detach(), None)
                    loss_Dgen       = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean()

                    real_logits,_   = self.dis(src_image2,None)
                    loss_Dreal      = (F.relu(torch.ones_like(real_logits) - real_logits)).mean()

                    loss_D          = loss_Dgen + loss_Dreal
                    self.d_optimizer.zero_grad()
                    loss_D.backward()
                    self.d_optimizer.step()
                else:
                    
                    # model.netD.requires_grad_(True)
                    img_fake        = self.gen(src_image1, latent_id)
                    # G loss
                    gen_logits,feat = self.dis(img_fake, None)
                    
                    loss_Gmain      = (-gen_logits).mean()
                    img_fake_down   = F.interpolate(img_fake, size=(112,112), mode='bicubic')
                    latent_fake     = self.arcface(img_fake_down)
                    latent_fake     = F.normalize(latent_fake, p=2, dim=1)
                    loss_G_ID       = (1 - cos_loss(latent_fake, latent_id)).mean()
                    real_feat       = self.dis.get_feature(src_image1)
                    feat_match_loss = l1_loss(feat["3"],real_feat["3"])
                    loss_G          = loss_Gmain + loss_G_ID * id_w + \
                                                feat_match_loss * feat_w
                    if step%2 == 0:
                        #G_Rec
                        loss_G_Rec  = l1_loss(img_fake, src_image1)
                        loss_G      += loss_G_Rec * rec_w

                    self.g_optimizer.zero_grad()
                    loss_G.backward()
                    self.g_optimizer.step()
                
            # Print out log info
            if (step + 1) % log_freq == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                
                epochinformation="[{}], Elapsed [{}], Step [{}/{}], \
                        G_loss: {:.4f}, Rec_loss: {:.4f}, Fm_loss: {:.4f}, \
                            D_loss: {:.4f}, D_fake: {:.4f}, D_real: {:.4f}". \
                            format(self.config["version"], elapsed, step, total_step, \
                                loss_G.item(), loss_G_Rec.item(), feat_match_loss.item(), \
                                    loss_D.item(), loss_Dgen.item(), loss_Dreal.item())
                print(epochinformation)
                self.reporter.writeInfo(epochinformation)

                if self.config["logger"] == "tensorboard":
                    self.logger.add_scalar('G/G_loss', loss_G.item(), step)
                    self.logger.add_scalar('G/Rec_loss', loss_G_Rec.item(), step)
                    self.logger.add_scalar('G/Fm_loss', feat_match_loss.item(), step)
                    self.logger.add_scalar('D/D_loss', loss_D.item(), step)
                    self.logger.add_scalar('D/D_fake', loss_Dgen.item(), step)
                    self.logger.add_scalar('D/D_real', loss_Dreal.item(), step)
                elif self.config["logger"] == "wandb":
                    self.logger.log({"G_loss": loss_G.item()}, step = step)
                    self.logger.log({"Rec_loss": loss_G_Rec.item()}, step = step)
                    self.logger.log({"Fm_loss": feat_match_loss.item()}, step = step)
                    self.logger.log({"D_loss": loss_D.item()}, step = step)
                    self.logger.log({"D_fake": loss_Dgen.item()}, step = step)
                    self.logger.log({"D_real": loss_Dreal.item()}, step = step)
            
            if (step + 1) % sample_freq == 0:
                self.__evaluation__(
                    step = step,
                    **{
                    "src1": src_image1,
                    "src2": src_image2
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
                torch.save(self.dis.state_dict(),
                        os.path.join(ckpt_dir, 'step{}_{}.pth'.format(step + 1, 
                                    self.config["checkpoint_names"]["discriminator_name"])))
                
                torch.save(self.g_optimizer.state_dict(),
                        os.path.join(ckpt_dir, 'step{}_optim_{}'.format(step + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))
                
                torch.save(self.d_optimizer.state_dict(),
                        os.path.join(ckpt_dir, 'step{}_optim_{}'.format(step + 1, 
                                    self.config["checkpoint_names"]["discriminator_name"])))
                print("Save step %d model checkpoint!"%(step+1))
                torch.cuda.empty_cache()

                self.__evaluation__(
                    step = step,
                    **{
                    "src1": src_image1,
                    "src2": src_image2
                })