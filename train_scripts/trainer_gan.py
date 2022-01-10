#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_condition_SN_multiscale.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 6th July 2021 7:36:42 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import  os
import  time

import  torch
from    torchvision.utils  import save_image

from    components.Transform import Transform_block
from    utilities.utilities import denorm

class Trainer(object):

    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        
        # Data loader
        #============build train dataloader==============#
        # TODO to modify the key: "your_train_dataset" to get your train dataset path
        self.train_dataset   = config["dataset_paths"][config["dataset_name"]]
        #================================================#
        print("Prepare the train dataloader...")
        dlModulename    = config["dataloader"]
        package         = __import__("data_tools.dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        self.dataloader_class = dataloaderClass
        # dataloader      = self.dataloader_class(self.train_dataset,
        #                                 config["batch_size_list"][0],
        #                                 config["imcrop_size_list"][0],
        #                                 **config["dataset_params"])
        
        # self.train_loader= dataloader

        #========build evaluation dataloader=============#
        # TODO to modify the key: "your_eval_dataset" to get your evaluation dataset path
        # eval_dataset = config["dataset_paths"][config["eval_dataset_name"]]

        # #================================================#
        # print("Prepare the evaluation dataloader...")
        # dlModulename    = config["eval_dataloader"]
        # package         = __import__("data_tools.eval_dataloader_%s"%dlModulename, fromlist=True)
        # dataloaderClass = getattr(package, 'EvalDataset')
        # dataloader      = dataloaderClass(eval_dataset,
        #                                 config["eval_batch_size"])
        # self.eval_loader= dataloader

        # self.eval_iter  = len(dataloader)//config["eval_batch_size"]
        # if len(dataloader)%config["eval_batch_size"]>0:
        #     self.eval_iter+=1

        #==============build tensorboard=================#
        if self.config["use_tensorboard"]:
            from utilities.utilities import build_tensorboard
            self.tensorboard_writer = build_tensorboard(self.config["project_summary"])
    
    # TODO modify this function to build your models
    def __init_framework__(self):
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

        class_name      = model_config["d_model"]["class_name"]
        package         = __import__(dscript_name, fromlist=True)
        dis_class       = getattr(package, class_name)
        self.dis        = dis_class(**model_config["d_model"]["module_params"])
        
        # print and recorde model structure
        self.reporter.writeInfo("Generator structure:")
        self.reporter.writeModel(self.gen.__str__())
        self.reporter.writeInfo("Discriminator structure:")
        self.reporter.writeModel(self.dis.__str__())
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
            self.gen.load_state_dict(torch.load(model_path))

            model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["discriminator_name"]))
            self.dis.load_state_dict(torch.load(model_path))
            
            print('loaded trained backbone model epoch {}...!'.format(self.config["project_checkpoints"]))
    

    # TODO modify this function to evaluate your model
    def __evaluation__(self, epoch, step = 0):
        # Evaluate the checkpoint
        self.network.eval()
        total_psnr = 0
        total_num  = 0
        with torch.no_grad():
            for _ in range(self.eval_iter):
                hr, lr = self.eval_loader()
                
                if self.config["cuda"] >=0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                hr     = (hr + 1.0)/2.0 * 255.0
                hr     = torch.clamp(hr,0.0,255.0)
                lr     = (lr + 1.0)/2.0 * 255.0
                lr     = torch.clamp(lr,0.0,255.0)
                res    = self.network(lr)
                # res     = (res + 1.0)/2.0 * 255.0
                # hr      = (hr + 1.0)/2.0 * 255.0
                res     = torch.clamp(res,0.0,255.0)
                diff    = (res-hr) ** 2
                diff    = diff.mean(dim=-1).mean(dim=-1).mean(dim=-1).sqrt()
                psnrs   = 20. * (255. / diff).log10()
                total_psnr+= psnrs.sum()
                total_num+=res.shape[0]
        final_psnr = total_psnr/total_num
        print("[{}], Epoch [{}], psnr: {:.4f}".format(self.config["version"],
                                                    epoch, final_psnr))
        self.reporter.writeTrainLog(epoch,step,"psnr: {:.4f}".format(final_psnr))
        self.tensorboard_writer.add_scalar('metric/loss', final_psnr, epoch)

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
                

    def train(self):

        ckpt_dir    = self.config["project_checkpoints"]
        log_frep    = self.config["log_step"]
        model_freq  = self.config["model_save_epoch"]
        total_epoch = self.config["total_epoch"]
        
        n_class     = len(self.config["selected_style_dir"])
        # prep_weights= self.config["layersWeight"]
        feature_w       = self.config["feature_weight"]
        transform_w     = self.config["transform_weight"]
        d_step          = self.config["d_step"]
        g_step          = self.config["g_step"]

        batch_size_list     = self.config["batch_size_list"] 
        switch_epoch_list   = self.config["switch_epoch_list"]
        imcrop_size_list    = self.config["imcrop_size_list"]
        sample_dir          = self.config["project_samples"]
        
        current_epoch_index = 0
        
        #===============build framework================#
        self.__init_framework__()

        #===============build optimizer================#
        # Optimizer
        # TODO replace below lines to build your optimizer
        print("build the optimizer...")
        self.__setup_optimizers__()
        
        #===============build losses===================#
        # TODO replace below lines to build your losses
        Transform   = Transform_block().cuda()
        L1_loss     = torch.nn.L1Loss()
        MSE_loss    = torch.nn.MSELoss()
        Hinge_loss  = torch.nn.ReLU().cuda()

            
        # set the start point for training loop
        if self.config["phase"] == "finetune":
            start = self.config["checkpoint_epoch"] - 1
        else:
            start = 0


        output_size = self.dis.get_outputs_len()

        print("prepare the fixed labels...")
        fix_label   = [i for i in range(n_class)]
        fix_label   = torch.tensor(fix_label).long().cuda()
        # fix_label       = fix_label.view(n_class,1)
        # fix_label       = torch.zeros(n_class, n_class).cuda().scatter_(1, fix_label, 1)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        from utilities.logo_class import logo_class
        logo_class.print_start_training()
        start_time = time.time()

        for epoch in range(start, total_epoch):

            # switch training image size
            if epoch in switch_epoch_list:
                print('Current epoch: {}'.format(epoch))
                print('***Redefining the dataloader for progressive training.***')
                print('***Current spatial size is {} and batch size is {}.***'.format(
                                imcrop_size_list[current_epoch_index], batch_size_list[current_epoch_index]))
                del self.train_loader
                self.train_loader = self.dataloader_class(self.train_dataset,
                                        batch_size_list[current_epoch_index],
                                        imcrop_size_list[current_epoch_index],
                                        **self.config["dataset_params"])
                # Caculate the epoch number
                step_epoch  = len(self.train_loader)
                step_epoch  = step_epoch // (d_step + g_step)
                print("Total step = %d in each epoch"%step_epoch)
                current_epoch_index += 1

            for step in range(step_epoch):
                self.dis.train()
                self.gen.train()
                
                # ================== Train D ================== #
                # Compute loss with real images
                for _ in range(d_step):
                    content_images,style_images,label  = self.train_loader.next()
                    label           = label.long()
                    
                    d_out = self.dis(style_images,label)
                    d_loss_real = 0
                    for i in range(output_size):
                        temp = Hinge_loss(1 - d_out[i]).mean()
                        d_loss_real += temp

                    d_loss_photo = 0
                    d_out = self.dis(content_images,label)
                    for i in range(output_size):
                        temp = Hinge_loss(1 + d_out[i]).mean()
                        d_loss_photo += temp
                        
                    fake_image,_= self.gen(content_images,label)
                    d_out       = self.dis(fake_image.detach(),label)
                    d_loss_fake = 0
                    for i in range(output_size):
                        temp = Hinge_loss(1 + d_out[i]).mean()
                        # temp *= prep_weights[i]
                        d_loss_fake += temp

                    # Backward + Optimize
                    d_loss = d_loss_real + d_loss_photo + d_loss_fake
                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                
                # ================== Train G ================== #
                for _ in range(g_step):

                    content_images,_,_      = self.train_loader.next()
                    fake_image,real_feature = self.gen(content_images,label)
                    fake_feature            = self.gen(fake_image, get_feature=True)
                    d_out                   = self.dis(fake_image,label.long())
                    
                    g_feature_loss          = L1_loss(fake_feature,real_feature)
                    g_transform_loss        = MSE_loss(Transform(content_images), Transform(fake_image))
                    g_loss_fake             = 0
                    for i in range(output_size):
                        temp = -d_out[i].mean()
                        # temp *= prep_weights[i]
                        g_loss_fake += temp

                    # backward & optimize
                    g_loss = g_loss_fake + g_feature_loss* feature_w + g_transform_loss* transform_w
                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                

                # Print out log info
                if (step + 1) % log_frep == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # cumulative steps
                    cum_step = (step_epoch * epoch + step + 1)
                    
                    epochinformation="[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, d_loss_real: {:.4f}, \\\
                            d_loss_photo: {:.4f}, d_loss_fake: {:.4f}, g_loss: {:.4f}, g_loss_fake: {:.4f}, \\\
                            g_feature_loss: {:.4f}, g_transform_loss: {:.4f}".format(self.config["version"], 
                                epoch + 1, total_epoch, elapsed, step + 1, step_epoch, 
                                d_loss.item(), d_loss_real.item(), d_loss_photo.item(), 
                                d_loss_fake.item(), g_loss.item(), g_loss_fake.item(),\
                                    g_feature_loss.item(), g_transform_loss.item())
                    print(epochinformation)
                    self.reporter.writeRawInfo(epochinformation)
                    
                    if self.config["use_tensorboard"]:
                        self.tensorboard_writer.add_scalar('data/d_loss', d_loss.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/d_loss_real', d_loss_real.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/d_loss_photo', d_loss_photo.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/d_loss_fake', d_loss_fake.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/g_loss', g_loss.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/g_loss_fake', g_loss_fake.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/g_feature_loss', g_feature_loss, cum_step)
                        self.tensorboard_writer.add_scalar('data/g_transform_loss', g_transform_loss, cum_step)
                
            #===============adjust learning rate============#
            if (epoch + 1) in self.config["lr_decay_step"] and self.config["lr_decay_enable"]:
                print("Learning rate decay")
                for p in self.optimizer.param_groups:
                    p['lr'] *= self.config["lr_decay"]
                    print("Current learning rate is %f"%p['lr'])

            #===============save checkpoints================#
            if (epoch+1) % model_freq==0:
                print("Save epoch %d model checkpoint!"%(epoch+1))
                torch.save(self.gen.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))
                torch.save(self.dis.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, 
                                    self.config["checkpoint_names"]["discriminator_name"])))
                
                torch.cuda.empty_cache()
                print('Sample images {}_fake.jpg'.format(step + 1))
                self.gen.eval()
                with torch.no_grad():
                    sample = content_images[0, :, :, :].unsqueeze(0)
                    saved_image1 = denorm(sample.cpu().data)
                    for index in range(n_class):
                        fake_images,_ = self.gen(sample, fix_label[index].unsqueeze(0))
                        saved_image1 = torch.cat((saved_image1, denorm(fake_images.cpu().data)), 0)
                    save_image(saved_image1,
                            os.path.join(sample_dir, '{}_fake.jpg'.format(step + 1)),nrow=3)