#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_condition_SN_multiscale.py
# Created Date: Saturday April 18th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th October 2021 2:18:26 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import  os
import  time

import  torch
from    torchvision.utils  import save_image

from    components.Transform import Transform_block
from    utilities.utilities import denorm, Gram, img2tensor255
from    pretrained_weights.vgg import VGG16

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
        package         = __import__("data_tools.data_loader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        self.dataloader_class = dataloaderClass
        dataloader      = self.dataloader_class(self.train_dataset,
                                        config["batch_size"],
                                        config["imcrop_size"],
                                        **config["dataset_params"])
        
        self.train_loader= dataloader

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

            # TODO To save the important scripts
            # save the yaml file
            import  shutil
            file1       = os.path.join("components", "%s.py"%model_config["g_model"]["script"])
            tgtfile1    = os.path.join(self.config["project_scripts"], "%s.py"%model_config["g_model"]["script"])
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
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.gen = self.gen.cuda()

        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["project_checkpoints"],
                                        "epoch%d_%s.pth"%(self.config["checkpoint_step"],
                                        self.config["checkpoint_names"]["generator_name"]))
            self.gen.load_state_dict(torch.load(model_path))
            
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
                

    def train(self):

        ckpt_dir    = self.config["project_checkpoints"]
        log_frep    = self.config["log_step"]
        model_freq  = self.config["model_save_epoch"]
        total_epoch = self.config["total_epoch"]
        batch_size  = self.config["batch_size"]
        style_img   = self.config["style_img_path"]
        
        # prep_weights= self.config["layersWeight"]
        content_w   = self.config["content_weight"]
        style_w     = self.config["style_weight"]
        crop_size   = self.config["imcrop_size"]

        sample_dir  = self.config["project_samples"]
        
        
        #===============build framework================#
        self.__init_framework__()

        #===============build optimizer================#
        # Optimizer
        # TODO replace below lines to build your optimizer
        print("build the optimizer...")
        self.__setup_optimizers__()
        
        #===============build losses===================#
        # TODO replace below lines to build your losses
        MSE_loss    = torch.nn.MSELoss()

            
        # set the start point for training loop
        if self.config["phase"] == "finetune":
            start = self.config["checkpoint_epoch"] - 1
        else:
            start = 0

        # print("prepare the fixed labels...")
        # fix_label   = [i for i in range(n_class)]
        # fix_label   = torch.tensor(fix_label).long().cuda()
        # fix_label       = fix_label.view(n_class,1)
        # fix_label       = torch.zeros(n_class, n_class).cuda().scatter_(1, fix_label, 1)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        from utilities.logo_class import logo_class
        logo_class.print_start_training()
        start_time = time.time()

        # Caculate the epoch number
        step_epoch  = len(self.train_loader)
        step_epoch  = step_epoch // batch_size
        print("Total step = %d in each epoch"%step_epoch)

        VGG = VGG16().cuda()

        MEAN_VAL = 127.5
        SCALE_VAL= 127.5
        # Get Style Features
        imagenet_neg_mean   = torch.tensor([-103.939, -116.779, -123.68], dtype=torch.float32).reshape(1,3,1,1).cuda()
        imagenet_neg_mean_11= torch.tensor([-103.939 + MEAN_VAL, -116.779 + MEAN_VAL, -123.68 + MEAN_VAL], dtype=torch.float32).reshape(1,3,1,1).cuda()

        style_tensor        = img2tensor255(style_img).cuda()
        style_tensor        = style_tensor.add(imagenet_neg_mean)
        B, C, H, W          = style_tensor.shape
        style_tensor        = VGG(style_tensor.expand([batch_size, C, H, W]))
        # style_features      = VGG(style_tensor)
        style_gram          = {}
        for key, value in style_tensor.items():
            style_gram[key] = Gram(value)
        del style_tensor
        # step_epoch = 2
        for epoch in range(start, total_epoch):
            for step in range(step_epoch):
                self.gen.train()

                content_images          = self.train_loader.next()
                fake_image              = self.gen(content_images)
                generated_features      = VGG((fake_image*SCALE_VAL).add(imagenet_neg_mean_11))
                content_features        = VGG((content_images*SCALE_VAL).add(imagenet_neg_mean_11))
                content_loss            = MSE_loss(generated_features['relu2_2'], content_features['relu2_2'])

                style_loss              = 0.0
                for key, value in generated_features.items():
                    s_loss = MSE_loss(Gram(value), style_gram[key])
                    style_loss += s_loss

                # backward & optimize
                g_loss = content_loss* content_w + style_loss* style_w
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                

                # Print out log info
                if (step + 1) % log_frep == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # cumulative steps
                    cum_step = (step_epoch * epoch + step + 1)
                    
                    epochinformation="[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], content_loss: {:.4f}, style_loss: {:.4f}, g_loss: {:.4f}".format(self.config["version"], elapsed, epoch + 1, total_epoch, step + 1, step_epoch, content_loss.item(), style_loss.item(), g_loss.item())
                    print(epochinformation)
                    self.reporter.writeInfo(epochinformation)
                    
                    if self.config["use_tensorboard"]:
                        self.tensorboard_writer.add_scalar('data/g_loss', g_loss.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/content_loss', content_loss.item(), cum_step)
                        self.tensorboard_writer.add_scalar('data/style_loss', style_loss, cum_step)
                
            #===============adjust learning rate============#
            # if (epoch + 1) in self.config["lr_decay_step"] and self.config["lr_decay_enable"]:
            #     print("Learning rate decay")
            #     for p in self.optimizer.param_groups:
            #         p['lr'] *= self.config["lr_decay"]
            #         print("Current learning rate is %f"%p['lr'])

            #===============save checkpoints================#
            if (epoch+1) % model_freq==0:
                print("Save epoch %d model checkpoint!"%(epoch+1))
                torch.save(self.gen.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, 
                                    self.config["checkpoint_names"]["generator_name"])))
                
                torch.cuda.empty_cache()
                print('Sample images {}_fake.jpg'.format(epoch + 1))
                self.gen.eval()
                with torch.no_grad():
                    sample = fake_image
                    saved_image1 = denorm(sample.cpu().data)
                    save_image(saved_image1,
                            os.path.join(sample_dir, '{}_fake.jpg'.format(epoch + 1)),nrow=4)