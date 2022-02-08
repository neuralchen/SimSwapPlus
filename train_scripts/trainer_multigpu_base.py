#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_base.py
# Created Date: Sunday January 16th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 6th February 2022 3:06:45 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

class TrainerBase(object):

    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        
        

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

        # #==============build tensorboard=================#
        # if self.config["logger"] == "tensorboard":
        #     from utilities.utilities import build_tensorboard
        #     tensorboard_writer = build_tensorboard(self.config["project_summary"])
        #     self.logger = tensorboard_writer
        # elif self.config["logger"] == "wandb":
        #     import wandb
        #     wandb.init(project="Simswap_HQ", entity="xhchen", notes="512",
        #         tags=[self.config["tag"]], name=self.config["version"])
    
        #     wandb.config = {
        #         "total_step": self.config["total_step"],
        #         "batch_size": self.config["batch_size"]
        #         }
        #     self.logger = wandb
    
    # TODO modify this function to build your models
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        pass
    
    # TODO modify this function to configurate the optimizer of your pipeline
    def __setup_optimizers__(self):
        pass
    

    # TODO modify this function to evaluate your model
    # Evaluate the checkpoint
    def __evaluation__(self,
            step = 0,
            **kwargs
            ):
        pass


    def __create_dataloader__(self,
            config,
            cur_gpu
            ):
        # Data loader
        #============build train dataloader==============#
        # TODO to modify the key: "your_train_dataset" to get your train dataset path
        dataset   = config["dataset_paths"][config["dataset_name"]]
        #================================================#
        print("Prepare the train dataloader...")
        dlModulename    = config["dataloader"]
        package         = __import__("data_tools.data_loader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        dataloader_class= dataloaderClass
        dataloader      = dataloader_class(dataset,
                                        cur_gpu,
                                        config["batch_size"],
                                        **config["dataset_params"])
        
        return dataloader
                

    def train(self):
        #===============build framework================#
        self.init_framework()

        #===============build optimizer================#
        # Optimizer
        # TODO replace below lines to build your optimizer
        print("build the optimizer...")
        self.__setup_optimizers__()

        # set the start point for training loop
        if self.config["phase"] == "finetune":
            self.start = self.config["checkpoint_step"]
        else:
            self.start = 0

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        from utilities.logo_class import logo_class
        logo_class.print_start_training()