#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: train.py
# Created Date: Tuesday April 28th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 23rd February 2022 2:30:03 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


from curses.panel import version
import  os
import  shutil
import  argparse 
from    torch.backends import cudnn
from    utilities.json_config import readConfig, writeConfig
from    utilities.reporter import Reporter
from    utilities.yaml_config import getConfigYaml



def str2bool(v):
    return v.lower() in ('true')

####################################################################################
# To configure the seting of training\finetune\test
#
####################################################################################
def getParameters():
    
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('-v', '--version', type=str, default='distillation',
                                            help="version name for train, test, finetune")
    parser.add_argument('-t', '--tag', type=str, default='distillation',
                                            help="tag for current experiment")

    parser.add_argument('-p', '--phase', type=str, default="train",
                                            choices=['train', 'finetune','debug'],
                                                help="The phase of current project")

    parser.add_argument('-c', '--gpus', type=int, nargs='+', default=[0,1]) # <0 if it is set as -1, program will use CPU
    parser.add_argument('-e', '--ckpt', type=int, default=74,
                                help="checkpoint epoch for test phase or finetune phase")

    # training
    parser.add_argument('--experiment_description', type=str,
                                default="测试蒸馏代码")

    parser.add_argument('--train_yaml', type=str, default="train_distillation.yaml")

    # system logger
    parser.add_argument('--logger', type=str,
                  default="none", choices=['tensorboard', 'wandb','none'], help='system logger')

    # # logs (does not to be changed in most time)
    # parser.add_argument('--dataloader_workers', type=int, default=6)
    # parser.add_argument('--use_tensorboard', type=str2bool, default='True',
    #                         choices=['True', 'False'], help='enable the tensorboard')
    # parser.add_argument('--log_step', type=int, default=100)
    # parser.add_argument('--sample_step', type=int, default=100)
    
    # # template (onece editing finished, it should be deleted)
    # parser.add_argument('--str_parameter', type=str, default="default", help='str parameter')
    # parser.add_argument('--str_parameter_choices', type=str,
    #               default="default", choices=['choice1', 'choice2','choice3'], help='str parameter with choices list')
    # parser.add_argument('--int_parameter', type=int, default=0, help='int parameter')
    # parser.add_argument('--float_parameter', type=float, default=0.0, help='float parameter')
    # parser.add_argument('--bool_parameter', type=str2bool, default='True', choices=['True', 'False'], help='bool parameter')
    # parser.add_argument('--list_str_parameter', type=str, nargs='+', default=["element1","element2"], help='str list parameter')
    # parser.add_argument('--list_int_parameter', type=int, nargs='+', default=[0,1], help='int list parameter')
    return parser.parse_args()

ignoreKey = [
        "dataloader_workers",
        "log_root_path",
        "project_root",
        "project_summary",
        "project_checkpoints",
        "project_samples",
        "project_scripts",
        "reporter_path",
        "use_specified_data",
        "specified_data_paths",
        "dataset_path","cuda", 
        "test_script_name",
        "test_dataloader",
        "test_dataset_path",
        "save_test_result",
        "test_batch_size",
        "node_name",
        "checkpoint_epoch",
        "test_dataset_path",
        "test_dataset_name",
        "use_my_test_date"]

####################################################################################
# This function will create the related directories before the 
# training\fintune\test starts
# Your_log_root (version name)
#   |---summary/...
#   |---samples/... (save evaluated images)
#   |---checkpoints/...
#   |---scripts/...
#
####################################################################################
def createDirs(sys_state):
    # the base dir
    if not os.path.exists(sys_state["log_root_path"]):
        os.makedirs(sys_state["log_root_path"])

    # create dirs
    sys_state["project_root"]        = os.path.join(sys_state["log_root_path"],
                                            sys_state["version"])
                                            
    project_root                     = sys_state["project_root"]
    if not os.path.exists(project_root):
        os.makedirs(project_root)
    
    sys_state["project_summary"]     = os.path.join(project_root, "summary")
    if not os.path.exists(sys_state["project_summary"]):
        os.makedirs(sys_state["project_summary"])

    sys_state["project_checkpoints"] = os.path.join(project_root, "checkpoints")
    if not os.path.exists(sys_state["project_checkpoints"]):
        os.makedirs(sys_state["project_checkpoints"])

    sys_state["project_samples"]     = os.path.join(project_root, "samples")
    if not os.path.exists(sys_state["project_samples"]):
        os.makedirs(sys_state["project_samples"])

    sys_state["project_scripts"]     = os.path.join(project_root, "scripts")
    if not os.path.exists(sys_state["project_scripts"]):
        os.makedirs(sys_state["project_scripts"])
    
    sys_state["reporter_path"] = os.path.join(project_root,sys_state["version"]+"_report")

def fetch_teacher_files(sys_state, env_config):

    version = sys_state["teacher_model"]["version"]
    if not os.path.exists(sys_state["log_root_path"]):
        os.makedirs(sys_state["log_root_path"])
    # create dirs                                     
    sys_state["teacher_model"]["project_root"] = os.path.join(sys_state["log_root_path"], version)
                                            
    project_root = sys_state["teacher_model"]["project_root"]
    if not os.path.exists(project_root):
        os.makedirs(project_root)

    sys_state["teacher_model"]["project_checkpoints"] = os.path.join(project_root, "checkpoints")
    if not os.path.exists(sys_state["teacher_model"]["project_checkpoints"]):
        os.makedirs(sys_state["teacher_model"]["project_checkpoints"])

    sys_state["teacher_model"]["project_scripts"]     = os.path.join(project_root, "scripts")
    if not os.path.exists(sys_state["teacher_model"]["project_scripts"]):
        os.makedirs(sys_state["teacher_model"]["project_scripts"])
    if sys_state["teacher_model"]["node_ip"] != "localhost":
        from    utilities.sshupload import fileUploaderClass
        machine_config = env_config["machine_config"]
        machine_config = readConfig(machine_config)
        nodeinf = None
        for item in machine_config:
            if item["ip"] == sys_state["teacher_model"]["node_ip"]:
                nodeinf = item
                break
        if not nodeinf:
            raise Exception(print("Configuration of node %s is unavaliable"%sys_state["node_ip"]))
        print("ready to fetch related files from server: %s ......"%nodeinf["ip"])
        uploader    = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])

        remotebase  = os.path.join(nodeinf['path'],"train_logs",version).replace('\\','/')
        
        # Get the config.json
        print("ready to get the teacher's config.json...")
        remoteFile  = os.path.join(remotebase, env_config["config_json_name"]).replace('\\','/')
        localFile   = os.path.join(project_root, env_config["config_json_name"])
        
        ssh_state   = uploader.sshScpGet(remoteFile, localFile)
        if not ssh_state:
            raise Exception(print("Get file %s failed! config.json does not exist!"%remoteFile))
        print("success get the teacher's config.json from server %s"%nodeinf['ip'])

        # Get scripts
        remoteDir   = os.path.join(remotebase, "scripts").replace('\\','/')
        localDir    = os.path.join(sys_state["teacher_model"]["project_scripts"])
        ssh_state   = uploader.sshScpGetDir(remoteDir, localDir)
        if not ssh_state:
            raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
        print("Get the teacher's scripts successful!")
    # Read model_config.json
    config_json = os.path.join(project_root, env_config["config_json_name"])
    json_obj    = readConfig(config_json)
    for item in json_obj.items():
        if item[0] in ignoreKey:
            pass
        else:
            sys_state["teacher_model"][item[0]] = item[1]
        
        # Get checkpoints
    if sys_state["teacher_model"]["node_ip"] != "localhost":        
        ckpt_name = "step%d_%s.pth"%(sys_state["teacher_model"]["checkpoint_step"],
                                    sys_state["teacher_model"]["checkpoint_names"]["generator_name"])
        localFile   = os.path.join(sys_state["teacher_model"]["project_checkpoints"],ckpt_name)
        if not os.path.exists(localFile):
            remoteFile  = os.path.join(remotebase, "checkpoints", ckpt_name).replace('\\','/')
            ssh_state = uploader.sshScpGet(remoteFile, localFile, True)
            if not ssh_state:
                raise Exception(print("Get file %s failed! Checkpoint file does not exist!"%remoteFile))
            print("Get the teacher's checkpoint %s successfully!"%(ckpt_name))
        else:
            print("%s exists!"%(ckpt_name))

def main():

    config = getParameters()
    # speed up the program
    cudnn.benchmark = True
    cudnn.enabled   = True

    from utilities.logo_class import logo_class
    logo_class.print_group_logo()

    sys_state = {}

    # set the GPU number
    gpus = [str(i) for i in config.gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus) 

    # read system environment paths
    env_config = readConfig('env/env.json')
    env_config = env_config["path"]

    # obtain all configurations in argparse
    config_dic = vars(config)
    for config_key in config_dic.keys():
        sys_state[config_key] = config_dic[config_key]
    
    #=======================Train Phase=========================#
    if config.phase == "train":
        # read training configurations from yaml file
        ymal_config = getConfigYaml(os.path.join(env_config["train_config_path"], config.train_yaml))
        for item in ymal_config.items():
            sys_state[item[0]] = item[1]

        # create related dirs
        sys_state["log_root_path"] = env_config["train_log_root"]
        createDirs(sys_state)
        
        # create reporter file
        reporter = Reporter(sys_state["reporter_path"])

        # save the config json
        config_json = os.path.join(sys_state["project_root"], env_config["config_json_name"])
        writeConfig(config_json, sys_state)

        # save the dependent scripts 
        # TODO and copy the scripts to the project dir
        
        # save the trainer script into [train_logs_root]\[version name]\scripts\
        file1       = os.path.join(env_config["train_scripts_path"],
                                    "trainer_%s.py"%sys_state["train_script_name"])
        tgtfile1    = os.path.join(sys_state["project_scripts"],
                                    "trainer_%s.py"%sys_state["train_script_name"])
        shutil.copyfile(file1,tgtfile1)

        # save the yaml file
        file1       = os.path.join(env_config["train_config_path"], config.train_yaml)
        tgtfile1    = os.path.join(sys_state["project_scripts"], config.train_yaml)
        shutil.copyfile(file1,tgtfile1)

        # TODO replace below lines, here to save the critical scripts

    #=====================Finetune Phase=====================#
    elif config.phase == "finetune":
        sys_state["log_root_path"]  = env_config["train_log_root"]
        sys_state["project_root"]   = os.path.join(sys_state["log_root_path"], sys_state["version"])

        config_json                 = os.path.join(sys_state["project_root"], env_config["config_json_name"])
        train_config                = readConfig(config_json)
        for item in train_config.items():
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]]  = item[1]
        
        createDirs(sys_state)
        reporter = Reporter(sys_state["reporter_path"])
        sys_state["com_base"]       = "train_logs.%s.scripts."%sys_state["version"]
    
    fetch_teacher_files(sys_state,env_config)    
    # get the dataset path
    sys_state["dataset_paths"] = {}
    for data_key in env_config["dataset_paths"].keys():
        sys_state["dataset_paths"][data_key] = env_config["dataset_paths"][data_key]

    # display the training information
    moduleName  = "train_scripts.trainer_" + sys_state["train_script_name"]
    if config.phase == "finetune":
        moduleName  = sys_state["com_base"] + "trainer_" + sys_state["train_script_name"]
    
    # print some important information
    # TODO
    # print("Start to run training script: {}".format(moduleName))
    # print("Traning version: %s"%sys_state["version"])
    # print("Dataloader Name: %s"%sys_state["dataloader"])
    # # print("Image Size: %d"%sys_state["imsize"])
    # print("Batch size: %d"%(sys_state["batch_size"]))
    # print("GPUs:", gpus)
    print("\n========================================================================\n")
    print(sys_state)
    for data_key in sys_state.keys():
        print("[%s]---[%s]"%(data_key,sys_state[data_key]))
    print("\n========================================================================\n")

    
    # Load the training script and start to train
    reporter.writeConfig(sys_state) 

    package     = __import__(moduleName, fromlist=True)
    trainerClass= getattr(package, 'Trainer')
    trainer     = trainerClass(sys_state, reporter)
    trainer.train()


if __name__ == '__main__':
    main()