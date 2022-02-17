#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test.py
# Created Date: Saturday July 3rd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 17th February 2022 10:23:54 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import  os
import  argparse
from    torch.backends import cudnn
from    utilities.json_config import readConfig
from    utilities.reporter import Reporter
from    utilities.sshupload import fileUploaderClass


def str2bool(v):
    return v.lower() in ('true')

####################################################################################
# To configure the seting of training\finetune\test
#
####################################################################################
def getParameters():
    
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('-v', '--version', type=str, default='depthwise',
                                            help="version name for train, test, finetune")

    parser.add_argument('-c', '--cuda', type=int, default=0) # >0 if it is set as -1, program will use CPU
    parser.add_argument('-s', '--checkpoint_step', type=int, default=300000,
                                            help="checkpoint epoch for test phase or finetune phase")

    # test
    parser.add_argument('-t', '--test_script_name', type=str, default='image')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-n', '--node_name', type=str, default='localhost', 
                                            choices=['localhost', '4card','8card','new4card'])


    parser.add_argument('-i', '--id_imgs', type=str, default='G:\\swap_data\\ID\\gxt3.jpeg')
    # parser.add_argument('-i', '--id_imgs', type=str, default='G:\\VGGFace2-HQ\\VGGface2_ffhq_align_256_9_28_512_bygfpgan\\n000002\\0027_01.jpg')
    parser.add_argument('-a', '--attr_files', type=str, default='G:\\swap_data\\8',
                                                help="file path for attribute images or video")
    
    parser.add_argument('--use_specified_data', action='store_true')
    parser.add_argument('--specified_data_paths', type=str, nargs='+', default=[""], help='paths to specified files')
    
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

def main():

    config = getParameters()
    # speed up the program
    cudnn.benchmark = True

    sys_state = {}

    # set the GPU number
    if config.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)

    # read system environment paths
    env_config = readConfig('env/env.json')
    env_config = env_config["path"]

    # obtain all configurations in argparse
    config_dic = vars(config)
    for config_key in config_dic.keys():
        sys_state[config_key] = config_dic[config_key]
    
    #=======================Test Phase=========================#

    # TODO modify below lines to obtain the configuration
    sys_state["log_root_path"]        = env_config["train_log_root"]
    
    sys_state["test_samples_path"]    = os.path.join(env_config["test_log_root"], 
                                        sys_state["version"] , "samples")
    # if not config.use_my_test_date:
    #     print("Use public benchmark...")
    #     data_key = config.test_dataset_name.lower()
    #     sys_state["test_dataset_path"] = env_config["test_dataset_paths"][data_key]
    #     if config.test_dataset_name.lower() == "set5" or config.test_dataset_name.lower() =="set14":
    #         sys_state["test_dataloader"] = "setx"
    #     else:
    #         sys_state["test_dataloader"] = config.test_dataset_name.lower()
            
    # sys_state["test_dataset_name"] = config.test_dataset_name    

    if not os.path.exists(sys_state["test_samples_path"]):
        os.makedirs(sys_state["test_samples_path"])
    
    # Create dirs
    createDirs(sys_state)
    config_json = os.path.join(sys_state["project_root"], env_config["config_json_name"])
    
    # Read model_config.json from remote machine
    if sys_state["node_name"]!="localhost":
        remote_mac  = env_config["remote_machine"]
        nodeinf     = remote_mac[sys_state["node_name"]]
        print("ready to fetch related files from server: %s ......"%nodeinf["ip"])
        uploader    = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])

        remotebase  = os.path.join(nodeinf['base_path'],"train_logs",sys_state["version"]).replace('\\','/')
        
        # Get the config.json
        print("ready to get the config.json...")
        remoteFile  = os.path.join(remotebase, env_config["config_json_name"]).replace('\\','/')
        localFile   = config_json
        
        ssh_state   = uploader.sshScpGet(remoteFile, localFile)
        if not ssh_state:
            raise Exception(print("Get file %s failed! config.json does not exist!"%remoteFile))
        print("success get the config.json from server %s"%nodeinf['ip'])

    # Read model_config.json
    json_obj    = readConfig(config_json)
    for item in json_obj.items():
        if item[0] in ignoreKey:
            pass
        else:
            sys_state[item[0]] = item[1]
        
    # Read scripts from remote machine
    if sys_state["node_name"]!="localhost":
        # # Get scripts
        # remoteFile  = os.path.join(remotebase, "scripts", sys_state["gScriptName"]+".py").replace('\\','/')
        # localFile   = os.path.join(sys_state["project_scripts"], sys_state["gScriptName"]+".py")
        # ssh_state   = uploader.sshScpGet(remoteFile, localFile)
        # if not ssh_state:
        #     raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
        # print("Get the scripts:%s.py successfully"%sys_state["gScriptName"])
        # Get checkpoint of generator
        localFile   = os.path.join(sys_state["project_checkpoints"],
                            "epoch%d_%s.pth"%(sys_state["checkpoint_epoch"],
                                    sys_state["checkpoint_names"]["generator_name"]))
        if not os.path.exists(localFile):
            remoteFile  = os.path.join(remotebase, "checkpoints",
                            "epoch%d_%s.pth"%(sys_state["checkpoint_epoch"],
                                    sys_state["checkpoint_names"]["generator_name"])).replace('\\','/')
            ssh_state = uploader.sshScpGet(remoteFile, localFile, True)
            if not ssh_state:
                raise Exception(print("Get file %s failed! Checkpoint file does not exist!"%remoteFile))
            print("Get the checkpoint %s successfully!"%("epoch%d_%s.pth"%(sys_state["checkpoint_epoch"],
                                    sys_state["checkpoint_names"]["generator_name"])))
        else:
            print("%s exists!"%("epoch%d_%s.pth"%(sys_state["checkpoint_epoch"],
                                    sys_state["checkpoint_names"]["generator_name"])))

    # TODO get the checkpoint file path
    sys_state["ckp_name"] = {}
    # for data_key in sys_state["checkpoint_names"].keys():
    #     sys_state["ckp_name"][data_key] = os.path.join(sys_state["project_checkpoints"],
    #                                 "%d_%s.pth"%(sys_state["checkpoint_epoch"],
    #                                     sys_state["checkpoint_names"][data_key]))

    # Get the test configurations
    sys_state["com_base"]       = "train_logs.%s.scripts."%sys_state["version"]

    # make a reporter
    report_path = os.path.join(env_config["test_log_root"], sys_state["version"],
                                sys_state["version"]+"_report")
    reporter    = Reporter(report_path)
    reporter.writeConfig(sys_state)
    
    # Display the test information
    # TODO modify below lines to display your configuration information
    moduleName  = "test_scripts.tester_" + sys_state["test_script_name"]
    print("Start to run test script: {}".format(moduleName))
    print("Test version: %s"%sys_state["version"])
    print("Test Script Name: %s"%sys_state["test_script_name"])

    package     = __import__(moduleName, fromlist=True)
    testerClass = getattr(package, 'Tester')
    tester      = testerClass(sys_state,reporter)
    tester.test()


if __name__ == '__main__':
    main()