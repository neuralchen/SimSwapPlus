#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: GUI copy 2.py
# Created Date: Wednesday December 22nd 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 10th February 2022 12:14:47 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################



import os
import sys
import time
import json
import tkinter
try:
    import paramiko
except:
    from pip._internal import main
    main(['install', 'paramiko'])
    import paramiko

try:
    import pyperclip
except:
    from pip._internal import main
    main(['install', 'pyperclip'])
    import pyperclip

import threading
import tkinter as tk
import tkinter.ttk as ttk

import subprocess
from pathlib import Path




#############################################################
# Predefined functions
#############################################################

def read_config(path):
    with open(path,'r') as cf:
        nodelocaltionstr = cf.read()
        nodelocaltioninf = json.loads(nodelocaltionstr)
        if isinstance(nodelocaltioninf,str):
            nodelocaltioninf = json.loads(nodelocaltioninf)
    return nodelocaltioninf

def write_config(path, info):
    with open(path, 'w') as cf:
        configjson  = json.dumps(info, indent=4)
        cf.writelines(configjson)

class fileUploaderClass(object):
    def __init__(self,serverIp,userName,passWd,port=22):
        self.__ip__         = serverIp
        self.__userName__   = userName
        self.__passWd__     = passWd
        self.__port__       = port
        self.__ssh__        = paramiko.SSHClient()
        self.__ssh__.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def sshScpPut(self,localFile,remoteFile):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        remoteDir  = remoteFile.split("/")
        if remoteFile[0]=='/':
            sftp.chdir('/')
            
        for item in remoteDir[0:-1]:
            if item == "":
                continue
            try:
                sftp.chdir(item)
            except:
                sftp.mkdir(item)
                sftp.chdir(item)
        sftp.put(localFile,remoteDir[-1])
        sftp.close()
        self.__ssh__.close()
        print("[To %s]:%s remotefile:%s success"%(self.__ip__,localFile,remoteFile))

    def sshScpPuts(self,localFiles,remoteFiles):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        for i_dir in range(len(remoteFiles)):
            remoteDir  = remoteFiles[i_dir].split("/")
            if remoteFiles[i_dir][0]=='/':
                sftp.chdir('/')
            for item in remoteDir[0:-1]:
                if item == "":
                    continue
                try:
                    sftp.chdir(item)
                except:
                    sftp.mkdir(item)
                    sftp.chdir(item)
            sftp.put(localFiles[i_dir],remoteDir[-1])
            print("[To %s]:%s remotefile:%s success"%(self.__ip__,localFiles[i_dir],remoteFiles[i_dir]))
        sftp.close()
        self.__ssh__.close()

    def sshExec(self, cmd):
        try:
            self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
            _, stdout, _ = self.__ssh__.exec_command(cmd)
            results = stdout.read().strip().decode('utf-8')
            self.__ssh__.close() 
            return results
        except Exception as e:
            print(e)
        finally:
            self.__ssh__.close()    

    def sshScpGetNames(self,remoteDir):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        wocao = sftp.listdir(remoteDir)
        # print(wocao.st_mtime)
        roots = {}
        for item in wocao:
            wocao = sftp.stat(remoteDir+"/"+item)
            roots[item] = {
                "t":wocao.st_mtime,
                "p":remoteDir+"/"+item
            }
            # temp= remoteDir+ "/"+item
            # child_dirs = sftp.listdir(temp)
            # child_dirs = ["save\\" +item + "\\" + i for i in child_dirs]
            # list_name += child_dirs
        sftp.close()
        self.__ssh__.close()
        return roots
    
    def sshScpGetRNames(self,remoteDir):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        wocao = sftp.listdir(remoteDir)
        # print(wocao.st_mtime)
        roots = {}
        for item in wocao:
            wocao = sftp.stat(remoteDir+"/"+item)
            roots[item] = {
                "t":wocao.st_mtime,
                "p":remoteDir+"/"+item
            }
            # temp= remoteDir+ "/"+item
            # child_dirs = sftp.listdir(temp)
            # child_dirs = ["save\\" +item + "\\" + i for i in child_dirs]
            # list_name += child_dirs
        sftp.close()
        self.__ssh__.close()
        return roots
    
    def sshScpGet(self, remoteFile, localFile, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        if showProgress:
            sftp.get(remoteFile, localFile,callback=self.__putCallBack__)
        else:
            sftp.get(remoteFile, localFile)
        sftp.close()
        self.__ssh__.close()
    
    def sshScpGetFiles(self, remoteFiles, localFiles, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        for i in range(len(remoteFiles)):
            if showProgress:
                sftp.get(remoteFiles[i], localFiles[i],callback=self.__putCallBack__)
            else:
                sftp.get(remoteFiles[i], localFiles[i])
            print("Get %s success!"%(remoteFiles[i]))
        sftp.close()
        self.__ssh__.close()
    
    def sshScpGetDir(self, remoteDir, localDir, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        files = sftp.listdir(remoteDir)
        for i_f in files:
            i_remote_file = Path(remoteDir,i_f).as_posix()
            local_file    = Path(localDir,i_f)
            if showProgress:
                sftp.get(i_remote_file, local_file,callback=self.__putCallBack__)
            else:
                sftp.get(i_remote_file, local_file)
        sftp.close()
        self.__ssh__.close()
    
    def __putCallBack__(self,transferred,total):
        print("current transferred %.1f percent"%(transferred/total*100))
    
    def sshScpRename(self, oldpath, newpath):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.rename(oldpath,newpath)
        sftp.close()
        self.__ssh__.close()
        print("ssh oldpath:%s newpath:%s success"%(oldpath,newpath))

    def sshScpDelete(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.remove(path)
        sftp.close()
        self.__ssh__.close()
        print("ssh delete:%s success"%(path))
    
    def sshScpDeleteDir(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        self.__rm__(sftp,path)
        sftp.close()
        self.__ssh__.close()
        
    def __rm__(self,sftp,path):
        try:
            files = sftp.listdir(path=path)
            print(files)
            for f in files:
                filepath = os.path.join(path, f).replace('\\','/')
                self.__rm__(sftp,filepath)
            sftp.rmdir(path)
            print("ssh delete:%s success"%(path))
        except:
            print(path)
            sftp.remove(path)
            print("ssh delete:%s success"%(path))

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see(tk.END)
    
    def flush(self):
        pass

#############################################################
# Main Class
#############################################################

class Application(tk.Frame):

    tab_info        = []
    tab_body        = None
    current_index   = 0
    gui_root        = "GUI/"
    machine_json    = gui_root + "machines.json"
    filesynlogroot  = "file_sync/"
    filesynlogroot  = gui_root + filesynlogroot
    ignore_json     = gui_root + "guiignore.json"
    machine_list    = []
    machine_dict    = {}

    ignore_text={
                "white_list":{
                    "extension":["py",
                                "yaml"
                                ],
                    "file":[],
                    "path":[]
                },
                "black_list":{
                    "extension":[
                        "png",
                        "yaml"
                        ],
                    "file":[],
                    "path":["save/", "GUI/",]
                }
            }
    env_text={
        "train_log_root":"./train_logs",
        "test_log_root":"./test_logs",
        "systemLog":"./system/system_log.log",
        "dataset_paths":{
            "train_dataset_root":"",
            "val_dataset_root":"",
            "test_dataset_root":""
        },
        "train_config_path":"./train_yamls",
        "train_scripts_path":"./train_scripts",
        "test_scripts_path":"./test_scripts",
        "config_json_name":"model_config.json"
    }
    machine_text = {
                "ip": "0.0.0.0",
                "user": "username",
                "port": 22,
                "passwd": "12345678",
                "path": "/path/to/remote_host",
                "ckp_path":"save",
                "logfilename": "filestate_machine0.json"
    }
    current_log = {}


    def __init__(self, master=None):
        tk.Frame.__init__(self, master,bg='black')
        # self.font_size = 16
        self.font_list = ("Times New Roman",14)
        self.padx = 5
        self.pady = 5
        if not Path(self.gui_root).exists():
            Path(self.gui_root).mkdir(parents=True)
        self.window_init()
    
    def __label_text__(self, usr, root):
        return "User Name:  %s\nWorkspace:  %s"%(usr, root)

    def window_init(self):
        cwd = os.getcwd()
        self.master.title('File Synchronize - %s'%cwd)
        # self.master.iconbitmap('./utilities/_logo.ico')
        self.master.geometry("{}x{}".format(640, 800))
        font_list = self.font_list
        try:
            self.machines = read_config(self.machine_json)
        except:
            self.machine_list = [self.machine_text,]
            write_config(self.machine_json,self.machine_list)
            # subprocess.call("start %s"%self.machine_json, shell=True)
        #################################################################################################
        list_frame    = tk.Frame(self.master)
        list_frame.pack(fill="both", padx=5,pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(1, weight=1)
        list_frame.columnconfigure(2, weight=1)

        self.mac_var = tkinter.StringVar()

        self.list_com = ttk.Combobox(list_frame, textvariable=self.mac_var)
        self.list_com.grid(row=0,column=0,sticky=tk.EW)
        

        open_button = tk.Button(list_frame, text = "Update",
                            font=font_list, command = self.Machines_Update, bg='#F4A460', fg='#F5F5F5')
        open_button.grid(row=0,column=1,sticky=tk.EW)

        open_button = tk.Button(list_frame, text = "Machines",
                            font=font_list, command = self.MachineConfig, bg='#F4A460', fg='#F5F5F5')
        open_button.grid(row=0,column=2,sticky=tk.EW)
        
        #################################################################################################
        self.mac_text   = tk.StringVar()
        mac_label = tk.Label(self.master, textvariable=self.mac_text,font=self.font_list,justify="left")
        mac_label.pack(fill="both", padx=5,pady=5)
        self.mac_text.set(self.list_com.get())
        self.machines_update()
        def xFunc(event):
            ip      = self.list_com.get()
            cur_mac = self.machine_dict[ip]
            str_temp= self.__label_text__(cur_mac["user"],cur_mac["path"])
            self.mac_text.set(str_temp)
            self.update_log_task()
            self.update_ckpt_task()
        self.list_com.bind("<<ComboboxSelected>>",xFunc)
        #################################################################################################
        run_frame    = tk.Frame(self.master)
        run_frame.pack(fill="both", padx=5,pady=5)
        run_frame.columnconfigure(0, weight=1)
        run_frame.columnconfigure(1, weight=1)
        run_test_button = tk.Button(run_frame, text = "Synch Files",
                            font=font_list, command = self.Synchronize, bg='#006400', fg='#FF0000')
        run_test_button.grid(row=0,column=0,sticky=tk.EW)

        open_button = tk.Button(run_frame, text = "Synch To All",
                            font=font_list, command = self.SynchronizeAll, bg='#F4A460', fg='#F5F5F5')
        open_button.grid(row=0,column=1,sticky=tk.EW)

        #################################################################################################
        ssh_frame    = tk.Frame(self.master)
        ssh_frame.pack(fill="both", padx=5,pady=5)
        ssh_frame.columnconfigure(0, weight=1)
        ssh_frame.columnconfigure(1, weight=1)
        ssh_frame.columnconfigure(2, weight=1)
        ssh_button = tk.Button(ssh_frame, text = "Open SSH",
                            font=font_list, command = self.OpenSSH, bg='#990033', fg='#F5F5F5')
        ssh_button.grid(row=0,column=0,sticky=tk.EW)

        ssh_button = tk.Button(ssh_frame, text = "Copy Passwd",
                            font=font_list, command = self.CopyPasswd, bg='#990033', fg='#F5F5F5')
        ssh_button.grid(row=0,column=1,sticky=tk.EW)

        ssh_button = tk.Button(ssh_frame, text = "Pull Log",
                            font=font_list, command = self.PullLog, bg='#990033', fg='#F5F5F5')
        ssh_button.grid(row=0,column=2,sticky=tk.EW)

        #################################################################################################
        config_frame    = tk.Frame(self.master)
        config_frame.pack(fill="both", padx=5,pady=5)
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(2, weight=1)

        cmd_btn          = tk.Button(config_frame, text = "Open CMD",
                            font=font_list, command = self.OpenCMD, bg='#0033FF', fg='#F5F5F5')
        cmd_btn.grid(row=0,column=0,sticky=tk.EW)

        cwd_btn          = tk.Button(config_frame, text = "CWD",
                            font=font_list, command = self.CWD, bg='#0033FF', fg='#F5F5F5')
        cwd_btn.grid(row=0,column=1,sticky=tk.EW)

        gpu_btn          = tk.Button(config_frame, text = "GPU Usage",
                            font=font_list, command = self.GPUUsage, bg='#0033FF', fg='#F5F5F5')
        gpu_btn.grid(row=0,column=2,sticky=tk.EW)

        ################################################################################################

        config_frame    = tk.Frame(self.master)
        config_frame.pack(fill="both", padx=5,pady=5)
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=1)
        # config_frame.columnconfigure(2, weight=1)

        machine_btn     = tk.Button(config_frame, text = "Ignore Conf",
                            font=font_list, command = self.IgnoreConfig, bg='#660099', fg='#F5F5F5')
        machine_btn.grid(row=0,column=0,sticky=tk.EW)

        machine_btn2     = tk.Button(config_frame, text = "Env Conf",
                            font=font_list, command = self.EnvConfig, bg='#660099', fg='#F5F5F5')
        machine_btn2.grid(row=0,column=1,sticky=tk.EW)

        #################################################################################################
        log_frame    = tk.Frame(self.master)
        log_frame.pack(fill="both", padx=5,pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(1, weight=1)
        log_frame.columnconfigure(2, weight=1)

        self.log_var = tkinter.StringVar()

        self.log_com = ttk.Combobox(log_frame, textvariable=self.log_var)
        self.log_com.grid(row=0,column=0,sticky=tk.EW)
        def select_log(event):
            self.update_ckpt_task()
        self.log_com.bind("<<ComboboxSelected>>",select_log)
        

        log_update_button = tk.Button(log_frame, text = "Fresh",
                            font=font_list, command = self.UpdateLog, bg='#F4A460', fg='#F5F5F5')
        log_update_button.grid(row=0,column=1,sticky=tk.EW)

        log_update_button = tk.Button(log_frame, text = "Pull Log",
                            font=font_list, command = self.PullLog, bg='#F4A460', fg='#F5F5F5')
        log_update_button.grid(row=0,column=2,sticky=tk.EW)

        #################################################################################################
        test_frame    = tk.Frame(self.master)
        test_frame.pack(fill="both", padx=5,pady=5)
        test_frame.columnconfigure(0, weight=1)
        test_frame.columnconfigure(1, weight=1)
        test_frame.columnconfigure(2, weight=1)

        self.test_var = tkinter.StringVar()

        self.test_com = ttk.Combobox(test_frame, textvariable=self.test_var)
        self.test_com.grid(row=0,column=0,sticky=tk.EW)
        

        test_update_button = tk.Button(test_frame, text = "Fresh CKPT",
                            font=font_list, command = self.UpdateCKPT, bg='#F4A460', fg='#F5F5F5')
        test_update_button.grid(row=0,column=1,sticky=tk.EW)

        test_update_button = tk.Button(test_frame, text = "Test",
                            font=font_list, command = self.Test, bg='#F4A460', fg='#F5F5F5')
        test_update_button.grid(row=0,column=2,sticky=tk.EW)


        # #################################################################################################
        # tbtext_frame    = tk.Frame(self.master)
        # tbtext_frame.pack(fill="both", padx=5,pady=5)
        # tbtext_frame.columnconfigure(0, weight=5)

        # self.tensorlog_str   = tk.StringVar()
        # tb_text         = tk.Entry(tbtext_frame,font=font_list, textvariable=self.tensorlog_str )
        # tb_text.grid(row=0,column=0,sticky=tk.EW)
        # config_tb       = tk.Button(tbtext_frame, text = "Config",
        #                     font=font_list, command = self.OpenConfig, bg='#003472', fg='#F5F5F5')
        # config_tb.grid(row=0,column=1,sticky=tk.EW)
        # #################################################################################################


        # #################################################################################################
        # tb_frame    = tk.Frame(self.master)
        # tb_frame.pack(fill="both", padx=5,pady=5)
        # tb_frame.columnconfigure(1, weight=1)
        # tb_frame.columnconfigure(0, weight=1)
        # open_tb     = tk.Button(tb_frame, text = "Open Tensorboard",
        #                     font=font_list, command = self.OpenTensorboard, bg='#003472', fg='#F5F5F5')
        # open_tb.grid(row=0,column=0,sticky=tk.EW)
        # download_tb = tk.Button(tb_frame, text = "Update Tensorboard Logs",
        #                     font=font_list, command = self.DownloadTBLogs, bg='#003472', fg='#F5F5F5')
        # download_tb.grid(row=0,column=1,sticky=tk.EW)

        # #################################################################################################

        text = tk.Text(self.master, wrap="word")
        text.pack(fill="both",expand="yes", padx=5,pady=5)
        

        sys.stdout = TextRedirector(text, "stdout")

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    # def __scaning_logs__(self):
    def UpdateCKPT(self):
        thread_update = threading.Thread(target=self.update_ckpt_task)
        thread_update.start()
    
    def update_ckpt_task(self):
        ip          = self.list_com.get()
        log         = self.log_com.get()
        cur_mac     = self.machine_dict[ip]
        files = Path('.',cur_mac["ckp_path"], log)
        files = files.glob('*.pth')
        all_files   = []
        for one_file in files:
            all_files.append(one_file.name)
        self.test_com["value"] =all_files
        self.test_com.current(0)
    
    def CopyPasswd(self):
        def copy():
            ip          = self.list_com.get()
            cur_mac     = self.machine_dict[ip]
            passwd      = cur_mac["passwd"]
            pyperclip.copy(passwd)

        thread_update = threading.Thread(target=copy)
        thread_update.start()
    
    def Test(self):
        def test_task():
            log         = self.log_com.get()
            ckpt        = self.test_com.get()
            cwd         = os.getcwd()
            files = str(Path(log, ckpt))
            print(files)
            subprocess.check_call("start cmd /k \"cd /d %s && conda activate base \
                && python test.py --model %s\""%(cwd, files), shell=True)
        thread_update = threading.Thread(target=test_task)
        thread_update.start()
    
    def Machines_Update(self):
        # self.update_log_task()
        thread_update = threading.Thread(target=self.machines_update)
        thread_update.start()

    def machines_update(self):
        self.machine_list = read_config(self.machine_json)
        print(self.machine_list)
        ip_list = []
        for item in self.machine_list:
            self.machine_dict[item["ip"]] = item
            ip_list.append(item["ip"])
        print(ip_list)
        self.list_com["value"] = ip_list
        self.list_com.current(0)
        ip      = self.list_com.get()
        cur_mac = self.machine_dict[ip]
        str_temp= self.__label_text__(cur_mac["user"],cur_mac["path"])
        self.mac_text.set(str_temp)
        print("Machine list update success!")
    
    def connection(self):
        ip              = self.list_com.get()
        cur_mac         = self.machine_dict[ip]
        ssh_ip          = cur_mac["ip"]
        ssh_username    = cur_mac["user"]
        ssh_passwd      = cur_mac["passwd"]
        ssh_port        = int(cur_mac["port"])
        print(ssh_ip)
        if ip.lower() == "local" or ip.lower() == "localhost":
            print("localhost no need to connect!")
            return [], cur_mac
        remotemachine   = fileUploaderClass(ssh_ip,ssh_username,ssh_passwd,ssh_port)
        return remotemachine, cur_mac

    def __decode_filestr__(self, filestr):
        cells = filestr.split("\n")
        print(cells)
    
    def update_log_task(self):
        remotemachine,mac = self.connection()
        remote_path = os.path.join(mac["path"],mac["ckp_path"]).replace("\\", "/")
        if remotemachine == []:
            files = Path(remote_path).glob("*/")
            first_level   = {}
            for one_file in files:
                first_level[one_file.name] = {
                "t":"",
                "p":""
                }
        else:
            first_level = remotemachine.sshScpGetNames(remote_path)
        logs = []
        for k,v in first_level.items():
            logs.append(k)
        logs = sorted(logs)
        self.log_com["value"] =logs
        self.log_com.current(0)
        self.current_log = first_level
        self.update_ckpt_task()

    def UpdateLog(self):
        thread_update = threading.Thread(target=self.update_log_task)
        thread_update.start()

    def PullLog(self):
        def pull_log_task():
            remotemachine,mac = self.connection()
            log             = self.log_com.get()
            remote_path     = self.current_log[log]["p"]
            if remotemachine == []:
                return
            all_level = remotemachine.sshScpGetRNames(remote_path)
            file_need_download = []
            local_position = []
            local_dir = Path("./",mac["ckp_path"],log)
            if not local_dir.exists():
                local_dir.mkdir()

            for k,v in all_level.items():
                local_file = Path("./",mac["ckp_path"],log,k)
                if  local_file.exists():
                    if int(local_file.stat().st_mtime) < v["t"]:
                        file_need_download.append(v["p"])
                        local_position.append(str(local_file))
                        # print(int(local_file.stat().st_mtime))
                        # print(v["t"])
                else:
                    file_need_download.append(v["p"])
                    local_position.append(str(local_file))
            if len(file_need_download) > 0 :
                remotemachine.sshScpGetFiles(file_need_download, local_position)
                
            else:
                print("No file need to pull......")
            self.update_ckpt_task()
            
        thread_update = threading.Thread(target=pull_log_task)
        thread_update.start()
        
    def OpenCMD(self):
        def open_cmd_task():
            subprocess.call("start cmd", shell=True)
        thread_update = threading.Thread(target=open_cmd_task)
        thread_update.start()
    
    def CWD(self):
        def open_cmd_task():
            cwd = os.getcwd()
            subprocess.call("explorer "+cwd, shell=False)
        thread_update = threading.Thread(target=open_cmd_task)
        thread_update.start()
    
    def OpenSSH(self):
        def open_ssh_task():
            ip              = self.list_com.get()
            if ip.lower() == "local" or ip.lower() == "localhost":
                print("localhost no need to connect!")
            cur_mac         = self.machine_dict[ip]
            ssh_ip          = cur_mac["ip"]
            ssh_username    = cur_mac["user"]
            ssh_passwd      = cur_mac["passwd"]
            ssh_port        = cur_mac["port"]
            # subprocess.call("start cmd", shell=True)
            subprocess.call("start cmd /k ssh %s@%s -p %s"%(ssh_username, ssh_ip, ssh_port), shell=True)
            # subprocess.call("start echo %s"%(ssh_passwd), shell=True)
            # p = Popen("cp -rf a/* b/", shell=True, stdout=PIPE, stderr=PIPE)  
            # proc = subprocess.Popen("ssh %s@%s -p %s"%(ssh_username, ssh_ip, ssh_port),
            #                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            #                 stderr=subprocess.PIPE,creationflags =subprocess.CREATE_NEW_CONSOLE)
            # # out, err = proc.communicate(ssh_passwd.encode("utf-8"))
            # proc.stdin.write(ssh_passwd.encode('utf-8'))
            # print(out.decode('utf-8'))

        thread_update = threading.Thread(target=open_ssh_task)
        thread_update.start()
    
    def GPUUsage(self):
        def gpu_usage_task():
            remotemachine,_ = self.connection()
            results         = remotemachine.sshExec("nvidia-smi")
            print(results)
            
        thread_update = threading.Thread(target=gpu_usage_task)
        thread_update.start()
    
    def IgnoreConfig(self):
        def ignore_config_task():
            if not os.path.exists(self.ignore_json):
                print("guiignore.json file does not exist...")
                
                if not os.path.exists(self.gui_root):
                    os.makedirs(self.gui_root)
                write_config(self.ignore_json,self.ignore_text)
            subprocess.call("start %s"%self.ignore_json, shell=True)
        thread_update = threading.Thread(target=ignore_config_task)
        thread_update.start()
    
    def EnvConfig(self):
        def env_config_task():
            root_dir = os.getcwd()
            logs_dir = os.path.join(root_dir,"env","env.json")
            if not os.path.exists(logs_dir):
                print("env.json file does not exist...")
                
                if not os.path.exists(os.path.join(root_dir,"env")):
                    os.makedirs(os.path.join(root_dir,"env"))
                write_config(logs_dir,self.env_text)
            subprocess.call("start env/env.json", shell=True)
            
        thread_update = threading.Thread(target=env_config_task)
        thread_update.start()
      
    def MachineConfig(self):
        def machine_config_task():
            subprocess.call("start %s"%self.machine_json, shell=True)
        thread_update = threading.Thread(target=machine_config_task)
        thread_update.start()
        
    def OpenConfig(self):
        def open_config_task():
            root_dir = os.getcwd()
            logs_dir = os.path.join(root_dir,"env","logs_position.json")
            if not os.path.exists(logs_dir):
                print("logs configuration file does not exist...")
                positions={
                    "template":{
                        "root_path":"./",
                        "machine_name":"localhost",
                    }
                }
                if not os.path.exists(os.path.join(root_dir,"env")):
                    os.makedirs(os.path.join(root_dir,"env"))
                write_config(logs_dir,positions)
            subprocess.call("start env/logs_position.json", shell=True)
            # time.sleep(5) 
            # subprocess.call("start http://localhost:6006/", shell=True)
        thread_update = threading.Thread(target=open_config_task)
        thread_update.start()
    
    def OpenTensorboard(self):
        thread_update = threading.Thread(target=self.open_tensorboard_task)
        thread_update.start()
    
    def open_tensorboard_task(self):
        self.download_tblogs()
        root_dir = os.getcwd()
        logs_dir = os.path.join(root_dir,"train_logs")
        subprocess.call("start cmd /k tensorboard --logdir=\"%s\""%(logs_dir), shell=True)
        time.sleep(5) 
        subprocess.call("start http://localhost:6006/", shell=True)
    
    def DownloadTBLogs(self):
        thread_update = threading.Thread(target=self.download_tblogs)
        thread_update.start()

    def download_tblogs(self):
        tb_monitor_logs = self.tensorlog_str.get()
        tb_monitor_logs = tb_monitor_logs.split(";")
        root_dir = os.getcwd()
        mach_dir = os.path.join(root_dir,"env","machine_config.json")
        machines = read_config(mach_dir)
        logs_dir = os.path.join(root_dir,"env","logs_position.json")
        tb_logs  = read_config(logs_dir)

        for i_logs in tb_monitor_logs:
            try:
                mac_name            = tb_logs[i_logs]["machine_name"]
                i_mac               = machines[mac_name]
                i_mac["log_name"]   = i_logs
                # mac_list.append(i_mac)
                remotemachine       = fileUploaderClass(i_mac["ip"],i_mac["usrname"],i_mac["passwd"],i_mac["port"])
                path_temp           = Path(i_mac["root"],"train_logs",i_logs,"summary").as_posix()
                local_dir           = Path(root_dir,"train_logs",i_logs,"summary")
                if not Path(local_dir).exists():
                    Path(local_dir).mkdir(parents=True)
                remotemachine.sshScpGetDir(path_temp,local_dir)
                print("%s log files download successful!"%i_logs)
            except Exception as e:
                print(e)

    def Synchronize(self):
        def update():
            self.update_action()
        thread_update = threading.Thread(target=update)
        thread_update.start()

    def SynchronizeAll(self):
        def update_all():
            for i_mach in range(len(self.tab_info["configs"])):
                self.update_action(i_mach)
        thread_update = threading.Thread(target=update_all)
        thread_update.start()

    def update_action(self):
        last_state = {}
        changed_files = []

        ip              = self.list_com.get()
        cur_mac         = self.machine_dict[ip]
        if ip.lower() == "local" or ip.lower() == "localhost":
            print("localhost no need to update!")
            return
        ssh_ip          = cur_mac["ip"]
        ssh_username    = cur_mac["user"]
        ssh_passwd      = cur_mac["passwd"]
        ssh_port        = cur_mac["port"]
        root_path       = cur_mac["path"]

        log_path    = os.path.join(self.filesynlogroot,cur_mac["logfilename"])

        if not Path(self.filesynlogroot).exists():
            Path(self.filesynlogroot).mkdir(parents=True)
        else:
            if Path(log_path).exists():
                with open(log_path,'r') as cf:
                    nodelocaltionstr = cf.read()
                    last_state = json.loads(nodelocaltionstr)

        all_files   = []
        # scan files
        file_filter = read_config("./GUI/guiignore.json")

        white_list  = file_filter["white_list"]

        black_list  = file_filter["black_list"]

        white_ext   = white_list["extension"]

        black_path  = black_list["path"]

        black_file  = black_list["file"]

        for item in white_ext:
            if item=="":
                print("something error in the white list")
                continue
            files = Path('.').rglob('*.%s'%item) # ./*
            for one_file in files:
                all_files.append(one_file)
            for i_dir in black_path:
                files = Path('.', i_dir).rglob('*.%s'%item)
                for one_file in files:
                    # print(one_file)
                    all_files.remove(one_file)
        for item in black_file:
            try:
                all_files.remove(Path('.', item))
            except:
                print("%s does not exist!"%item)
            
        # check updated files
        for item in all_files:
            temp = item.stat().st_mtime
            if item._str in last_state:
                last_mtime = last_state[item._str]
                if last_mtime != temp:
                    changed_files.append(item._str)
                    last_state[item._str] = temp
            else:
                changed_files.append(item._str)
                last_state[item._str] = temp

        print("[To %s]"%ssh_ip,changed_files)

        localfiles  = []
        remotefiles = []

        for item in changed_files:
            localfiles.append(item)
            remotefiles.append(Path(root_path,item).as_posix())

        try:
            remotemachine = fileUploaderClass(ssh_ip,ssh_username,ssh_passwd,ssh_port)
            remotemachine.sshScpPuts(localfiles,remotefiles)
            with open(log_path, 'w') as cf:
                configjson  = json.dumps(last_state, indent=4)
                cf.writelines(configjson)
        except Exception as e:
            print(e)
            print("File Synchronize Failed!")

    # def __save_config__(self):

    #     previous_info = read_config(self.log_path)

    #     for i in range(len(self.tab_info["names"])):

    #         databind    = self.tab_info["databind"][i]

    #         data_aquire = {
    #             "name":         self.tab_info["names"][i],
    #             "remote_ip":    databind["remote_ip"].get(),
    #             "remote_user":  databind["remote_user"].get(),
    #             "remote_port":  databind["remote_port"].get(),
    #             "remote_passwd":databind["remote_passwd"].get(),
    #             "remote_path":  databind["remote_path"].get(),
    #             "logfilename":  "filestate_%s.json"%self.tab_info["names"][i]
    #         }
    #         if self.tab_info["names"][i] in previous_info["names"]:
    #             location = previous_info["names"].index(self.tab_info["names"][i])
    #             previous_info["configs"][location] = data_aquire
                
    #         else:
    #             previous_info["names"].append(self.tab_info["names"][i])
    #             previous_info["configs"].append(data_aquire)

    #     previous_info["databind"] = []
    #     write_config(self.log_path,previous_info)

    def on_closing(self):

        # self.__save_config__()
        self.master.destroy()
    


if __name__ == "__main__":
    app = Application()
    app.mainloop()