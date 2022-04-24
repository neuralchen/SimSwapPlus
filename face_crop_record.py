#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: face_crop.py
# Created Date: Tuesday February 1st 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 22nd April 2022 8:43:40 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import os
import cv2
import sys
import glob
import json
import tkinter
from tkinter.filedialog import askdirectory

import threading
import tkinter as tk
import tkinter.ttk as ttk

import subprocess
from pathlib import Path

from insightface_func.face_detect_crop_multi import Face_detect_crop

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


    def __init__(self, master=None):
        tk.Frame.__init__(self, master,bg='black')
        # self.font_size = 16
        self.font_list = ("Times New Roman",14)
        self.padx = 5
        self.pady = 5
        self.window_init()
    
    def __label_text__(self, usr, root):
        return "User Name:  %s\nWorkspace:  %s"%(usr, root)

    def window_init(self):
        cwd = os.getcwd()
        self.master.title('Face Crop - %s'%cwd)
        # self.master.iconbitmap('./utilities/_logo.ico')
        self.master.geometry("{}x{}".format(640, 600))

        font_list = self.font_list
        
        #################################################################################################
        list_frame    = tk.Frame(self.master)
        list_frame.pack(fill="both", padx=5,pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.columnconfigure(1, weight=1)
        list_frame.columnconfigure(2, weight=1)

        self.img_path = tkinter.StringVar()

        tk.Label(list_frame, text="Image/Video Path:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        
        tk.Entry(list_frame, textvariable= self.img_path, font=font_list)\
                    .grid(row=0,column=1,sticky=tk.EW)
        

        tk.Button(list_frame, text = "Select Path", font=font_list,
                    command = self.Select, bg='#F4A460', fg='#F5F5F5')\
                    .grid(row=0,column=2,sticky=tk.EW)
        #################################################################################################
        list_frame1    = tk.Frame(self.master)
        list_frame1.pack(fill="both", padx=5,pady=5)
        list_frame1.columnconfigure(0, weight=1)
        list_frame1.columnconfigure(1, weight=1)
        list_frame1.columnconfigure(2, weight=1)

        self.save_path = tkinter.StringVar()

        tk.Label(list_frame1, text="Target Path:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        
        tk.Entry(list_frame1, textvariable= self.save_path, font=font_list)\
                    .grid(row=0,column=1,sticky=tk.EW)
        

        tk.Button(list_frame1, text = "Select Path", font=font_list,
                    command = self.Select_Target, bg='#F4A460', fg='#F5F5F5')\
                    .grid(row=0,column=2,sticky=tk.EW)
        
        #################################################################################################
        label_frame    = tk.Frame(self.master)
        label_frame.pack(fill="both", padx=5,pady=5)
        label_frame.columnconfigure(0, weight=1)
        label_frame.columnconfigure(1, weight=1)
        label_frame.columnconfigure(2, weight=1)

        tk.Label(label_frame, text="Crop Size:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        
        tk.Label(label_frame, text="Align Mode:",font=font_list,justify="left")\
                    .grid(row=0,column=1,sticky=tk.EW)

        tk.Label(label_frame, text="Target Format:",font=font_list,justify="left")\
                    .grid(row=0,column=2,sticky=tk.EW)
        
        #################################################################################################

        test_frame    = tk.Frame(self.master)
        test_frame.pack(fill="both", padx=5,pady=5)
        test_frame.columnconfigure(0, weight=1)
        test_frame.columnconfigure(1, weight=1)
        test_frame.columnconfigure(2, weight=1)

        self.test_var = tkinter.StringVar()

        self.test_com = ttk.Combobox(test_frame, textvariable=self.test_var)
        self.test_com.grid(row=0,column=0,sticky=tk.EW)
        self.test_com["value"] = [256,512,768,1024]
        self.test_com.current(1)

        self.align_var = tkinter.StringVar()
        self.align_com = ttk.Combobox(test_frame, textvariable=self.align_var)
        self.align_com.grid(row=0,column=1,sticky=tk.EW)
        self.align_com["value"] = ["VGGFace","ffhq"]
        self.align_com.current(0)

        self.format_var = tkinter.StringVar()

        self.format_com = ttk.Combobox(test_frame, textvariable=self.format_var)
        self.format_com.grid(row=0,column=2,sticky=tk.EW)
        self.format_com["value"] = ["png","jpg"]
        self.format_com.current(0)

        

        #################################################################################################
        scale_frame    = tk.Frame(self.master)
        scale_frame.pack(fill="both", padx=5,pady=5)
        scale_frame.columnconfigure(0, weight=2)
        label_frame.columnconfigure(1, weight=1)
        # label_frame.columnconfigure(2, weight=1)

        tk.Label(scale_frame, text="Min Size:",font=font_list,justify="left")\
                    .grid(row=0,column=0,sticky=tk.EW)
        self.min_scale = tkinter.StringVar()
        tk.Scale(scale_frame, from_=0.5, to=2.0, length=500, orient=tk.HORIZONTAL, variable= self.min_scale,\
                    font=font_list, resolution=0.1).grid(row=0,column=1,sticky=tk.EW)
        
        #################################################################################################
        test_frame1    = tk.Frame(self.master)
        test_frame1.pack(fill="both", padx=5,pady=5)
        test_frame1.columnconfigure(0, weight=1)
        # test_frame1.columnconfigure(1, weight=1)

        test_update_button = tk.Button(test_frame1, text = "Crop",
                            font=font_list, command = self.Crop, bg='#F4A460', fg='#F5F5F5')
        test_update_button.grid(row=0,column=0,sticky=tk.EW)

        

        #################################################################################################

        text = tk.Text(self.master, wrap="word")
        text.pack(fill="both",expand="yes", padx=5,pady=5)
        

        sys.stdout = TextRedirector(text, "stdout")
        
        self.init_algorithm()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def init_algorithm(self):
        self.detect = Face_detect_crop(name='antelope', root='./insightface_func/models')
        
    
    # def __scaning_logs__(self):
    def Select(self):
        thread_update = threading.Thread(target=self.select_task)
        thread_update.start()
    
    def select_task(self):
        path = askdirectory()
        print("Selected source directory: %s"%path)
        self.img_path.set(path)
    
    def Select_Target(self):
        thread_update = threading.Thread(target=self.select_target_task)
        thread_update.start()
    
    def select_target_task(self):
        path = askdirectory()
        print("Selected target directory: %s"%path)
        self.save_path.set(path)
    
    def Crop(self):
        thread_update = threading.Thread(target=self.crop_task)
        thread_update.start()
    
    def crop_task(self):
        mode        = self.align_com.get()
        crop_size   = int(self.test_com.get())
        
        path        = self.img_path.get()
        tg_path     = self.save_path.get()
        if not os.path.exists(tg_path):
            os.makedirs(tg_path)
        tg_format   = self.format_com.get()
        min_scale   = float(self.min_scale.get())
        blur_t      = 10.0
        font        = cv2.FONT_HERSHEY_SIMPLEX 
        self.detect.prepare(ctx_id = 0, det_thresh=0.6,\
                        det_size=(640,640),mode = mode,crop_size=crop_size,ratio=min_scale)
        if path and tg_path:
            imgs_list = []
            if os.path.isdir(path):
                print("Input a dir....")
                imgs = glob.glob(os.path.join(path,"*"))
                for item in imgs:
                    imgs_list.append(item)
                # print(imgs_list)
                index = 0
                for img in imgs_list:
                    print(img)
                    attr_img_ori= cv2.imread(img)
                    try:
                        attr_img_align_crop, _ = self.detect.get(attr_img_ori)
                        sub_index = 0
                        if len(attr_img_align_crop) < 1:
                            print("Small face")
                        for face_i in attr_img_align_crop:
                            imageVar = cv2.Laplacian(face_i, cv2.CV_64F).var()
                            f_path =os.path.join(tg_path, str(index).zfill(6)+"_%d.%s"%(sub_index,tg_format))
                            if imageVar < blur_t:
                                print("Over blurry image!")
                                continue
                            # face_i = cv2.putText(face_i, '%.1f'%imageVar,(50, 50), font, 0.8, (15, 9, 255), 2)
                            cv2.imwrite(f_path,face_i)
                            sub_index += 1
                        index += 1
                    except:
                        print("Detect no face!")
                        continue
            else:
                print("Input an image....")
                imgs_list.append(path)
            print("Process finished!")
        else:
            print("Pathes are invalid!")

    def on_closing(self):

        # self.__save_config__()
        self.master.destroy()
    


if __name__ == "__main__":
    app = Application()
    app.mainloop()