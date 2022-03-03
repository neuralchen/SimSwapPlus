#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Reporter.py
# Created Date: Tuesday September 24th 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 3rd March 2022 8:42:13 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import datetime
import os
import json

class Reporter:
    def __init__(self,reportPath):
        self.path           = reportPath
        self.withTimeStamp  = False
        self.index          = 1
        self.timeStrFormat  = '%Y-%m-%d %H:%M:%S'
        timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
        self.path = self.path + "-%s.log"%timeStr 
        if not os.path.exists(self.path):
            f = open(self.path,'w')
            f.close()
    
    def writeInfo(self,strLine):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),self.timeStrFormat)
            logf.writelines("[%d]-[%s]-[info] %s\n"%(self.index,timeStr,strLine))
            self.index += 1
    
    def writeConfig(self,config):
        with open(self.path,'a+') as logf:
            for item in config.items():
                text = "[%d]-[parameters] %s--%s\n"%(self.index,item[0],str(item[1]))
                logf.writelines(text)
                self.index +=1
    
    def writeModel(self,modelText):
        with open(self.path,'a+') as logf:
            logf.writelines("[%d]-[model] %s\n"%(self.index,modelText))
            self.index += 1
    
    def writeRawInfo(self, strLine):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),self.timeStrFormat)
            logf.writelines("[%d]-[info] %s\n"%(self.index,timeStr,strLine))
            self.index += 1
    
    def writeTrainLog(self, epoch, step, logText):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),self.timeStrFormat)
            logf.writelines("[%d]-[%s]-[logInfo]-epoch[%d]-step[%d] %s\n"%(self.index,timeStr,epoch,step,logText))
            self.index += 1
    
    def writeJson(self, info):
        with open(self.path, 'a+') as cf:
            configjson  = json.dumps(info, indent=4)
            cf.writelines(configjson)
