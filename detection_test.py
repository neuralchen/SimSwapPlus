#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: detection_test.py
# Created Date: Tuesday February 15th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 15th February 2022 10:31:52 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

from insightface.app import FaceAnalysis

import insightface

if __name__ == "__main__":
    app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640))
    