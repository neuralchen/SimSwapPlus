#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: KA.py
# Created Date: Wednesday February 23rd 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 23rd February 2022 12:12:05 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


def KA(X, Y):
    X_ = X.view(X.size(0), -1)
    Y_ = Y.view(Y.size(0), -1)
    assert X_.shape[0] == Y_.shape[
        0], f'X_ and Y_ must have the same shape on dim 0, but got {X_.shape[0]} for X_ and {Y_.shape[0]} for Y_.'
    X_vec = X_ @ X_.T
    Y_vec = Y_ @ Y_.T
    ret = (X_vec * Y_vec).sum() / ((X_vec**2).sum() * (Y_vec**2).sum())**0.5
    return ret