#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :demo5.py
# @Time      :2024/4/25 13:04
# @Author    :zhouxiaochuan
# @Description:
import numpy as np
d = np.array([0, 1, 2, 3, 4, 1])
# a = ( d == 1)
# a[[True, False]] = 200
a = np.full((10,3),np.random.rand(1, 3)[0])
print(a)
# print(np.random.rand(1, 3)[0])