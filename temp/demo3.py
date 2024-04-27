#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :demo3.py
# @Time      :2024/4/25 10:18
# @Author    :zhouxiaochuan
# @Description: 
# 加载两帧点云数据
import open3d as o3d
import numpy as np
pcd1 = o3d.io.read_point_cloud("./pcd/09_54_59_629.back_pcd")
# pcd2 = o3d.io.read_point_cloud("./back_pcd/obs.back_pcd")

# 将点云转换为NumPy数组以便处理
points1 = np.asarray(pcd1.points, dtype=np.float64)
# points2 = np.asarray(pcd2.points, dtype=np.float64)
print(points1[1])