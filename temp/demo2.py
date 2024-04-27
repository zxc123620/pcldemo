#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :demo2.py
# @Time      :2024/4/25 9:39
# @Author    :zhouxiaochuan
# @Description:
# -*-coding:utf-8 -*-
import numpy as np
import open3d as o3d

# 读取点云
back_pcd = o3d.io.read_point_cloud("./pcd/obs.pcd")
obs_pcd = o3d.io.read_point_cloud("./pcd/back.pcd")
# 将点云上成灰色
back_pcd.paint_uniform_color([0.5, 0.5, 0.5])
# 对点云进行kdtree检索
pcd_tree = o3d.geometry.KDTreeFlann(back_pcd)
back_pcd.colors[1500] = [1, 0, 0]
# 检索喵点附近的200个最近邻点, 并将其涂上蓝色
[k, idx, _] = pcd_tree.search_knn_vector_3d(back_pcd.points[1500], 200)
np.asarray(back_pcd.colors)[idx[1:], :] = [0, 0, 1]
# 检索以喵点为圆心半径小于0.2米的点云, 并将其涂上绿色
[k, idx, _] = pcd_tree.search_radius_vector_3d(back_pcd.points[1500], 0.2)
np.asarray(back_pcd.colors)[idx[1:], :] = [0, 1, 0]
# 可视化点云
o3d.visualization.draw_geometries([back_pcd],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])
