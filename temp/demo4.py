#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :demo4.py
# @Time      :2024/4/25 10:54
# @Author    :zhouxiaochuan
# @Description:

import random

import open3d as o3d
import numpy as np

#  ========================== 点云对比 =============================
# 加载两帧点云数据
pcd1 = o3d.io.read_point_cloud("./pcd/back.pcd")
pcd2 = o3d.io.read_point_cloud("./pcd/obs.pcd")

# 将点云转换为NumPy数组以便处理
points1 = pcd1.points
points2 = pcd2.points
tree = o3d.geometry.KDTreeFlann(pcd1)
# 找出只在第二帧中出现的点（新增的点）
appeared_points = []
for point2 in points2:
    [k, idx, _] = tree.search_radius_vector_3d(point2, radius=0.05)
    if k == 0:  # 没有找到对应的点
        appeared_points.append(point2)

appeared_pcd = o3d.geometry.PointCloud()
appeared_pcd.points = o3d.utility.Vector3dVector(appeared_points)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(appeared_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
# 可视化结果
# o3d.visualization.draw_geometries([disappeared_pcd, appeared_pcd])
o3d.visualization.draw_geometries([appeared_pcd])

# # 保存结果
# o3d.io.write_point_cloud("disappeared_points.ply", disappeared_pcd)
o3d.io.write_point_cloud("appeared_points.pcd", appeared_pcd)

# ============================ 点云聚类 ============================
# 读取点云数据 appeared_pcd
# 转换点云数据为NumPy数组
# 定义DBSCAN参数
eps = 15  # 邻域半径
min_points = 5  # 最小点数

# 使用DBSCAN进行聚类
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    cluster_labels = np.array(appeared_pcd.cluster_dbscan(eps, min_points, print_progress=True))

# 获取聚类结果
n_clusters = cluster_labels.max() + 1
print(f"Detected {n_clusters} clusters.")

# 为每个点赋予颜色，不同聚类赋予不同颜色
# print(f"point cloud has {max_label + 1} clusters")  # label = -1 为噪声，因此总聚类个数为 max_label + 1
colors = np.full_like(appeared_pcd.points, [0, 0, 0])
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
for i in range(n_clusters):
    if i == -1:
        continue
    mask = (cluster_labels == i)
    x = np.sum(mask)
    colors[mask] = np.full((x,3),np.random.rand(1, 3)[0])

# colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
appeared_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([appeared_pcd])
# 创建带颜色的点云
# colored_pcd = o3d.geometry.PointCloud()
# colored_pcd.points = o3d.utility.Vector3dVector(appeared_pcd.points)
# colored_pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化结果
# o3d.visualization.draw_geometries([appeared_pcd])

# 如果需要，可以保存带颜色的点云到文件
o3d.io.write_point_cloud("clustered_point_cloud.pcd", appeared_pcd)
