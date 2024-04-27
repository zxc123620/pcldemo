#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :MainWindow.py
# @Time      :2024/4/20 13:14
# @Author    :zhouxiaochuan
# @Description:
import os
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib import pyplot as plt

from QtGui.ui import Ui_Form

import numpy as np
import open3d as o3d
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import vtk


class VtkPointCloud:
    def __init__(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkPolyData = vtk.vtkPolyData()
        self.clear_points()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(-10, 10)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def add_points(self, points):
        for point in points:
            point_id = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(point_id)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clear_points(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


class PclMainWindow(QMainWindow, Ui_Form):
    def __init__(self):
        super(PclMainWindow, self).__init__(parent=None)
        self.pcd_files_path = None
        self.setupUi(self)
        self.timer = QTimer(parent=None)
        self.timer.timeout.connect(self.update_points)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.pcd_layout.addWidget(self.vtk_widget)
        self.point_cloud = VtkPointCloud()
        self.render = vtk.vtkRenderer()
        self.render.AddActor(self.point_cloud.vtkActor)
        self.render.SetBackground(0, 0, 0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.render)
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
        self.show()
        interactor.Initialize()

    def load_pcd(self, file_path):
        """
        显示pcd点云信息
        :param file_path: pcd文件路径
        :return:
        """
        pcd = o3d.io.read_point_cloud(file_path)
        pcd_points = np.asarray(pcd.points)  # x, y, z
        return pcd_points

    def select_pcd_and_show(self):
        """
        选择文件并显示PCD点云信息
        :return:
        """
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                          "All Files (*);;Text Files (*.txt)")
        pcd_points = self.load_pcd(file_path[0])
        self.render_points(pcd_points)

    def render_points(self, pcd_points):
        """
        加载点云
        :param pcd_points:
        :return:
        """
        self.point_cloud.clear_points()
        self.point_cloud.add_points(pcd_points)
        self.vtk_widget.GetRenderWindow().Render()

    def update_points(self):
        """
        更新点云数据
        :return:
        """
        pcd_file_path = self.pcd_files_path.pop()
        self.point_cloud.clear_points()
        pcd_points = self.load_pcd(pcd_file_path)
        self.render_points(pcd_points)

    def loop_pcd_show(self):
        """
        循环查看点云信息
        :return:
        """
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选取文件夹", f"{os.getcwd()}")  # 起始路径
        self.pcd_files_path = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith("back_pcd")]
        self.timer.start(10000)

    def point_cluster(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                          "All Files (*);;Text Files (*.txt)")
        pcd = o3d.io.read_point_cloud(file_path)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        o3d.visualization.draw_geometries([pcd],
                                          zoom=0.455,
                                          front=[-0.4999, -0.1659, -0.8499],
                                          lookat=[2.1813, 2.0619, 2.0999],
                                          up=[0.1204, -0.9852, 0.1215])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pcl_main_window = PclMainWindow()
    sys.exit(app.exec_())
