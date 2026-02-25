# /------------------------------------------------------------/
# **Data Import Helper Function: Point Cloud**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which imports point clouds
# stored in common formats, and adds them to the area object.
# -----
# Best use case: point clouds, convex spaces, missing points.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import open3d as o3d
import numpy as np

## 2. Base Method
def import_pcd(pcd_path:str,downsample:int|None,remove_outliers:tuple|None):
    pcd = o3d.io.read_point_cloud(pcd_path)
    if type(downsample) != int:
        raise TypeError('Downsampling k should be an integer. Usage: every kth point is eliminated')
    if type(remove_outliers) != tuple:
        raise TypeError('To remove outliers: define remove_outliers = (# neighbors to consider, σ outside distribution)')
    if downsample != None:
        pcd = o3d.geometry.uniform_down_sample(pcd,downsample)
    if remove_outliers == True:
        pcd = o3d.geometry.statistical_outlier_removal(pcd,remove_outliers[0],remove_outliers[1])
    return