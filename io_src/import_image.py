# /------------------------------------------------------------/
# **Data Import Helper Function: Image**
# -----
# *pysovist-dev* under MIT License
# -----
# This is an experimental helper function which imports image
# files with floor plans and converts detected lines into
# line instances usable by the Data2D class.
# -----
# Best use case: images of floor plans; PNG, JPG etc.
# -----
# Notes:
# * Please adjust the scale when importing the file.
# * Parameters may need tuning to work as intended.
# -----
# What's in the file:
# 1. imports
# 2. helper functions
# 3. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
import pandas as pd
import skfmm
from typing import List, Optional
from pathlib import Path
from scipy.ndimage import gaussian_filter, sobel

## 2. Helper Functions
##TODO: use Fast Marching Method to extract edges
def fmm_edges(img:np.array, sigma:float=1.5,alpha:int=20,p:int=2,percentile:int=95):
    img_f = gaussian_filter(img,sigma=sigma)
    gx = sobel(img_f,axis=1)
    gy = sobel(img_f,axis=0)
    grad = np.sqrt(gx**2+gy**2)
    grad_norm = grad/(grad.max()+1e-8)
    F = 1.0 / (1.0+alpha*grad_norm**p)
    phi = np.ones_like(img,dtype=int)
    phi[0,:] = -1
    phi[-1,:] = -1
    phi[:,0] = -1
    phi[:,-1] = -1
    t = skfmm.travel_time(phi,speed=F)
    tx = sobel(t,axis=1)
    ty = sobel(t,axis=0)
    t_grad = np.sqrt(tx**2+ty**2)
    edges = t_grad > np.percentile(t_grad,percentile)
    return edges, t, F


## 3. Base Method
def import_img(path:Path|str,scale,crop_extents:Optional[List[str]]=[0,1,0,1]):
    lines_df = pd.read_json(path)
    lines_2d = []
    for row in lines_df.iterrows():
        start = row[1][delimiter[0]][:2]
        end = row[1][delimiter[1]][:2]
        lines_2d.append([start,end])
    lines_2d = np.array(lines_2d)
    return lines_2d # [N,2,2] array