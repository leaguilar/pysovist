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
import cv2

## 2. Helper Functions
##use Fast Marching Method to extract edges
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
def import_img(path:Path|str,scale,crop_extents:Optional[List[str]]=[0,1,0,1],res:float=50,thr:float=0.7):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # res is based on true scale. 100 -> 100 pixels per m
    W = scale*res
    H = img.shape[0]*(W/img.shape[1])
    img = cv2.resize(img,(int(W),int(H)),interpolation=cv2.INTER_CUBIC)
    img = img[int(H*crop_extents[0]):int(H*crop_extents[1]),int(W*crop_extents[0]):int(W*crop_extents[1])]
    _, _, edges = fmm_edges(img)
    mask = edges > thr
    edges[mask] = 0
    edges_pts = np.argwhere(edges)
    return edges_pts # [N,N] array