# /------------------------------------------------------------/
# **Floor Plan Boundary Detection with Fast Marching**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which identifies indoor spaces
# using the fast marching method and assigns a navigable area
# to them.
# -----
# Best use case: floor plan *pysovist* workflows.
# -----
# What's in the file:
# 1. imports
# 2. helper functions
# 3. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
from scipy.spatial import KDTree
from src.m2d_segments_angle import visibility_area_np
from scipy.ndimage import gaussian_filter, sobel
import skfmm

## 2. Helper Functions
#TODO: detect boundaries with FMM and generate grid in valid area
def fmm_edges(img:np.array, sigma:float=20,alpha:int=1,p:int=0.5,percentile:int=95):
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
    td = np.log(np.linalg.norm(np.diff(np.diff(t)),axis=-1))
    tf = np.isfinite(td)
    td[~tf] = -1
    return td


## 3. Base Method
def identify_noninf(plan_lines:np.array,global_res:float,res:float=0.5) -> np.array:
    # global_res: global upscaling factor --> scale accordingly
    
    # sample points from a uniform grid within the boundary
    plan_lines = np.argwhere(td>-50)
    xmin,xmax,ymin,ymax = plan_lines[...,0].min(), plan_lines[...,0].max(), plan_lines[...,1].min(), plan_lines[...,1].max()
    xspace = np.linspace(xmin,xmax,int((xmax-xmin)/(global_res*res))) 
    yspace = np.linspace(ymin,ymax,int((ymax-ymin)/(global_res*res)))
    grid = np.meshgrid(xspace,yspace)
    grid = np.stack([g.ravel() for g in grid], axis=-1)
    tree_plan = KDTree(plan_lines)
    dists,idx = tree_plan.query(grid,k=1)
    idx = idx[dists<=res*global_res]
    valid_grid = plan_lines[idx]
    valid_vals = td[plan_lines[idx,0],plan_lines[idx,1]]
    tree = KDTree(valid_grid)
    # filter points with distant neighbors
    dists, _ = tree.query(valid_grid, k=3)
    dist_mask = dists[:, -1] <= np.percentile(dists[:, -1], 90)

    return valid_grid[dist_mask], valid_vals[dist_mask]

def densify_grid(points: np.ndarray,multiplier:float=3) -> np.ndarray:
    tree = KDTree(points)

    pairs = tree.query_pairs(r=1)

    new_pts = []
    for i, j in pairs:
        p0 = points[i]
        p1 = points[j]

        for t in np.linspace(0, 1, multiplier + 2)[1:-1]:
            new_pts.append((1 - t) * p0 + t * p1)

    if not new_pts:
        return points

    dense = np.vstack([points, np.asarray(new_pts)])
    #return np.unique(np.round(dense, 8), axis=0)
    return dense