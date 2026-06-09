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

## 2. Helper Functions
#TODO: detect boundaries with FMM and generate grid in valid area
def fmm_edges(img:np.array, sigma:float=1,alpha:int=20,p:int=2,percentile:int=95):
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

t_mask = t<300
t[t_mask] = 0

## 3. Base Method
def identify_noninf(plan_lines:np.array,res:float=0.5,thr:float=0.8) -> np.array:
    # sample points from a uniform grid within the boundary
    xmin,xmax,ymin,ymax = plan_lines[...,0].min(), plan_lines[...,0].max(), plan_lines[...,1].min(), plan_lines[...,1].max()
    # res: grid resolution
    # thr: non-infinite threshold
    xspace = np.linspace(xmin,xmax,int((xmax-xmin)/res)) 
    yspace = np.linspace(ymin,ymax,int((ymax-ymin)/res))
    grid = np.meshgrid(xspace,yspace)
    grid = np.stack([g.ravel() for g in grid], axis=-1)
    valids = []
    for j in grid:
        # simple raycasting --> find infinites
        valid = visibility_valid(plan_lines,[j[0],j[1]])
        valids.append(valid)
    # select nodes within threshold
    valids = np.asarray(valids)
    valid_mask = valids >= thr
    valid_grid = grid[valid_mask]
    valid_vals = valids[valid_mask]
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