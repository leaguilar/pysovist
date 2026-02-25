# /------------------------------------------------------------/
# **3D Volume Calculation Algorithm: 'Spherical'**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a visible volume calculation algorithm which takes
# in point clouds and returns the volume of the corresponding 
# spherical slice, visible without obstruction from 
# given points, by integration.
# -----
# Best use case: point clouds, concave shapes.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from scipy.stats import qmc
from scipy.interpolate import CubicSpline
from sklearn.neighbors import KNeighborsRegressor


## 2. Base Method
def visibility_spherical(pcd_points:np.ndarray, origin:np.ndarray, max_unknown:bool, min_distance = 0, max_distance = 40, num_samples = 1200, **kwargs):
    rel_positions = pcd_points-origin
    pcd_distances = np.linalg.norm(rel_positions,axis=1,keepdims=True)
    mask = (pcd_distances[:,0] >= min_distance) & (pcd_distances[:,0] <= max_distance)
    pcd_dirs = rel_positions/(pcd_distances+1e-5) #unit sphere
    pcd_dirs = pcd_dirs[mask]
    rel_positions = rel_positions[mask]
    pcd_distances = pcd_distances[mask]
    pcd_rgb = pcd_rgb[mask]
    
    indices = np.arange(0, num_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_samples)
    theta = np.pi * (1 + 5**0.5) * indices
    dirs = np.vstack([np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),np.cos(phi)]).T.astype(np.float32)
    
    # truncate view directions according to intrinsics
    if kwargs.view_dir == None:
        fov_x = 2*np.pi
        fov_y = 2*np.pi
        view_dir = [0.0,0.0,1.0]
    else:
        fov_x = kwargs.fov_x
        fov_y = kwargs.fov_y
        view_dir = kwargs.view_dir

    forward = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if np.abs(np.dot(world_up, forward)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, forward)
    x = dirs @ right
    y = dirs @ up
    z = dirs @ forward 
    h_ang = np.arctan2(x, z)
    v_ang = np.arctan2(y, z)
    mask_dir = (z > 0) & (np.abs(h_ang) <= fov_x * 0.5) & (np.abs(v_ang) <= fov_y * 0.5)
    dirs = dirs[mask_dir]
    ndirs = dirs.shape[0]
    tree = KDTree(dirs, leafsize=8)
    _, idx = tree.query(pcd_dirs, k=1)
    
    dist_per_bin = np.ones(ndirs)*1e5
    np.minimum.at(dist_per_bin, idx, pcd_distances[:,0])

    if kwargs.heightlims != None:
        # flatten floor and calculated ceiling height --> clip bins
        #heightlims = [floor,ceiling] --> relative to vantage point height
        heightlims = kwargs.heightlims
        #extrapolate heights from dist_per_bin, pcd_dirs
        rel_heights = dist_per_bin * dirs[:, 2]
        height_o_mask = rel_heights > (heightlims[1] - origin[2])
        height_u_mask = rel_heights < (heightlims[0] - origin[2])
        #calculate differences
        height_o_diff = np.abs(dist_per_bin[height_o_mask] - (heightlims[1] - origin[2]))
        height_u_diff = np.abs(dist_per_bin[height_u_mask] + (heightlims[0] - origin[2]))
        # safeguard against sqrt of negative values
        val_o = dist_per_bin[height_o_mask]**2 - height_o_diff**2
        val_u = dist_per_bin[height_u_mask]**2 - height_u_diff**2
        val_o[val_o < 0] = 0
        val_u[val_u < 0] = 0
        #get norm -> subtract difference from norm^2
        dist_per_bin[height_o_mask] = np.sqrt(val_o)
        dist_per_bin[height_u_mask] = np.sqrt(val_u)
    
    if max_unknown == False:
        dist_per_bin[dist_per_bin>max_distance] = 0
    if max_unknown == True:
        dist_per_bin[dist_per_bin>max_distance] = max_distance
    ts = dist_per_bin

    interpolate = kwargs.interpolate
    if interpolate != 'none':
        known_val = dist_per_bin[dist_per_bin != 0]
    if interpolate == 'linear':
        ts = np.interp(all_idx,known_idx,known_val) #linear
    if interpolate == 'cubic':
        spl = CubicSpline(known_idx,known_val) #cubic spline
        ts = spl(all_idx)
    if 'knn' in interpolate:
        k = int(interpolate.split('knn')[1])
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(known_idx.reshape(-1,1), known_val)
        ts = knn.predict(all_idx.reshape(-1,1))
        
    all_idx   = np.arange(dist_per_bin.size)
    known_idx = all_idx[dist_per_bin != 0]

    dist_per_bin[(~np.isfinite(dist_per_bin)) | (dist_per_bin > max_distance)] = 0.0
    valid_rays = dist_per_bin > 0.0

    rel_pts = dirs[valid_rays] * dist_per_bin[valid_rays, None]
    
    tree_relpts = KDTree(rel_positions,leafsize=8)
    _, rgb_idx = tree_relpts.query(rel_pts,k=1)
    pts_colors = pcd_rgb[rgb_idx]

    alpha = 0.5 * fov_x
    beta  = 0.5 * fov_y
    Omega = 4.0 * np.arcsin(np.sin(alpha) * np.sin(beta))
    N = dist_per_bin.size
    vol = (Omega / (3.0 * N)) * np.sum(dist_per_bin**3)
    min_dist = dist_per_bin.min()
    max_dist = dist_per_bin.max()
    mean_dist = dist_per_bin.mean()
    dist_stdev = dist_per_bin.std()

    if kwargs.return_stats == True:
        return rel_pts, pts_colors, vol, min_dist, max_dist, mean_dist, dist_stdev
    else:
        return vol