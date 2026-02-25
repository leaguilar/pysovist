# /------------------------------------------------------------/
# **2D Area Calculation Algorithm: 'Discretized'**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a visibility area calculation algorithm which
# transforms segments into discrete points and finds
# intersections by querying a k-dimensional tree. Accuracy
# may be compromised near boundaries.
# -----
# Best use case: Complex geometries with short segments,
# point cloud slices.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Polygon

## 2. Base Method
def visibility_discretized(segments, origin, max_dist, num_rays, **kwargs):
    ### set point grid resolution --> change if units are too large/small
    resolution = 15/num_rays 
    
    ### exclude curves with both endpoints outside visible range
    segments_diffs = segments-origin
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_dist) | (segments_dists[:,1] <= max_dist)
    segments = segments[segment_mask]
    
    ### get norms of each curve
    norms_curves = np.linalg.norm(segments[:,1,:]-segments[:,0,:],axis=1)/resolution
    norms_curves = np.array(norms_curves,dtype=int)

    ### generate points
    idx    = np.arange(norms_curves.max())
    T = idx[None,:] / (norms_curves[:,None] - 1 + 1e-6)
    starts = segments[:, 0, :]
    deltas = segments[:, 1, :] - starts 
    pts = starts[:, None, :] + deltas[:, None, :] * T[:, :, None]
    pts_flat  = pts.reshape(-1, 2) 
    mask = idx[None, :] < norms_curves[:, None]
    mask_flat = mask.flatten()
    valid_pts = pts_flat[mask_flat]
    
    ### raycasting directions
    angles = np.linspace(0,2*np.pi,num_rays)
    angles = np.vstack([np.sin(angles),np.cos(angles)]).T.astype(np.float32)
    
    ### k-d tree query
    tree = KDTree(angles,leafsize=8)
    rel_positions = valid_pts-origin
    distances = np.linalg.norm(rel_positions,axis=1,keepdims=True)
    rel_dirs = rel_positions/(distances+1e-6)
    dist_mask = (distances[:,0] <= max_dist)
    rel_dirs = rel_dirs[dist_mask]
    rel_positions = rel_positions[dist_mask]
    distances = distances[dist_mask]
    _, idx = tree.query(rel_dirs, k=1)
    dist_per_bin = np.full(num_rays, 1e3) 
    np.minimum.at(dist_per_bin, idx, distances[:,0])
    dist_per_bin[dist_per_bin>max_dist] = max_dist
    known_pts = (angles*dist_per_bin[:,None])+origin
    polygon = Polygon(known_pts)
    if kwargs.return_pts == True:
        return (polygon.area, known_pts)
    else:
        return polygon.area

##TODO: adaptive higher resolution on closer curves, knn5 interpolation
#number of rays increases -> res increases, spacing decreases
#number of rays increases -> maximum distance increases
#proximity_mask = (segments_dists[:,0] <= num_rays/60) | (segments_dists[:,1] <= num_rays/60)