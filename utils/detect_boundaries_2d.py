# /------------------------------------------------------------/
# **Floor Plan Boundary Detection**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which identifies indoor spaces and
# assigns a navigable area to them.
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
def visibility_valid(segments, vantage_point, max_distance=100.0, num_rays=36):
    x0, y0 = vantage_point
    P = np.array([x0, y0])
    segments_diffs = segments-vantage_point
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_distance) | (segments_dists[:,1] <= max_distance)
    segments = segments[segment_mask]
    ### Extract 2D segment endpoints (n_segments × 2)
    A = segments[:, 0]
    B = segments[:, 1]
    D = B - A # segment direction vectors (n_segments × 2)

    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    valids = []

    for θ in angles:
        dir_vec = np.array([np.cos(θ), np.sin(θ)])
        ray_endpoint = P + dir_vec * max_distance

        rhs = A - P  # shape (n_segments, 2)

        det = dir_vec[0] * (-D[:, 1]) - dir_vec[1] * (-D[:, 0])

        ### Cramer's numerators:
        # det_t = (A_x − x0)*(-D_y) − (A_y − y0)*(-D_x)
        det_t = rhs[:, 0] * (-D[:, 1]) - rhs[:, 1] * (-D[:, 0])
        # det_u = dir_x*(A_y − y0) − dir_y*(A_x − x0)
        det_u = dir_vec[0] * rhs[:, 1] - dir_vec[1] * rhs[:, 0]

        ### Solve t and u where valid
        valid = det != 0
        t = np.full_like(det, np.inf, dtype=float)
        u = np.full_like(det, -1.0, dtype=float)

        t[valid] = det_t[valid] / det[valid]
        u[valid] = det_u[valid] / det[valid]

        hit_mask = (t > 0) & (t < max_distance) & (u >= 0) & (u <= 1)
        valids.append(hit_mask.sum())

    valids = np.asarray(valids)
    return np.count_nonzero(valids)/num_rays


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