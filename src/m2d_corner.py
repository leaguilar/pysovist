# /------------------------------------------------------------/
# **2D Area Calculation Algorithm: 'Corner'**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a quick visibility area calculation algorithm which
# takes advantage of the JSON data format and uses endpoints 
# of line segments to return the area of an approximate
# visibility polygon, projected onto 2π degrees.
# -----
# Best use case: quick approximation, very large floor plans.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np

## 2. Base Method
def visibility_polygon_corner(segments:np.ndarray, origin, max_distance = 100, num_samples = 120, **kwargs):
    segments = np.array(segments, dtype=float) # shape [N,2,2]
    segments_diffs = segments-origin
    segments_dists = np.hypot(segments_diffs[:,:,0],segments_diffs[:,:,1])
    segment_mask = (segments_dists[:,0] <= max_distance) | (segments_dists[:,1] <= max_distance)
    valid_segments = segments_diffs[segment_mask]
    valid_dists = segments_dists[segment_mask]
    
    ### break up long segments --> longer than 5 units, change if units are too large/small
    norms_curves = np.linalg.norm(valid_segments[:,1,:]-valid_segments[:,0,:],axis=1)
    max_subd = 5
    subdivision_mask = (norms_curves >= max_subd)
    long_segments = valid_segments[subdivision_mask]
    short_segments = valid_segments[~subdivision_mask]
    long_norms = norms_curves[subdivision_mask]
    short_dists = valid_dists[~subdivision_mask]
    n_segs  = np.ceil(long_norms / max_subd).astype(int)
    start = long_segments[:, 0, :]
    end   = long_segments[:, 1, :]
    vec   = end - start
    t = np.concatenate([np.linspace(0, 1, ns+1)[:-1] for ns in n_segs])
    t_next = np.concatenate([np.linspace(0, 1, ns+1)[1:] for ns in n_segs])
    start_rep = np.repeat(start, n_segs, axis=0)
    vec_rep   = np.repeat(vec, n_segs, axis=0)
    seg_start = start_rep + vec_rep * t[:, None]
    seg_end   = start_rep + vec_rep * t_next[:, None]
    long_subdivided = np.stack([seg_start, seg_end], axis=1)
    long_dists = np.hypot(long_subdivided[:,:,0],long_subdivided[:,:,1])
    valid_segments = np.vstack([short_segments, long_subdivided])
    valid_dists = np.vstack([short_dists, long_dists])
    segments_dirs = valid_segments/valid_dists[:,:,None]
    segments_angles = np.arctan2(segments_dirs[:,:,1], segments_dirs[:,:,0])
    
    ### adjust segments_angles such that col1 -> col2 := CCW
    angle_diff = (segments_angles[:,1] - segments_angles[:,0]) % (2*np.pi)
    swap_mask = angle_diff < 0
    swap_mask |= angle_diff > np.pi
    out = segments_angles.copy()
    out_d = valid_dists.copy()
    out[swap_mask, 0] = segments_angles[:,1][swap_mask]
    out[swap_mask, 1] = segments_angles[:,0][swap_mask]

    ### update distances array accordingly
    out_d[swap_mask,0] = valid_dists[:,1][swap_mask]
    out_d[swap_mask,1] = valid_dists[:,0][swap_mask]

    ### sort by angle CCW
    idx = np.argsort((out[:, 0] - 0.0) % (2*np.pi), kind='mergesort')
    out_dists = out_d[idx]
    out_angles = out[idx]

    ### if difference in a row >= π and (+) to (-) -> split and flip direction
    out_flip_mask = (out_angles[:,0] > out_angles[:,1]) & (out_angles[:,0]/out_angles[:,1] < 0) & (np.abs(out_angles[:,0]-out_angles[:,1])>=np.pi)
    ordered_angles = out_angles[~out_flip_mask]
    flipped_angles = out_angles[out_flip_mask]
    ordered_dists = out_dists[~out_flip_mask]
    flipped_dists = out_dists[out_flip_mask]
    n = flipped_angles.shape[0]
    flipped_1 = np.column_stack([flipped_angles[:, 0], np.pi * np.ones(n)])
    flipped_2 = np.column_stack([-np.pi * np.ones(n), flipped_angles[:, 1]])
    flipped_angles = np.vstack([flipped_1, flipped_2])
    flipped_d1 = np.column_stack([flipped_dists[:, 0], flipped_dists[:,1]])
    flipped_d2 = np.column_stack([flipped_dists[:, 0], flipped_dists[:,1]])
    flipped_dists = np.vstack([flipped_d1, flipped_d2])
    out_angles = np.vstack([ordered_angles,flipped_angles])
    out_dists = np.vstack([ordered_dists,flipped_dists])
    
    ### calculate area: project onto a line 0 -> 2π
    tri_points1 = np.column_stack((out_angles[:,0],out_dists[:,0]))
    tri_points2 = np.column_stack((out_angles[:,1],out_dists[:,1]))
    tri_points = np.column_stack((tri_points1,tri_points2))
    tri_points = np.concatenate((tri_points,np.array([[out_angles.min(), max_distance, out_angles.max(), max_distance]])))
    segments = list(zip(tri_points[:,:2], tri_points[:,2:]))
    
    ### intersection of intervals -> select lower bound
    xs_all = [x for seg in segments for (x,_) in seg]
    xmin, xmax = min(xs_all), max(xs_all)
    xs = np.linspace(xmin, xmax, num_samples)
    ys = np.full_like(xs, np.inf, dtype=float)

    for (x1, y1), (x2, y2) in segments:
        mask = (xs >= min(x1,x2)) & (xs <= max(x1,x2))
        if x2 != x1:
            slope = (y2-y1)/(x2-x1)
            ys_seg = y1 + slope*(xs[mask]-x1)
        else:
            ys_seg = np.full(np.sum(mask), min(y1,y2))
        ys[mask] = np.minimum(ys[mask], ys_seg)
    ys_clipped = np.clip(ys, 0, None)
    #area = np.trapz(ys_clipped, xs)
    area = (ys_clipped**2).mean()*np.pi

    if kwargs.return_pts == True:
        r = ys_clipped
        theta = xs

        x = origin[0] + r * np.cos(theta)
        y = origin[1] + r * np.sin(theta)

        hit_pts_calc = np.column_stack([x, y])
        return (area, hit_pts_calc)

    return area

##TODO:
#add FOV truncation
#vars: view_center=np.pi,fov_x=2*np.pi