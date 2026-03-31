# /------------------------------------------------------------/
# **2D Area Calculation Algorithm: 'Segments-Angle'**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a visibility area calculation algorithm which relies
# on segment intersection using Cramer's rule using array
# operations. It takes in line segments and returns the area of
# a visibility polygon generated with shapely.
# -----
# Best use case: accurate visibility polygon generation.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
from shapely.geometry import Polygon

## 2. Base Method
def visibility_area_np(segments, vantage_point, max_distance=100.0, num_rays=3600, **kwargs):
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
    hit_pts = []

    for θ in angles:
        ### Ray direction and endpoint
        dir_vec = np.array([np.cos(θ), np.sin(θ)])
        ray_endpoint = P + dir_vec * max_distance

        ### Right‑hand side: A − P
        rhs = A - P  # shape (n_segments, 2)

        ### Compute determinant of the 2×2 system for each segment:
        # det = dir_x * (−D_y) − dir_y * (−D_x)
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

        ### Keep intersections with 0 ≤ u ≤ 1 (on segment) and 0 < t < max_distance
        mask = (t > 0) & (t < max_distance) & (u >= 0) & (u <= 1)
        if np.any(mask):
            t_min = t[mask].min()
            hit = P + dir_vec * t_min
        else:
            hit = ray_endpoint
        hit_pts.append((hit[0], hit[1]))

    visibility_poly = Polygon(hit_pts)
    if kwargs.get("return_pts", False):
        return (visibility_poly.area, np.asarray(hit_pts, dtype=float))
    return visibility_poly.area
