# /--------------------------------------------------/
# **3D Visible Volume Calculation: Wrapper Function**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a wrapper function which brings together
# methods for calculating visible volume for point
# clouds or meshes, for a single point.
# -----
# Best use case: visible volume calculation for a
# single point
# -----
# What's in the file:
# 1. imports
# 2. base class
# 2.1. volume.single_point
# 2.2. volume.visibility_p
# - wrapper
# /--------------------------------------------------/

## 1. Imports
import numpy as np
from typing import Optional

from m3d_spherical import visibility_spherical

## 2. Base Class
class volume:
    # 2.1. Single Point Calculation
    def single_point(dist_min:float, dist_max:float, N:int, origin: np.ndarray, pcd_points: np.ndarray, FOV:None|np.ndarray, view_dir:None|np.ndarray,**kwargs):
        """
        **3D Visible Volume Wrapper Function:** *compute_visibility_3d*
        
        :param dist_min: Minimum view radius
        :type dist_min: float
        :param dist_max: Maximum view radius
        :type dist_max: float
        :param N: Number of rays to cast
        :type N: int
        :param origin: Vantage point, 3-long array
        :type origin: np.ndarray
        :param input: Input points, [N,3] array
        :type input: np.ndarray
        :param FOV: Degree of view in radians, set to (2π,2π) by default. *Disabled if field is left empty*
        :type FOV: np.ndarray | None
        :param view_dir: View direction for each point, in unit vector format. *Only enabled if FOV is enabled.*
        :type view_dir: np.ndarray | None
        :param kwargs: Additional keyword arguments. *See documentation for details.*
        """
        if not isinstance(pcd_points, np.ndarray):
            raise TypeError('Points should be a [N,3] NumPy array; please check input/preprocessing.')
        if pcd_points.shape[1] != 3:
            raise ValueError(f"Segments should be a [N,3] NumPy array; got {pcd_points.shape}")
        if not isinstance(origin, np.ndarray):
            raise TypeError('Origin should be a 3-long NumPy array; please check input/preprocessing.')
        if origin.ndim != 3:
            raise ValueError(f"Origin should be a 3-long NumPy array; got {origin.shape}")
        
        if FOV != None and type(FOV) != float:
            raise TypeError('FOV should be empty or a 2-long NumPy array in [rad].')
        if view_dir != None and type(view_dir) != float or view_dir != None and type(view_dir) != np.ndarray:
            raise TypeError('View direction should be a vector expressed as a 3-long NumPy array.')
        if view_dir != None and np.linalg.norm(view_dir) != 1:
            norm = np.linalg.norm(view_dir)
            view_dir /= norm
        if view_dir != None and np.linalg.norm(view_dir) == 0:
            raise ValueError('View direction cannot be all zeros.')


        methods_list = ['corner', 'discretized', 'segments_angle']
        if kwargs.method != None and kwargs.method not in methods_list:
            raise NameError(f'Method should be one of: f{methods_list} or left empty.')
        

        if kwargs.method == 'discretized':
            area = visibility_polygon_corner(segments,origin,dist_max,N,kwargs)
        elif kwargs.method == 'cvxhull':
            vol = 
        elif kwargs.method == 'meshes':
            area = visibility_discretized(segments,origin,dist_max,N,kwargs)
        else:
            vol = visibility_spherical(pcd_points, origin:np.ndarray, max_unknown:bool, min_distance = 0, max_distance = 40, num_samples = 1200, **kwargs):
            