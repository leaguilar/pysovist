# /--------------------------------------------------/
# **2D Visible Area Calculation: Wrapper Function**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a wrapper function which brings together
# methods for calculating visible area for floor
# plans, for a single point.
# -----
# Best use case: visible area calculation for a
# single point
# -----
# What's in the file:
# 1. imports
# 2. base class 'area'
# 2.1. area.single_point
# 2.2. area.visibility_polygon
# 2.3. area.area_array
# 2.4. area.boundary
# - wrapper
# /--------------------------------------------------/

## 1. Imports
import numpy as np
from typing import Optional

from m2d_corner import visibility_polygon_corner
from m2d_segments_angle import visibility_area_np
from m2d_discretized import visibility_discretized

## 2. Base Class
class area:
    ## 2.1. Single Point Calculation
    def single_point(dist_max:float, N:int, origin: np.ndarray, segments: np.ndarray, FOV:float|None, view_dir:float|None|np.ndarray,**kwargs):
        """
        **2D Visible Area Wrapper Function:** *compute_visibility_2d*
        
        :param dist_max: Maximum view radius
        :type dist_max: float
        :param N: Number of rays to cast
        :type N: int
        :param origin: Vantage point, 2-long array
        :type origin: np.ndarray
        :param input: Input lines, [N,2,2] array
        :type input: np.ndarray
        :param FOV: Degree of view, set to 2π by default. *Disabled if field is left empty*
        :type FOV: float | None
        :param view_dir: View direction for each point. *Only enabled if FOV is enabled.* Can be unit vector or in radians.
        :type view_dir: float | None | np.ndarray
        :param kwargs: Additional keyword arguments. *See documentation for details.*
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError('Segments should be a [N,2,2] NumPy array; please check input/preprocessing.')
        if segments.shape[1:] != (2,2):
            raise ValueError(f"Segments should be a [N,2,2] NumPy array; got {segments.shape}")
        if not isinstance(origin, np.ndarray):
            raise TypeError('Origin should be a 2-long NumPy array; please check input/preprocessing.')
        if origin.ndim != 2:
            raise ValueError(f"Origin should be a 2-long NumPy array; got {origin.shape}")
        
        if FOV != None and type(FOV) != float:
            raise TypeError('FOV should be empty or a float value.')
        if view_dir != None and type(view_dir) != float or view_dir != None and type(view_dir) != np.ndarray:
            raise TypeError('View direction should either be a float value or a unit vector expressed as a 2-long NumPy array.')
        if view_dir != None and np.linalg.norm(view_dir) != 1:
            norm = np.linalg.norm(view_dir)
            view_dir /= norm
        if view_dir != None and np.linalg.norm(view_dir) == 0:
            raise ValueError('View direction cannot be all zeros.')

        methods_list = ['corner', 'discretized', 'segments_angle']
        if kwargs.method != None and kwargs.method not in methods_list:
            raise NameError(f'Method should be one of: f{methods_list} or left empty.')
        
        if kwargs.method == 'corner':
            area = visibility_polygon_corner(segments,origin,dist_max,N,kwargs)
        elif kwargs.method == 'discretized':
            area = visibility_discretized(segments,origin,dist_max,N,kwargs)
        else:
            area = visibility_area_np(segments,origin,dist_max,N,kwargs)
        
        
        if kwargs.return_stats == True:
            ###TODO: Calculate min-max, mean, standard deviation for hit point distances
            ## space syntax stats: depth,
            continue

        if kwargs.return_pts == True:
            area, pts = area
            return area, pts
        else:
            return area
    
    ## Visibility Polygon 
    def visibility_polygon():
        return
    def area_array(**kwargs): # calculate area for a given area, defined by corner points
        if kwargs.two_step == True:
            ##TODO: 2-step calculation corner->segments-angle
            continue
        return
    def boundary(): # calculate area for a given area, defined by bounding lines
        if kwargs.two_step == True:
            continue
        return