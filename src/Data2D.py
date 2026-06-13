# /------------------------------------------------------------/
# **2D Data Class**
# -----
# *pysovist-dev* under MIT License
# -----
# This module defines the dictionary-like data class used for
# 2D visibility calculations and their related workflows.
# /------------------------------------------------------------/

from __future__ import annotations

from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, List, Optional, Tuple
from pathlib import Path
import numpy as np
from .calculate_2d_sp import area_array, boundary, single_point

##TODO: desired metrics
#Core Set
#1. Angular Choice
#2. Angular Integration
#3. Connectivity
#4. Intelligibility
#5. Mean Depth

@dataclass
class Data2D(MutableMapping[str, Any]):
    data: dict[str, Any] = field(default_factory=dict)
    plan: np.ndarray | None = None
    
    results: dict[str, Any] = field(default_factory=dict)
    columns: list[str] | None = None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def import_json(self,path:Path|str,delimiter:Optional[List]=None) -> None:
        '''
        Import JSON files
        ---
        <u>Inputs</u>

        - **Path to JSON file** | *str, required*
        - **Delimiter**: start-end points of lines. By default `['start', 'end']` | *list of str (len: 2), optional*
        '''
        from io_src.import_json import import_json
        array = import_json(path,delimiter)
        self.data['plan'] = array
        return

    def import_rhino(self,filepath:Path|str,layer_name:str='Default') -> None:
        '''
        Import lines from Rhino document
        ---
        <u>Inputs</u>

        - **Path to Rhino doc** | *str, required*
        - **Delimiter**: start-end points of lines. By default `['start', 'end']` | *list of str (len: 2), optional*
        '''
        from io_src.import_rhino import from_rhino
        array = from_rhino(filepath,layer_name)
        self.data['plan'] = array
        return
    
    def import_csv(self,filepath:Path|str,start_xy:Tuple|None,end_xy:Tuple|None,delimiter:Optional[str]=',') -> None:
        '''
        Import lines from comma-separated values file
        ---
        <u>Inputs</u>

        - **Path to CSV file** | *str, required*
        - **Start**: columns indicating starting point of lines in the XY plane. By default `0,1`; can be int or str | *tuple, optional*
        - **End**: column indicating starting point of lines in the XY plane. By default `2,3`; can be int or str | *tuple, optional*
        - **Delimiter**: CSV delimiter. By default `','` | *str, optional*
        '''
        from io_src.import_csv import import_csv
        array = import_csv(filepath,start_xy,end_xy,delimiter)
        self.data['plan'] = array
        return

    def import_svg(self,filepath:Path|str,pagewidth:float=21,curve_step:float=0.2,crop_extents:Optional[List[str]]=[0,1,0,1]) -> None:
        '''
        Import lines from scalable vector graphics file
        ---
        <u>Inputs</u>

        - **Path to SVG file** | *str, required*
        - **Page width**: plan width represented in the full page width. `e.g.` if the plan has `1:100` scale and is printed on portrait `A4` paper, width equals `21`. By default `21`; can be int or str | *float, optional*
        - **Curve step**: segment length for sampling curves into lines; increase if curvature is high. By default `0.2` | *float, optional*
        - **Crop extents**: area which will be processed in normalized page coordinates `[W_min,W_max,H_min,H_max]`; any segments with endpoints outside are removed. By default `0,1,0,1` | *list, optional*
        '''
        from io_src.import_svg import import_svg
        array = import_svg(filepath,pagewidth,curve_step,crop_extents)
        self.data['plan'] = array
        return
    
    def import_pdf(self,filepath:Path|str,pagewidth:float=21,page:int=1,curve_step:float=0.2,crop_extents:Optional[List[str]]=[0,1,0,1]) -> None:
        '''
        Import lines from PDF
        ---
        <u>Inputs</u>

        - **Path to PDF file** | *str, required*
        - **Page**: page number, index starts with 1. By default `1` | *int, optional*
        - **Page width**: plan width represented in the full page width. `e.g.` if the plan has `1:100` scale and is printed on portrait `A4` paper, width equals `21`. By default `21` | *float, optional*
        - **Curve step**: segment length for sampling curves into lines; increase if curvature is high. By default `0.2` | *float, optional*
        - **Crop extents**: area which will be processed in normalized page coordinates `[W_min,W_max,H_min,H_max]`; any segments with endpoints outside are removed. By default `0,1,0,1` | *list, optional*
        '''
        from io_src.import_pdf import import_pdf
        array = import_pdf(filepath,pagewidth,page,curve_step,crop_extents)
        self.data['plan'] = array
        return
    
    def import_image(self,filepath:Path|str,pagewidth:float=21,crop_extents:Optional[List[str]]=[0,1,0,1],res:float=50,thr:float=0.7) -> None:
        from io_src.import_image import import_img
        raster = import_img(filepath,pagewidth,crop_extents,res,thr)
        self.data['raster'] = raster #process raster with discretized method
        self.data['raster_res'] = res
        return

    def view_plan(self,dark:bool=False,**kwargs) -> None:
        from utils.visualize_2d import view_baseplan
        if 'raster' in self.data.keys():
            view_baseplan(self.data['raster'],dark,raster=True,res=self.data['raster_res'],**kwargs)
        else:
            view_baseplan(self.data['plan'],dark,**kwargs)
        return
    
    def view_areas(self,grid:np.array|None=None,show_grid:bool=False,dark:bool=False,**kwargs) -> None:
        grid = self.get_grid(grid)
        from utils.visualize_2d import view_areas
        if 'raster' in self.data.keys():
            view_areas(self.data['raster'],grid,dark,show_grid,raster=True,res=self.data['raster_res'],**kwargs)
        else:
            view_areas(self.data['plan'],grid,dark,show_grid,**kwargs)
        return
    
    def detect_boundaries(self,return_dense:bool=False,res:float=0.5) -> Any:
        if 'raster' in self.data.keys():
            from utils.detect_boundaries_fmm import fmm_edges, identify_noninf, densify_grid

        else:
            from utils.detect_boundaries_2d import identify_noninf, densify_grid
            grid, grid_value = identify_noninf(self.data['plan'],res)
        self.results['grid_bounds'] = grid
        self.results['grid_confidence'] = grid_value
        if return_dense == True:
            dense = densify_grid(grid)
            self.results['grid_dense'] = dense
        return
    
    def img_to_vec():
        return
    
    def get_grid(self, grid:np.array|None=None) -> np.array:
        if grid is None:
            grid = self.results.get('grid_dense')
            if grid is None:
                grid = self.results.get('grid_bounds')
                if grid is not None:
                    print('Using sparse grid. Run detect_boundaries with return_dense=True for higher resolution.')
                else:
                    print('No grid found. Run detect_boundaries or modify parameters.')
        return grid

    def dense_calc(self,grid:np.array|None=None,reduce_outliers:float|None=2,return_perimeter:bool=False) -> Any:
        '''
        Automatically detect grid and calculate metrics
        ---
        <u>Inputs</u>
        '''
        # if dense grid is available use it automatically
        grid = self.get_grid(grid)

        if 'raster' not in self.data.keys():
            m = 'discretized_img'
        else:
            m = 'corner'

        self.visibility_batch(100,3600,grid,self.data['plan'],fov=2*np.pi,method=m)
        outlier_mask = np.abs(self.results['area']['result'] - self.results['area']['result'].mean()) <= reduce_outliers * self.results['area']['result'].std()
        self.results['area'] = {k: v[outlier_mask] if isinstance(v, np.ndarray) and len(v) == len(outlier_mask) else v for k, v in self.results['area'].items()}
        if self.data['res'] != None:
            self.results['area'] = self.results['area']/self.data['res']**2
        #TODO: area calculation also returns:
        # perimeter
        # closed perimeter
        # compactness
        # occlusivity
        # vista length
        # drift
        # average radial
        # variance
        # skewness
        return
    
    def get_graph() -> Any:

        return
    
    def graph_centrality(self, **kwargs) -> Any:
        kwargs.method
        return

    def visibility(self,dist_max: float,
        N: int,origin: np.ndarray,
        segments: np.ndarray | None = None,
        FOV: float | None = None,
        view_dir: float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run a single-point 2D visibility workflow.
        """
        segments_arr = self._segments_or_default(segments)
        result = single_point(
            dist_max,
            N,
            origin,
            segments_arr,
            FOV=FOV,
            view_dir=view_dir,
            **kwargs,
        )
        self._store_result(
            workflow="single_point",
            result=result,
            dist_max=dist_max,
            N=N,
            FOV=FOV,
            method=kwargs.get("method", "segments_angle"),
        )
        return result

    def visibility_batch(
        self,
        dist_max: float,
        N: int,
        origins: np.ndarray,
        segments: np.ndarray | None = None,
        FOV: float | None = None,
        view_dir: float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the batched 2D visibility workflow.
        """
        segments_arr = self._segments_or_default(segments)
        result = area_array(
            dist_max,
            N,
            origins,
            segments_arr,
            FOV=FOV,
            view_dir=view_dir,
            **kwargs,
        )

        record = {"X": origins[:,0], "Y": origins[:,1], "result": result, 'dist_max':dist_max,
            'N':N, 'FOV':FOV, 'method':kwargs.get("method", "corner")}
        self.results['area'] = record
        return result

    def calculate_boundary(
        self,
        dist_max: float,
        N: int,
        boundary_segments: np.ndarray,
        segments: np.ndarray | None = None,
        FOV: float | None = None,
        view_dir: float | np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the boundary-sampling 2D visibility workflow.
        """
        scene_segments = self._segments_or_default(segments)
        result = boundary(
            dist_max,
            N,
            boundary_segments,
            segments=scene_segments,
            FOV=FOV,
            view_dir=view_dir,
            **kwargs,
        )
        self._store_result(
            workflow="boundary",
            result=result,
            dist_max=dist_max,
            N=N,
            FOV=FOV,
            method=kwargs.get("method", "segments_angle"),
        )
        return result

    def _segments_or_default(self, segments: np.ndarray | None) -> np.ndarray:
        if segments is not None:
            return np.asarray(segments, dtype=float)
        if self.array is None:
            raise ValueError("No segments provided and no array has been imported.")
        return np.asarray(self.array, dtype=float)

    def _store_result(self, *, workflow: str, result: Any, **metadata: Any) -> None:
        record = {"workflow": workflow, "result": result, **metadata}
        self.results.append(record)
        self.data["last_result"] = record

    area_array = visibility
    boundary = calculate_boundary
