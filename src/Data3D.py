# /------------------------------------------------------------/
# **3D Data Class**
# -----
# *pysovist* under MIT License
# -----
# This module defines the dictionary-like data class used for
# 3D visibility calculations and their related workflows. Most
# methods in this script are an extension of the established
# 2-dimensional workflows from literature. For detailed 
# descriptions of the methods, see the documentation.
# /------------------------------------------------------------/

from __future__ import annotations

from collections.abc import Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, List, Optional, Tuple
from pathlib import Path
import open3d as o3d
import numpy as np

@dataclass
class Data3D(MutableMapping[str, Any]):
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
        return

    def import_pcd(self,filepath:Path|str,lower_xyz:Tuple|None,upper_xyz:Tuple|None,delimiter:Optional[str],xyz_columns:Optional[tuple],downsampling_voxel:Optional[float],
                   offset:Optional[tuple]=(0,0,0),downsampling_uniform:Optional[int]=1,color:Optional[tuple]=None,
                   scale:Optional[float]=1.0,reduce_outliers:Optional[float]=2.0,**kwargs) -> None:
        '''
        Import point clouds from document
        ---
        <u>Inputs</u>

        - **Path to file**. Supported formats: `{XYZ-(N)/(RGB),PTS,PLY,PCD,CSV,TXT}`. Text-based formats handled by Pandas, remaining by Open3D. | *str, required*
        - **Lower XYZ**: lower bound of the evaluated volume in `X,Y,Z` coordinates. Disabled by default; accepts `(float,float,float)` | *tuple, optional*
        - **Upper XYZ**: upper bound of the evaluated volume in `X,Y,Z` coordinates. Enabled if `lower_xyz` is present; accepts `(float,float,float)` | *tuple, optional*
        - **Delimiter**: CSV delimiter. By default `','` | *str, optional*
        - **XYZ columns**: columns of the text file indicating `X,Y,Z` values of points. By default `0,1,2`; enabled if file is a text file | *tuple, optional*
        - **Offset**: distance by which to shift the point cloud in `X,Y,Z` format; accepts `(float,float,float)` | *tuple, optional*
        - **Downsampling**: reduction factor for uniform downsampling. Indicates ratio of kept points to original points. By default `1.0` | *float, optional*
        - **Color**: columns of text file indicating `R,G,B`color data if available. By default `None`; accepts `(int,int,int)` or `(str,str,str)` | *tuple, optional*
        - **Scale**: scaling factor by which to enlarge/shrink the point cloud. Downstream calculations depend on new coordinates; by default `1.0` | *float, optional*
        - **Reduce outliers**: reduction factor for eliminating point cloud outliers, in standard deviation ratio. By default `2.0`, higher values relax filtering | *float, optional*

        '''
        
        if any(item in str(filepath) for item in ['.txt','.csv']):
            from io_src.import_csv import import_csv_3d
            from io_src.import_json import import_json_3d
            try:
                pts_3d = import_csv_3d(filepath,delimiter,xyz_columns,color)
            except:
                pts_3d = import_json_3d(filepath,xyz_columns,color)
                print('Provided file is in serial format; using JSON helper.')
            finally:
                print(f'File {filepath} is not in a recognized format.')
        else:
            try:
                pcd = o3d.io.read_point_cloud(filepath)
                pts_3d = pcd.points
            except:
                print(f'File {filepath} is not in a recognized format. Only (.xyz, .xyzn, .xyzrgb, .pts, .ply, .pcd, .csv, .txt) accepted')

            if not lower_xyz == None:
                pt_mask = (pts_3d[:,0]>=lower_xyz[0])&(pts_3d[:,1]>=lower_xyz[1])&(pts_3d[:,2]>=lower_xyz[2])&(pts_3d[:,0]<=upper_xyz[0])&(pts_3d[:,1]<=upper_xyz[1])&(pts_3d[:,2]<=upper_xyz[2])
                pts_xyz = pts_3d[pt_mask,:2]
                if color != None:
                    pts_rgb = pts_3d[pt_mask,3:]
            else:
                pts_xyz = pts_3d[:,:2]
                if color != None:
                    pts_rgb = pts_3d[:,3:]
            if not offset == (0,0,0):
                pts_xyz += offset
            pts_cen = np.array([pts_xyz[:,0].mean(),pts_xyz[:,1].mean(),pts_xyz[:,2].mean()])
            pts_dists = pts_xyz-pts_cen
            pts_dists_scaled = pts_dists*scale
            pts_scaled = pts_dists_scaled+pts_cen

            opcd = o3d.geometry.PointCloud()
            opcd.points = o3d.utility.Vector3dVector(pts_scaled.asype(np.float64))
            if color != None:
                pts_rgb255 = pts_rgb/((pts_rgb.max()-pts_rgb.min())/255)-pts_rgb.min()
                opcd.colors = o3d.utility.Vector3dVector(pts_rgb255.asype(np.float64))

            #downsampling routine --> uses open3d
            opcd = opcd.uniform_down_sample(every_k_points=downsampling_uniform)
            if downsampling_voxel != None:
                opcd = opcd.voxel_down_sample(voxel_size=downsampling_voxel)
            #outliers routine --> uses open3d
            opcd.remove_statistical_outlier(nb_neighbors=kwargs.get('outliers_neighbors',20),std_ratio=reduce_outliers)

            self.data['pcd_points'] = opcd.points
            if color != None:
                self.data['pcd_colors'] = opcd.colors
            return

    def import_mesh(self,filepath:Path|str,lower_xyz:Tuple|None,upper_xyz:Tuple|None,offset:Optional[tuple],downsampling:Optional[float]=1.0,
                    color:Optional[bool]=False,scale:Optional[float]=1.0) -> None:
        '''
        - **Path to file**. Supported formats: `{PLY,STL,OBJ,OFF,GLTF/GLB}` | *str, required*
        - **Lower XYZ**: lower bound of the evaluated volume in `X,Y,Z` coordinates. Disabled by default; accepts `(float,float,float)` | *tuple, optional*
        - **Upper XYZ**: upper bound of the evaluated volume in `X,Y,Z` coordinates. Enabled if `lower_xyz` is present; accepts `(float,float,float)` | *tuple, optional*
        - **Offset**: distance by which to shift the point cloud in `X,Y,Z` format; accepts `(float,float,float)` | *tuple, optional*
        - **Downsampling**: reduction factor for uniform downsampling. Indicates ratio of kept points to original points. By default `1.0` | *float, optional*
        - **Color**: toggle to keep color data if available. By default `True` | *bool, optional*
        - **Scale**: scaling factor by which to enlarge/shrink the point cloud. Downstream calculations depend on new coordinates; by default `1.0` | *float, optional*
        '''
        if not any(filepath.endswith(i) for i in ['.ply','.stl','.obj','.off','.gltf','.glb']):
            raise ValueError('Data format is not supported. Choose from (.ply, .stl, .obj, .off, .gltf, .glb)')
        mesh = o3d.io.read_triangle_mesh(filepath)

        return

    def import_rhino() -> None:
        return

    def view() -> None:
        return

    def detect_boundaries():
        #TODO: returns a 3D hull
        return

    def get_grid():
        #TODO: define analysis coordinate system here
        return

    def get_graph():
        return

    def calc_vol():
        from m3d_cvxhull import visibility_spherical
        return


# TODO: 3d metrics
# visibility volume
# polygon area
# radial lengths: stats
# visibility centroid --drift
# compactness - 36πV^2/S^3
# isotropy (circularity)
# occlusivity
# jaggedness
# vertical openness: average elevation angle
# sky visibility factor?
# angular width --> solid angle
# visibility entropy: -∑(i)p_i x log(p_i)
# visibility tensor: ∫r^2uu^TdΩ
# volumetric connectivity
