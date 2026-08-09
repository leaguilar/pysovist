# /------------------------------------------------------------/
# **Data Import Helper Function: CSV**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which imports lines of a floor
# plan stored in a CSV file and adds them to the area object.
# -----
# Best use case: discretized floor plans saved as CSV.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path

## 2. Base Method
def import_csv(path:Path|str,start:Optional[Tuple]=(0,1),end:Optional[Tuple]=(3,4),delimiter:Optional[str]=','):
    lines_df = pd.read_csv(path,delimiter=delimiter)
    lines_2d = []
    for row in lines_df.iterrows():
        start_val = (row[1][start[0]],row[1][start[1]])
        end_val = (row[1][end[0]],row[1][end[1]])
        lines_2d.append([start_val,end_val])
    lines_2d = np.array(lines_2d)
    return lines_2d # [N,2,2] array

def import_csv_3d(path:Path|str,delimiter:Optional[str]=',',xyz_columns:Optional[tuple]=(0,1,2),color_cols:Optional[tuple]=(3,4,5)):
    pts = pd.read_csv(path, delimiter=delimiter).to_numpy()
    pts_xyz = pts[xyz_columns]
    if color_cols != None:
        pts_rgb = pts[color_cols]
        pts_3d = np.stack(pts_xyz,pts_rgb,axis=1)
    else:
        pts_3d = pts_xyz
    return pts_3d # [N,3] or [N,6] array