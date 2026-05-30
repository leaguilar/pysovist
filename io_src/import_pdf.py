# /------------------------------------------------------------/
# **Data Import Helper Function: PDF**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which imports lines of a floor
# plan stored in a SVG and adds them to the area object.
# -----
# Best use case: vectorized floor plans in SVG files.
# -----
# Notes:
# * Please adjust the scale when importing the file.
# * Lines with high curvature may not be reproduced faithfully.
# -----
# What's in the file:
# 1. imports
# 2. helper functions
# 3. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path

## 2. Helper Functions
##TODO

## 3. Base Method
def import_svg(path:Path|str,scale,crop_extents:Optional[List[str]]=[0,1,0,1]):
    lines_df = pd.read_json(path)
    lines_2d = []
    for row in lines_df.iterrows():
        start = row[1][delimiter[0]][:2]
        end = row[1][delimiter[1]][:2]
        lines_2d.append([start,end])
    lines_2d = np.array(lines_2d)
    return lines_2d # [N,2,2] array