# /------------------------------------------------------------/
# **Data Import Helper Function: Image**
# -----
# *pysovist-dev* under MIT License
# -----
# This is an experimental helper function which imports image
# files with floor plans and converts detected lines into
# line instances usable by the Data2D class.
# -----
# Best use case: images of floor plans; PNG, JPG etc.
# -----
# Notes:
# * Please adjust the scale when importing the file.
# * Parameters may need tuning to work as intended.
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