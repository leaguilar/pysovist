# /------------------------------------------------------------/
# **Data Import Helper Function: JSON**
# -----
# *pysovist-dev* under MIT License
# -----
# This is a helper function which imports lines of a floor
# plan stored in a JSON and adds them to the area object.
# -----
# Best use case: discretized floor plans saved as JSON.
# -----
# What's in the file:
# 1. imports
# 2. base method
# /------------------------------------------------------------/

## 1. Imports
import numpy as np
import pandas as pd
from typing import List, Optional
from pathlib import Path

## 2. Base Method
def import_json(path:Path|str,delimiter:Optional[List[str]]=None):
    if delimiter is None:
        delimiter = ['start', 'end']
    lines_df = pd.read_json(path)
    lines_2d = []
    for row in lines_df.iterrows():
        start = row[1][delimiter[0]][:2]
        end = row[1][delimiter[1]][:2]
        lines_2d.append([start,end])
    lines_2d = np.array(lines_2d)
    return lines_2d # [N,2,2] array