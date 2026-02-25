# /------------------------------------------------------------/
# **Import Helper: Rhino Files**
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
# 2. helper function to find matching layers
# 3. helper function to find layer index for matching layers
# 4. main function
# /------------------------------------------------------------/

## 1. Imports
import json
import math
import numpy as np
import os
from pathlib import Path
import rhino3dm

#workflow
#1. use lines/curves from selected layer
#2. (optional) write to json
#3. output ndarray

## 2. Layer Matching Function
def _layer_matches(layer, name_or_path: str) -> bool:
    # Match either short name or full path when available (Parent::Child)
    if layer is None:
        return False
    if getattr(layer, "Name", None) == name_or_path:
        return True
    full_path = getattr(layer, "FullPath", None)
    if full_path and full_path == name_or_path:
        return True
    return False

## 3. Layer Index Finder
def _find_layer_index(model, layer_name: str) -> int:
    # Returns layer index in model.Layers or -1 if not found
    for i in range(model.Layers.Count):
        layer = model.Layers[i]
        if _layer_matches(layer, layer_name):
            return i
    return -1

## 4. Base Import Function
def from_rhino(filepath:str,layer_name:str,save_path:None|str):
    """
    Inputs:
        PLines: List of polylines
        Path: File path for JSON export (string)
        L: Length of marker lines
    Outputs:
        JSON_out: The JSON string (for preview)
        Lines_out: The marker lines (Rhino geometry)
    """
    p = Path(filepath).expanduser()
    if not p.exists():
        raise FileNotFoundError

PLines = x
all_lines, all_segments, json_entries = [], [], []

if PLines:
    for pl in PLines:
        #print(type(pl))
        # Ensure we have a Polyline object
        if isinstance(pl, rg.PolyCurve):
            pl = pl.ToPolyline(0,3,math.pi,0.5,1,0,0,0,True)
        if isinstance(pl, rg.PolylineCurve):
            pl = pl.ToPolyline()
        if isinstance(pl, rg.Polyline):
            sub_pls = pl.BreakAtAngles(1e-3)
            for sub in sub_pls:
                pts = list(sub)
                for i in range(len(pts)-1):
                    p0, p1 = pts[i], pts[i+1]
                    line = rg.Line(p0, p1)
                    all_lines.append(line)
                    json_entries.append({
                        "start": [p0.X, p0.Y, p0.Z],
                        "end": [p1.X, p1.Y, p1.Z]
                    })

# Build JSON
JSON_out = json.dumps(json_entries, indent=4)

Path = '/Users/bartutuncay/cog/eth3d/anlieferung/anlieferung_plan.json'
folder = os.path.dirname(Path)

# Write to file if path is given
#try:
folder = os.path.dirname(Path)
with open(Path, 'w') as f:
    f.write(JSON_out)
#except Exception as e:
#    print("Error writing file:", e)

######

if task.empty == False:
    N = int(task['dirs'])
    D = int(task['max_distance'])
    M = task['method'].values[0]
    print(N,D,M)

    ## Floor plan to list
    lines_json = '../lines_hospital.json'
    lines_df = pd.read_json(lines_json)
    lines_2d = []
    for row in lines_df.iterrows():
        start = row[1]['start'][:2]
        end = row[1]['end'][:2]
        lines_2d.append([start,end])
    lines_2d = np.array(lines_2d)

    