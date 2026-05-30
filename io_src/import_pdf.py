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
from math import ceil, hypot
import fitz

## 2. Helper Functions
def cubic_point(p0, p1, p2, p3, t):
    u = 1 - t
    x = (u**3*p0.x+3*u**2*t*p1.x+3*u*t**2*p2.x+t**3*p3.x)
    y = (u**3*p0.y+3*u**2*t*p1.y+3*u*t**2*p2.y+t**3*p3.y)
    return (x, y)

def sample_cubic(p0, p1, p2, p3, curve_step=10.0):
    length = (
        hypot(p1.x - p0.x, p1.y - p0.y) +
        hypot(p2.x - p1.x, p2.y - p1.y) +
        hypot(p3.x - p2.x, p3.y - p2.y)
    )

    steps = max(1, ceil(length / curve_step))

    return [
        cubic_point(p0, p1, p2, p3, i / steps)
        for i in range(steps + 1)
    ]

## 2. Base Method
def import_pdf(path:Path|str,pagewidth:float=20,page:int=1,curve_step:float=0.2,crop_extents:Optional[List[str]]=[0,1,0,1]):
    doc = fitz.open(path)
    segments = []
    page = doc[page-1]
    scale = pagewidth/page.rect.width
    curve_step = curve_step/scale
    drawings = page.get_drawings()

    for drawing in drawings:
        for item in drawing["items"]:
            cmd = item[0]

            if cmd == "l":
                _, p1, p2 = item
                segments.append({"start":(p1.x, p1.y),"end":(p2.x, p2.y),"kind":"line"})

            elif cmd == "re":
                _, rect, *_ = item

                corners = [(rect.x0, rect.y0),(rect.x1, rect.y0),(rect.x1, rect.y1),(rect.x0, rect.y1)]

                for a, b in zip(corners, corners[1:] + corners[:1]):
                    segments.append({"start":a,"end":b,"kind":"rect"})

            elif cmd == "c":
                _, p0, p1, p2, p3 = item
                points = sample_cubic(p0, p1, p2, p3, curve_step)

                for a, b in zip(points, points[1:]):
                    segments.append({
                        "start": a,
                        "end": b,
                        "kind": "curve_sample",
                    })

            elif cmd == "qu":
                _, quad = item

                points = [quad.ul, quad.ur, quad.lr, quad.ll]

                for a, b in zip(points, points[1:] + points[:1]):
                    segments.append(((a.x, a.y), (b.x, b.y)))

    lines_2d = np.array([[segment['start'],segment['end']] for segment in segments])*scale
    W, H = scale*page.rect.width, scale*page.rect.height
    w_min = crop_extents[0]*W
    w_max = crop_extents[1]*W
    h_min = crop_extents[2]*H
    h_max = crop_extents[3]*H
    lines_mask = ((lines_2d[...,0] >= w_min) & (lines_2d[...,0] <= w_max) & (lines_2d[...,1] >= h_min) & (lines_2d[...,1] <= h_max))
    array_mask = lines_mask.all(axis=1)
    return lines_2d[array_mask] # [N,2,2] array