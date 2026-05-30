# /------------------------------------------------------------/
# **Data Import Helper Function: SVG**
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
from svgelements import SVG, Line, Polyline, Polygon, Rect, Path, Close

## 2. Helper Functions
def point_tuple(p):
    return (float(p.x), float(p.y))

def add_segment(segments, p1, p2, kind="line"):
    segments.append({
        "start": point_tuple(p1),
        "end": point_tuple(p2),
        "kind": kind,
    })

def sample_curve(segment, step):
    points = []
    steps = int(segment.length()/step+1)
    for i in range(steps + 1):
        t = i / steps
        p = segment.point(t)
        points.append(p)
    return points


def path_to_segments(path, curve_steps):
    segments = []
    for seg in path:
        # Straight SVG path commands: L, H, V usually become Line-like segments.
        if isinstance(seg, Line):
            add_segment(segments, seg.start, seg.end, kind="line")
        # Close path is usually a straight segment back to the subpath start.
        elif isinstance(seg, Close):
            add_segment(segments, seg.start, seg.end, kind="line")
        # Curves: CubicBezier, QuadraticBezier, Arc, etc.
        else:
            points = sample_curve(seg, step=curve_steps)
            for a, b in zip(points, points[1:]):
                add_segment(segments, a, b, kind="curve_sample")
    return segments

## 3. Base Method
def import_svg(path:Path|str,pagewidth:float,curve_step:float=0.2,crop_extents:Optional[List[str]]=[0,1,0,1]):
    svg = SVG.parse(path)
    scale = pagewidth/svg.width
    segments = []

    for element in svg.elements():
        if isinstance(element, Line):
            add_segment(segments, element.start, element.end, kind="line")
        elif isinstance(element, Polyline):
            points = list(element.points)
            for a, b in zip(points, points[1:]):
                add_segment(segments, a, b, kind="polyline")
        elif isinstance(element, Polygon):
            points = list(element.points)
            for a, b in zip(points, points[1:]):
                add_segment(segments, a, b, kind="polygon")
            if len(points) > 1:
                add_segment(segments, points[-1], points[0], kind="polygon")
        elif isinstance(element, Rect):
            x = float(element.x)
            y = float(element.y)
            w = float(element.width)
            h = float(element.height)
            corners = [(x, y),(x + w, y),(x + w, y + h),(x, y + h)]

            for a, b in zip(corners, corners[1:] + corners[:1]):
                segments.append({"start": a,"end": b,"kind": "rect"})

        elif isinstance(element, Path):
            segments.extend(path_to_segments(element, curve_steps=curve_step))

    lines_2d = np.array([[segment['start'],segment['end']] for segment in segments])*scale
    W, H = scale*svg.width, scale*svg.height
    w_min = crop_extents[0]*W
    w_max = crop_extents[1]*W
    h_min = crop_extents[2]*H
    h_max = crop_extents[3]*H
    lines_mask = ((lines_2d[...,0] >= w_min) & (lines_2d[...,0] <= w_max) & (lines_2d[...,1] >= h_min) & (lines_2d[...,1] <= h_max))
    array_mask = lines_mask.all(axis=1)
    return lines_2d[array_mask] # [N,2,2] array


