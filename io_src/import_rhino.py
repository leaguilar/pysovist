# /------------------------------------------------------------/
# **Data Import Helper Function: Rhino**
# -----
# *pysovist-dev* under MIT License
# -----
# This helper imports Rhino curve geometry, optionally filters
# by layer, optionally exports the extracted segments as JSON,
# and returns a [N,2,2] NumPy array for 2D workflows.
# /------------------------------------------------------------/

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rhino3dm


def _layer_matches(layer: Any, name_or_path: str) -> bool:
    if layer is None:
        return False
    if getattr(layer, "Name", None) == name_or_path:
        return True
    full_path = getattr(layer, "FullPath", None)
    return bool(full_path and full_path == name_or_path)


def _iter_collection(collection: Any):
    count = getattr(collection, "Count", None)
    if count is not None:
        for index in range(count):
            yield collection[index]
        return
    yield from collection


def _find_layer_index(model: rhino3dm.File3dm, layer_name: str) -> int:
    for index, layer in enumerate(_iter_collection(model.Layers)):
        if _layer_matches(layer, layer_name):
            return index
    return -1


def _point_to_xyz(point: Any) -> list[float]:
    return [float(point.X), float(point.Y), float(point.Z)]


def _polyline_segments(polyline: Any) -> list[list[list[float]]]:
    points = [_point_to_xyz(point) for point in polyline]
    return [[start, end] for start, end in zip(points[:-1], points[1:])]


def _try_get_polyline_segments(geometry: Any) -> list[list[list[float]]]:
    try_get_polyline = getattr(geometry, "TryGetPolyline", None)
    if try_get_polyline is None:
        return []

    result = try_get_polyline()
    if isinstance(result, tuple) and len(result) == 2:
        ok, polyline = result
        if ok:
            return _polyline_segments(polyline)
    elif result is not None:
        return _polyline_segments(result)
    return []


def _line_like_segments(geometry: Any) -> list[list[list[float]]]:
    start = getattr(geometry, "PointAtStart", None)
    end = getattr(geometry, "PointAtEnd", None)
    if start is None or end is None:
        return []
    return [[_point_to_xyz(start), _point_to_xyz(end)]]


def _extract_segments(geometry: Any) -> list[list[list[float]]]:
    segments = _try_get_polyline_segments(geometry)
    if segments:
        return segments
    return _line_like_segments(geometry)


def from_rhino(
    filepath: str,
    layer_name: str | None = None,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Read Rhino geometry and return 2D line segments as a [N,2,2] array.
    """
    path = Path(filepath).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Rhino file not found: {path}")

    model = rhino3dm.File3dm.Read(str(path))
    if model is None:
        raise ValueError(f"Could not read Rhino file: {path}")

    layer_index = None
    if layer_name is not None:
        layer_index = _find_layer_index(model, layer_name)
        if layer_index < 0:
            raise ValueError(f"Layer '{layer_name}' was not found in {path.name}.")

    segments_3d: list[list[list[float]]] = []
    for obj in _iter_collection(model.Objects):
        if layer_index is not None and obj.Attributes.LayerIndex != layer_index:
            continue
        segments_3d.extend(_extract_segments(obj.Geometry))

    if not segments_3d:
        layer_label = layer_name if layer_name is not None else "all layers"
        raise ValueError(f"No supported line or polyline geometry found in {layer_label}.")

    segments_3d_arr = np.asarray(segments_3d, dtype=float)
    lines_2d = segments_3d_arr[:, :, :2]

    if save_path is not None:
        output_path = Path(save_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_entries = [
            {"start": start.tolist(), "end": end.tolist()}
            for start, end in segments_3d_arr
        ]
        output_path.write_text(json.dumps(json_entries, indent=4), encoding="utf-8")

    return lines_2d
