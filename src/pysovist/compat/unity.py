"""The Unity view-volume estimator of the hospital study, as a preset.

The reference volumes were computed in a Unity scene that treats every scan
point as a solid ball and casts random rays from each eye point. This module
states that setup in pysovist terms, with the C# source and scene line that
fixes each choice.

Occluders
    Every point is a ball of radius 0.05 m (``pointRadius``, scene line 458),
    built from the transformed points, so the radius holds in the plan frame
    (``NativePointCloud.cs:107, 211``). A ray that starts inside a ball gets
    depth 0 (``NativePointCloud.cs:164-167``), so an eye inside a ball has V = 0.
Rays
    Directions are uniform on the sphere by rejection sampling in the unit
    cube (``PointCloudIsovistAnalysisRunner.cs:215-222``). Each eye received
    exactly 20070 rays: 10 per step (scene line 459) and 281020140 rays over
    14002 eyes (scene lines 466-468). No range limit and no near clip
    (``NativePointCloud.cs:38``, ``NativeOctreeRaycastQuery.cs:25``).
Seeds
    ``System.Random(1337)`` draws one seed per query file
    (``PointCloudIsovistAnalyzer.cs:88, 99``) and worker thread ``i`` starts
    from that seed plus ``i`` (``PointCloudIsovistAnalysisRunner.cs:70``).
    Eyes go to threads in batches of 32 (line 101), so the directions of an
    eye depend on thread scheduling and cannot be replayed. The preset's seed
    1337 is a convention: parity with Unity is statistical.
Escaping rays
    A ray that hits nothing returns 1e31 (``NativePointCloud.cs:43``). The
    writer drops depths above 1e9 from the sum (``IsovistDataWriter.cs:18``,
    scene line 239, ``IsovistExtensions.cs:30``) but keeps them in the count
    (``IsovistExtensions.cs:44``), so they contribute 0:
    V = (4 pi / 3) mean(r^3) with escaped r = 0 (``IsovistExtensions.cs:35-44``).
Frame
    ``.pts`` rows ``x y z`` load as Unity ``(x, z, y)``
    (``PointCloudDataLoader.cs:136``). One transform (scene lines 182-184,
    used at line 1122) rotates by 180 degrees about the vertical, scales by
    0.99 and translates by (9.65, 1.074, 10.45) in Unity ``(x, y, z)``. A
    second transform is the identity. In the plan frame (z up)::

        x = 9.65 - 0.99 x_pts,   y = 10.45 - 0.99 y_pts,   z = 1.074 + 0.99 z_pts.

Eye points
    Query files hold plan ``x, y`` and the eye height ``z``. They load as Unity
    ``(x, z, y)`` (``ObservationPointsLoader.cs:97-99``) and the height offset
    is 0 (scene line 456), so the eye stands at the query point itself. Output
    files list Unity ``x, y, z``: ``y`` is the eye height and ``z`` the plan y
    (``IsovistDataWriter.cs:169-173``).

One Unity detail is not reproduced. The octree search returns the nearest
hit among the balls stored in the first octree leaf along the ray that holds
any hit (``NativeOctreeRaycastQuery.cs:74-88, 131-157``), and a ball stored
in that leaf can be hit beyond the leaf's exit. pysovist returns the exact
first hit, so Unity depths can exceed the exact depth by up to about one
ball diameter where a ray grazes a surface.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

UNITY_RADIUS = 0.05
UNITY_RAYS = 20070

UNITY_PRESET = {
    "directions": "random",
    "n_rays": UNITY_RAYS,
    "seed": 1337,
    "max_distance": math.inf,
    "escape": "zero",
    "inside": "zero",
    "radius": UNITY_RADIUS,
}
"""Keyword arguments of ``view_volume`` and ``view_volume_field`` that match Unity."""


def unity_transform() -> np.ndarray:
    """4 x 4 matrix from the ``.pts`` frame of the hospital scan to the plan frame."""
    s = 0.99
    return np.array([
        [-s, 0.0, 0.0, 9.65],
        [0.0, -s, 0.0, 10.45],
        [0.0, 0.0, s, 1.074],
        [0.0, 0.0, 0.0, 1.0],
    ])


def read_unity_volumes(path) -> pd.DataFrame:
    """Read a Unity output file into the plan frame.

    Returns columns ``x, y`` (plan), ``z`` (eye height) and ``volume`` (cubic metres).
    """
    df = pd.read_csv(path)
    return pd.DataFrame({"x": df["x"].to_numpy(float), "y": df["z"].to_numpy(float),
                         "z": df["y"].to_numpy(float), "volume": df["volume"].to_numpy(float)})
