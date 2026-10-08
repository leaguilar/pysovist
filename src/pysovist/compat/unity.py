"""Preset for the Unity scene that produced the reference view volumes.

That scene treats every scan point as a solid ball and casts random rays from
each eye point. This module states its setup in pysovist terms.

Occluders
:   Every point is a ball of radius 0.05 m, built from the transformed points,
    so the radius holds in the plan frame. A ray that starts inside a ball gets
    depth 0, so an eye inside a ball has V = 0.

Rays
:   Directions are uniform on the sphere, by rejection sampling of the unit
    ball inside the cube [-1, 1]^3. Each eye received exactly 20070 rays. No
    range limit and no near clip.

Seeds
:   One generator seeded with 1337 draws one seed per query file, and worker
    thread ``i`` starts from that seed plus ``i``. Eyes go to threads in
    batches of 32, so the directions of an eye depend on thread scheduling and
    cannot be replayed. The preset's seed 1337 is a convention: parity with
    Unity is statistical.

Escaping rays
:   A ray that hits nothing returns 1e31. The writer drops depths above 1e9
    from the sum but keeps them in the count, so they contribute 0:
    V = (4 pi / 3) mean(r^3) with escaped r = 0.

Frame
:   ``.pts`` rows ``x y z`` load as Unity ``(x, z, y)``. One transform rotates
    by 180 degrees about the vertical, scales by 0.99 and translates by
    (9.65, 1.074, 10.45) in Unity ``(x, y, z)``. A second transform is the
    identity. In the plan frame (z up):

        x = 9.65 - 0.99 x_pts,   y = 10.45 - 0.99 y_pts,   z = 1.074 + 0.99 z_pts.

Eye points
:   Query files hold plan ``x, y`` and the eye height ``z``. They load as Unity
    ``(x, z, y)`` with no height offset, so the eye stands at the query point
    itself. Output files list Unity ``x, y, z``: ``y`` is the eye height and
    ``z`` the plan y.

Two Unity details are not reproduced. First, Unity drew new directions for
every eye, while ``view_volume_field`` shares one direction set across all
eyes. Call ``view_volume`` with a different seed per eye for independent
errors. Second, the octree search returns the nearest hit among the balls
stored in the first octree leaf along the ray that holds any hit, and a ball
stored in that leaf can be hit beyond the leaf's exit. pysovist returns the
exact first hit, so Unity depths can exceed the exact depth where a ray grazes
a surface.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

UNITY_RADIUS = 0.05
"""Ball radius of every scan point in the Unity scene, in metres."""

UNITY_RAYS = 20070
"""Number of rays per eye point in the Unity scene."""

UNITY_PRESET = {
    "directions": "random",
    "n_rays": UNITY_RAYS,
    "seed": 1337,
    "max_distance": math.inf,
    "escape": "zero",
    "inside": "zero",
    "radius": UNITY_RADIUS,
}
"""Keyword arguments of [`view_volume`][pysovist.view_volume] and
[`view_volume_field`][pysovist.view_volume_field] that reproduce the Unity estimator."""


def unity_transform() -> np.ndarray:
    """4 x 4 matrix from the ``.pts`` frame of the scan to the plan frame.

    The scan is the one used in the Unity scene that produced the reference
    view volumes, and the constants belong to that scene. The map is
    ``x = 9.65 - 0.99 x_pts``, ``y = 10.45 - 0.99 y_pts`` and
    ``z = 1.074 + 0.99 z_pts``.

    Returns
    -------
    ndarray of shape (4, 4)
        Affine matrix for [`PointCloud.transform`][pysovist.PointCloud.transform].
    """
    s = 0.99
    return np.array([
        [-s, 0.0, 0.0, 9.65],
        [0.0, -s, 0.0, 10.45],
        [0.0, 0.0, s, 1.074],
        [0.0, 0.0, 0.0, 1.0],
    ])


def read_unity_volumes(path) -> pd.DataFrame:
    """Read a Unity output file into the plan frame.

    Parameters
    ----------
    path : str or os.PathLike
        CSV file with the Unity columns ``x, y, z`` and ``volume``, where
        ``y`` is the eye height and ``z`` the plan y.

    Returns
    -------
    DataFrame
        Columns ``x, y`` (plan), ``z`` (eye height) and ``volume`` (cubic metres).
    """
    df = pd.read_csv(path)
    return pd.DataFrame({"x": df["x"].to_numpy(float), "y": df["z"].to_numpy(float),
                         "z": df["y"].to_numpy(float), "volume": df["volume"].to_numpy(float)})
