# Unity compatibility

A set of reference view volumes was computed in a Unity scene that treats every scan point as a solid ball and casts random rays from each eye point. `pysovist.compat.unity` states that setup in pysovist terms, so that `view_volume` reproduces the estimator and its results can be read in the same frame.

## The preset

`UNITY_PRESET` holds the keyword arguments of `view_volume` and `view_volume_field` that match the Unity estimator.

| Argument | Value | Unity behaviour |
|---|---|---|
| `radius` | 0.05 | Every point is a ball of radius 5 cm, built from the transformed points, so the radius holds in the plan frame. |
| `directions` | `"random"` | Directions are uniform on the sphere, drawn by rejection sampling in the unit cube. |
| `n_rays` | 20070 | Each eye received exactly 20,070 rays. |
| `seed` | 1337 | A convention. Unity draws one seed per query file from a generator seeded with 1337, worker thread i starts from that seed plus i, and eyes go to threads in batches of 32. The directions of an eye therefore depend on thread scheduling and cannot be replayed. Parity with Unity is statistical. |
| `max_distance` | inf | No range limit. There is no near clip either. |
| `escape` | `"zero"` | A ray that hits nothing returns 1e31. The writer drops depths above 1e9 from the sum but keeps them in the count, so escaped rays contribute 0. |
| `inside` | `"zero"` | A ray that starts inside a ball gets depth 0, so an eye inside a ball has V = 0. |

With escaped rays counted at depth 0, the Unity volume is

$$
V = \frac{4\pi}{3} \cdot \frac{1}{N} \sum_{i=1}^{N} r_i^3, \qquad r_i = 0 \text{ for an escaped ray},
$$

with \(N = 20070\) rays and \(r_i\) the depth along ray \(i\). This is the general estimator of the [metric definitions](metrics.md#3d-metrics) page under `escape="zero"`. A run with the preset sets `flags.unbounded` whenever a ray escapes, because the range is infinite, and still returns a finite volume.

## The frame

The scan's `.pts` rows `x y z` load into Unity as `(x, z, y)`, since Unity's vertical axis is y. One transform of the scene then rotates the points by 180 degrees about the vertical, scales them by 0.99 and translates them by (9.65, 1.074, 10.45) in Unity `(x, y, z)`. A second transform is the identity. In the plan frame, with z up:

$$
x = 9.65 - 0.99\, x_\text{pts}, \qquad
y = 10.45 - 0.99\, y_\text{pts}, \qquad
z = 1.074 + 0.99\, z_\text{pts}.
$$

`unity_transform()` returns this map as a 4 x 4 matrix for `PointCloud.transform`. The constants belong to the reference scene.

Query files hold plan `x, y` and the eye height `z`. They load into Unity as `(x, z, y)` with a height offset of 0, so the eye stands at the query point itself. Output files list Unity `x, y, z`, where `y` is the eye height and `z` the plan y. `read_unity_volumes` reads such a file into plan columns `x, y`, the eye height `z` and `volume` in cubic metres.

```python
import pysovist
from pysovist.compat import unity

cloud = pysovist.PointCloud.read("scan.pts").transform(unity.unity_transform())
vv = pysovist.view_volume(cloud, (5.0, 4.0, 1.7), **unity.UNITY_PRESET)
reference = unity.read_unity_volumes("unity_volumes.csv")  # x, y, z, volume
```

## The one difference

The Unity octree search returns the nearest hit among the balls stored in the first octree leaf along the ray that holds any hit, and a ball stored in that leaf can be hit beyond the leaf's exit. pysovist returns the exact first hit. Unity depths can therefore exceed the exact depth by up to about one ball diameter where a ray grazes a surface.
