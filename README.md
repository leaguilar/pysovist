# pysovist

Exact isovists and view volumes from floor plans, meshes and raw point clouds.

An isovist is the region visible from one point. pysovist computes it in two settings:

- **2D, from wall segments.** The visible polygon is computed exactly, with no ray count. Area, perimeter, occlusivity, radial statistics, drift, elongation and convex deficiency follow in closed form.
- **3D, from a mesh or a point cloud.** The view volume is the volume visible from an eye point. Point clouds are used as scanned: every point is a small ball, so furniture, equipment and people occlude as they did on the day of the scan.

## Install

```
pip install pysovist                    # 2D isovists
pip install "pysovist[pointcloud]"      # point clouds (numba, laspy)
pip install "pysovist[mesh]"            # meshes (open3d)
pip install "pysovist[all]"             # everything, including e57 and Rhino files
```

Python 3.10 or newer.

## Quick start

```python
import numpy as np
import pysovist

# A 10 m x 10 m room with a 2 m x 2 m pillar, as wall segments [[x0, y0], [x1, y1]].
def rect(x0, y0, x1, y1):
    v = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)

plan = pysovist.Plan(np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)]))
iso = pysovist.isovist(plan, (3.0, 0.0))
iso.area                       # 70.0, exact
iso.metrics["occlusivity"]     # 13.416..., the two edges of the pillar's shadow

# Many observers at once, one row each.
field = pysovist.isovist_field(plan, plan.grid(0.5), max_distance=40.0, n_jobs=-1)
```

View volumes from a point cloud:

```python
cloud = pysovist.PointCloud.read("scan.las", radius=0.05)   # every point is a 5 cm ball
vv = pysovist.view_volume(cloud, (3.0, 0.0, 1.7), n_rays=262_144)
vv.volume, vv.metrics["escape_fraction"]
```

Units are metres and radians. z points up.

## What makes it exact

Seen from the observer, the angles of all wall endpoints and wall crossings cut the full turn into wedges. Within one wedge the nearest wall never changes, so each wedge is a triangle or, under a range limit, a circular sector. Every metric is a sum of closed-form integrals over these pieces.

Ray-sampling tools describe the same isovist by the polygon through N hit points. `pysovist.sampled_metrics` reproduces that description, so its error can be measured against the exact value. The area error falls as N^-2 in convex rooms and as N^-1 where the boundary has occluding edges.

## Verification

- 2D: closed-form scenes (square, hallway, regular polygons, L-shaped room, room with a pillar, open room with a range limit, fields of view) agree to a relative 1e-9. Random rooms agree with the visilibity library and with a polygon shadow construction to 1e-7, and the results are invariant under rigid motion and scaling.
- 3D: closed-form volumes (cube, pyramid, sphere mesh, convex hulls, a cube clipped by a range sphere) converge with the number of directions. On an extruded floor plan the view volume equals the ceiling height times the exact 2D isovist area, to 1e-3 at 65,536 directions.
- Point clouds: the ball ray caster agrees with a brute-force first hit to 1e-12.

## Licence

MIT.
