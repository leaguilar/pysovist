# pysovist

Exact isovists and view volumes from floor plans, meshes and raw point clouds.

An isovist is the region visible from one point. pysovist computes it in two settings:

- **2D, from wall segments.** The visible polygon is computed exactly, with no ray count. Area, perimeter, occlusivity, radial statistics, drift and elongation follow in closed form, and so does convex deficiency without a range limit.
- **3D, from a mesh or a point cloud.** The view volume is the volume visible from an eye point. Point clouds are used as scanned: every point is a small ball, so furniture, equipment and people occlude as they did on the day of the scan.

## Install

```
pip install pysovist                    # 2D isovists
pip install "pysovist[pointcloud]"      # point clouds (numba, laspy)
pip install "pysovist[mesh]"            # meshes (open3d)
pip install "pysovist[all]"             # everything, including e57 files
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
cloud = pysovist.PointCloud.read("scan.las", radius=0.05)   # every point is a ball of radius 5 cm
vv = pysovist.view_volume(cloud, (3.0, 0.0, 1.7), n_rays=262_144)
vv.volume, vv.metrics["escape_fraction"]
```

Units are metres and radians. z points up.

## What makes it exact

Seen from the observer, the angles of all wall endpoints, wall crossings and, under a range limit, crossings of walls with the range circle cut the full turn into wedges. Within one wedge the nearest wall never changes, so each wedge is a triangle or, under a range limit, a circular sector. Every metric is built from closed forms over these pieces.

Ray-sampling tools describe the same isovist by the polygon through N hit points. `pysovist.sampled_metrics` reproduces that description, so its error can be measured against the exact value. The area error falls as N^-2 in convex rooms and as N^-1 where the boundary has occluding edges.

## Verification

- 2D: closed-form scenes (square, hallway, regular polygons, L-shaped room, room with a pillar, open room with a range limit, fields of view) agree to a relative 1e-9 (absolute 1e-9 for values of zero). Random rooms agree with the visilibity library and with a polygon shadow construction to 1e-7, and the results are invariant under rigid motion and scaling.
- 3D: closed-form volumes (cube, pyramid, sphere mesh, convex hulls, a cube clipped by a range sphere) agree to a relative 1e-4 at 65,536 directions (2e-4 for random hulls), and the error on the cube falls faster than N^-1/2. On an extruded floor plan the view volume equals the ceiling height times the exact 2D isovist area, to 1e-3 at 65,536 directions.
- Point clouds: the ball ray caster agrees with a brute-force first hit to 1e-12.

## How to cite

pysovist is free to use and modify under the MIT licence. If it helps your work, please cite the
software and the methods paper. If you use the emergency-department results, please cite the
behaviour paper. GitHub's "Cite this repository" button gives the software reference from
`CITATION.cff`.

```bibtex
@software{pysovist,
  author  = {Aguilar, Leonel and Tuncay, Bartu},
  title   = {pysovist: exact isovists and view volumes from floor plans, meshes and point clouds},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/leaguilar/pysovist}
}

@unpublished{pysovist_methods,
  author = {Aguilar, Leonel and Baur, Rapha{\"e}l and Wheele, Theresa and Tuncay, Bartu and Ladouce, Simon and Li, Chenyang and Schmutz, Jan and Sailer, Kerstin and Gr{\"u}bel, Jascha and Slankamenac, Ksenija and Honegger, Patrik and Conroy-Dalton, Ruth and Sax, Hugo and H{\"o}lscher, Christoph and Gath-Morad, Michal},
  title  = {Exact isovists and view volumes from raw point clouds},
  note   = {In preparation},
  year   = {2026}
}

@unpublished{pysovist_ed_behaviour,
  author = {Aguilar, Leonel and Baur, Rapha{\"e}l and Wheele, Theresa and Ladouce, Simon and Li, Chenyang and Schmutz, Jan and Sailer, Kerstin and Gr{\"u}bel, Jascha and Slankamenac, Ksenija and Honegger, Patrik and Conroy-Dalton, Ruth and Sax, Hugo and H{\"o}lscher, Christoph and Gath-Morad, Michal},
  title  = {Visibility and staff occupancy in an emergency department},
  note   = {In preparation},
  year   = {2026}
}
```

## Licence

MIT. Copyright (c) 2025-2026 Bartu Tuncay and Leonel Aguilar.
