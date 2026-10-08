# Quick start

All examples on this page use one scene: a 10 m x 10 m room with a 2 m x 2 m pillar in its centre. The Python blocks build on each other and run in order as one script, except the scan-file block, which needs a scan of your own. On macOS, Windows and Python 3.14 or newer, put the script under `if __name__ == "__main__":`.

## A 2D isovist

A plan is a set of wall segments `[[x0, y0], [x1, y1]]`, in metres.

```python
import numpy as np
import pysovist


def rect(x0, y0, x1, y1):
    v = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


plan = pysovist.Plan(np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)]))
iso = pysovist.isovist(plan, (3.0, 0.0))
iso.area                       # 70.0, exact
iso.metrics["occlusivity"]     # 13.416..., the two edges of the pillar's shadow
iso.perimeter                  # 47.416..., 34 m of wall plus two shadow edges
```

The observer stands 2 m to the right of the pillar. The pillar hides a shadow of 26 m², so the visible area is 100 - 4 - 26 = 70 m². Each edge of the shadow runs from a corner of the pillar to the outer wall and has length \(\sqrt{45}\) m, so the occlusivity is \(2\sqrt{45} = 13.416\) m. `iso.metrics` holds all 17 metrics of the [metric definitions](metrics.md) page, and `iso.polygon()` returns the boundary as a closed ring.

## Many observers

`isovist_field` returns one row per observer, with the observer's position, every metric, the clearance to the nearest wall and the flags.

```python
pts = plan.grid(0.5)                       # 400 points, 0.5 m apart
pts = pts[~(np.abs(pts) < 1).all(axis=1)]  # drop the 16 inside the pillar
field = pysovist.isovist_field(plan, pts, max_distance=40.0, n_jobs=-1)
len(field)                                 # 384
field[["x", "y", "area", "occlusivity", "drift"]].head()
```

Walls have no solid side. An observer inside the pillar sees the pillar's 4 m² interior, which is why the grid points inside it are dropped. `n_jobs=-1` spreads the observers over all cores in separate processes.

## A view volume from an extruded plan

`Mesh.from_plan` turns every wall into a vertical rectangle between a floor and a ceiling. Inside such a prism, an eye at any height between floor and ceiling sees the height of the prism times the 2D isovist area of the plan closed along the same outline.

```python
mesh = pysovist.Mesh.from_plan(plan, floor=0.0, ceiling=2.5)
vv = pysovist.view_volume(mesh, (3.0, 0.0, 1.7), n_rays=65_536)
vv.volume                      # 174.98, against 2.5 m x 70 m² = 175 m³
vv.metrics["escape_fraction"]  # 0.0, every ray hits a surface
```

The 3D volume is estimated from 65,536 directions, and the residual of 9e-5 is the error of that estimate.

By default, `Mesh.from_plan` closes the sides of the prism along the convex hull of the plan, so no ray from an eye inside the hull escapes. A polygon given as `close=` closes the domain along its own outline instead. A building footprint keeps a concave building concave, even when the plan holds only interior walls.

```python
footprint = np.array([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)], float)
inner = pysovist.Plan([[(2.5, 2.0), (2.5, 3.5)], [(6.0, 1.0), (7.5, 1.0)]])
building = pysovist.Mesh.from_plan(inner, ceiling=2.7, close=footprint)
vv = pysovist.view_volume(building, (1.5, 1.2, 1.6), n_rays=65_536)
vv.volume                      # 149.58

outline = np.stack([footprint, np.roll(footprint, -1, axis=0)], axis=1)
closed = pysovist.Plan(np.concatenate([inner.segments, outline]))
2.7 * pysovist.isovist(closed, (1.5, 1.2)).area   # 149.61
```

The volume equals the height of 2.7 m times the isovist area of the plan closed by the same footprint, to the 2e-4 error of 65,536 directions.

## A view volume from a point cloud

A point cloud needs no surfaces. Every point stands for a ball of radius `radius`, and the balls of a dense scan close into opaque surfaces. The cloud below samples the floor, the ceiling and the walls of the same room every 4 cm.

```python
s, h = 0.04, 2.5
u = np.arange(-5, 5 + s / 2, s)
x, y = (g.ravel() for g in np.meshgrid(u, u))
free = ~((np.abs(x) < 1) & (np.abs(y) < 1))  # no floor under the pillar
pts = [np.c_[x[free], y[free], np.full(free.sum(), z)] for z in (0.0, h)]
z = np.arange(0, h + s / 2, s)
for (x0, y0), (x1, y1) in plan.segments:
    t = np.linspace(0, 1, int(round(np.hypot(x1 - x0, y1 - y0) / s)) + 1)
    tt, zz = (g.ravel() for g in np.meshgrid(t, z))
    pts.append(np.c_[x0 + tt * (x1 - x0), y0 + tt * (y1 - y0), zz])
cloud = pysovist.PointCloud(np.concatenate(pts), radius=0.04)

cloud.spacing()["p99"]         # 0.04 m between neighbouring points
vv = pysovist.view_volume(cloud, (3.0, 0.0, 1.7), n_rays=65_536)
vv.volume                      # 163.76
```

The balls fill the free space up to 4 cm in front of every surface, so the cloud sees 163.76 m³ where the mesh sees 174.98 m³. A smaller radius takes less free space, and the radius must stay above the spacing divided by \(\sqrt{2}\) for the surfaces to stay closed. The [conventions](conventions.md#ball-radius-of-a-point-cloud) page explains the choice.

A scan file is read in one line. `.pts`, `.las`, `.laz`, `.e57`, `.ply` and `.pcd` are supported.

```python
# Every point of the scan becomes a ball of radius 5 cm.
cloud = pysovist.PointCloud.read("scan.las", radius=0.05)
vv = pysovist.view_volume(cloud, (3.0, 0.0, 1.7), n_rays=262_144)
vv.volume, vv.metrics["escape_fraction"]
```

A ray that leaves through an opening of the scan makes the volume infinite under the default `max_distance=inf`. Set `max_distance`, or choose another `escape` policy, for scans that are not closed. `view_volume_field` evaluates many eye points in one call and returns one row per eye.

## Command line

The `pysovist` command reads a plan, a mesh or a point cloud and a CSV of points, and writes one CSV row per point. The script below writes the walls of the room and three observers.

```python
import json

import pandas as pd

walls = [{"start": a, "end": b} for a, b in plan.segments.tolist()]
with open("walls.json", "w") as f:
    json.dump(walls, f)
points = pd.DataFrame({"x": [3.0, -3.0, 0.0], "y": [0.0, 2.0, -4.0]})
points.to_csv("points.csv", index=False)
```

```bash
pysovist isovist --plan walls.json --points points.csv --out isovists.csv
pysovist volume --plan walls.json --ceiling 2.5 --eye-height 1.7 \
    --points points.csv --n-rays 65536 --out volumes.csv
pysovist volume --cloud scan.las --radius 0.05 --eye-height 1.7 \
    --points points.csv --out cloud_volumes.csv
```

Every run also writes a JSON file next to the CSV (`isovists.json` for `isovists.csv`). It holds the pysovist version, all parameters and the sha256 of every input file. The columns of the CSV are those of `isovist_field` or `view_volume_field`.

| Option | Command | Default | Meaning |
|---|---|---|---|
| `--plan` | both | | Wall segments: JSON records with `start` and `end`, or a CSV with columns `x0, y0, x1, y1`. For `volume`, the plan is extruded with `Mesh.from_plan`. |
| `--close` | both | | JSON polygon `[[x, y], ...]` that closes the domain. For `isovist` its edges are added as walls. For `volume` it is the `close=` polygon of `Mesh.from_plan`, and without it the plan closes along its convex hull. |
| `--points` | both | | CSV with columns `x, y` and, for `volume`, `z`. |
| `--out` | both | | Output CSV. |
| `--max-distance` | both | inf | Range limit in metres. |
| `--n-jobs` | both | 1 for `isovist`, -1 for `volume` | Processes for `isovist`, threads for `volume`. -1 uses all cores. |
| `--cloud` | `volume` | | Point cloud file. |
| `--mesh` | `volume` | | Mesh file in any format open3d reads. |
| `--floor`, `--ceiling` | `volume` | 0.0, 2.5 | Heights of an extruded plan. |
| `--eye-height` | `volume` | | Eye height that replaces the `z` column. |
| `--n-rays` | `volume` | 262144 | Directions per eye. |
| `--radius` | `volume` | 0.05 | Ball radius of cloud points in metres. |
| `--escape` | `volume` | clip | `clip`, `zero` or `nan`. |
| `--inside` | `volume` | nan | `nan` or `zero`. |
| `--near-clip` | `volume` | 0.0 | Ignore occluders closer than this to the eye. |

`volume` takes exactly one of `--cloud`, `--mesh` and `--plan`, and always uses Fibonacci directions.

### As an ODTP component

`pysovist odtp` runs the same commands with settings taken from environment variables, as the Open Digital Twin Platform passes them to a component. Input files are read from `ODTP_INPUT` (default `/odtp/odtp-input`). The results are written to `ODTP_OUTPUT` (default `/odtp/odtp-output`) as `<OUTPUT_PREFIX>_<TASK>.csv` and `.json`.

```bash
export ODTP_INPUT="$PWD" ODTP_OUTPUT="$PWD/output"
export TASK=isovist PLAN_FILE=walls.json QUERY_FILE=points.csv MAX_DISTANCE=40
pysovist odtp                  # writes output/pysovist_isovist.csv and .json
```

| Variable | Option | Meaning |
|---|---|---|
| `TASK` | | `isovist` or `volume`. |
| `QUERY_FILE` | `--points` | CSV of points. |
| `PLAN_FILE` | `--plan` | Wall segments. |
| `CLOSE_FILE` | `--close` | Closing polygon. |
| `MAX_DISTANCE` | `--max-distance` | Range limit in metres (default: none). |
| `N_JOBS` | `--n-jobs` | Processes or threads. |
| `POINTCLOUD_FILE` | `--cloud` | Point cloud, for `TASK=volume`. |
| `MESH_FILE` | `--mesh` | Mesh, for `TASK=volume`. |
| `EYE_HEIGHT` | `--eye-height` | Eye height that replaces the `z` column. |
| `N_RAYS` | `--n-rays` | Directions per eye (default 262144). |
| `SPLAT_RADIUS` | `--radius` | Ball radius of cloud points in metres (default 0.05). |
| `ESCAPE`, `INSIDE`, `NEAR_CLIP` | `--escape`, `--inside`, `--near-clip` | Policies of `view_volume`. |
| `FLOOR`, `CEILING` | `--floor`, `--ceiling` | Heights of an extruded plan. |
| `OUTPUT_PREFIX` | | Prefix of the output files (default `pysovist`). |

File names are relative to `ODTP_INPUT`. The variables from `POINTCLOUD_FILE` to `CEILING` apply to `TASK=volume` only.
