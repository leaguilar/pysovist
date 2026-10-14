# pysovist

Exact isovists and view volumes from floor plans, meshes and raw point clouds.

An isovist is the region visible from one point. pysovist computes it in two settings:

- **2D, from wall segments.** The visible polygon is computed exactly, with no ray count. Area, perimeter, occlusivity, radial statistics, drift and elongation follow in closed form, and so does convex deficiency without a range limit.
- **3D, from a mesh or a point cloud.** The view volume is the volume visible from an eye point. Point clouds are used as scanned: every point is a small ball, so furniture, equipment and people occlude as they did on the day of the scan.

The 3D volume is an integral over the sphere of directions, and pysovist evaluates it with a set of rays. For a cube, a pyramid and a sphere mesh, 65,536 directions give a relative error below 1e-4.

## Install

```bash
pip install pysovist                    # 2D isovists (numpy, scipy, pandas)
pip install "pysovist[pointcloud]"      # point clouds (numba, laspy)
pip install "pysovist[mesh]"            # meshes (open3d)
pip install "pysovist[e57]"             # .e57 scans (pye57)
pip install "pysovist[ifc]"             # BIM models in IFC files (ifcopenshell)
pip install "pysovist[all]"             # all of the above
```

Python 3.10 or newer. Point clouds in `.ply` or `.pcd` files are read through open3d, so they need `[mesh]` as well as `[pointcloud]`. On a Linux machine without a desktop, open3d also needs the system libraries `libegl1`, `libgl1` and `libgomp1`.

## What makes it exact

Seen from the observer, the angles of all wall endpoints, wall crossings and, under a range limit, crossings of walls with the range circle cut the full turn into wedges. Within one wedge the nearest wall never changes, so each wedge is a triangle or, under a range limit, a circular sector. Every metric is built from closed forms over these pieces.

Ray-sampling tools describe the same isovist by the polygon through \(N\) hit points. `pysovist.sampled_metrics` reproduces that description, so its error can be measured against the exact value. The area error falls as \(N^{-2}\) in convex rooms and as \(N^{-1}\) where the boundary has occluding edges.

## Contents

- [Quick start](quickstart.md): a 2D isovist, a field of isovists on a grid, view volumes from a point cloud and from an extruded plan, a plan from a BIM model, and the command line.
- [BIM models](ifc.md): plans and meshes from IFC files, the class presets, units, storeys and openings.
- [Conventions](conventions.md): units, plans, eye heights, flags, the policies of `view_volume` and the choice of the ball radius.
- [Metric definitions](metrics.md): the formula and source of every metric, and how the outputs of the Grasshopper isovist components map to pysovist.
- [Verification](verification.md): what the test suite checks, and to which tolerance.
- [Obstacle visibility](obstacle-visibility.md): which walls an observer sees and how much of each, summed along a path, and ray hits per wall.
- [Directed view volumes](directed-volumes.md): view volumes within a field of view, objects hit by each ray, and the sky, ground and object factors.
- [Plan tools](plan-tools.md): single rays, an analysis grid flooded from a seed, isovists along a path and isovists in a vertical plane.
- [Unity compatibility](unity.md): a preset that reproduces a Unity view-volume estimator.
- [DeCodingSpaces compatibility](decodingspaces.md): the properties of the DeCodingSpaces isovist components for Grasshopper, reproduced from the exact isovist.
- [API reference](api/isovists.md): every public class and function, and the [command line](api/cli.md).
- [How to cite](cite.md).
