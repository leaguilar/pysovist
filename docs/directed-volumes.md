# Directed view volumes and view factors

A full view volume counts everything an eye can see in every direction. Many questions concern a narrower view: what a person sees ahead of them, how much of a window's view is sky, how much of a street view a facade takes up. pysovist answers them with three additions to `view_volume`: a field of view, per-ray object ids, and the sky, ground and object factors. A compatibility module reproduces the Isovist3D component of the DeCodingSpaces toolbox for Grasshopper, so that its results can be compared and migrated.

## A view within a field of view

A field of view is a band of directions. Its azimuths lie within `hfov / 2` of the view `direction` (radians counter-clockwise from +x), and its elevations lie within `vfov / 2` of the centre `elevation` (radians above the eye's horizontal plane). The centre elevation is the vertical offset of the view. The band must lie between the poles. Its solid angle is

$$
\Omega = h\,(\sin e_1 - \sin e_0),
$$

where \(h\) is the horizontal field of view and \(e_0\), \(e_1\) are the elevations of the lower and upper edge. The directed view volume is the part of the full view volume that lies in these directions,

$$
V = \frac{1}{3} \int_{\text{field}} r(\omega)^3 \, d\omega \approx \frac{1}{3} \sum_i w_i\, r_i^3,
$$

with \(r(\omega)\) the visible depth in direction \(\omega\), and \(r_i\) the depth along ray \(i\), which carries the solid angle \(w_i\). Two direction sets give the weights.

- **Equal-area directions.** With the default Fibonacci directions (or random ones), only the directions inside the field are cast, and each keeps the share \(4\pi / N\) of the \(N\) directions on the sphere. `n_rays` counts the directions on the whole sphere, so a field keeps the density of a full view.
- **An angle grid.** `angle_grid(n_azimuth, n_elevation, hfov=..., vfov=..., direction=..., elevation=...)` cuts the field into equal steps of azimuth and elevation and returns the centre of each cell with the exact solid angle of the cell, \((h / n_\text{azimuth})(\sin e_\text{hi} - \sin e_\text{lo})\). The weights sum to \(\Omega\), and passing them as `weights` to `view_volume` makes the volume the midpoint rule over the cells.

Over a field, the depth statistics `r_mean` and `r_std` are averages weighted by solid angle, `escape_fraction` is the share of the field's solid angle in which nothing is hit, and `volume_up` and `volume_down` split the directed volume at the eye's horizontal plane. Every field includes its lower edges and leaves its upper edges to the neighbouring field, so adjacent fields partition the sphere and their volumes add up to the full view volume. Without a field of view and without weights, `view_volume` and `view_volume_field` return the full view volume with the same arithmetic as before, to the last bit.

In a closed box room the directed volume has a closed form for each azimuth, which makes it an exact reference. Over five fields, from 30 x 20 degrees to a full turn 40 degrees tall, one of them straddling the azimuth \(\pm\pi\), a grid of 768 x 384 cells lands within 2e-5 of the exact wedge volume, and 2^20 Fibonacci directions within 1e-4.

## Objects and view factors

A scene often consists of named parts: the ground, a building, its neighbours. `Mesh.combine([ground, building, neighbour])` joins meshes into one scene and gives every triangle the index of the mesh it came from as its object id. With `object_ids=True`, `view_volume` reports for every ray the object it hits, or -1 when it hits nothing within the range. For a point cloud the object is the point whose ball the ray hits, so per-point labels map the rays to classes.

From the object ids, a `ViewVolume` gives

- `hits_per_object()`: the number of rays that hit each object,
- `solid_angle_per_object()`: the solid angle in which each object is seen, the sum of the weights of the rays that hit it, which counts only the visible part of the object inside the field,
- `factors(ground=[...])`: the shares of the field's solid angle in which a ray escapes (`sky_factor`), hits an object designated as ground (`ground_factor`), or hits any other object (`object_factor`). The three sum to 1. The sky factor equals `escape_fraction`.

`view_volume_field(..., ground=[...])` adds the three factors as columns, one row per eye. Against exact solid angles of a ground square and a wall seen from one eye, the factors agree to 5e-4 with 2^18 Fibonacci directions and with a grid of 720 x 360 cells.

## A worked example

An eye stands 1.7 m above a street, between a 12 m building on its left and a 9 m building on its right, and looks along the street with a view 120 degrees wide and 60 degrees tall, raised by 10 degrees.

```python
import numpy as np
from pysovist import Mesh, view_volume
from pysovist.directions import angle_grid


def box(x0, y0, z0, x1, y1, z1):
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    return Mesh(v, [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


ground = Mesh([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)], [(0, 1, 2), (0, 2, 3)])
scene = Mesh.combine([ground, box(-30, 6, 0, 30, 16, 12), box(-30, -16, 0, 30, -6, 9)])
eye = (0.0, 0.0, 1.7)
field = dict(hfov=np.radians(120), vfov=np.radians(60), direction=0.0, elevation=np.radians(10))

vv = view_volume(scene, eye, n_rays=1 << 18, max_distance=100.0, object_ids=True, **field)
vv.n_rays                           # 43028 of the 262144 directions lie in the field
vv.volume                           # 137479.4 m3
vv.factors(ground=[0])              # sky 0.1979, ground 0.2111, object 0.5910
vv.solid_angle_per_object(3)        # [0.4354, 0.6356, 0.5834] sr

d, w = angle_grid(240, 120, **field)
w.sum()                             # 2.062577 sr, the solid angle of the field
vg = view_volume(scene, eye, directions=d, weights=w, max_distance=100.0, object_ids=True)
vg.factors(ground=[0])              # sky 0.1990, ground 0.2107, object 0.5903
```

The building on the left, taller and seen higher in the field, takes 0.64 sr of the 2.06 sr in view, more than the 0.58 sr of the lower building on the right. The sky takes a fifth of the view. With 2^24 directions the volume settles at 137,406 m3, and the estimate above, from 43,028 rays, lies 0.05 % above it.

## DeCodingSpaces Isovist3D

`pysovist.compat.decodingspaces3d.isovist3d` reproduces the outputs of the Isovist3D component (DeCodingSpaces 2021.10.11) for one vantage point: the hit vector, length and surface id of every ray, the hits per obstacle mesh, the volume and the isovist mesh. It takes the component's inputs in its own units, with angles in degrees and the view direction as a vector, and the same defaults: range 1000 m, 180 x 90 degrees, 10 x 10 rays, an implicit ground plane at z = 0.

```python
from pysovist import Mesh
from pysovist.compat.decodingspaces3d import isovist3d

walls = [[(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)]]
room = Mesh.from_plan(walls, floor=0.0, ceiling=3.0)
res = isovist3d((5, 5, 1.5), [room], (1, 0, 0), horizontal_angle=360, vertical_angle=180,
                horizontal_rays=360, vertical_rays=180, ground_at_zero=False)
res.volume                 # 299.08 m3 for a closed 10 x 10 x 3 m room, as the component
res.object_visibility      # rays per obstacle mesh (output OV)
res.rays, res.lengths, res.surface_ids
```

The component's rules, which differ from those of `view_volume`:

| Output | Rule |
|---|---|
| Ray layout | Azimuth steps of 360 / HPr degrees and elevation steps of 360 / VPr degrees, whatever the viewing angles. Azimuths run from half the horizontal angle left of the view direction to half of it right, elevations from half the vertical angle below the offset to half of it above, all ends included. 12 x 6 rays give 52 rays over 360 x 180 degrees and 14 over 180 x 90 degrees. |
| Lengths | The distance to the first hit, or the full range for a ray that hits nothing. An open scene therefore has a volume of hundreds of millions of cubic metres. |
| Surface ids | Obstacle meshes are numbered from 0 and ground meshes after them. -1 marks a ray that hits nothing, the implicit ground plane, or the last triangle of an obstacle mesh, which in a closed box room from Rhino is one half of the floor. |
| Hits per mesh | The number of rays with each obstacle id. Ground meshes and the plane do not count. |
| Volume | The polyhedron through the hit points: each cell of the ray grid is cut into two triangles, and the tetrahedra from the eye to the triangles are summed. |

The volume is a polyhedron and not the solid-angle integral. On flat walls the two agree: 12 x 6 rays from the centre of the closed room give a 12-sided prism of 60.75 m3. Where a cell spans a near and a far surface, the polyhedron cuts the corner, so the component underestimates scenes with many occluding edges. On a hospital floor plan extruded to 3 m, from an eye with an exact volume of 200.01 m3, it returns 3.3 % less at 360 x 180 rays and 1.1 % less at 720 x 360. The same rule gives 0.5 % less at 1,440 x 720.

| Measurement | Largest relative difference to the component |
|---|---|
| Volumes of 12 views of three scenes (room, room with a block, block on a ground mesh), 12 x 6 and 360 x 180 rays | 2.4e-6 |
| Lengths of all 240 rays of the nine 12 x 6 views | 1.1e-6 |
| Closed room, five ray counts from 10 x 10 to 720 x 360 | 1.2e-6 |
| Hospital plan, 10 x 10 to 720 x 360 rays | 1.8e-5 |

The surface ids agree on every ray except 13 that meet the room's floor exactly on the diagonal its two triangles share, where rounding picks the triangle. The hits per mesh agree exactly at 12 x 6 rays and, at 360 x 180 rays, for the room with a block (30,800 and 404) and the block on the ground (654). The closed room has 437 rays on that diagonal at 360 x 180 rays, and its count differs by 398 of them (25,813 against 26,211).

At 1,440 x 720 rays on the hospital plan the component returns 193.23 m3, 2.9 % less than its own rules give (199.02 m3). The rules with 1,440 azimuth steps and any number of elevation steps up to 720, or with 720 elevation steps and any number of azimuth steps up to 1,440, miss that number by at least 8e-5, four times the agreement at coarser counts. The shortfall therefore comes from a step outside the layout and the volume formula. The compatibility module returns the value of the rules.
