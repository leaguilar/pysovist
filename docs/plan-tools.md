# Plan tools

Four tools extend the exact 2D kernel beyond one isovist per point: single rays, an analysis grid flooded from a seed point, isovists along a walking path, and isovists in a vertical plane. They cover the plan tools of the DeCodingSpaces toolbox for Grasshopper: IsovistRay, Analysis Grid, and the path and vertical isovists of its tutorial. Every result is exact, and none depends on a number of rays.

The examples share one scene, an L-shaped room of 60 m², and run in order as one script.

```python
import numpy as np
import pysovist
from pysovist.paths import path_isovists
from pysovist.rays2d import cast_rays
from pysovist.vertical import plane_to_world, vertical_isovist

outline = np.array([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)], float)
plan = pysovist.Plan(np.stack([outline, np.roll(outline, -1, axis=0)], axis=1))
```

Wall i runs from vertex i of the outline to vertex i + 1. Wall 0 is the bottom wall, wall 1 the short wall at x = 10, wall 4 the top wall of the upper wing and wall 5 the wall at x = 0.

## Single rays

`cast_rays` sends rays from origins at given angles. For each ray it returns the distance to the first wall, the hit point and the index of that wall.

```python
hits = cast_rays(plan, (2.0, 2.0), [0.0, np.pi / 2, np.pi])
hits.distance   # [[8., 7., 2.]]
hits.segment    # [[1, 4, 5]]
hits.point      # [[[10., 2.], [2., 9.], [0., 2.]]]
```

With m origins and k angles, every array has the shape (m, k). A 1-D array of angles goes to every origin. A 2-D array of shape (m, k) gives row i to origin i, so `angles[:, None]` gives each origin a direction of its own.

`max_distance` is the range. A ray that meets no wall within it has the distance `inf`, a NaN hit point and the wall index `NO_HIT`, which is -1. `hits.depth` is the visible depth along each ray, the distance capped at the range. It equals the exact isovist depth at the same angle.

```python
own = cast_rays(plan, [(2.0, 2.0), (2.0, 7.0)], np.array([[0.0], [0.0]]), max_distance=5.0)
own.segment     # [[-1], [3]]: the first ray reaches no wall within 5 m
own.depth       # [[5.], [2.]]
```

Counting the rays that end on each wall gives the obstacle visibility of IsovistRay. Here 360 rays leave the observer, offset by half a degree so that none passes through a corner.

```python
hits = cast_rays(plan, (2.0, 2.0), np.radians(np.arange(360) + 0.5))
np.bincount(hits.segment[hits.hit], minlength=len(plan))  # [121, 28, 31, 29, 32, 119]
```

Walls are closed segments with no thickness. A ray through the shared endpoint of two walls ends there, and a ray through the free end of a wall hits that wall, so no ray slips through a closed corner. A wall that lies along a ray is seen edge-on and is not hit. Only hits at a positive distance count, so a ray from an origin on a wall leaves through that wall. Wall indices refer to `plan.segments`, from which zero-length segments are removed. An origin or angle that is not finite, or a range that is not positive, raises a `ValueError`.

## An analysis grid flooded from a seed

`Plan.flood_grid(spacing, seed)` returns the cells of `Plan.grid(spacing)` that a flood from the seed reaches without crossing a wall. It gives their centres and their bounds `(xmin, ymin, xmax, ymax)`.

```python
centres, cells = plan.flood_grid(0.5, (2.0, 2.0))
len(centres), len(plan.grid(0.5))   # (240, 360): the notch of the L stays empty
field = pysovist.isovist_field(plan, centres)
```

The flood starts in the cell that contains the seed. It passes from a cell to each of its four side neighbours when the straight segment between their centres stays farther than `eps` from every wall, endpoints included. The default `eps` is 1e-9 m. The flood stops at the walls and at the edge of the grid, so in an open plan, where the walls enclose nothing, it fills the bounding rectangle of the walls. `domain=` sets another rectangle, as for `Plan.grid`.

- A cell is reached or not as a whole, by its centre. A cell cut by a wall belongs to the side of the wall on which its centre lies.
- A centre closer than `eps` to a wall is never reached, so every centre of the fill is a valid isovist observer. No row of `field` above is flagged `on_wall`.
- A gap between walls lets the flood through when a row or column of centres passes through it. A door narrower than the spacing therefore lets it through at some grid positions and not at others. A larger `eps` closes every gap narrower than 2 `eps`, such as walls in a drawing that miss each other by a millimetre.
- The centres are the points of `Plan.grid` with the same spacing and domain, bit for bit and in the same order.
- The seed must see the centre of its own cell. A wall between them raises a `ValueError`, and so does a seed outside the grid.

## Isovists along a path

`path_isovists` places observers every `spacing` metres of arc length along a polyline and centres each field of view on the direction of the path at that point. Each row holds the arc length `s`, the position, the `direction` and the columns of `isovist_field`.

```python
walk = [(8, 2), (2, 2), (2, 8)]
df = path_isovists(plan, walk, 1.0, fov=np.radians(120))
df[["s", "x", "y", "direction", "area"]].iloc[4:8]
```

| s | x | y | direction | area |
|---|---|---|---|---|
| 4.0 | 4.0 | 2.0 | 3.142 | 20.702 |
| 5.0 | 3.0 | 2.0 | 3.142 | 12.640 |
| 6.0 | 2.0 | 2.0 | 1.571 | 26.309 |
| 7.0 | 2.0 | 3.0 | 1.571 | 21.691 |

Walking left along the lower wing, the observer sees less of the room as the end wall comes closer, from 33.7 m² at the start of the path to 12.6 m² one metre before the corner. At the corner the view turns up the other wing and opens to 26.3 m².

The observers stand at the arc lengths 0, `spacing`, 2 `spacing` and so on. When the length of the path is not a multiple of `spacing`, an observer also stands on the last vertex, unless `endpoint=False`. An observer on a vertex looks along the segment that leaves it, and the observer on the last vertex looks along the segment that ends there. Repeated vertices are skipped. A spacing that is not positive and finite, a vertex that is not finite, or a path with fewer than two distinct vertices raises a `ValueError`. `sample_path` returns the arc lengths, points and directions alone. With the default full turn as the field of view, the direction plays no role, and the rows equal those of `isovist_field` at the same points.

## Vertical isovists

A vertical isovist is the region of a vertical plane through the eye that the eye sees. The DeCodingSpaces tutorial obtains it by turning the building by 90 degrees and computing a 2D isovist. `vertical_isovist` cuts the mesh with the vertical plane through the eye directly and computes the exact isovist in that plane.

Here the room is extruded to a ceiling at 2.7 m, and a table 1 m deep and 0.75 m high stands in the lower wing.

```python
room = pysovist.Mesh.from_plan(plan, floor=0.0, ceiling=2.7, close=None)
top = np.array([(4, 1), (5, 1), (5, 3), (4, 3)], float)
table = pysovist.Mesh.from_plan(np.stack([top, np.roll(top, -1, axis=0)], axis=1),
                                floor=0.0, ceiling=0.75, close=None)
mesh = pysovist.Mesh(np.vstack([room.vertices, table.vertices]),
                     np.vstack([room.triangles, table.triangles + len(room.vertices)]))

eye = (1.0, 2.0, 1.6)
iso = vertical_isovist(mesh, eye, 0.0)   # the plane along +x through the eye
iso.area                                 # 24.926...
```

The plane runs along the lower wing, 9 m ahead of the eye and 1 m behind it, from the floor to the ceiling: 27 m². The table fills 0.75 m² of it. The sight line over its far edge, 4 m ahead at a height of 0.75 m, meets the floor 1.6 × 4 / 0.85 = 7.53 m ahead of the eye, and the triangle below that line is hidden: 1.32 m². The isovist is 27 - 0.75 - 1.32 = 24.93 m².

Results are in plane coordinates (u, z). The coordinate u is the horizontal distance from the eye along the azimuth, negative behind the eye, and z is the height of the scene. The eye stands at (0, z_eye). Angles in the plane are elevations, counter-clockwise from the forward horizontal: 0 looks ahead and π/2 up. `elevation=` centres a field of view, and `plane_to_world` maps points of the plane back to the scene.

```python
down = vertical_isovist(mesh, eye, 0.0, elevation=np.radians(-30), fov=np.radians(60))
down.area                                       # 11.587...
ring = plane_to_world(iso.polygon(), eye, 0.0)  # the boundary in the scene, shape (k, 3)
```

`vertical_section` returns the section itself as a plan in plane coordinates. Every triangle that crosses the plane gives one segment. A vertex in the plane counts as lying to the right of the azimuth, the rule of `Mesh.section` turned upright, and a triangle in the plane gives no segment. In an extruded plan without furniture, the vertical isovist is the rectangle from the floor to the ceiling between the first wall ahead and the first wall behind, the two distances that `cast_rays` gives in the plan. A section that is open, for example one with no ceiling, needs `max_distance`.

## API

The functions are documented on the [plan tools API page](api/plan-tools.md), and `Plan.flood_grid` with
`Plan` under [2D isovists](api/isovists.md).
