# Obstacle visibility

An isovist says how much space an observer sees. Obstacle visibility says which walls bound that space, and how much of each. For every wall segment of a plan, pysovist returns the length of the segment that the observer sees, the visible fraction of the segment, the angle the visible part fills in the view, and the visible stretches along the segment. They come from the same exact geometry as the [isovist](metrics.md), so they do not depend on a ray count. Summed over many viewpoints, such as the points of a path, they give the cumulative visibility of every facade. A sampled version counts ray hits per wall, as the obstacle visibility output (OV) of the DeCodingSpaces Isovist component for Grasshopper does.

The Python blocks on this page run in order as one script.

## Which walls a person sees

The scene is a 10 m x 10 m room with a 1 m x 2 m pillar. The observer stands at (2, 5), 4 m in front of the pillar's west face.

```python
import math

import numpy as np
from pysovist.visibility2d import segment_ray_hits, segment_visibility, segment_visibility_field


def rect(x0, y0, x1, y1):
    v = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


walls = np.concatenate([rect(0, 0, 10, 10), rect(6, 4, 7, 6)])  # room, then pillar
vis = segment_visibility(walls, (2.0, 5.0))
vis.visible_length     # [10, 6, 10, 10, 0, 0, 0, 2]
vis.visible_fraction   # [1, 0.6, 1, 1, 0, 0, 0, 1]
vis.intervals[1]       # [[0, 0.3], [0.7, 1]]
index, pieces = vis.pieces()
pieces[index == 1]     # [[(10, 0), (10, 3)], [(10, 7), (10, 10)]]
```

The rays through the pillar's corners (6, 4) and (6, 6) have slopes of -1/4 and +1/4, so they meet the east wall at y = 3 and y = 7. The pillar hides the 4 m between them, and the observer sees 6 m of that wall. Of the pillar, only its west face is visible, all 2 m of it. The visible lengths add up to 38 m, the part of the isovist perimeter that lies on walls.

Every array has one entry per input segment, in input order:

`visible_length`
:   The length of the segment that the observer sees, in metres.

`visible_fraction`
:   The visible length divided by the segment length.

`visible_angle`
:   The angle that the visible part fills at the observer, in radians.

`visible`
:   True where the visible length is positive.

`intervals`
:   The visible stretches as parameter ranges along the segment. The parameter runs from 0 at the first endpoint to 1 at the second, in the order the segment was given. `pieces()` returns the same stretches as coordinates.

The range and the field of view work as in `isovist`. A range of 4 m reaches only the west wall, 2 m away, and cuts it where the range circle crosses it, at y = 5 ± √12:

```python
segment_visibility(walls, (2.0, 5.0), max_distance=4).visible_length[3]   # 6.93, which is 2 √12
segment_visibility(walls, (2.0, 5.0), fov=math.pi / 2, direction=0.0).visible_length
# [3, 6, 3, 0, 0, 0, 0, 2]: a quarter view to the east
```

## Cumulative visibility along a path

The scene is a street 12 m wide and 60 m long, with three shop fronts of 20 m on its north side, two building fronts of 30 m on its south side and an 8 m x 2 m kiosk in the middle. A pedestrian walks along the south pavement, 3 m from the buildings, and stops every 2 m.

```python
north = np.array([[(0, 12), (20, 12)], [(20, 12), (40, 12)], [(40, 12), (60, 12)]], float)
south = np.array([[(0, 0), (30, 0)], [(30, 0), (60, 0)]], float)
street = np.concatenate([north, south, rect(26, 5, 34, 7)])
path = np.c_[np.arange(1.0, 60.0, 2.0), np.full(30, 3.0)]   # 30 viewpoints
table = segment_visibility_field(street, path)
table[["length", "seen_by", "visible_length_sum", "visible_fraction_mean"]]
```

The table has one row per segment. `seen_by` counts the viewpoints from which the segment's visible length is positive. `visible_length_sum` adds the visible length over the viewpoints, and `visible_length_mean` divides that sum by the number of viewpoints. `visible_fraction_mean` divides the mean by the segment length, which makes long and short facades comparable.

| Segment | Length (m) | Seen from | Summed visible length (m) | Mean visible fraction |
|---|---:|---:|---:|---:|
| 0, north front, x 0 to 20 | 20 | 24 of 30 | 373.8 | 0.623 |
| 1, north front, x 20 to 40 | 20 | 28 of 30 | 447.0 | 0.745 |
| 2, north front, x 40 to 60 | 20 | 24 of 30 | 373.8 | 0.623 |
| 3, south front, x 0 to 30 | 30 | 30 of 30 | 900.0 | 1.000 |
| 4, south front, x 30 to 60 | 30 | 30 of 30 | 900.0 | 1.000 |
| 5, kiosk, south face | 8 | 30 of 30 | 240.0 | 1.000 |
| 6, kiosk, east face | 2 | 13 of 30 | 26.0 | 0.433 |
| 7, kiosk, north face | 8 | 0 of 30 | 0.0 | 0.000 |
| 8, kiosk, west face | 2 | 13 of 30 | 26.0 | 0.433 |

The middle shop stands right behind the kiosk and is still the best seen of the three. The kiosk's shadow on the north side moves along with the pedestrian. It covers the middle front only while the pedestrian passes the kiosk, at 2 of the 30 viewpoints. Seen from farther along the street, the shadow falls on the outer front beyond the kiosk, and each outer front is hidden from 6 viewpoints.

With `per_viewpoint=True`, a second table lists every viewpoint and every segment it sees, with the visible length, fraction and angle. A pivot turns it into a visibility profile along the path, one column per segment:

```python
table, pairs = segment_visibility_field(street, path, per_viewpoint=True)
profile = pairs.pivot_table(index="viewpoint", columns="segment",
                            values="visible_length", fill_value=0.0)
profile.loc[14, [0, 1, 2]]   # 15.5, 0, 8.5: at x = 29 the kiosk hides the middle front
```

A viewpoint closer than `eps` to a wall sees nothing and is left out of the mean. `table.attrs["viewpoints"]` holds the number of viewpoints used and `table.attrs["on_wall"]` the number left out. `n_jobs` spreads the viewpoints over several processes, as in `isovist_field`.

## Ray hits and DeCodingSpaces

The DeCodingSpaces Isovist component casts rays from the viewpoint and reports, for every obstacle line, how many rays hit it first. `segment_ray_hits` counts the same hits. With `layout="decodingspaces"` it casts the component's rays:

- The angular step is 2π / n for a requested ray count n with 10 < n < 3000, and one degree for any other n.
- The rays start at the left edge of the view, the view direction plus half the field of view, and step clockwise while the angle stays short of the right edge. The step is summed in floating point, as in the component, and the rounding decides whether a ray lands on the right edge. In the default view direction a full turn casts n + 1 rays for n = 360 or 2,999, so the ray straight behind the view direction is cast twice, and n rays for n = 15 or 100.
- The component's default view direction (0, 1) is `direction=math.pi / 2`, and its default range is 1,000 m. `direction` also takes the component's direction vector, such as `(0, 1)`, which gives its rays bit for bit. The component measures the direction clockwise from +y, between 0 and 2π.

```python
hits = segment_ray_hits(walls, (2.0, 5.0), 2999, max_distance=1000,
                        direction=math.pi / 2, layout="decodingspaces")
hits                                        # [666, 299, 665, 1136, 0, 0, 0, 234]
vis.visible_angle / (2 * math.pi / 2999)    # [664.7, 299.4, 664.7, 1136.3, 0, 0, 0, 233.9]
```

These are the component's own counts for this scene. The exact visible angle divided by the angular step predicts them to within one ray per visible stretch. The south wall gets one ray more because the doubled ray behind the observer points at it.

`segment_ray_hits` reproduces every count of the component on 132 reference records. Twelve cover four scenes (a square seen from its centre and from off-centre, a 20 m x 5 m rectangle, and the room with the pillar), each at 2,999 and at 360 rays with a range of 1,000 m, and at 2,999 rays with a range of 4 m, all full turns in the default view direction. The other 120 vary the view direction around the full circle, the field of view (90, 180 and 359 degrees) and the ray count (11 to 2,400, and 100.5), and add an interior wall that ends on the middle of another wall and two walls 1 mm apart.

A ray that passes exactly through the corner where one segment ends and the next starts counts once, for the segment that starts there. In a closed outline every corner thus belongs to exactly one wall. From the centre of a square at 360 rays, four rays hit corners, and the walls get 91, 90, 90 and 90 hits, as in the component. The default `layout="even"` casts n rays evenly over the full turn, starting at `direction`, or n rays from one edge of the field of view to the other.

## Special cases

- **Overlapping walls.** Where two segments share a stretch, such as a wall drawn twice, the stretch counts once, for the segment listed first. Two segments overlap where their distances from the observer agree to a relative 1e-9.
- **Walls seen edge-on.** A wall that points straight at the observer shows no length and is not visible. A ray that runs along a wall does not hit it.
- **Open scenes.** Rays that meet no wall add nothing, so a facade seen across an open square needs no range limit.
- **Observer on a wall.** Closer than `eps` (1e-9 m by default) to a wall, the lengths are NaN and `flags.on_wall` is set, or a `ValueError` is raised with `on_wall="raise"`.
- **Segments of zero length** keep their row and are never visible.
- **Arguments outside their domain** raise a `ValueError` that names the argument: an origin that is not finite, a range, field of view or `eps` that is not positive, a direction that is not finite, a ray count below 1 or not whole for the even layout, and a `chunk` or `n_jobs` below 1 (`n_jobs=-1` uses all cores). A field of view above 2π is the full turn.
