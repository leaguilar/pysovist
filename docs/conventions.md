# Conventions

## Units and frames

Lengths are in metres. Angles are in radians, measured counter-clockwise from the +x axis. The z axis points up. Every angle that pysovist returns, such as `drift_angle` or the wedge bounds of an `Isovist`, is absolute in the plan frame.

## Floor plans

A `Plan` is an array of wall segments of shape (n, 2, 2): segment `i` runs from `segments[i, 0] = [x0, y0]` to `segments[i, 1] = [x1, y1]`. A third coordinate, if present, is dropped. Segments shorter than 1e-12 m are removed. Walls may cross each other, and every crossing point becomes a vertex of the plan.

`Plan.from_json` reads a list of records `{"start": [x, y], "end": [x, y]}`, with an optional z in each point. The command line also reads a CSV with columns `x0, y0, x1, y1`.

A wall is a two-sided segment with no thickness and no inside. An observer inside a closed outline, such as a pillar, sees the inside of that outline. Grid points that fall inside solid objects should be removed before an `isovist_field` call.

## Range and field of view

`max_distance` is the range limit R, in metres. The default is no limit. In 2D, a ray that meets no wall within R ends on the circle of radius R. In 3D, the `escape` policy decides what such a ray counts.

`fov` is the 2D field of view in radians, centred on `direction`. The default is the full turn. With a partial field of view, the perimeter includes the two radial edges that bound it.

## Eye heights

The origin of a view volume is a 3D point `(x, y, z)` in the frame of the scene. Its z is an absolute coordinate. It is a height above the floor only when the floor lies at z = 0, as in a plan extruded with the default `floor=0.0`. The `--eye-height` option of the command line replaces the `z` column of the points file with one value.

In an extruded plan the view volume does not depend on the eye height. Every eye strictly between floor and ceiling sees a volume equal to the height of the prism times the 2D isovist area at its plan position.

## Flags

Some observers have no ordinary metrics. pysovist flags them and keeps their row, so that a field of results keeps the order of its input.

| Flag | Set when | Effect on the metrics |
|---|---|---|
| `on_wall` | The 2D observer is closer than `eps` (1e-9 m) to a wall, or the 3D eye is closer than `eps` to a mesh surface. | NaN. In 2D, `on_wall="raise"` raises a `ValueError` instead. In 3D, `inside="zero"` sets them to 0. |
| `unbounded` | Some ray hits nothing and `max_distance` is infinite. | In 2D, area and perimeter are `inf` and the other metrics NaN. In 3D, the escape policy decides. |
| `inside_occluder` | The eye lies inside a ball of a point cloud, or within `eps` of a ball's surface. | NaN, or 0 with `inside="zero"`. |
| `near_clipped` | The clearance of a 3D eye is below a positive `near_clip`. | None. The row reports a volume that ignores the nearby occluders. |

The first three flags are fields of `result.flags`, and `near_clipped` is an attribute of `ViewVolume`. The table of `isovist_field` has the columns `on_wall` and `unbounded`, and the table of `view_volume_field` has all four. The column `clearance` holds the distance from the observer to the nearest wall, surface or ball surface. It is negative inside a ball.

## Policies of `view_volume`

Four parameters decide what a view volume counts.

`max_distance`
:   The range limit R. Each depth is clipped to R.

`escape`
:   The depth counted for a ray that hits nothing within R. `"clip"` (the default) counts R, so with `max_distance=inf` a single escaping ray makes the volume infinite. `"zero"` counts 0, as the [Unity estimator](unity.md) does. `"nan"` makes the metrics NaN. `flags.unbounded` is set whenever a ray escapes with R infinite, whatever the policy.

`inside`
:   The metrics of an eye inside a ball of a point cloud or closer than `eps` to an occluder. `"nan"` (the default) makes them NaN, and `"zero"` sets them to 0.

`near_clip`
:   A distance below which occluders are ignored. For a point cloud, the balls whose centre lies within `near_clip` of the eye are ignored. For a mesh, each ray starts at distance `near_clip` from the eye. An eye inside clutter, such as the scan of a person standing at the eye point, then sees past it.

A scan with openings, such as windows or open doors, needs `max_distance` or `escape="zero"` to give finite volumes. `escape_fraction` reports the share of the full solid angle through which rays escaped.

## Ray directions

A view volume is a sum over \(N\) ray directions, each carrying the solid angle \(4\pi/N\).

- `directions="fibonacci"` (the default) is an equal-area spiral. It is deterministic, and its error falls faster than \(N^{-1/2}\).
- `directions="random"` draws independent uniform directions from `seed`. Its error is a sampling error with standard deviation of order \(N^{-1/2}\).
- An array of shape (N, 3) is normalised and weighted \(4\pi/N\) per row, so it should be an equal-area design.

`view_volume` uses 16,384 directions by default, and the command line uses 262,144. All eyes of one `view_volume_field` call share the same directions.

## Ball radius of a point cloud

A scan has no surfaces, only samples of them. pysovist turns each point into a solid ball of radius r, so that a dense enough cloud closes into opaque surfaces. A surface sampled on a square lattice of spacing s is sealed when

$$
r > \frac{s}{\sqrt{2}},
$$

and the balls then fill the free space in front of it to a depth between \(\sqrt{r^2 - s^2/2}\) and \(r\). A radius just above the bound closes the surfaces and takes the least free space.

Real scans are not lattices, so the spacing to use is a high percentile of the distance from each point to its nearest neighbour. `PointCloud.spacing()` returns the median, the 90th and the 99th percentile of that distance, measured for 200,000 random points against the whole cloud. Choose the radius above the 99th percentile divided by \(\sqrt{2}\). For the cloud of the [quick start](quickstart.md#a-view-volume-from-a-point-cloud):

```python
sp = cloud.spacing()           # {'median': 0.04, 'p90': 0.04, 'p99': 0.04}
bound = sp["p99"] / np.sqrt(2)  # 0.028 m
cloud.radius > bound            # True: 0.04 m seals the surfaces
```

The radius belongs to the frame of the points. `PointCloud.transform` keeps it, so it is set in the target frame, after any scaling. `PointCloud.with_radius` and the `radius=` argument of `view_volume` change it without copying the points.

## Parallel runs

`isovist_field` distributes observers over `n_jobs` processes. `view_volume_field` casts the rays of all eyes on `n_jobs` threads. On macOS and Windows, and on Linux from Python 3.14, worker processes import the main script again. A script that calls `isovist_field` with `n_jobs` other than 1 must then guard its entry point with `if __name__ == "__main__":`.
