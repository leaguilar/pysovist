# Changelog

## 0.3.0 (unreleased)

View volumes have shape metrics, the 3D counterparts of the 2D isovist metrics. `view_volume(..., shape=True)` adds them
to the metrics and `view_volume_field(..., shape=True)` adds them as the last columns, in the order of
`SHAPE_METRIC_NAMES`: the surface and occluding area, sphericity and jaggedness, the drift with its horizontal and
vertical parts and its direction, two elongations, the convex deficiency, and the variance, skewness, mean absolute
deviation and dispersion of the depth weighted by solid angle. They hold for fields of view and angle grids and follow
the escape and inside policies. Neighbouring rays mark an occluding face where their inverse depth changes faster than
any plane at the eye's clearance allows, so a wall seen at a grazing angle never counts as occluding, and `jump_factor`
scales that bound. Each metric is tested against a closed form, the exact 2D isovist of a prism room or shadow faces
built in the tests, and the surface and occluding areas converge with the ray count. Calls without `shape=True` return
what 0.2.1 returns.

Angles given in degrees by mistake now stop with a message, and plan files may list segments. The checked angles are
every field of view (`fov`, `hfov`, `vfov`), every `direction`, the `azimuth` of `vertical_section`, `vertical_isovist`
and `plane_to_world`, and the `elevation` of `vertical_isovist` and `view_volume`. A field of view above its limit
raises a `ValueError`. From 7 on, such as `fov=120` meant as degrees, the message suggests `math.radians`, where 0.2.1
returned the full turn in `isovist`, `segment_visibility`, `segment_ray_hits`, `path_isovists` and the fields. Below 7
it says to pass `math.tau` for the full turn, and a `vfov` up to 2 pi is told that pole to pole is pi. A value up to a
relative 1e-6 above its limit is the limit, so 2 pi and pi in float32 are the full turn and pole to pole, in 2D and in
3D, where 0.2.1 refused them in 3D. A direction, an azimuth or the elevation of `vertical_isovist` beyond 2 pi in size
raises with two cures, `math.radians` for degrees and `math.remainder(a, math.tau)` for radians of more than a turn,
where 0.2.1 took any finite value, also a direction that a full turn leaves unused. Every `direction` and `azimuth` of
one observer also takes a vector `(dx, dy)`, which gives the result of the angle `atan2(dy, dx)` to the last bit, so a
caller who knows a target point needs no angle convention. The DeCodingSpaces layout of `segment_ray_hits` is the one
exception: its vector gives the plugin's rays bit for bit, and the angle may move the last ray across the far edge of
the view. `isovist_field`, `segment_visibility_field` and `view_volume_field` take one angle for every observer. A
vector or a list of angles, which two observers cannot tell apart, raises a `ValueError` that names `math.atan2(dy, dx)`
and the loop to write, where 0.2.1 raised an error that named neither the argument nor the loop. `Plan.from_json` and
the command line read a list of segments `[[x0, y0], [x1, y1]]` as well as records, and refuse a file with one message.
Any other content, a file that is not JSON in UTF-8, and an x or y that is missing or not finite raise a `ValueError`
that names the file, both formats or the record counted from 1. In 0.2.1, `Plan.from_json` raised errors that named no
format and read `{}` as an empty plan, and the command line ended with a traceback on a file that is not UTF-8 or is
nested too deep to parse.

Four other inputs of 0.2.1 give another result. `segment_visibility_field` given 2 pi in float32 counts the visible
angle of each wall over the full turn once, where 0.2.1 counted a sliver of 1.75e-7 rad twice. `segment_ray_hits`
refuses a direction vector of four or more components, where 0.2.1 dropped all but the first two. `plane_to_world` given
an azimuth vector maps the points along it, where 0.2.1 broadcast the vector against two points and returned wrong
coordinates without a message. `sample_path` and `path_isovists` sample the last vertex of a path shorter than 1e-9
spacing, where 0.2.1 returned its first vertex alone. Every other result is that of 0.2.1.

## 0.2.1

Text files are read and written as UTF-8 on every platform: wall plans and closing polygons in JSON, point clouds in
`.pts` files, and the records of the command line. A plan with a non-ASCII name now reads on Windows, whose default
encoding is not UTF-8. A closing polygon that cuts the plan now stops the command line with a message that names
`--close`, as every other inconsistent input does, where 0.2.0 ended with a Python traceback. Every result is that of
0.2.0.

## 0.2.0

The visibility tools of the DeCodingSpaces toolbox for Grasshopper, in pysovist, and checked inputs everywhere. Every
public function is tested against an independent reference: a closed form, an independent library, a brute-force
search, an identity between two routes that share no code, or measured outputs of the plugin. Every code example of
the documentation runs as a test. Every valid input of 0.1.2 gives the same result, which the pin tests assert, with
two exceptions. Point clouds in `.pts` files and the CSV files of the command line are now parsed to the nearest float,
where the fast parser returned some coordinates one unit in the last place off. `Isovist.polygon` of a field of view
now ends at the observer, so every ring repeats its first point.

New functions

- Obstacle visibility: `segment_visibility` gives the exact visible length, fraction and parts of every wall seen from a
  viewpoint, `segment_visibility_field` sums it over many viewpoints (for example along a path), and `segment_ray_hits`
  counts ray hits per wall.
- Directed view volumes: `view_volume` and `view_volume_field` take a horizontal and vertical field of view, a view
  direction and an elevation, and angle-grid directions with exact cell solid angles (`angle_grid`). Rays report the
  object they hit (`Mesh.combine`, `Mesh.cast_hits`, `ViewVolume.hits_per_object`, `ViewVolume.n_objects`), and the
  sky, ground and object factors give the shares of the field that escape, meet the ground or meet other objects.
- Plan tools: `cast_rays` casts single 2D rays and returns the distance, hit point and wall index. `Plan.flood_grid`
  builds an analysis grid flooded from a seed point. `path_isovists` computes exact isovists along a path, looking along
  it. `vertical_isovist` computes exact isovists in the vertical plane through the eye, cut from a mesh.
- DeCodingSpaces compatibility: `pysovist.compat.decodingspaces` reproduces the plugin's 2D Isovist component (its 17
  properties, polygon, radials and hits per line, ray by ray), and `pysovist.compat.decodingspaces3d` reproduces its
  Isovist3D component (ray layout, lengths, surface ids, hits per mesh and volume). The conventions of both are measured
  from the plugin and listed on the DeCodingSpaces page, with the agreement on every measured record.

Checked inputs

- A closing polygon must hold the plan. Every edge of the polygon becomes a wall, the edge from the last vertex back to
  the first included, so a polygon that leaves walls outside cuts through the building. `Mesh.from_plan`, the new
  `Mesh.check_closing` and the `--close` option of the commands raise a `ValueError` when a wall lies mostly (more than
  half its length) outside the polygon. Walls that end on it, run along it or reach past it pass. A plan that already
  includes its outline needs no polygon, and the error says so.
- Inputs that returned a number without a message, or crashed, now raise a `ValueError` that says what is wrong and
  what to do: coordinates that are not finite, arrays of the wrong shape, an empty scene, a missing or empty mesh or
  point-cloud file, walls on one line, a closing polygon without area, a range, field of view, spacing or eps of zero or
  below, counts that are not whole numbers, a near clip at or beyond the range, ground ids the scene does not have, and
  a ball radius of NaN or infinity, which crashed the interpreter.
- The commands check every input before they compute and stop with a message before they write anything: an output
  that would overwrite an input, a points file without finite coordinates, a plan without valid walls, an option that
  does not apply to the scene, a section plane that cuts no ball, and an eye outside an extruded plan. The ODTP mode
  names the variable it needs.
- Defined results instead of errors: `Isovist.depth` and `Isovist.radial` give NaN for an isovist flagged `on_wall` or
  `unbounded`, `sampled_metrics` of a flagged isovist follows its flags, and an eye outside a closed mesh is flagged
  `unbounded` with its escape fraction. `isovist_field` and `segment_visibility_field` take a single point.

## 0.1.2

- BIM models in IFC files, read through IfcOpenShell with the new `ifc` extra. `Mesh.from_ifc` reads the elements of the chosen classes in world coordinates and in metres, whatever the file's length unit. `Plan.from_ifc` cuts them at a height above a storey's elevation, for exact 2D isovists. The presets `"walls"` and `"furnished"` name the classes of a wall plan and of a furnished plan, and lists of IFC classes select any others.
- `Mesh.section(z)` cuts any mesh at the height z into a plan.
- `pysovist isovist --ifc FILE --height H` and `pysovist volume --ifc FILE` read IFC models, with `--ifc-classes` and `--ifc-storey`. The JSON record holds the sha256 of the IFC file.

## 0.1.1

- `pysovist section` gives the isovist area of the horizontal section of a point cloud at eye height, for every point of a CSV. Each row holds one `section_isovist_area` result: area, clearance and the `inside_occluder` and `unbounded` flags. The JSON record holds the version, the parameters and the input checksums, as for the other commands.
- `pysovist odtp` runs it as `TASK=section`. `SECTION_RAYS` sets the rays per eye (default 3600), separately from the `N_RAYS` of view volumes.
- The crossing points of near-parallel walls no longer raise an overflow warning. Isovists are unchanged.

## 0.1.0

- First release.
