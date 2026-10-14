# Changelog

## 0.2.0

The visibility tools of the DeCodingSpaces toolbox for Grasshopper, in pysovist. Every result of 0.1.2 is unchanged: the
existing functions keep their arguments, defaults and values, which tests pin.

- Obstacle visibility: `segment_visibility` gives the exact visible length, fraction and parts of every wall seen from a
  viewpoint, `segment_visibility_field` sums it over many viewpoints (for example along a path), and `segment_ray_hits`
  counts ray hits per wall.
- Directed view volumes: `view_volume` and `view_volume_field` take a horizontal and vertical field of view, a view
  direction and an elevation, and angle-grid directions with exact cell solid angles (`angle_grid`). Rays report the
  object they hit (`Mesh.combine`, `Mesh.cast_hits`, `ViewVolume.hits_per_object`), and the sky, ground and object
  factors give the shares of the field that escape, meet the ground or meet other objects.
- Plan tools: `cast_rays` casts single 2D rays and returns the distance, hit point and wall index. `Plan.flood_grid`
  builds an analysis grid flooded from a seed point. `path_isovists` computes exact isovists along a path, looking along
  it. `vertical_isovist` computes exact isovists in the vertical plane through the eye, cut from a mesh.
- DeCodingSpaces compatibility: `pysovist.compat.decodingspaces` reproduces the 17 properties of the plugin's 2D Isovist
  component, with its ray layout and its ray-count rule, and `pysovist.compat.decodingspaces3d` reproduces its Isovist3D
  component (ray layout, lengths, surface ids, hits per mesh and volume).

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
