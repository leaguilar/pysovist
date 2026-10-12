# Changelog

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
