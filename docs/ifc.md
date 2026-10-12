# BIM models

A BIM model describes a building as typed elements: walls, columns, furniture, slabs, doors and windows. pysovist reads IFC files, the open exchange format of BIM models, through IfcOpenShell. The elements of the chosen classes become a floor plan cut at a height, for exact 2D isovists, or a triangle mesh, for view volumes.

```bash
pip install "pysovist[ifc]"
```

```python
import pysovist

# The walls and furniture of one storey, cut 1.2 m above its floor.
plan = pysovist.Plan.from_ifc("model.ifc", 1.2, classes="furnished", storey="Level 1")
field = pysovist.isovist_field(plan, plan.grid(0.5), max_distance=40.0)

# The same elements with floors, ceilings, windows and doors, for view volumes.
classes = ("furnished", "IfcSlab", "IfcCovering", "IfcWindow", "IfcDoor")
mesh = pysovist.Mesh.from_ifc("model.ifc", classes=classes)
vv = pysovist.view_volume(mesh, (3.0, 4.0, 1.6), n_rays=65_536)
```

## What is read

pysovist reads the body geometry of every element of the chosen classes, as IfcOpenShell triangulates it. Each class includes its subtypes. An element also brings the parts it aggregates, so a curtain wall is read through its plates and members.

Coordinates are in the world frame of the file and in metres, whatever the file's length unit. A model in millimetres gives the same plan and the same mesh as the model in metres.

## Classes

`classes` takes a preset, an IFC class, or a list that mixes both.

| Preset | Classes | Use |
|---|---|---|
| `"walls"` (default) | IfcWall with its subtypes, IfcColumn, IfcCurtainWall | The wall plan. |
| `"furnished"` | The walls, IfcFurniture, IfcFurnishingElement | The plan with the furniture that fills the rooms. |

A preset class that the file's schema does not define is left out. `"furnished"` therefore reads the furnishing elements of an IFC2X3 file, which has no IfcFurniture. A class given by name must exist in the schema. `("walls", "IfcWindow", "IfcDoor")` reads the wall plan with its windows and doors.

## Heights and storeys

`Plan.from_ifc(path, height)` cuts the model with a horizontal plane `height` metres above the project origin. With `storey=`, it reads only the elements of that storey and measures `height` from the storey's elevation. A storey is named by its Name or its GlobalId. Its elevation is the height of its placement in world coordinates, or its Elevation attribute when it has no placement.

An element that spans the heights \(z_0\) to \(z_1\) is cut by every plane at a height \(z\) with \(z_0 \le z < z_1\). Walls that stand on the floor are cut at height 0, and a table whose top lies at the cutting height is not.

The plan holds the outline of every element the plane cuts. Walls in a BIM model have a thickness, so an observer in a room sees the inner faces of its walls. A room of 6 m x 4 m outside, enclosed by walls 0.2 m thick, gives an isovist of 5.6 m x 3.6 m = 20.16 m² from any point inside it. As in any plan, an observer inside a wall or a piece of furniture sees the inside of its outline, so grid points that fall inside elements should be removed.

`Mesh.section(z)` gives the plan of any mesh at the height `z`, by the same rule.

## Openings, doors and windows

Openings are cut from the walls that host them. The presets hold no doors and no windows, so an observer sees through every door opening and every window, glazed or not, at the heights the opening spans. Add `IfcDoor` to close the doors in the position the model shows, and `IfcWindow` to make the windows opaque.

## View volumes

`Mesh.from_ifc` keeps the world coordinates of the file. The z of an eye is absolute, as for [any mesh](conventions.md#eye-heights): an eye 1.6 m above a storey at elevation e stands at z = e + 1.6.

The presets hold no floors and no ceilings, so most rays from an eye escape upward or downward. Add the classes that close the rooms, usually `IfcSlab` and `IfcCovering`, or set `max_distance`. The ceiling of a storey is often the floor slab of the storey above, which belongs to that storey, so a view volume is read without `storey=`.

A room enclosed by walls of height \(H\), with a floor slab and a ceiling slab, has the view volume \(V = H A\) of an extruded plan, with \(A\) the isovist area of the room cut at any height between floor and ceiling.

## Command line

```bash
pysovist isovist --ifc model.ifc --height 1.2 --ifc-storey "Level 1" \
    --ifc-classes furnished IfcWindow IfcDoor --points points.csv --out isovists.csv
pysovist volume --ifc model.ifc --ifc-classes furnished IfcSlab IfcCovering IfcWindow IfcDoor \
    --eye-height 1.6 --points points.csv --out volumes.csv
```

The JSON record of the run holds the sha256 of the IFC file, the height, the classes and the storey. The [quick start](quickstart.md#command-line) lists every option.

## Scope

- Geometry is all that is read. A glass pane, a railing and a solid wall occlude alike, and spaces, materials and properties are not used.
- Elements without a body representation are skipped.
- Coordinates stay in the file's own world frame. A map conversion to a geographic frame is not applied.
