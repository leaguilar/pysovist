# Verification

The test suite compares pysovist with answers computed another way: closed forms, independent libraries, brute-force searches and an exact identity between the 2D and the 3D code. The tolerances below are the ones the tests assert.

```bash
pip install -e ".[all]" pytest hypothesis shapely visilibity
pytest
```

`visilibity` needs SWIG to build. The suite needs hypothesis, and numba for the point clouds (the `pointcloud` extra). Tests that need open3d, visilibity, shapely, laspy, pye57 or ifcopenshell are skipped when the package is missing. The Unity tests and the DeCodingSpaces tests on a hospital floor plan are skipped without their reference data.

## 2D isovists

**Closed-form scenes agree to 1e-9, relative for values above 1 and absolute below.**

| Scene | Observer | Checked |
|---|---|---|
| 10 m x 10 m square | centre | \(A = 100\), \(P = 40\), occlusivity 0, \(r_\text{min} = 5\), \(r_\text{max} = 5\sqrt{2}\), \(r_\text{mean} = \frac{20}{\pi}\ln(1 + \sqrt{2})\), \(r_\text{var} = \frac{100}{\pi} - r_\text{mean}^2\), compactness \(\pi/4\), drift 0 |
| 10 m x 10 m square | four other points, one 1e-6 m from a wall | \(A = 100\), \(P = 40\), drift = distance to the centre |
| 40 m x 2 m hallway | three points | \(A = 80\), \(P = 84\) |
| Regular n-gons of circumradius \(\rho\), n = 3, 6 and 17 | off-centre | \(A = \frac{n}{2}\rho^2 \sin(2\pi/n)\), \(P = 2n\rho\sin(\pi/n)\) |
| Hallway with range 10 m | centre | \(A = 2(\sqrt{99} + 100 \arcsin 0.1)\), \(P = 4\sqrt{99} + 40\arcsin 0.1\), with no wall endpoint in range |
| Empty plan with range 3 m | (1, 2) | the disc, \(A = 9\pi\), \(P = 6\pi\) |
| L-shaped room | in one arm | \(A = 20.5\), occlusivity \(\sqrt{4.25}\), \(P = 22.5 + \sqrt{4.25}\) |
| Room with a pillar | 2 m beside the pillar | \(A = 70\), occlusivity \(2\sqrt{45}\), \(P = 34 + 2\sqrt{45}\) |
| 10 m square open on one side, range 40 m | centre | \(A = 75 + 400\pi\), occlusivity \(2(40 - \sqrt{50})\) |
| Square, quarter field of view | centre | \(A = 25\), \(P = 10 + 10\sqrt{2}\) |
| Two crossing walls in a room | below the crossing | \(A\) = 100 minus the hidden polygon |

Six fields of view of 60 degrees add up to the area of the full isovist. A subdivided and partly duplicated outline gives the same isovist as the plain one. A wall seen almost edge-on, which gives a wedge of 5e-6 rad, scales area, perimeter and occlusivity under factors 0.01, 3 and 7, to a relative 1e-9. An observer on a wall is flagged `on_wall` with a NaN area, and an open room without a range limit is flagged `unbounded` with infinite area.

**Random scenes agree with two independent references to 1e-7.** Rectangular rooms with up to nine rectangular pillars agree with the visibility polygon of the visilibity library. Square rooms with up to ten interior walls, which may cross, agree with the room's area minus the union of the walls' shadows, computed with shapely. Each property is checked on up to 150 random scenes.

**Invariants hold on the random rooms with interior walls.**

- A rigid motion (any rotation and a shift of up to 1,000 m along each axis) leaves area, perimeter, occlusivity, `r_mean`, `r_min`, `r_max` and drift unchanged, to 1e-7.
- Scaling by a factor k from 0.01 to 100 multiplies the area by k² and the perimeter by k, and leaves compactness unchanged, to 1e-7.
- Adding a wall never increases the area. A longer range never decreases it, and the area under range R never exceeds \(\pi R^2\), each to a relative 1e-9.
- The angular index used for large plans gives the same area and perimeter as the direct search, to 1e-12, on 25 observers among 800 random walls.

**The metric definitions match direct computation.** `r_mean`, `r_mad` and `r_skew` agree with a numerical integration over 2,000,001 rays to 1e-5. A 40 m x 2 m hallway has elongation 20, its aspect ratio. The L-shaped room has convex deficiency \((22.5 - 20.5)/22.5\), and a convex room has 0. The boundary ring of the room with a pillar has area 70.

## Ray sampling

Four rays from the centre of a square give the inscribed diamond: area 50 and perimeter \(20\sqrt{2}\). The error of the sampled area, averaged over 32 positions of the first ray, falls with slope \(-2 \pm 0.2\) in log-log over 64 to 1,024 rays in a convex room. With a pillar it falls with slope \(-1 \pm 0.25\) over 101 to 1,601 rays.

## Obstacle visibility

**The visible part of every wall agrees with closed forms to 1e-9.** The scenes are a square room seen from three points, a room with a pillar, a hallway cut by a range limit, quarter fields of view, fields of view that tile the full turn, a wall split into touching pieces, a wall drawn twice, overlapping collinear walls, a wall seen edge-on and segments of zero length.

**Random rooms agree with the exact isovist and with shadow subtraction.** In square rooms with up to 8 interior walls, which may cross, the visible lengths add up to the part of the isovist perimeter that lies on walls, to a relative 1e-9. The visible length of each wall equals the wall minus the union of the other walls' shadows, computed with shapely, to 1e-6. A shift of up to 1,000 m along each axis and a scaling by a factor from 0.01 to 100 change the lengths as they change the scene, to 1e-7. Each property is checked on up to 120 random scenes.

**Fields and ray counts follow the single viewpoints.** The sums and means of `segment_visibility_field` equal those of single `segment_visibility` calls to 1e-9, and two processes give the table of one, bit for bit. With 20,000 even rays, the hits on each wall times the angular step lie within one step per visible stretch, plus one, of the exact visible angle, on up to 40 random rooms. With the DeCodingSpaces layout, the hit counts equal those of the plugin, as the [DeCodingSpaces 2D layer](#decodingspaces-2d-layer) states.

## Plan tools

**Single rays end at the exact isovist depth.** In a square room, rays along the axes and a diagonal end at their closed-form distances, to a relative 1e-12, and rays through the corners of regular polygons of 3, 4, 7 and 12 sides end on the corners, to 1e-12. In square rooms with up to 10 interior walls, the distance of every ray equals the depth of the exact isovist at the same angle to 1e-9, and its hit point lies on the wall it reports, on up to 150 random scenes.

**Paths sample their arc length and look along it.** Along a straight corridor, with half a turn as the field of view, the areas and perimeters of the rows equal their closed forms to 1e-12. So do the areas walking the other way, and along a path that turns a corner in a square room. Every row has the area and the drift angle of the isovist looking along the tangent of the path, exactly, on up to 40 random paths. On up to 100 random polylines the samples lie on the path at their arc length, to 1e-9. With the full turn as the field of view, the rows equal those of `isovist_field` at the same points, to the relative 1e-5 of the table comparison of pandas.

**The flood grid equals a breadth-first fill.** On up to 60 random scenes with up to 12 walls, spacings from 0.15 to 1.5 m and three values of `eps`, the cells reached are those of a breadth-first fill whose link test is the segment distance of GEOS, computed with shapely, bit for bit. Any cell of the fill, taken as the seed, fills the same cells. An L-shaped room fills its 60 cells of 1 m from three seeds, and no cell of its notch. No centre of the fill of a closed regular polygon lies outside it, on up to 60 random polygons of 3 to 12 sides.

**Vertical isovists are exact.** In a box room the vertical isovist is the rectangle of the room's depth and height, along, across and diagonally, to 1e-12, and a table casts the closed-form shadow behind it. It equals, to 1e-12, the horizontal isovist of the building turned by a quarter turn, as the DeCodingSpaces tutorial computes it. In an L-shaped room and a room with a pillar, extruded to full height, it is the rectangle between the first walls ahead and behind in the plan, to 1e-9, for up to 60 random eyes and azimuths.

## DeCodingSpaces 2D layer

**The layer reproduces the records of the plugin's Isovist component.** The 17 properties agree to a relative 1e-9 (absolute 1e-9 near zero). Each ray points in the direction of the plugin's ray of the same rank to 1e-12 rad, and its length agrees to a relative 1e-9. The ring has the area of the plugin's Isovist Polygon to 1e-9, and the hits per line are equal. On the floor plan of a hospital the areas agree to 1e-8, and to 1e-7 where the plugin's areas carry six decimals. The hospital tests run where the plan and the plugin's areas are present. The plugin's ray layout is implemented twice, in the layer and in `segment_ray_hits`, and the two give the same hits per wall for every Precision, view angle and view direction tested. The [DeCodingSpaces compatibility](decodingspaces.md#agreement-with-the-plugin) page lists the records and the largest difference of each.

## 3D view volumes from meshes

A closed convex enclosure is star-shaped from any eye inside it, so the view volume equals the enclosed volume. At 65,536 Fibonacci directions the relative error is below 1e-4 unless stated otherwise.

- A cube of half-side \(a = 1.5\) m gives \(8a^3 = 27\) m³ from the centre and from two off-centre eyes. The error from an off-centre eye falls from 3.7e-4 at 1,024 directions to 3.2e-6 at 65,536 (measured), and the test asserts a log-log slope steeper than -0.75. Independent random directions would give -0.5.
- A square pyramid gives 4 m³, and a faceted sphere of radius 2 m gives the volume of its own tetrahedra.
- Random convex hulls of 8 to 40 points agree to 2e-4.
- A range limit of 1.2 m in a cube of half-side 1 m gives the ball minus six caps, 6.38372 m³.
- A floor plate 1 m below the eye with a range of 2 m gives \(9\pi\), with `volume_up` \(= 16\pi/3\).
- A box with its top removed is `unbounded`, and its `escape_fraction` is 1/6, the share of the sphere that the open face covers, to 2e-3 at 4,096 directions.
- A cube shifted by 2.6e6 m in x and 1.2e6 m in y, as in a georeferenced frame, keeps the 1e-4 error.
- Rays cast on a mesh from inside a cube end at the closed-form distances to its faces, to a relative 2e-6, with and without a range limit and near clipping.
- From the centre of a cube, `r_mean` and `r_std` agree with integrals over its faces computed with scipy, to 1e-4 and 1e-3.
- The `equivalent_radius` of a faceted sphere of radius 2 m is 2 m to 1e-3.
- Independent random directions estimate the cube's volume without bias: 5 runs of 20,000 directions each lie within 4 standard errors of it.
- `view_volume_field` gives the volume of a cube from 3 eyes to 1e-4, on one thread, on three and on all cores.

`view_volume_field` gives the same metrics as single `view_volume` calls to 1e-12. The clearance of an eye to random triangles is never larger than a dense sampling of the triangles, and smaller by at most one sampling step.

## The extruded-plan identity

Inside a prism of height H (vertical walls between a horizontal floor and ceiling), the segment from the eye to any point between floor and ceiling crosses a wall exactly when its plan projection does. Every eye therefore sees \(V = H A\), with \(A\) the exact 2D isovist area.

- A room with a pillar and an L-shaped room, 4 eyes each at heights 0.3, 1.6 and 2.4 m (24 eyes), satisfy the identity to a relative 1e-3 at 65,536 directions. The largest residual is 2.4e-4 (measured), and it falls to 6e-5 at 262,144 directions.
- For three eyes of the L-shaped room, the residual falls with the number of directions, from 4,096 to 65,536 to 262,144, so it is the quadrature error of the 3D estimate.
- A concave building whose plan holds only interior walls, closed with `close=` along its L-shaped footprint, satisfies the identity against the plan closed by the same footprint.
- The command line reproduces the identity: 250 m³ for a 10 m x 10 m room with a 2.5 m ceiling, to 2e-3 at 65,536 directions.

## Directed view volumes and view factors

**Directed volumes converge to the exact wedge of a box room.** The reference integrates the cubed depth over each field in closed form in elevation and by adaptive quadrature in azimuth, to 1e-13. Over 5 fields, from 30 x 20 degrees to a full turn 40 degrees tall, one of them across the azimuth \(\pm\pi\), an angle grid of 768 x 384 cells agrees to 2e-5, and 2^20 Fibonacci directions to 1e-4. The weights of an angle grid sum to the solid angle of its field to 1e-13, and to \(4\pi\) over the sphere to 1e-14. Eight octants add up to the full view volume to 1e-12.

**View factors agree with exact solid angles.** The sky, ground and object factors of an eye that sees a ground square and a wall agree with the exact solid angles of the two rectangles (Van Oosterom and Strackee 1983) to an absolute 5e-4, with 2^18 Fibonacci directions and with an angle grid of 720 x 360 cells. The three factors add up to 1 to 1e-14. In a field of view, the rows of `view_volume_field`, factors included, equal single `view_volume` calls to 1e-12.

## 3D shape metrics

**Closed forms.** Rays all clipped at a range of 2 m give the ball: a surface of \(4\pi R^2\) to 1e-3 at 16,384 rays, sphericity 1, no drift, elongation 1 and a depth without spread. A 6 m x 4 m x 3 m box seen from an interior eye gives its 108 m² to 1e-2 at 65,536 rays, the distance to its centre as the drift to 1e-4, the elongations 6/3 and 6/4 to 1e-3, and no occluding face. A cube 2 m in front of the eye, in empty space with a range of 6 m, casts four planar shadow faces out to the range sphere, \(R^2 \arctan(a/p) - p a\) each for an edge of half-length \(a\) at distance \(p\), which the occluding area matches to 1e-2. With `escape="zero"`, the rays through the open top of a box end at the eye, and the occluding area matches the four triangles from the eye to the rim to 1e-2. The same box sampled every 4 cm by balls of 3 cm, through the ball ray caster, gives a surface between those of the box eroded by the two bounds of the balls, the drift of the box centre to 1e-3 and elongations between those of the two erosions.

**The extruded-plan identity carries over.** In a prism room every eye sees the extrusion of its exact 2D isovist. In a room with a pillar from two eyes, a room with a pillar 0.3 m wide and an L-shaped room, at 65,536 rays, the occluding area equals the height times the 2D occlusivity to 1e-2. The surface area equals the floor and the ceiling plus the height times the 2D perimeter to 1.5e-2. The horizontal drift and its azimuth equal those of the 2D isovist to 1e-3, and the convex deficiency equals the 2D one to 2e-3. Shadow faces of a full-height pillar, built from the plan in the test, agree to 1e-2 from two eyes. From the square where its arms overlap, the L-shaped room gives the convex deficiency of its two boxes, 18/82, to 1e-3.

**Both areas converge with the ray count.** At 4,096, 16,384, 65,536 and 262,144 rays, the occluding area of these rooms lies within 10 %, 1.5 %, 0.6 % and 0.6 % of the exact faces. The surface area of these rooms and of the box falls short at every count, with a log-log slope between -0.6 and -0.4, and by less than 0.8 % at 262,144 rays. A 40 m x 2 m x 3 m corridor seen from one end has an occluding area of exactly 0 at 4,096, 16,384 and 65,536 rays. A pillar 19 m from the eye, in a hall of 60 m x 40 m x 6 m, is missed at 4,096 rays and found to 2e-2 at 262,144.

**Directed views and statistics.** For the part of a ball inside four fields of view, with 65,536 Fibonacci directions and with a grid of 360 x 180 cells, the volume agrees with its closed form to 1e-3, the surface with its side faces to 2e-2 and the centroid to 2e-3. Seen from 1.7 m above a ground plane, a view raised 10 degrees whose nearest ground lies 5 m away has no occluding face. Far surfaces seen through 60 holes of three rays in a near cap, from an eye 2 cm from a surface, keep their occluding faces below the area that the deepest ray allows around the holes, for 6 random scenes. From the centre of a cube, `r_var`, `r_skew`, `r_mad` and `dispersion` agree with integrals over its faces computed with scipy to 2e-3, for equal-area directions and for an angle grid. For equal weights they equal the population statistics of numpy and scipy to 1e-9. The rows of `view_volume_field` equal single `view_volume` calls to 1e-12, for a full view and for a field of view with view factors.

## DeCodingSpaces Isovist3D layer

**The layer reproduces the Isovist3D component to the component's single precision.** Each ray points in the direction of the component's ray to 1e-6, its length agrees to a relative 2e-6, and its hit point to 2e-6 of its length. The surface ids are equal except on rays that meet a floor exactly on the diagonal its two triangles share, and the hits per mesh differ by at most those rays. The volumes of the measured views agree to a relative 2e-6, or an absolute 1e-5 for a grid that encloses nothing. They agree to 5e-6 for a closed room and an extruded hospital plan up to 720 x 360 rays. From there to 2,880 x 1,440 rays they agree to 2e-4, except at the few counts that the [directed view volumes](directed-volumes.md#decodingspaces-isovist3d) page lists.

**The component's volume is a polyhedron, not the solid-angle integral.** 12 x 6 rays from the centre of the closed room give a prism of 60.75 m³, to 1e-6. On the hospital plan, the exact volume is 3 m times the exact isovist area, 200.009 m³. The component's volumes lie 3.30 %, 1.08 % and 3.39 % below it at 360 x 180, 720 x 360 and 1,440 x 720 rays, to 0.05 %.

## BIM models

**A horizontal section of a mesh is exact.** The section of an extruded plan, at the floor, at 0.3 and 1.6 m and just below the ceiling, gives the isovist areas of the plan to a relative 1e-9, for the room with a pillar and the L-shaped room, four eyes each. With the default closure, the section adds the walls along the convex hull. Sections below the floor, at the ceiling and above it are empty. The section of a random convex hull of 8 to 40 points gives the area of the polygon that its half-spaces cut out at that height, computed with scipy, to 1e-9 on up to 25 random hulls.

**IFC models built in the tests give closed-form isovists.** The models are written with IfcOpenShell. A room of 6 m x 4 m outside, enclosed by four walls 0.2 m thick, gives 20.16 m² from three eyes, to a relative 1e-9.

- A table 0.75 m high in the room hides, at 0.5 m, the same area as its outline in a 2D plan, to 1e-9. At 1.2 m the area is 20.16 m² again, and the `"walls"` preset leaves the table out.
- The same model in millimetres gives the same mesh to 1e-12 and the same isovists to 1e-9.
- In a model with two storeys 3 m apart, a height is measured from the storey's elevation: 0.5 m above the upper storey gives the same segments as 3.5 m above the origin. A storey is found by its Name or its GlobalId, and a storey without a placement takes its Elevation attribute.
- The presets read exactly their classes and the subtypes of those classes, a curtain wall is read through its plate, and `"furnished"` reads the furnishing elements of an IFC2X3 model. An unknown class or storey is an error.
- With a floor slab and a ceiling slab, the view volume of the room equals the wall height times its isovist area, to the 1e-3 of the extruded-plan identity at 65,536 directions, from eyes at 0.4, 1.6 and 2.6 m.
- The four walls enclose 11.52 m³, the volume of their boxes, to a relative 1e-12, in metres and in millimetres. A door opening of 1 m x 2.1 m takes its 0.42 m³ out of the wall, and an eye in the room sees out through it at 1.2 m. With `IfcDoor` the door closes the opening, and the room gives 20.16 m² again.
- A cut at the top of the table leaves the table out, and a cut at 0 holds the walls. Parts are read through any depth of aggregation, a storey name held by two storeys asks for a GlobalId, and a map conversion in the file leaves the coordinates unchanged.
- The `isovist` command gives exactly the table of `isovist_field` on `Plan.from_ifc`, and the `volume` command that of `view_volume_field` on `Mesh.from_ifc`, also for one storey of two. The `volume` command gives 3 m x 20.16 m² to 1e-3. Both record the sha256 of the IFC file.

## Point clouds

**The ball ray caster agrees with a brute-force first hit to 1e-12.** The grid traversal returns the same misses and the same depths, to an absolute 1e-12, as the minimum over all balls. The tests cover clouds of 1 to 10,000 balls with radii from 0.005 to 0.5 m, eyes in free space, inside balls and outside the cloud, cell sizes from 0.3 to 7 radii, lattice clouds with axis-aligned rays, range limits and near clipping. The 2D caster of horizontal sections agrees with its own brute force to the same tolerance.

**Clouds of balls lie between their erosion bounds.** The faces of a cube of half-side \(a\), sampled on a lattice of spacing \(s\) with balls of radius \(r\), give a view volume between \(8(a - r)^3\) and \(8(a - \sqrt{r^2 - s^2/2})^3\), for \(r\) = 0.03 and 0.05 m and \(s\) = 0.04 m, within the 1e-3 quadrature tolerance. A horizontal section of a 10 m square room sampled every 1 cm, with balls of radius 2 cm, gives an area between the room eroded by \(r\) and the room eroded by \(\sqrt{r^2 - s^2/2}\), a bracket 5.2e-4 of the area wide. The `section` command gives an area between the two erosion bounds for the same room sampled every 2 cm, with balls of radius 3 cm. In a sampled room with a pillar, its areas, clearances and flags equal those of `section_isovist_area` bit for bit.

**Policies behave as specified.** An eye inside a ball gets NaN or 0. Near clipping gives exactly the volume of the cloud without the near points. Over an open floor, the `escape_fraction` matches the share of the sphere that the floor leaves open, to 0.01. With `escape="zero"` the volume is \(\frac{4\pi}{3}\) times the mean cubed depth, with escaped depths at 0, and `escape="nan"` gives NaN.

## Directions and the Unity preset

Fibonacci directions are unit vectors whose mean is within 2/N of zero, and the share of directions in a polar cap about z equals the cap's share of the sphere to 1/N. Random directions are reproducible from their seed. The weights sum to \(4\pi\) to 1e-14.

The Unity frame transform equals the composition of the Unity loader's axis swap, the scene's rotation, scale and translation, and the swap back. The [Unity preset](unity.md) runs through `view_volume` with 20,070 random directions. Where the Unity reference volumes are available, a further test compares 100 standing eye points: the same eyes get a volume of 0, and the Spearman rank correlation exceeds 0.99. The mean relative difference from Unity lies within three standard errors of zero, with the standard error taken from the spread between two pysovist runs, and its spread is at most 1.5 times the spread between two pysovist runs with different seeds.

## Checked inputs

**An input outside the domain of a function raises a ValueError that names the argument and what to pass.** Four examples, each asserted by a test:

- `isovist(plan, (5, 5), max_distance=-1)` raises "max_distance must be positive (inf for no limit), got -1.0".
- `isovist(plan, (5, 5), fov=120)`, an angle in degrees, raises "fov must be an angle in radians of at most 2 pi, the full turn, got 120.0. An angle in degrees converts with math.radians, such as math.radians(120.0)."
- `PointCloud(points, radius=float("nan"))` raises "radius must be a positive, finite distance in metres, not nan".
- `Mesh(vertices, quads)` with four indices per face raises "triangles must have shape (m, 3), not (1, 4): split each quad (a, b, c, d) into (a, b, c) and (a, c, d)".

Where a result is defined instead, the docstring states it and a test asserts it: an eye outside the closed domain of an extruded plan is flagged `unbounded`, and a 2D isovist flagged `on_wall` or `unbounded` has NaN depths.

## Command line and documentation

**Every command gives the table of the library call it wraps, bit for bit.** `isovist` on a plan in JSON records, a JSON list of segments or CSV, closed by a polygon or not, with a range and on two processes, equals `isovist_field`. `volume` on an extruded plan, a mesh file, a point cloud and an IFC model, with every policy option, equals `view_volume_field`, and `section` equals one `section_isovist_area` call per point. ODTP mode maps every variable to its option and writes the same table as the command. Every kind of inconsistent input listed in the [quick start](quickstart.md#command-line) stops the command with a message, before it writes anything.

**Every example in this documentation runs.** The Python blocks of each page run in order as one script in an empty directory, and the `pysovist` commands of the page run after them. Every value that a page states in a comment or in a printed table agrees with the output to the digits written, and so do the results that the quick start states in its sentences. The blocks that read a scan or a BIM model of the reader's own are not run, and the names they use are defined by the blocks before them. The tolerances and counts that this page states for obstacle visibility, the plan tools, the two DeCodingSpaces layers and directed view volumes equal those in the tests that assert them, and the messages it quotes under checked inputs are those the library raises.
