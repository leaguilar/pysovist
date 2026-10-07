# Verification

The test suite compares pysovist with answers computed another way: closed forms, independent libraries, brute-force searches and an exact identity between the 2D and the 3D code. The tolerances below are the ones the tests assert.

```bash
pip install -e ".[pointcloud,mesh]" pytest hypothesis shapely visilibity
pytest
```

`visilibity` needs SWIG to build. Without `open3d` or `visilibity` the tests that need them are skipped.

## 2D isovists

**Closed-form scenes agree to a relative 1e-9.**

| Scene | Observer | Checked |
|---|---|---|
| 10 m x 10 m square | centre | \(A = 100\), \(P = 40\), occlusivity 0, \(r_\text{min} = 5\), \(r_\text{max} = 5\sqrt{2}\), \(r_\text{mean} = \frac{20}{\pi}\ln(1 + \sqrt{2})\), \(r_\text{var} = \frac{100}{\pi} - r_\text{mean}^2\), compactness \(\pi/4\), drift 0 |
| 10 m x 10 m square | four other points, one 1e-6 m from a wall | \(A = 100\), \(P = 40\), drift = distance to the centre |
| 40 m x 2 m hallway | three points | \(A = 80\), \(P = 84\) |
| Regular n-gons of circumradius \(\rho\), n = 3, 6 and 17 | off-centre | \(A = \frac{n}{2}\rho^2 \sin(2\pi/n)\), \(P = 2n\rho\sin(\pi/n)\) |
| Hallway with range 10 m | centre | \(A = 2(\sqrt{99} + 100 \arcsin 0.1)\), \(P = 4\sqrt{99} + 40\arcsin 0.1\), with no wall endpoint in range |
| Empty plan with range 3 m | any | the disc, \(A = 9\pi\), \(P = 6\pi\) |
| L-shaped room | in one arm | \(A = 20.5\), occlusivity \(\sqrt{4.25}\), \(P = 22.5 + \sqrt{4.25}\) |
| Room with a pillar | 2 m beside the pillar | \(A = 70\), occlusivity \(2\sqrt{45}\), \(P = 34 + 2\sqrt{45}\) |
| 10 m square open on one side, range 40 m | centre | \(A = 75 + 400\pi\), occlusivity \(2(40 - \sqrt{50})\) |
| Square, quarter field of view | centre | \(A = 25\), \(P = 10 + 10\sqrt{2}\) |
| Two crossing walls in a room | below the crossing | \(A\) = 100 minus the hidden polygon |

Six fields of view of 60 degrees add up to the full isovist. A subdivided and partly duplicated outline gives the same isovist as the plain one. A wall seen almost edge-on, which gives a wedge of 5e-6 rad, scales area, perimeter and occlusivity exactly under factors 0.01, 3 and 7. An observer on a wall is flagged `on_wall` with NaN metrics, and an open room without a range limit is flagged `unbounded` with infinite area.

**Random scenes agree with two independent references to 1e-7.** Rectangular rooms with up to nine rectangular pillars agree with the visibility polygon of the visilibity library. Square rooms with up to ten interior walls, which may cross, agree with the room's area minus the union of the walls' shadows, computed with shapely. Each property is checked on up to 150 random scenes.

**Invariants hold on the same random scenes.**

- A rigid motion (any rotation and a shift of up to 1,000 m) leaves area, perimeter, occlusivity, `r_mean`, `r_min`, `r_max` and drift unchanged, to 1e-7.
- Scaling by a factor k from 0.01 to 100 multiplies the area by k² and the perimeter by k, and leaves compactness unchanged, to 1e-7.
- Adding a wall never increases the area. A longer range never decreases it, and the area under range R never exceeds \(\pi R^2\).
- The angular index used for large plans gives the same area and perimeter as the direct search, to 1e-12, on 25 observers among 800 random walls.

**The metric definitions match direct computation.** `r_mean`, `r_mad` and `r_skew` agree with a numerical integration over 2,000,001 rays to 1e-5. A 40 m x 2 m hallway has elongation 20, its aspect ratio. The L-shaped room has convex deficiency \((22.5 - 20.5)/22.5\), and a convex room has 0. The boundary ring of the room with a pillar has area 70.

## Ray sampling

Four rays from the centre of a square give the inscribed diamond: area 50 and perimeter \(20\sqrt{2}\). The error of the sampled area, averaged over 32 positions of the first ray, falls with slope \(-2 \pm 0.2\) in log-log over 64 to 1,024 rays in a convex room. With a pillar it falls with slope \(-1 \pm 0.25\) over 101 to 1,601 rays.

## 3D view volumes from meshes

A closed convex enclosure is star-shaped from any eye inside it, so the view volume equals the enclosed volume. At 65,536 Fibonacci directions the relative error is below 1e-4 unless stated otherwise.

- A cube of half-side \(a = 1.5\) m gives \(8a^3 = 27\) m³ from the centre and from two off-centre eyes. The error from an off-centre eye falls from 3.7e-4 at 1,024 directions to 3.2e-6 at 65,536, with a log-log slope steeper than -0.75. Independent random directions would give -0.5.
- A square pyramid gives 4 m³, and a faceted sphere of radius 2 m gives the volume of its own tetrahedra.
- Random convex hulls of 8 to 40 points agree to 2e-4.
- A range limit of 1.2 m in a cube of half-side 1 m gives the ball minus six caps, 6.38372 m³.
- A floor plate 1 m below the eye with a range of 2 m gives \(9\pi\), with `volume_up` \(= 16\pi/3\).
- A box with its top removed is `unbounded`, and its `escape_fraction` is 1/6, the share of the sphere that the open face covers.
- A cube shifted by 2.6e6 m in x and 1.2e6 m in y, as in a georeferenced frame, keeps the 1e-4 error.

`view_volume_field` gives the same metrics as single `view_volume` calls to 1e-12. The clearance of an eye to random triangles is never larger than a dense sampling of the triangles, and smaller by at most one sampling step.

## The extruded-plan identity

Inside a prism of height H (vertical walls between a horizontal floor and ceiling), the segment from the eye to any point between floor and ceiling crosses a wall exactly when its plan projection does. Every eye therefore sees \(V = H A\), with \(A\) the exact 2D isovist area.

- A room with a pillar and an L-shaped room, 4 eyes each at heights 0.3, 1.6 and 2.4 m (24 eyes), satisfy the identity to a relative 1e-3 at 65,536 directions. The largest residual is 2.4e-4, and it falls to 6e-5 at 262,144 directions.
- The residual falls with the number of directions, from 4,096 to 65,536 to 262,144, so it is the quadrature error of the 3D estimate.
- A concave building whose plan holds only interior walls, closed with `close=` along its L-shaped footprint, satisfies the identity against the plan closed by the same footprint.
- The command line reproduces the identity: 250 m³ for a 10 m x 10 m room with a 2.5 m ceiling, to 2e-3 at 65,536 directions.

## Point clouds

**The ball ray caster agrees with a brute-force first hit to 1e-12.** The grid traversal returns the same misses and the same depths, to an absolute 1e-12, as the minimum over all balls. The tests cover clouds of 1 to 10,000 balls with radii from 0.005 to 0.5 m, eyes in free space, inside balls and outside the cloud, cell sizes from 0.3 to 7 radii, lattice clouds with axis-aligned rays, range limits and near clipping. The 2D caster of horizontal sections agrees with its own brute force to the same tolerance.

**Clouds of balls lie between their erosion bounds.** The faces of a cube of half-side \(a\), sampled on a lattice of spacing \(s\) with balls of radius \(r\), give a view volume between \(8(a - r)^3\) and \(8(a - \sqrt{r^2 - s^2/2})^3\), for \(r\) = 0.03 and 0.05 m and \(s\) = 0.04 m, within the 1e-3 quadrature tolerance. A horizontal section of a 10 m square room sampled every 1 cm, with balls of radius 2 cm, gives an area between the room eroded by \(r\) and the room eroded by \(\sqrt{r^2 - s^2/2}\), a bracket 5.2e-4 wide.

**Policies behave as specified.** An eye inside a ball gets NaN or 0. Near clipping gives exactly the volume of the cloud without the near points. Over an open floor, the `escape_fraction` matches the share of the sphere that the floor leaves open, to 0.01. With `escape="zero"` the volume is \(\frac{4\pi}{3}\) times the mean cubed depth, with escaped depths at 0, and `escape="nan"` gives NaN.

## Directions and the Unity preset

Fibonacci directions are unit vectors with zero mean, and the share of directions in any polar cap equals the cap's share of the sphere to 1/N. Random directions are reproducible from their seed. The weights sum to \(4\pi\) to 1e-14.

The Unity frame transform equals the composition of the Unity loader's axis swap, the scene's rotation, scale and translation, and the swap back. The [Unity preset](unity.md) runs through `view_volume` with 20,070 random directions. Where the Unity reference volumes are available, a further test compares 100 standing eye points: the same eyes get a volume of 0, and the Spearman rank correlation exceeds 0.99. The mean relative difference from Unity lies within three standard errors of zero, and its spread is at most 1.5 times the spread between two pysovist runs with different seeds.
