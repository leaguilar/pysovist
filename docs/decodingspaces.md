# DeCodingSpaces compatibility

DeCodingSpaces is a Grasshopper toolbox for Rhino. Its 2D Isovist component casts rays from each viewpoint against obstacle lines and reports 17 Isovist Properties. `pysovist.compat.decodingspaces` reproduces these properties from pysovist's exact isovist. It samples the exact depth along the plugin's rays and applies the plugin's formulas, so a Grasshopper definition and a Python script give the same numbers.

The plugin's radial statistics, Circularity, Dispersion and Drift Magnitude are the isovist measures of OmniVista (Conroy Dalton and Dalton 2001), computed on the rays and on the polygon through the ray ends, with the radial statistics left as sums over the rays instead of means. Two rules follow for anyone comparing numbers. The radial statistics grow with the ray count, and dividing them by the number of rays recovers the means that pysovist computes exactly.

## Reproducing the component

```python
from pysovist.compat.decodingspaces import ds_isovist, ds_isovist_properties

walls = [[(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)]]
table = ds_isovist_properties(walls, [(5, 5), (2, 3)], precision=360)  # one row per viewpoint
one = ds_isovist(walls, (2, 3), precision=360)
one.properties["Drift Magnitude"]  # 3.6052 m, the exact drift is sqrt(13) = 3.6056 m
one.polygon                        # the ring the component returns as Isovist Polygon
```

`ds_isovist_properties` returns a table with columns `x`, `y` and the 17 property names spelled as in the plugin. `ds_isovist` returns one viewpoint with its ray angles, depths, hit lines, ray ends, polygon and properties. The arguments mirror the component's inputs, so the view angle is in degrees and the direction is a vector, unlike the radians used elsewhere in pysovist.

| Component input | Argument | Default | Handling |
|---|---|---|---|
| Vantage Point(s) | `viewpoints` | | One row per point. A third coordinate is ignored. |
| Obstacle Lines | `occluders` | | Line segments, or a `Plan`. |
| Viewing Direction | `view_direction` | (0, 1, 0) | A vector. A list as long as the viewpoints gives each its own direction, and any other list gives every viewpoint the first. |
| Viewing Range | `view_range` | 1000 | Metres. Values of 0 and below mean 1000. |
| Viewing Angle | `view_angle_deg` | 360 | Degrees. Values outside (0, 360] mean 360. |
| Precision | `precision` | 360 | Sets the angular step, see below. |

## The rays

Precision sets an angular step, not a ray count. With \(\mathrm{Pr}\) the Precision input, the step is

$$
\Delta\theta = \begin{cases} 2\pi / \mathrm{Pr} & \text{if } 10 < \mathrm{Pr} < 3000, \\ 1^\circ & \text{otherwise.} \end{cases}
$$

The plugin gives no warning outside the accepted range. A Precision of 3,000, 3,600 or 36,000 therefore returns the 360-ray isovist. `ds_angular_step` returns \(\Delta\theta\).

The rays sweep counter-clockwise from \(\delta - \alpha/2\) to \(\delta + \alpha/2\) in steps of \(\Delta\theta\), both ends included, where \(\delta\) is the angle of the view direction and \(\alpha\) the view angle. A full view has \(\mathrm{Pr} + 1\) rays. Its first ray points opposite the view direction, along \(-y\) for the default, and its last ray repeats the first. The repeated ray enters every mean and every sum over the rays. `ds_ray_angles` returns these directions. A ray ends at the first obstacle line it meets, or at the view range when it meets none.

## The 17 properties

The \(m\) rays end at distances \(r_1, \dots, r_m\) from the viewpoint \(\mathbf{v}\), with mean \(\bar r\). The polygon through the ray ends has area \(A\), perimeter \(P\) and centroid \(\mathbf{c}\). \(A_\text{hull}\) is the area of its convex hull and \(\mathbf{d}\) is the view direction vector.

| IP | Property | Plugin's value | pysovist's metric of the same idea |
|---|---|---|---|
| 0 | Area | \(A\) | `area`, the exact area |
| 1 | Distance Weighted Area | \(A\) | none |
| 2 | Perimeter | \(P\) | `perimeter`, the exact boundary length |
| 3 | Compactness | \(4\pi A / P^2\) | `compactness`, the same formula on the exact isovist |
| 4 | Circularity | \(\pi \bar r^2 / A\) | none in the standard set |
| 5 | Convex Deficiency | \(A / A_\text{hull}\), 1 for a convex isovist | `convex_deficiency` \(= 1 - A / A_\text{hull}\), 0 for a convex isovist |
| 6 | Occlusivity | length of the occluding polygon edges, see below | `occlusivity`, the exact depth jumps |
| 7 | Min Radial | \(\min_i r_i\) | `r_min` |
| 8 | Max Radial | \(\max_i r_i\) | `r_max` |
| 9 | Mean Radial | \(\bar r\) | `r_mean`, the mean over the angle |
| 10 | Standard Deviation | \(\sum_i \lvert r_i - \bar r \rvert\) | `r_mad`, the mean absolute deviation |
| 11 | Variance | \(\sum_i (r_i - \bar r)^2\) | `r_var`, the population variance |
| 12 | Skewness | \(\sum_i (r_i - \bar r)^3\) | `r_skew`, the standardised third moment |
| 13 | Dispersion | \(\bar r - \sum_i \lvert r_i - \bar r \rvert\) | `dispersion` \(= r_\text{std} / r_\text{mean}\) |
| 14 | Elongation | \(\bar r / \max_i r_i\) | `elongation`, the ratio of the principal axes |
| 15 | Drift Magnitude | \(\lvert \mathbf{c} - \mathbf{v} \rvert\) | `drift` |
| 16 | Drift Angle | the angle at \(\mathbf{v}\) between two points, see below | `drift_angle`, the direction of the drift |

**Radial statistics.** OmniVista defines the standard deviation and the skewness of the radials as sums divided by the number of radials. The plugin keeps the sums. Dividing Standard Deviation, Variance and Skewness by \(m = \mathrm{Pr} + 1\) gives the mean absolute deviation, the population variance and the third central moment of the sampled depths, which converge to `r_mad`, `r_var` and `r_skew` \(\cdot\, r_\text{std}^3\). In a 10 m square with a 1 m x 2 m pillar, seen from 2 m off one wall and 4 m in front of the pillar, Standard Deviation divided by 3,000 is 1.7063 m at 2,999 rays, and the exact `r_mad` is 1.7069 m. Dispersion follows OmniVista's definition, the mean minus the standard deviation. With the sum in place of the standard deviation it is negative and grows in magnitude with the ray count.

**Circularity.** Conroy Dalton and Dalton (2001) define circularity as the area of the circle whose radius is the mean radial length, divided by the isovist area. Benedikt (1979) uses the name for \(P^2 / 4\pi A\), whose inverse is pysovist's `compactness`. The OmniVista form follows from the exact metrics as \(\pi\, r_\text{mean}^2 / \text{area}\).

**Occlusivity.** The plugin walks the polygon edges counter-clockwise. An edge counts with its full length when its first ray hits a line and the next ray either misses every line or hits a line that shares no end point with the first. Corners between connected walls therefore do not count, and the two edges that bridge the silhouette of a pillar do. An edge from a hit to a miss counts and an edge from a miss to a hit does not, so a wall that leaves the view range adds a short edge on one side only. pysovist's `occlusivity` is the sum of the exact depth jumps, which a wall leaving the range does not produce.

**Elongation.** The plugin divides the mean radial by the longest. A 20 m x 5 m room seen from its centre has Elongation 0.476 at 2,999 rays and a pysovist `elongation` of 4, its aspect ratio.

**Drift Angle.** The plugin measures the angle at the viewpoint \(\mathbf{v}\) between the point \(\mathbf{c} - \mathbf{v}\) and the point \(\mathbf{d}\). Both vectors are read as positions in the plan, so the value depends on where the scene lies. In a 10 m square seen from its centre, the drift is zero and Drift Angle is 0.1107 rad, set by the coordinates of the centre. The same room moved by 100 m gives another value. For the direction of the drift use `drift_angle`, \(\operatorname{atan2}\) of \(\mathbf{c} - \mathbf{v}\).

## Agreement with the plugin

The reference values come from DeCodingSpaces 2021.10.11 in Rhino 7.

| Scenes | Settings | Compared | Worst relative difference |
|---|---|---|---|
| 10 m square from its centre and off-centre, 20 m x 5 m room from its centre, 10 m square with a pillar | 2,999 and 360 rays at range 1,000 m, 2,999 rays at range 4 m | all 17 properties | 1e-9 |
| floor plan of 183 walls, 20 viewpoints | range 40 m, 40 ray counts from 12 to 36,000 | Area | 7e-9, and 3e-8 where the plugin's areas carry six decimals |

The worst case among the 17 properties is Drift Magnitude where the drift is 3e-6 m. Every other property agrees to 2e-11. Where the plugin reports rounding noise for a zero, such as a Standard Deviation of 7e-13 for a view that ends on the range circle in every direction, the module returns 0. The eight ray counts from 3,000 upward give the 360-ray areas, as the fallback rule says. The test suite asserts a relative 1e-9 on the properties and 1e-8 on the areas (1e-7 on those with six decimals).

## Scope

The agreement above covers full views under the default direction. Four behaviours extend the same rules beyond it, without a plugin reference value. For another view direction, the first ray points opposite that direction and the Drift Angle uses that vector. A view angle below 360 degrees sweeps from edge to edge at the same step, and the polygon closes through the viewpoint. Lines that touch at a T-junction, without a shared end point, count as not connected for Occlusivity. Drift Angle is reported without a sign. The component's GPU option, active with more than ten viewpoints, runs a different single-precision computation and is not reproduced.

## References

- Benedikt, M. L. (1979). To take hold of space: isovists and isovist fields. *Environment and Planning B*, 6(1), 47-65.
- Conroy Dalton, R. and Dalton, N. (2001). OmniVista: an application for isovist field and path analysis. *Proceedings of the 3rd International Space Syntax Symposium*, Atlanta, 25.1-25.10.
