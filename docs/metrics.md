# Metric definitions

## The visible depth

Every 2D metric is a functional of one function: the visible depth \(r(\theta)\), the distance from the observer to the first wall in direction \(\theta\). The observer sits at the origin of the formulas below. \(\Theta\) is the measure of the field of view, \(2\pi\) for a full turn.

Seen from the observer, the angles of the plan's vertices (wall endpoints and wall crossings) and of the crossings between walls and the range circle cut the field of view into wedges. Within one wedge the nearest wall never changes. A wedge is therefore bounded either by a range arc, where \(r = R\), or by one wall piece. On a wall piece

$$
r(\theta) = \frac{p}{\cos u}, \qquad u = \theta - \varphi,
$$

where \(p\) is the perpendicular distance from the observer to the line that carries the wall and \(\varphi\) is the direction of that perpendicular. Over a wall piece from \(u_0\) to \(u_1\), every power of \(r\) that the metrics need has a closed form:

$$
\begin{aligned}
\int_{u_0}^{u_1} r \, du &= p \, \Big[\operatorname{arsinh}(\tan u)\Big]_{u_0}^{u_1}, \\
\int_{u_0}^{u_1} r^2 \, du &= p^2 \, \Big[\tan u\Big]_{u_0}^{u_1}, \\
\int_{u_0}^{u_1} r^3 \, du &= \frac{p^3}{2} \, \Big[\sec u \tan u + \operatorname{arsinh}(\tan u)\Big]_{u_0}^{u_1}, \\
\int_{u_0}^{u_1} r^4 \, du &= p^4 \, \Big[\tan u + \tfrac{1}{3}\tan^3 u\Big]_{u_0}^{u_1}.
\end{aligned}
$$

Here \(\operatorname{arsinh}(\tan u) = \ln(\sec u + \tan u)\). On a range arc of angular width \(\Delta\theta\), \(\int r^k \, d\theta = R^k \, \Delta\theta\). Every 2D metric below is built from these terms and the wedge geometry, so no result depends on a number of rays.

## 2D metrics

The keys of `Isovist.metrics` and the columns of `isovist_field`, in the order of `pysovist.METRIC_NAMES`. Lengths are in metres, areas in square metres and angles in radians.

### Area, perimeter and occlusivity

`area`
:   The area of the isovist (Benedikt 1979), \(A = \frac{1}{2} \int_\Theta r^2 \, d\theta\).

`perimeter`
:   The length \(P\) of the whole isovist boundary: visible wall length + range-arc length + occlusivity + field-of-view edges. Benedikt (1979) splits this boundary into real surfaces, occluding radials and the boundary of the region. His perimeter measure counts the real surfaces only, which here is the visible wall length. The field-of-view edges are the two radial edges \(r(\theta_\text{start}) + r(\theta_\text{end})\) that bound a partial field of view.

`occlusivity`
:   The total length of the occluding radial edges, where the boundary jumps in depth (Benedikt 1979). At each wedge boundary \(\theta_i\), the jump is \(\lvert r(\theta_i^+) - r(\theta_i^-) \rvert\). Jumps below \(10^{-9} \max(r_\text{max}, 1)\) are rounding and are not counted.

### Radial statistics

The statistics of the depth over the field of view, each angle weighted equally. With the moments \(m_k = \frac{1}{\Theta} \int_\Theta r^k \, d\theta\):

`r_min`, `r_max`
:   The smallest and largest depth. On a wall piece the smallest depth is \(p\) when the foot of the perpendicular lies inside the wedge, and the depth at an end of the piece otherwise.

`r_mean`
:   \(m_1\).

`r_var`
:   \(m_2 - m_1^2\), the population variance of \(r\).

`r_std`
:   \(\sqrt{m_2 - m_1^2}\).

`r_skew`
:   \((m_3 - 3 m_1 m_2 + 2 m_1^3) / r_\text{std}^3\), and 0 when \(r_\text{std} = 0\).

`r_mad`
:   The mean absolute deviation \(\frac{1}{\Theta} \int_\Theta \lvert r - m_1 \rvert \, d\theta = \frac{2}{\Theta} \int_{r > m_1} (r - m_1) \, d\theta\). On a wall piece \(r > m_1\) where \(\lvert u \rvert > \arccos(p / m_1)\), so the integral is again a closed form.

`dispersion`
:   \(r_\text{std} / r_\text{mean}\).

### Shape

`compactness`
:   \(4 \pi A / P^2\), the inverse of the circularity \(P^2 / 4 \pi A\) of Benedikt (1979), so it equals 1 for a disc. A square seen from its centre has \(\pi / 4\).

`jaggedness`
:   \(P^2 / A\) (Wiener and Franz 2005).

`drift`, `drift_angle`
:   `drift` is the distance \(\lvert \mathbf{c} \rvert\) from the observer to the centroid \(\mathbf{c}\) of the isovist (Conroy 2001). `drift_angle` is its direction, \(\operatorname{atan2}(c_y, c_x)\). A wall wedge is the triangle spanned by the observer and the two ends of its wall piece, with its centroid at one third of the sum of those ends. A range wedge of width \(\Delta\theta\) is a circular sector, with its centroid at distance \(4 R \sin(\Delta\theta / 2) / (3 \Delta\theta)\) along its bisector. Both are exact.

`elongation`
:   \(\sqrt{\lambda_1 / \lambda_2}\), where \(\lambda_1 \ge \lambda_2\) are the principal second moments of area of the isovist about its centroid. It equals the aspect ratio for a rectangle: a 40 m x 2 m hallway has elongation 20.

`convex_deficiency`
:   \((A_\text{hull} - A) / A_\text{hull}\), with \(A_\text{hull}\) the area of the convex hull of the isovist. Range arcs enter the hull as polygons with a step of 0.5 degrees, so this is the one 2D metric with a discretisation.

## 3D metrics

A view volume is star-shaped around the eye and bounded by the visible depth \(r(\omega)\) in each direction \(\omega\) of the unit sphere \(S^2\). Its volume is the 3D analogue of the isovist area:

$$
V = \frac{1}{3} \int_{S^2} r(\omega)^3 \, d\omega \approx \frac{4\pi}{3N} \sum_{i=1}^{N} \min(r_i, R)^3 .
$$

The estimate uses \(N\) directions, each carrying the solid angle \(4\pi / N\), with \(r_i\) the depth along direction \(i\) and \(R\) the range limit. A ray that hits nothing within \(R\) counts \(R\), 0 or NaN, as the `escape` policy says (see [conventions](conventions.md#policies-of-view_volume)). The keys of `ViewVolume.metrics` and the columns of `view_volume_field`, in the order of `pysovist.VOLUME_METRIC_NAMES`:

`volume`
:   \(V\) in cubic metres.

`r_mean`, `r_std`, `r_min`, `r_max`
:   The mean, population standard deviation, minimum and maximum of the depth over the \(N\) directions. All directions carry the same solid angle, so these weight the sphere evenly.

`equivalent_radius`
:   \((3V / 4\pi)^{1/3}\), the radius of the ball with the same volume.

`escape_fraction`
:   The share of the full solid angle in which no occluder is hit within \(R\).

`volume_up`, `volume_down`
:   The parts of \(V\) above and below the eye's horizontal plane. Directions in the plane count half to each, so `volume_up + volume_down = volume`.

`section_isovist_area` gives the isovist area of a horizontal section through a point cloud. The plane \(z = z_\text{eye}\) cuts every ball of radius \(r\) whose centre lies within \(r\) of it in a disk of radius \(\sqrt{r^2 - (z - z_\text{eye})^2}\). With \(n\) rays at equal angle steps \(\Delta\theta = 2\pi / n\), ray \(i\) meets the first disk at depth \(r_i\), and the area is \(\frac{\Delta\theta}{2} \sum_i r_i^2\).

## Ray-sampled metrics

Ray-casting tools cast \(N\) equally spaced rays and describe the isovist by the polygon through the hit points. `sampled_metrics(iso, n_rays)` computes that polygon from the exact depth function, so the error of any ray count can be read off against the exact value of the same scene. In the room with a pillar of the [quick start](quickstart.md):

```python
import numpy as np
import pysovist


def rect(x0, y0, x1, y1):
    v = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


plan = pysovist.Plan(np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)]))
iso = pysovist.isovist(plan, (3.0, 0.0))
sampled = pysovist.sampled_metrics(iso, n_rays=360)
sampled["area"]                # 69.686, against the exact 70.0
sampled["perimeter"]           # 47.233, against the exact 47.416
```

The area of the sampled polygon converges to the exact area as \(N^{-2}\) in a convex room. Where the boundary has occluding edges it converges as \(N^{-1}\), because a ray that straddles a depth jump cuts a triangle off or adds one. The radial statistics of `sampled_metrics` are population statistics over the \(N\) depths. The returned keys are `area`, `perimeter`, `r_min`, `r_max`, `r_mean`, `r_var`, `r_std`, `r_mad` and `compactness`.

## Grasshopper isovist outputs

The isovist components of the DeCodingSpaces toolbox for Grasshopper return 17 outputs. The tool casts a fixed number \(N\) of rays. The table maps each output to pysovist. `sampled_metrics(iso, N)` gives the values of the polygon through those \(N\) hit points.

| Grasshopper output | pysovist | Note |
|---|---|---|
| `Area` | `area` | The sampled value is `sampled_metrics(iso, N)["area"]`. |
| `Perimeter` | `sampled_metrics(iso, N)["perimeter"]` | The perimeter of the sampled polygon. |
| `MinRadial` | `r_min` | |
| `MaxRadial` | `r_max` | |
| `MeanRadial` | `r_mean` | |
| `Variance` | `r_var` | |
| `StandardDeviation` | `r_mad` | The output is smaller than the standard deviation, as a mean absolute deviation is. |
| `Circularity` | `compactness` | \(4 \pi A / P^2\). |
| `Compactness` | no equivalent | Equals \(\sqrt{\text{Circularity}} / \pi^2\), so it ranks observers as `compactness` does. |
| `Occlusivity` | `occlusivity` | |
| `Skewness` | not comparable | Its values track `Variance` in that tool, so it is not a skewness. |
| `Dispersion` | not identified | |
| `Elogation` | not identified | |
| `DistanceWeightedArea` | about \(\pi \cdot\) `r_mean` | |
| `Drift` | `drift` | Not computed by that tool. pysovist computes it. |
| `DriftAngle` | `drift_angle` | Not computed by that tool. pysovist computes it. |
| `ConvexDeficiency` | `convex_deficiency` | Not computed by that tool. pysovist computes it. |

## References

- Benedikt, M. L. (1979). To take hold of space: isovists and isovist fields. *Environment and Planning B*, 6(1), 47-65.
- Conroy, R. (2001). *Spatial navigation in immersive virtual environments*. PhD thesis, University College London. The author now publishes as Ruth Conroy Dalton.
- Wiener, J. M. and Franz, G. (2005). Isovists as a means to predict spatial experience and behavior. In *Spatial Cognition IV*, Lecture Notes in Computer Science 3343, 42-57. doi:10.1007/978-3-540-32255-9_3.
