# Metric definitions

## The visible depth

Every 2D metric is a functional of one function: the visible depth \(r(\theta)\), the distance from the observer to the first wall in direction \(\theta\). The observer sits at the origin of the formulas below. \(\Theta\) is the measure of the field of view, \(2\pi\) for a full turn.

Seen from the observer, the angles of the plan's vertices (wall endpoints and wall crossings) and of the crossings between walls and the range circle cut the field of view into wedges. Within one wedge the nearest wall never changes. A wedge is therefore bounded either by a range arc, where \(r = R\), or by one wall piece. On a wall piece

$$
r(\theta) = \frac{p}{\cos u}, \qquad u = \theta - \varphi,
$$

where \(p\) is the perpendicular distance from the observer to the line that carries the wall and \(\varphi\) is the direction of that perpendicular. Over a wall piece from \(u_0\) to \(u_1\), the three powers of \(r\) that the metrics need have closed forms:

$$
\begin{aligned}
\int_{u_0}^{u_1} r \, du &= p \, \Big[\operatorname{arsinh}(\tan u)\Big]_{u_0}^{u_1}, \\
\int_{u_0}^{u_1} r^2 \, du &= p^2 \, \Big[\tan u\Big]_{u_0}^{u_1}, \\
\int_{u_0}^{u_1} r^3 \, du &= \frac{p^3}{2} \, \Big[\sec u \tan u + \operatorname{arsinh}(\tan u)\Big]_{u_0}^{u_1}.
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

## 3D shape metrics

`view_volume(..., shape=True)` and `view_volume_field(..., shape=True)` add the 3D counterparts of the 2D metrics, after the volume metrics, in the order of `pysovist.SHAPE_METRIC_NAMES`. Each comes from three numbers per ray: its unit direction \(\mathbf{u}_i\), the solid angle \(w_i\) it carries, and the depth \(r_i\) at which it ends, after the `escape` policy has set the depth of the rays that hit nothing. A ray clipped by the range ends at \(R\). The definitions therefore hold alike for equal-area directions, angle grids and fields of view. Below, \(\Omega = \sum_i w_i\) is the solid angle of the view, \(V\) its volume, and the eye sits at the origin.

### Radial statistics

The statistics of the depth, each ray weighted by its solid angle, like `r_mean` and `r_std`.

`r_var`
:   \(\frac{1}{\Omega} \sum_i w_i (r_i - r_\text{mean})^2\) in m², the square of `r_std`.

`r_skew`
:   \(\frac{1}{\Omega} \sum_i w_i (r_i - r_\text{mean})^3 / r_\text{std}^3\), and 0 when \(r_\text{std} = 0\).

`r_mad`
:   \(\frac{1}{\Omega} \sum_i w_i \lvert r_i - r_\text{mean} \rvert\) in m.

`dispersion`
:   \(r_\text{std} / r_\text{mean}\).

### Centroid and elongation

The cone over the solid angle \(d\omega\), out to the depth \(r\), has the volume \(r^3 d\omega / 3\) and the first moment \(r^4 \mathbf{u}\, d\omega / 4\). The centroid of the view volume is therefore

$$
\mathbf{c} = \frac{1}{V} \int \frac{r^4}{4}\, \mathbf{u} \, d\omega \approx \frac{1}{V} \sum_i w_i \frac{r_i^4}{4}\, \mathbf{u}_i .
$$

`drift`
:   \(\lvert \mathbf{c} \rvert\) in m, the distance from the eye to the centroid (Conroy 2001).

`drift_horizontal`, `drift_vertical`
:   \(\sqrt{c_x^2 + c_y^2}\) and \(c_z\) in m. The second is negative when the centroid lies below the eye.

`drift_azimuth`, `drift_elevation`
:   The direction of \(\mathbf{c}\) in radians: \(\operatorname{atan2}(c_y, c_x)\) counter-clockwise from +x, the counterpart of `drift_angle`, and the angle of \(\mathbf{c}\) above the eye's horizontal plane.

`elongation`, `elongation_12`
:   \(\sqrt{\lambda_1 / \lambda_3}\) and \(\sqrt{\lambda_1 / \lambda_2}\), where \(\lambda_1 \ge \lambda_2 \ge \lambda_3\) are the principal second moments of the volume about its centroid, the eigenvalues of \(\int \frac{r^5}{5} \mathbf{u}\mathbf{u}^\top d\omega - V \mathbf{c}\mathbf{c}^\top\). A box of sides \(a \ge b \ge c\) has the elongation \(a / c\) and the elongation_12 \(a / b\). A corridor of 40 m x 2 m x 3 m has 20 and 13.3, and a hall of 20 m x 20 m x 3 m has 6.7 and 1: the pair tells a rod from a slab.

### Boundary

`surface_area`
:   The area \(S\) in m² of the boundary of the view volume, the counterpart of the perimeter. It holds the visible surfaces, the occluding faces and, for a field of view, the side faces from the eye to the edges of the field.

`occluding_area`
:   The area in m² of the occluding faces, the counterpart of occlusivity. Where a near object hides a far surface, the boundary follows the rays that graze the object, from the object out to the far surface. Along an occluding contour of angular length element \(ds\), the face has the area \(\int (r_\text{far}^2 - r_\text{near}^2) / 2 \, ds\). A full-height pillar in a prism room casts vertical rectangles: its occluding edges in plan times the height.

`sphericity`
:   \(36 \pi V^2 / S^3\), the counterpart of compactness. It equals 1 for a ball seen from its centre, the most compact view.

`jaggedness`
:   \(S^3 / V^2\), the counterpart of \(P^2 / A\), and \(36 \pi\) for a ball.

`convex_deficiency`
:   \((H - V) / H\), with \(H\) the volume of the convex hull of the eye and the ray ends, and 0 when \(H < V\). The hull of the ray ends lies inside the true hull, so the deficiency runs low, by at most 2e-3 at 65,536 rays in the rooms of the [verification](verification.md#3d-shape-metrics).

### The surface through the ray ends

The two areas need a surface, which the rays sample. The unit directions are triangulated by their convex hull, which is their Delaunay triangulation on the sphere, and each triangle is mapped to the triangle through its three ray ends \(r_i \mathbf{u}_i\).

Where a near object hides a far surface, neighbouring rays end on different surfaces. Two neighbouring rays \(i\) and \(j\) see such a depth jump when

$$
\left\lvert \frac{1}{r_i} - \frac{1}{r_j} \right\rvert > k \, \frac{\lvert \mathbf{u}_i - \mathbf{u}_j \rvert}{c},
$$

with \(c\) the clearance of the eye, its distance to the nearest occluder, and \(k\) the `jump_factor`, 1 by default. The bound comes from one identity. A plane at distance \(p\) from the eye, with unit normal \(\mathbf{n}\), is seen at the inverse depth \(1/r = \mathbf{n} \cdot \mathbf{u} / p\), which changes by at most \(\lvert \mathbf{u}_i - \mathbf{u}_j \rvert / p\) between two rays. No wall, floor or ceiling at least the clearance away from the eye passes the test, however grazing the angle at which it is seen, so a 40 m corridor seen from one end has an occluding area of exactly 0. A jump from \(r_\text{near}\) to \(r_\text{far}\) is seen once neighbouring rays are closer than \((1/r_\text{near} - 1/r_\text{far})\, c / k\) apart.

The triangles with a side across a jump form a band along each occluding contour. In each, the face is rebuilt from radial segments from the near surface to the far surface, in the direction halfway along each side across the jump. The depth of each surface there is extrapolated from its own rays, through the fit of \(1/r = \mathbf{a} \cdot \mathbf{u}\) to a ray and its neighbours on the same surface, which is exact on a plane. The extrapolated depth stays within the range of the depths the fit used, widened by that range on either side. The direction halfway along a side lies half a side from the ray and its neighbours a full side away, so any linear trend that a neighbour within 60 degrees of the gradient reveals stays inside the range. A fit tilted by neighbours on other surfaces, such as balls of a point cloud seen through gaps from an eye close to a surface, cannot send a face out to an arbitrary depth. The band follows the lattice of the rays in steps. Each piece of a face therefore counts with its vector area projected onto the mean vector area of the pieces within \(N^{-1/4}\) radians that cross the jump in the same sense, where \(N = 4\pi / \bar{w}\) is the number of directions of the same density over the whole sphere. The steps cancel. The side faces of a field of view, from the eye to its outermost rays, are measured the same way.

Both areas are estimates from sampled ray ends, like a sampled 2D perimeter, and both converge with the ray count. In a room with a pillar, a room with a thin pillar and an L-shaped room, extruded to 3 m, the occluding area lies within 10 % of the exact faces at 4,096 rays, within 1.5 % at 16,384 and within 0.6 % from 65,536 on. The surface area falls short where two surfaces meet at an edge, since the triangles cut the edge, and its error falls as \(N^{-1/2}\), to less than 0.8 % at 262,144 rays. Three prices remain. A plane that passes nearer to the eye than the clearance, with its nearest point off the surface, such as a wall end or a door jamb seen almost edge-on, can count as occluding. For a field of view, the surface ends at the outermost rays, half a ray spacing inside the field's edges. A point cloud is a surface of balls, and once neighbouring rays come closer than the spacing of the balls, the grooves between them count as occluding faces: a 6 m x 4 m x 3 m room sampled every 4 cm by balls of 3 cm has none at 16,384 rays and less than 2 % of its surface at 65,536.

The shape metrics cost about 30 ms per eye at 16,384 rays, on one core. The triangulation of the directions is computed once and shared by all eyes of a call.

### Example

The room with a pillar of the quick start, 3 m high, seen from an eye 2 m beside the pillar:

```python
import numpy as np
import pysovist


def rect(x0, y0, x1, y1):
    v = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], float)
    return np.stack([v, np.roll(v, -1, axis=0)], axis=1)


plan = np.concatenate([rect(-5, -5, 5, 5), rect(-1, -1, 1, 1)])
mesh = pysovist.Mesh.from_plan(plan, floor=0.0, ceiling=3.0)
vv = pysovist.view_volume(mesh, (3.0, 0.0, 1.2), n_rays=65_536, shape=True)
m = vv.metrics
m["occluding_area"]     # 40.39, against 3 m x 2 sqrt(45) m = 40.25 m²
m["surface_area"]       # 279.3, against 2 x 70 m² + 3 m x 47.416 m = 282.25 m²
m["drift_horizontal"]   # 1.8857, the drift of the 2D isovist
m["drift_vertical"]     # 0.300, from the eye at 1.2 m to mid-height
m["convex_deficiency"]  # 0.3000, that of the 2D isovist
m["sphericity"]         # 0.229
```

Inside a prism room every eye sees the extrusion of its 2D isovist, so the 3D metrics follow from the 2D ones. The occluding faces are the two shadow edges of the pillar times the height, and the boundary is the floor and the ceiling plus the height times the 2D perimeter. The surface area falls 1 % short at 65,536 rays, where the triangles cut the edges of the room.

A field of eyes gives one row per eye, with the shape metrics as its last columns:

```python
eyes = [(3.0, 0.0, 1.6), (-2.5, 3.0, 1.6), (0.0, -4.0, 1.6)]
table = pysovist.view_volume_field(mesh, eyes, n_rays=16_384, shape=True)
list(table.columns[-16:]) == list(pysovist.SHAPE_METRIC_NAMES)  # True
```

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

The Isovist component of the DeCodingSpaces toolbox for Grasshopper returns 17 Isovist Properties computed from its rays. `pysovist.compat.decodingspaces` reproduces all 17, and the [DeCodingSpaces compatibility](decodingspaces.md) page maps each to the metric above that measures the same idea. Its Area and Perimeter are those of the sampled polygon: for a full view in the default direction at a Precision \(N\) from 11 to 2,999, `sampled_metrics(iso, N, offset=-np.pi / 2)` gives both.

## References

- Benedikt, M. L. (1979). To take hold of space: isovists and isovist fields. *Environment and Planning B*, 6(1), 47-65.
- Conroy, R. (2001). *Spatial navigation in immersive virtual environments*. PhD thesis, University College London. The author now publishes as Ruth Conroy Dalton.
- Wiener, J. M. and Franz, G. (2005). Isovists as a means to predict spatial experience and behavior. In *Spatial Cognition IV*, Lecture Notes in Computer Science 3343, 42-57. doi:10.1007/978-3-540-32255-9_3.
