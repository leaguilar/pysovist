# Changelog

## 0.1.1

- `pysovist section` gives the isovist area of the horizontal section of a point cloud at eye height, for every point of a CSV. Each row holds one `section_isovist_area` result: area, clearance and the `inside_occluder` and `unbounded` flags. The JSON record holds the version, the parameters and the input checksums, as for the other commands.
- `pysovist odtp` runs it as `TASK=section`. `SECTION_RAYS` sets the rays per eye (default 3600), separately from the `N_RAYS` of view volumes.
- The crossing points of near-parallel walls no longer raise an overflow warning. Isovists are unchanged.

## 0.1.0

- First release.
