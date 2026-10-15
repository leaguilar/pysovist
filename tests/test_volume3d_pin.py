"""Pins of the 0.1.2 view volumes, so that later additions cannot move them.

The values were written by pysovist 0.1.2 before any 3D addition. The last
bits of a metric can depend on the processor's instruction set: meshes are
cast by Embree in single precision, and the double-precision metrics of
point clouds differ by a few units in the last place between processors.
Both scenes are therefore pinned twice: to the bit against the 0.1.2
arithmetic applied to ``cast`` on the same machine, and against the stored
values, to 1e-6 for meshes and to 1e-12 for point clouds.
"""

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("numba")

from pysovist.directions import fibonacci_sphere, random_sphere  # noqa: E402
from pysovist.mesh import Mesh  # noqa: E402
from pysovist.pointcloud import PointCloud  # noqa: E402
from pysovist.volume3d import VOLUME_METRIC_NAMES, view_volume, view_volume_field  # noqa: E402

from .geometry import polygon_walls, room_walls  # noqa: E402

L_PLAN = polygon_walls([(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)])

MESH_CASES = {
    "default": ((2.0, 2.0, 1.6), {}),
    "random": ((8.0, 1.0, 0.4), {"n_rays": 2048, "directions": "random", "seed": 5}),
    "gap_clip": ((5.0, 1.0, 1.2), {"n_rays": 4096, "max_distance": 30.0}),
    "gap_zero": ((5.0, 1.0, 1.2), {"n_rays": 4096, "escape": "zero"}),
    "gap_nan": ((5.0, 1.0, 1.2), {"n_rays": 4096, "escape": "nan"}),
}
CLOUD_CASES = {
    "default": ((3.0, 2.5, 1.2), {"n_rays": 4096}),
    "clip": ((3.0, 2.5, 1.2), {"n_rays": 4096, "max_distance": 20.0}),
    "zero_radius": ((3.0, 2.5, 1.2), {"n_rays": 2048, "escape": "zero", "radius": 0.06,
                                       "directions": "random", "seed": 3}),
}
MESH_FIELD_ORIGINS = [(2.0, 2.0, 1.6), (8.0, 1.0, 0.4), (0.0, 5.0, 1.0)]
CLOUD_FIELD_ORIGINS = [(3.0, 2.5, 1.2), (1.0, 1.0, 1.0), (0.0, 3.0, 1.0)]

MESH_PINS = {
    'default': {
        'volume': '0x1.43ff6f4597d21p+7',
        'r_mean': '0x1.4ee3137e1c000p+1',
        'r_std': '0x1.63c2a3b054fc8p+0',
        'r_min': '0x1.199e000000000p+0',
        'r_max': '0x1.0b4aba0000000p+3',
        'equivalent_radius': '0x1.b0dd183856327p+1',
        'escape_fraction': '0x0.0p+0',
        'volume_up': '0x1.07e60901655a0p+6',
        'volume_down': '0x1.8018d589ca4a3p+6',
    },
    'random': {
        'volume': '0x1.0f5fd9eaf8d1ep+7',
        'r_mean': '0x1.00e69d6d38000p+1',
        'r_std': '0x1.9f3844503e5acp+0',
        'r_min': '0x1.9999e60000000p-2',
        'r_max': '0x1.47f5640000000p+3',
        'equivalent_radius': '0x1.9807fea096424p+1',
        'escape_fraction': '0x0.0p+0',
        'volume_up': '0x1.cd8efc9e3f6c2p+6',
        'volume_down': '0x1.44c2dcdec8de4p+4',
    },
    'gap_clip': {
        'volume': '0x1.fe701fa3bbe71p+14',
        'r_mean': '0x1.5416e4b514000p+3',
        'r_std': '0x1.8c0ff56209e4ap+3',
        'r_min': '0x1.3346680000000p+0',
        'r_max': '0x1.e000000000000p+4',
        'equivalent_radius': '0x1.3d4bceb659bc6p+4',
        'escape_fraction': '0x1.2680000000000p-2',
        'volume_up': '0x1.0d251d926f6e0p+14',
        'volume_down': '0x1.e296042298f22p+13',
    },
    'gap_zero': {
        'volume': '0x1.1b0129c1cdd88p+7',
        'r_mean': '0x1.fff725a8a0000p+0',
        'r_std': '0x1.d1274fc54879ep+0',
        'r_min': '0x0.0p+0',
        'r_max': '0x1.2e4f200000000p+3',
        'equivalent_radius': '0x1.9dc75f87af40cp+1',
        'escape_fraction': '0x1.2680000000000p-2',
        'volume_up': '0x1.39c43b2955087p+6',
        'volume_down': '0x1.f87c30b48d50fp+5',
    },
    'gap_nan': {
        'volume': 'nan',
        'r_mean': 'nan',
        'r_std': 'nan',
        'r_min': 'nan',
        'r_max': 'nan',
        'equivalent_radius': 'nan',
        'escape_fraction': '0x1.2680000000000p-2',
        'volume_up': 'nan',
        'volume_down': 'nan',
    },
}
CLOUD_PINS = {
    'default': {
        'volume': 'inf',
        'r_mean': 'inf',
        'r_std': 'nan',
        'r_min': '0x1.3b499c32a91bep+1',
        'r_max': 'inf',
        'equivalent_radius': 'inf',
        'escape_fraction': '0x1.4560000000000p-1',
        'volume_up': 'inf',
        'volume_down': 'inf',
    },
    'clip': {
        'volume': '0x1.4dadac3703f84p+14',
        'r_mean': '0x1.bd794c4e42fb0p+3',
        'r_std': '0x1.01067f66a985dp+3',
        'r_min': '0x1.3b499c32a91bep+1',
        'r_max': '0x1.4000000000000p+4',
        'equivalent_radius': '0x1.136051900d601p+4',
        'escape_fraction': '0x1.4560000000000p-1',
        'volume_up': '0x1.47321bb97c71ep+13',
        'volume_down': '0x1.54293cb48b7eap+13',
    },
    'zero_radius': {
        'volume': '0x1.d0b8fc4bd4d56p+5',
        'r_mean': '0x1.3230cdc29bf6cp+0',
        'r_std': '0x1.9c8f2ca56c000p+0',
        'r_min': '0x0.0p+0',
        'r_max': '0x1.27fc1c732dedap+2',
        'equivalent_radius': '0x1.3386a18df345bp+1',
        'escape_fraction': '0x1.4600000000000p-1',
        'volume_up': '0x1.e43ef2aa1a534p+4',
        'volume_down': '0x1.bd3305ed8f579p+4',
    },
}
MESH_FIELD_PINS = {
    'volume': ['0x1.4457d00e14de4p+7', '0x1.f23b1e9ffdbabp+6', 'nan'],
    'r_mean': ['0x1.4ee14ba300000p+1', '0x1.fc3dab10e0000p+0', 'nan'],
    'r_std': ['0x1.63fa6287e6faap+0', '0x1.8eecff1a4b888p+0', 'nan'],
    'r_min': ['0x1.19e0120000000p+0', '0x1.9a001e0000000p-2', 'nan'],
    'r_max': ['0x1.0994960000000p+3', '0x1.3e0a6a0000000p+3', 'nan'],
    'equivalent_radius': ['0x1.b104704633c06p+1', '0x1.8c8e368db572fp+1', 'nan'],
    'escape_fraction': ['0x0.0p+0', '0x0.0p+0', 'nan'],
    'volume_up': ['0x1.0b2695fd037eep+6', '0x1.b03e816ef9b7ap+6', 'nan'],
    'volume_down': ['0x1.7d890a1f263e1p+6', '0x1.07f274c4100bep+4', 'nan'],
    'clearance': ['0x1.199999999999ap+0', '0x1.999999999999ap-2', '0x1.1e3779b97f4a8p-51'],
}
MESH_FIELD_FLAGS = {
    'on_wall': [False, False, True],
    'inside_occluder': [False, False, False],
    'unbounded': [False, False, False],
    'near_clipped': [False, False, False],
}
CLOUD_FIELD_PINS = {
    'volume': ['0x1.4c488319d883cp+14', '0x1.e3795eb05db9cp+13', 'nan'],
    'r_mean': ['0x1.bc17e1eb6acb2p+3', '0x1.4810b02704ae0p+3', 'nan'],
    'r_std': ['0x1.015b9caf1f696p+3', '0x1.21e71ec46abf7p+3', 'nan'],
    'r_min': ['0x1.3b08b5a1c7e4ep+1', '0x1.ed7e959a7e411p-1', 'nan'],
    'r_max': ['0x1.4000000000000p+4', '0x1.4000000000000p+4', 'nan'],
    'equivalent_radius': ['0x1.12fdedf4e5872p+4', '0x1.eea5797d6c07bp+3', 'nan'],
    'escape_fraction': ['0x1.4400000000000p-1', '0x1.d700000000000p-2', 'nan'],
    'volume_up': ['0x1.4836bc5b84697p+13', '0x1.ad9cac5db24c5p+12', 'nan'],
    'volume_down': ['0x1.505a49d82c9dap+13', '0x1.0cab088184936p+13', 'nan'],
    'clearance': ['0x1.3ae147ae147aep+1', '0x1.eb851eb851eb8p-1', '-0x1.47ae147ae147bp-5'],
}
CLOUD_FIELD_FLAGS = {
    'on_wall': [False, False, False],
    'inside_occluder': [False, False, True],
    'unbounded': [False, False, False],
    'near_clipped': [False, False, False],
}


@pytest.fixture(scope="module")
def closed():
    return Mesh.from_plan(L_PLAN, floor=0.0, ceiling=2.7)


@pytest.fixture(scope="module")
def gap():
    # The first wall is left out, so rays escape through the opening.
    return Mesh.from_plan(L_PLAN[1:], floor=0.0, ceiling=2.7, close=None)


@pytest.fixture(scope="module")
def cloud():
    return PointCloud(room_walls(side=6.0, s=0.05, z0=0.0, z1=2.5), radius=0.04)


def metrics_0_1_2(t, dz, R, escape):
    """The metric arithmetic of pysovist 0.1.2, copied verbatim, for one eye."""
    hit = np.isfinite(t) & (t <= R)
    fill = {"clip": R, "zero": 0.0, "nan": np.nan}[escape]
    r = np.where(hit, np.minimum(t, R), fill)
    k = 4 * np.pi / (3 * t.shape[1])
    up, down, flat = dz > 0, dz < 0, dz == 0
    with np.errstate(invalid="ignore"):
        r3 = r**3
        half = 0.5 * r3[:, flat].sum(axis=1)
        volume = k * r3.sum(axis=1)
        return {
            "volume": volume,
            "r_mean": r.mean(axis=1),
            "r_std": r.std(axis=1),
            "r_min": r.min(axis=1),
            "r_max": r.max(axis=1),
            "equivalent_radius": np.cbrt(3 * volume / (4 * np.pi)),
            "escape_fraction": 1.0 - hit.mean(axis=1),
            "volume_up": k * (r3[:, up].sum(axis=1) + half),
            "volume_down": k * (r3[:, down].sum(axis=1) + half),
        }


def unhex(d):
    return {k: float.fromhex(v) for k, v in d.items()}


def same_bits(a, b):
    """Equal to the last bit, with every NaN equal to every NaN (its sign bit is not pinned)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    return a[~np.isnan(a)].tobytes() == b[~np.isnan(b)].tobytes()


@pytest.mark.parametrize("case", sorted(MESH_CASES))
def test_mesh_view_volume_keeps_the_0_1_2_arithmetic(case, closed, gap):
    origin, kw = MESH_CASES[case]
    mesh = gap if case.startswith("gap") else closed
    vv = view_volume(mesh, origin, **kw)
    n = kw.get("n_rays", 16384)
    d = random_sphere(n, kw["seed"]) if kw.get("directions") == "random" else fibonacci_sphere(n)
    R = kw.get("max_distance", np.inf)
    t = mesh.cast(np.array([origin], dtype=float), d, max_distance=R)
    ref = metrics_0_1_2(t, d[:, 2], R, kw.get("escape", "clip"))
    for name in VOLUME_METRIC_NAMES:
        assert same_bits(vv.metrics[name], ref[name][0]), name
    assert same_bits(vv.directions, d)
    assert same_bits(vv.weights, np.full(n, 4 * np.pi / n))
    assert same_bits(vv.distances, np.minimum(t[0], R))


@pytest.mark.parametrize("case", sorted(MESH_CASES))
def test_mesh_view_volume_matches_the_stored_values(case, closed, gap):
    origin, kw = MESH_CASES[case]
    vv = view_volume(gap if case.startswith("gap") else closed, origin, **kw)
    expected = unhex(MESH_PINS[case])
    for name in VOLUME_METRIC_NAMES:
        np.testing.assert_allclose(vv.metrics[name], expected[name], rtol=1e-6, err_msg=name)


@pytest.mark.parametrize("case", sorted(CLOUD_CASES))
def test_cloud_view_volume_keeps_the_0_1_2_arithmetic(case, cloud):
    origin, kw = CLOUD_CASES[case]
    vv = view_volume(cloud, origin, **kw)
    n = kw["n_rays"]
    d = random_sphere(n, kw["seed"]) if kw.get("directions") == "random" else fibonacci_sphere(n)
    R = kw.get("max_distance", np.inf)
    occ = cloud.with_radius(kw["radius"]) if "radius" in kw else cloud
    t = occ.cast(np.array([origin], dtype=float), d, max_distance=R)
    ref = metrics_0_1_2(t, d[:, 2], R, kw.get("escape", "clip"))
    for name in VOLUME_METRIC_NAMES:
        assert same_bits(vv.metrics[name], ref[name][0]), name


@pytest.mark.parametrize("case", sorted(CLOUD_CASES))
def test_cloud_view_volume_matches_the_stored_values(case, cloud):
    origin, kw = CLOUD_CASES[case]
    vv = view_volume(cloud, origin, **kw)
    expected = unhex(CLOUD_PINS[case])
    for name in VOLUME_METRIC_NAMES:
        np.testing.assert_allclose(vv.metrics[name], expected[name], rtol=1e-12, err_msg=name)


def test_mesh_field_is_pinned(closed):
    df = view_volume_field(closed, np.array(MESH_FIELD_ORIGINS), n_rays=1024)
    for name, values in MESH_FIELD_PINS.items():
        expected = [float.fromhex(v) for v in values]
        np.testing.assert_allclose(df[name], expected, rtol=1e-6, err_msg=name)
    for name, values in MESH_FIELD_FLAGS.items():
        assert df[name].tolist() == values, name
    d = fibonacci_sphere(1024)
    t = closed.cast(np.array(MESH_FIELD_ORIGINS[:2]), d)
    ref = metrics_0_1_2(t, d[:, 2], np.inf, "clip")
    for name in VOLUME_METRIC_NAMES:
        assert same_bits(df[name].to_numpy()[:2], ref[name]), name


def test_cloud_field_is_pinned(cloud):
    df = view_volume_field(cloud, np.array(CLOUD_FIELD_ORIGINS), n_rays=1024, max_distance=20.0)
    for name, values in CLOUD_FIELD_PINS.items():
        expected = [float.fromhex(v) for v in values]
        np.testing.assert_allclose(df[name], expected, rtol=1e-12, err_msg=name)
    for name, values in CLOUD_FIELD_FLAGS.items():
        assert df[name].tolist() == values, name
    d = fibonacci_sphere(1024)
    t = cloud.cast(np.array(CLOUD_FIELD_ORIGINS[:2]), d, max_distance=20.0)
    ref = metrics_0_1_2(t, d[:, 2], 20.0, "clip")
    for name in VOLUME_METRIC_NAMES:
        assert same_bits(df[name].to_numpy()[:2], ref[name]), name
