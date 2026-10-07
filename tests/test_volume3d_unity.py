"""The Unity preset: frame, reference volumes and parity on the hospital scan."""

import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, spearmanr

from pysovist.compat.unity import UNITY_PRESET, read_unity_volumes, unity_transform
from pysovist.pointcloud import PointCloud
from pysovist.volume3d import view_volume

DATA = Path(os.environ.get("PYSOVIST_DATA", Path(__file__).resolve().parents[2] / "data" / "raw"))
SCAN = DATA / "scan" / "Hospital2 1.pts"
UNITY = DATA / "unity"
needs_data = pytest.mark.skipif(not (SCAN.exists() and UNITY.exists()),
                                reason="hospital scan and Unity outputs not available")


@pytest.fixture(scope="module")
def hospital():
    return PointCloud.read(SCAN).transform(unity_transform())


# --- frame ------------------------------------------------------------------------------------

def test_transform_is_the_scene_chain():
    # Loader: pts (x, y, z) -> Unity (x, z, y). Transform: T R S with R = 180 deg about Unity y
    # (quaternion (0, 1, 0, 0)), position (9.65, 1.074, 10.45), scale 0.99. Plan = Unity (x, z, y).
    swap = np.eye(4)[[0, 2, 1, 3]]
    T = np.eye(4)
    T[:3, 3] = (9.65, 1.074, 10.45)
    R = np.diag([-1.0, 1.0, -1.0, 1.0])
    S = np.diag([0.99, 0.99, 0.99, 1.0])
    np.testing.assert_allclose(unity_transform(), swap @ T @ R @ S @ swap, atol=1e-15)
    m = unity_transform()
    np.testing.assert_allclose(m @ [0, 0, 0, 1], [9.65, 10.45, 1.074, 1])
    np.testing.assert_allclose(m @ [1, 2, 3, 1], [9.65 - 0.99, 10.45 - 1.98, 1.074 + 2.97, 1])
    assert np.linalg.det(m[:3, :3]) == pytest.approx(0.99**3)  # a rotation, no mirror


def test_preset_runs_view_volume():
    assert UNITY_PRESET == {"directions": "random", "n_rays": 20070, "seed": 1337,
                            "max_distance": math.inf, "escape": "zero", "inside": "zero",
                            "radius": 0.05}
    pc = PointCloud(np.random.default_rng(0).uniform(-1, 1, size=(200, 3)), radius=0.3)
    vv = view_volume(pc, (5.0, 5.0, 5.0), **UNITY_PRESET)
    assert vv.n_rays == 20070 and vv.flags.unbounded and np.isfinite(vv.volume)


@needs_data
def test_reference_volumes_are_in_the_plan_frame():
    for posture, height in (("standing", 1.7), ("sitting", 0.8)):
        q = pd.read_csv(UNITY / "input" / f"query_points_{posture}.csv")
        v = read_unity_volumes(UNITY / "output" / f"query_points_{posture}_volume.csv")
        assert list(v.columns) == ["x", "y", "z", "volume"]
        assert len(v) == len(q) == 7001
        np.testing.assert_allclose(v[["x", "y"]], q[["x", "y"]], atol=1e-5)  # written as float32
        np.testing.assert_array_equal(v["z"], height)


@needs_data
def test_anchor_points_land_on_walls_at_floor_level(hospital):
    """The four registration anchors are wall ends on the plan at floor level (z = 0).

    Each has wall-height points within 0.3 m in plan and the floor within 0.1 m of z = 0
    within 1 m. Shifting the cloud by 0.3 m already breaks the first condition.
    """
    anchors = pd.read_csv(UNITY / "input" / "anchor_points.csv")

    def wall_and_floor(points):
        tree = cKDTree(points[:, :2])
        out = []
        for x, y in anchors[["x", "y"]].to_numpy():
            z = points[tree.query_ball_point((x, y), 0.3), 2]
            zf = points[tree.query_ball_point((x, y), 1.0), 2]
            out.append((int(((z > 0.3) & (z < 2.0)).sum()), float(np.percentile(zf, 5))))
        return out

    for walls, floor in wall_and_floor(hospital.points):
        assert walls >= 300
        assert abs(floor) < 0.1
    shifted = wall_and_floor(hospital.points + [0.3, 0.3, 0.0])
    assert min(w for w, _ in shifted) < 100


# --- parity with the Unity volumes ------------------------------------------------------------

@needs_data
@pytest.mark.slow
def test_parity_with_unity_on_100_standing_points(hospital):
    """Unity's volumes behave like one more draw of the same Monte Carlo estimator.

    Unity drew independent directions for every eye, so each eye gets its own seed here too
    (a shared direction set would correlate the errors across eyes). Two seed sets give the
    spread of a difference of two independent estimates, which is heavy-tailed: a few long
    rays carry much of a volume. Unity's differences must match that spread and show no
    bias beyond it.
    """
    ref = read_unity_volumes(UNITY / "output" / "query_points_standing_volume.csv")
    sample = ref.iloc[np.linspace(0, len(ref) - 1, 100).round().astype(int)]
    sample = sample.reset_index(drop=True)
    origins = sample[["x", "y", "z"]].to_numpy()

    def volumes(first_seed):
        return np.array([view_volume(hospital, o, **{**UNITY_PRESET, "seed": first_seed + i}).volume
                         for i, o in enumerate(origins)])

    t0 = time.perf_counter()
    va = volumes(1337)
    seconds = time.perf_counter() - t0
    vb = volumes(1_000_000)
    vu = sample["volume"].to_numpy()

    # The same eyes stand inside a ball (V = 0), and the ranking agrees.
    np.testing.assert_array_equal(va == 0, vu == 0)
    rho = spearmanr(va, vu).statistic
    assert rho > 0.99

    live = vu > 0
    rel_seed = (va - vb)[live] / va[live]
    rel_unity = (vu - va)[live] / va[live]
    noise, bias = rel_seed.std(), rel_unity.mean()
    assert abs(bias) < 3 * noise / math.sqrt(live.sum())
    assert rel_unity.std() < 1.5 * noise
    assert ks_2samp(rel_seed, rel_unity).pvalue > 0.01
    loa = (bias - 1.96 * rel_unity.std(), bias + 1.96 * rel_unity.std())
    print(f"\nparity on {live.sum()} live of {len(vu)} eyes, {(~live).sum()} zeros agree: "
          f"bias {bias:+.4f}, limits of agreement [{loa[0]:+.4f}, {loa[1]:+.4f}], "
          f"seed-to-seed sd {noise:.4f}, Spearman {rho:.5f}, "
          f"{seconds / len(origins) * 1e3:.1f} ms per eye")
