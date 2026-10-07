"""Point clouds: reading, transforming, cropping, spacing and clearance."""

import math

import numpy as np
import pytest

from pysovist.pointcloud import PointCloud


def lattice(spacing, n):
    ax = np.arange(n) * spacing
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)


@pytest.fixture
def cloud_xyz():
    rng = np.random.default_rng(0)
    return np.round(rng.uniform(-5, 5, size=(500, 3)), 3)


# --- reading ----------------------------------------------------------------------------------

def test_read_pts_with_count_line(tmp_path, cloud_xyz):
    path = tmp_path / "scan.pts"
    lines = [f"{len(cloud_xyz)}             "]
    lines += [f"{x:.3f} {y:.3f} {z:.3f} -1883 87 92 93" for x, y, z in cloud_xyz]
    path.write_text("\n".join(lines) + "\n")
    pc = PointCloud.read(path, radius=0.02)
    np.testing.assert_allclose(pc.points, cloud_xyz, atol=1e-12)
    assert pc.radius == 0.02
    assert len(pc) == len(cloud_xyz)


def test_read_las(tmp_path, cloud_xyz):
    laspy = pytest.importorskip("laspy")
    las = laspy.create(point_format=3, file_version="1.2")
    las.header.scales = np.array([0.001, 0.001, 0.001])
    las.header.offsets = np.zeros(3)
    las.x, las.y, las.z = cloud_xyz.T
    las.write(tmp_path / "scan.las")
    pc = PointCloud.read(tmp_path / "scan.las")
    np.testing.assert_allclose(pc.points, cloud_xyz, atol=1e-9)
    assert pc.radius == 0.05


@pytest.mark.parametrize("ext", [".ply", ".pcd"])
def test_read_open3d_formats(tmp_path, cloud_xyz, ext):
    o3d = pytest.importorskip("open3d")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_xyz))
    assert o3d.io.write_point_cloud(str(tmp_path / f"scan{ext}"), pcd)
    pc = PointCloud.read(tmp_path / f"scan{ext}")
    np.testing.assert_allclose(pc.points, cloud_xyz, atol=1e-5)


def test_read_e57_with_two_scans(tmp_path, cloud_xyz):
    pye57 = pytest.importorskip("pye57")
    e57 = pye57.E57(str(tmp_path / "scan.e57"), mode="w")
    for part in (cloud_xyz[:200], cloud_xyz[200:]):
        e57.write_scan_raw({"cartesianX": part[:, 0], "cartesianY": part[:, 1],
                            "cartesianZ": part[:, 2]})
    e57.close()
    pc = PointCloud.read(tmp_path / "scan.e57")
    np.testing.assert_allclose(pc.points, cloud_xyz, atol=1e-9)


def test_read_unknown_extension(tmp_path):
    (tmp_path / "scan.xyz9").write_text("1 2 3\n")
    with pytest.raises(ValueError, match="xyz9"):
        PointCloud.read(tmp_path / "scan.xyz9")


# --- geometry ---------------------------------------------------------------------------------

def test_transform_rotation_scale_translation(cloud_xyz):
    c, s = math.cos(0.3), math.sin(0.3)
    m = np.array([[2 * c, -2 * s, 0, 1.0], [2 * s, 2 * c, 0, -2.0], [0, 0, 2, 0.5], [0, 0, 0, 1]])
    pc = PointCloud(cloud_xyz, radius=0.07)
    out = pc.transform(m)
    np.testing.assert_allclose(out.points, cloud_xyz @ m[:3, :3].T + m[:3, 3], atol=1e-12)
    assert out.radius == 0.07  # the radius is set in the target frame, after scaling
    np.testing.assert_array_equal(pc.points, cloud_xyz)
    with pytest.raises(ValueError):
        pc.transform(np.eye(3))


def test_crop_box_and_polygon(cloud_xyz):
    shapely = pytest.importorskip("shapely")
    pc = PointCloud(cloud_xyz)
    box2 = pc.crop((-1, -2, 3, 4))
    keep = ((cloud_xyz[:, 0] >= -1) & (cloud_xyz[:, 0] <= 3)
            & (cloud_xyz[:, 1] >= -2) & (cloud_xyz[:, 1] <= 4))
    np.testing.assert_array_equal(box2.points, cloud_xyz[keep])
    box3 = pc.crop((-1, -2, 0, 3, 4, 2))
    keep3 = keep & (cloud_xyz[:, 2] >= 0) & (cloud_xyz[:, 2] <= 2)
    np.testing.assert_array_equal(box3.points, cloud_xyz[keep3])
    ring = np.array([(-4, -4), (3, -4), (3, -1), (-1, -1), (-1, 4), (-4, 4)], dtype=float)
    poly = pc.crop(ring, z=(-3, 3))
    shp = shapely.Polygon(ring)
    inside = np.array([shp.contains(shapely.Point(x, y)) for x, y in cloud_xyz[:, :2]])
    inside &= (cloud_xyz[:, 2] >= -3) & (cloud_xyz[:, 2] <= 3)
    np.testing.assert_array_equal(poly.points, cloud_xyz[inside])


def test_spacing_of_a_lattice():
    pc = PointCloud(lattice(0.02, 12))
    sp = pc.spacing()
    for key in ("median", "p90", "p99"):
        assert sp[key] == pytest.approx(0.02, rel=1e-9)


def test_clearance_is_distance_minus_radius():
    pc = PointCloud([[0.0, 0, 0], [3.0, 0, 0]], radius=0.1)
    cl = pc.clearance([[1.0, 0, 0], [0.05, 0, 0], [3.0, 4.0, 0]])
    np.testing.assert_allclose(cl, [0.9, -0.05, 3.9], atol=1e-12)
    np.testing.assert_allclose(pc.clearance([[1.0, 0, 0]], radius=0.5), [0.5], atol=1e-12)
