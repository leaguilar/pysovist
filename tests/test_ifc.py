"""BIM models read from IFC files.

The models are built in the tests with ifcopenshell.api. The room has four
walls 0.2 m thick around an inner floor of 5.6 m x 3.6 m = 20.16 m², laid
out so that the inner faces close at the corners. Every placement is an
exact quarter turn and shift, so the section of the walls is the inner
rectangle to the last bits, and an eye in the room sees exactly 20.16 m².
"""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ifcopenshell")
import ifcopenshell  # noqa: E402
import ifcopenshell.api.aggregate  # noqa: E402
import ifcopenshell.api.attribute  # noqa: E402
import ifcopenshell.api.context  # noqa: E402
import ifcopenshell.api.geometry  # noqa: E402
import ifcopenshell.api.owner.settings  # noqa: E402
import ifcopenshell.api.project  # noqa: E402
import ifcopenshell.api.root  # noqa: E402
import ifcopenshell.api.spatial  # noqa: E402
import ifcopenshell.api.unit  # noqa: E402
import ifcopenshell.util.unit  # noqa: E402

from pysovist import Mesh, Plan, isovist, isovist_field  # noqa: E402
from pysovist.cli import main  # noqa: E402

from .geometry import rect  # noqa: E402

api = ifcopenshell.api
# ifcopenshell 0.8 stamps new files with datetime.utcnow(), which Python 3.12 deprecates.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:ifcopenshell")
OUTER = (6.0, 4.0)  # outer size of the room, m
T = 0.2  # wall thickness, m
H = 3.0  # wall height, m
INNER_AREA = (OUTER[0] - 2 * T) * (OUTER[1] - 2 * T)  # 20.16 m²
TABLE = (2.0, 1.5, 3.0, 2.1)  # plan of the table, xmin, ymin, xmax, ymax
TABLE_HEIGHT = 0.75
EYES = [(1.0, 1.8), (4.5, 2.9), (2.5, 3.3)]


def quarter_turn(k, x, y, z):
    """Placement matrix: a turn of k x 90 degrees about z, then a shift, with exact entries."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    m[:3, 3] = (x, y, z)
    return m


class Model:
    """A small IFC model: storeys holding boxes of any class. Lengths are given in metres."""

    def __init__(self, unit="m", version="IFC4", monkeypatch=None):
        self.f = api.project.create_file(version=version)
        if version == "IFC2X3":  # IFC2X3 entities need an owner history, from a user and an app
            org = self.f.createIfcOrganization(Name="test")
            user = self.f.createIfcPersonAndOrganization(self.f.createIfcPerson(), org)
            app = self.f.createIfcApplication(org, "0", "test", "test")
            settings = ifcopenshell.api.owner.settings
            monkeypatch.setattr(settings, "get_user", lambda f: user)
            monkeypatch.setattr(settings, "get_application", lambda f: app)
        project = api.root.create_entity(self.f, ifc_class="IfcProject", name="test")
        prefix = {"m": None, "mm": "MILLI"}[unit]
        api.unit.assign_unit(self.f, units=[api.unit.add_si_unit(self.f, prefix=prefix)])
        model = api.context.add_context(self.f, context_type="Model")
        self.body = api.context.add_context(self.f, context_type="Model",
                                            context_identifier="Body", target_view="MODEL_VIEW",
                                            parent=model)
        self.building = api.root.create_entity(self.f, ifc_class="IfcBuilding", name="building")
        api.aggregate.assign_object(self.f, relating_object=project, products=[self.building])
        api.geometry.edit_object_placement(self.f, product=self.building)
        self.elevation = {}

    def storey(self, name, elevation, *, placed=True):
        """A storey at ``elevation`` metres, given by its placement or, if not placed, by its
        Elevation attribute alone."""
        st = api.root.create_entity(self.f, ifc_class="IfcBuildingStorey", name=name)
        api.aggregate.assign_object(self.f, relating_object=self.building, products=[st])
        if placed:
            api.geometry.edit_object_placement(self.f, product=st,
                                               matrix=quarter_turn(0, 0, 0, elevation))
        scale = ifcopenshell.util.unit.calculate_unit_scale(self.f)
        api.attribute.edit_attributes(self.f, product=st, attributes={"Elevation":
                                                                      elevation / scale})
        self.elevation[st] = elevation
        return st

    def box(self, storey, ifc_class, length, width, height, *, at=(0.0, 0.0, 0.0), turn=0,
            parent=None):
        """A box ``length`` along its own x, ``width`` along its own y, standing at ``at``
        (metres above the storey) and turned by ``turn`` quarter turns."""
        el = api.root.create_entity(self.f, ifc_class=ifc_class)
        if parent is None:
            api.spatial.assign_container(self.f, relating_structure=storey, products=[el])
        else:
            api.aggregate.assign_object(self.f, relating_object=parent, products=[el])
        rep = api.geometry.add_wall_representation(self.f, context=self.body, length=length,
                                                   height=height, thickness=width)
        api.geometry.assign_representation(self.f, product=el, representation=rep)
        x, y, z = at
        api.geometry.edit_object_placement(
            self.f, product=el, matrix=quarter_turn(turn, x, y, z + self.elevation[storey]))
        return el

    def room(self, storey, *, table=False, slabs=False):
        lx, ly = OUTER
        self.box(storey, "IfcWall", lx, T, H)  # south, the full outer length
        self.box(storey, "IfcWall", lx, T, H, at=(0, ly - T, 0))  # north
        # West and east walls run between them, turned so that their thickness points to -x.
        self.box(storey, "IfcWall", ly - 2 * T, T, H, at=(T, T, 0), turn=1)
        self.box(storey, "IfcWall", ly - 2 * T, T, H, at=(lx, T, 0), turn=1)
        if table:
            x0, y0, x1, y1 = TABLE
            self.box(storey, "IfcFurniture", x1 - x0, y1 - y0, TABLE_HEIGHT, at=(x0, y0, 0))
        if slabs:
            self.box(storey, "IfcSlab", lx, ly, T, at=(0, 0, -T))  # floor, top at 0
            self.box(storey, "IfcSlab", lx, ly, T, at=(0, 0, H))  # ceiling, bottom at H

    def write(self, path):
        self.f.write(str(path))
        return path


def room_file(path, *, unit="m", elevation=0.0, **kw):
    m = Model(unit)
    m.room(m.storey("Ground", elevation), **kw)
    return m.write(path)


def inner_plan(table=False):
    walls = [rect(T, T, OUTER[0] - T, OUTER[1] - T)]
    if table:
        walls.append(rect(*TABLE))
    return Plan(np.concatenate(walls))


# --- plans from IFC ---------------------------------------------------------------------------

def test_isovist_in_a_room_of_four_walls_is_its_inner_floor(tmp_path):
    plan = Plan.from_ifc(room_file(tmp_path / "room.ifc"), 1.2)
    assert len(plan) > 0
    for eye in EYES:
        assert isovist(plan, eye).area == pytest.approx(INNER_AREA, rel=1e-9)


def test_table_occludes_below_its_top_and_not_above(tmp_path):
    path = room_file(tmp_path / "room.ifc", table=True)
    low = Plan.from_ifc(path, 0.5, classes="furnished")
    high = Plan.from_ifc(path, 1.2, classes="furnished")
    for eye in EYES:
        hidden = isovist(inner_plan(table=True), eye).area
        assert hidden < INNER_AREA - 0.6  # the table's own 0.6 m² at least
        assert isovist(low, eye).area == pytest.approx(hidden, rel=1e-9)
        assert isovist(high, eye).area == pytest.approx(INNER_AREA, rel=1e-9)
    # The walls preset leaves the furniture out.
    walls = Plan.from_ifc(path, 0.5)
    assert isovist(walls, EYES[0]).area == pytest.approx(INNER_AREA, rel=1e-9)


def test_millimetre_file_gives_the_same_result_as_metres(tmp_path):
    paths = {u: room_file(tmp_path / f"room_{u}.ifc", unit=u, elevation=3.0, table=True,
                          slabs=True) for u in ("m", "mm")}
    text = paths["mm"].read_text()
    assert "MILLI" in text and "3000." in text  # the file holds millimetres
    meshes = {u: Mesh.from_ifc(p, classes=("furnished", "IfcSlab")) for u, p in paths.items()}
    np.testing.assert_allclose(meshes["mm"].vertices, meshes["m"].vertices, rtol=1e-12,
                               atol=1e-12)
    np.testing.assert_array_equal(meshes["mm"].triangles, meshes["m"].triangles)
    for h in (0.5, 1.2):
        plans = {u: Plan.from_ifc(p, h, classes="furnished", storey="Ground")
                 for u, p in paths.items()}
        for eye in EYES:
            assert isovist(plans["mm"], eye).area == pytest.approx(
                isovist(plans["m"], eye).area, rel=1e-9)
    assert isovist(plans["mm"], EYES[0]).area == pytest.approx(INNER_AREA, rel=1e-9)


@pytest.mark.parametrize("unit", ["m", "mm"])
def test_heights_are_above_the_storey_elevation(tmp_path, unit):
    m = Model(unit)
    m.room(m.storey("Ground", 0.0), table=True)
    upper = m.storey("Level 1", H)
    m.room(upper)
    path = m.write(tmp_path / "two_storeys.ifc")
    ground = Plan.from_ifc(path, 0.5, classes="furnished", storey="Ground")
    level1 = Plan.from_ifc(path, 0.5, classes="furnished", storey="Level 1")
    hidden = isovist(inner_plan(table=True), EYES[0]).area
    assert isovist(ground, EYES[0]).area == pytest.approx(hidden, rel=1e-9)
    assert isovist(level1, EYES[0]).area == pytest.approx(INNER_AREA, rel=1e-9)
    # Without a storey the height is taken from the project origin, and the elements of all
    # storeys are read: 3.5 m cuts the upper walls only, and gives the same segments.
    absolute = Plan.from_ifc(path, H + 0.5, classes="furnished")
    np.testing.assert_array_equal(absolute.segments, level1.segments)
    np.testing.assert_array_equal(Plan.from_ifc(path, 0.5, classes="furnished").segments,
                                  ground.segments)
    # A storey is named by its Name or its GlobalId.
    by_id = Plan.from_ifc(path, 0.5, classes="furnished", storey=upper.GlobalId)
    np.testing.assert_array_equal(by_id.segments, level1.segments)
    lo, hi = Mesh.from_ifc(path, storey="Level 1").bounds
    assert lo[2] == pytest.approx(H) and hi[2] == pytest.approx(2 * H)


def test_storey_without_placement_uses_its_elevation_attribute(tmp_path):
    m = Model("mm")
    m.room(m.storey("Ground", 2.0, placed=False))
    path = m.write(tmp_path / "room.ifc")
    assert isovist(Plan.from_ifc(path, 0.5, storey="Ground"), EYES[0]).area == pytest.approx(
        INNER_AREA, rel=1e-9)
    assert len(Plan.from_ifc(path, 0.5)) == 0  # 0.5 m above the origin lies below the walls


def test_unknown_storey_or_class_is_an_error(tmp_path):
    path = room_file(tmp_path / "room.ifc")
    with pytest.raises(ValueError, match="storey"):
        Plan.from_ifc(path, 1.2, storey="Basement")
    with pytest.raises(ValueError, match="IfcWal"):
        Plan.from_ifc(path, 1.2, classes=["IfcWal"])
    with pytest.raises(ValueError, match="no geometry"):
        Mesh.from_ifc(path, classes="IfcDoor")


# --- class presets ----------------------------------------------------------------------------

ROW = ["IfcWall", "IfcWallStandardCase", "IfcColumn", "IfcCurtainWall", "IfcFurniture",
       "IfcFurnishingElement", "IfcSlab", "IfcDoor", "IfcBuildingElementProxy"]


@pytest.fixture
def row_of_boxes(tmp_path):
    """One 1 m box of every class in ROW, box i at x from 2i to 2i + 1. The curtain wall has
    no body of its own, only a plate as its part, as curtain walls usually are."""
    m = Model()
    st = m.storey("Ground", 0.0)
    for i, cls in enumerate(ROW):
        if cls == "IfcCurtainWall":
            cw = api.root.create_entity(m.f, ifc_class=cls)
            api.spatial.assign_container(m.f, relating_structure=st, products=[cw])
            api.geometry.edit_object_placement(m.f, product=cw)
            m.box(st, "IfcPlate", 1.0, 1.0, 2.0, at=(2 * i, 0, 0), parent=cw)
        else:
            m.box(st, cls, 1.0, 1.0, 2.0, at=(2 * i, 0, 0))
    return m.write(tmp_path / "row.ifc")


def boxes_read(path, classes):
    x = Mesh.from_ifc(path, classes=classes).vertices[:, 0]
    return [ROW[i] for i in sorted(set(np.floor(x / 2).astype(int)))]


@pytest.mark.parametrize("classes, expected", [
    ("walls", ROW[:4]),
    ("furnished", ROW[:6]),
    ("IfcWall", ROW[:2]),  # with its subtypes
    (["IfcSlab", "IfcDoor"], ["IfcSlab", "IfcDoor"]),
    (("walls", "IfcBuildingElementProxy"), [*ROW[:4], "IfcBuildingElementProxy"]),
])
def test_presets_and_class_lists(row_of_boxes, classes, expected):
    assert boxes_read(row_of_boxes, classes) == expected


def test_furnished_preset_reads_ifc2x3(tmp_path, monkeypatch):
    m = Model(version="IFC2X3", monkeypatch=monkeypatch)
    st = m.storey("Ground", 0.0)
    m.room(st)
    x0, y0, x1, y1 = TABLE
    m.box(st, "IfcFurnishingElement", x1 - x0, y1 - y0, TABLE_HEIGHT, at=(x0, y0, 0))
    path = m.write(tmp_path / "room2x3.ifc")
    # IFC2X3 has no IfcFurniture: the preset reads its furnishing elements.
    low = Plan.from_ifc(path, 0.5, classes="furnished")
    hidden = isovist(inner_plan(table=True), EYES[0]).area
    assert isovist(low, EYES[0]).area == pytest.approx(hidden, rel=1e-9)
    with pytest.raises(ValueError, match="IfcFurniture"):
        Plan.from_ifc(path, 0.5, classes=["IfcFurniture"])


# --- view volumes -----------------------------------------------------------------------------

def test_view_volume_of_the_room_is_height_times_isovist_area(tmp_path):
    pytest.importorskip("open3d")
    from pysovist import view_volume

    path = room_file(tmp_path / "room.ifc", slabs=True)
    mesh = Mesh.from_ifc(path, classes=("walls", "IfcSlab"))
    plan = Plan.from_ifc(path, 1.2)
    for eye, z in zip(EYES, (0.4, 1.6, 2.6), strict=True):
        vv = view_volume(mesh, (*eye, z), n_rays=65_536)
        assert vv.metrics["escape_fraction"] == 0.0
        # The relative 1e-3 of the extruded-plan identity tests.
        assert vv.volume == pytest.approx(H * isovist(plan, eye).area, rel=1e-3)


# --- command line -----------------------------------------------------------------------------

@pytest.fixture
def eyes_csv(tmp_path):
    path = tmp_path / "eyes.csv"
    pd.DataFrame({"x": [e[0] for e in EYES], "y": [e[1] for e in EYES]}).to_csv(path,
                                                                                index=False)
    return path


def test_isovist_command_reads_ifc_and_records_its_checksum(tmp_path, eyes_csv):
    path = room_file(tmp_path / "room.ifc", table=True)
    out = tmp_path / "iso.csv"
    main(["isovist", "--ifc", str(path), "--height", "0.5", "--ifc-classes", "furnished",
          "--points", str(eyes_csv), "--out", str(out)])
    df = pd.read_csv(out, float_precision="round_trip")
    ref = isovist_field(Plan.from_ifc(path, 0.5, classes="furnished"),
                        pd.read_csv(eyes_csv).to_numpy())
    pd.testing.assert_frame_equal(df, ref, check_exact=True, check_dtype=False)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["inputs"]["ifc"] == {"path": str(path),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert meta["parameters"] == {"max_distance": math.inf, "n_jobs": 1, "height": 0.5,
                                  "ifc_classes": ["furnished"], "ifc_storey": None}


def test_isovist_command_needs_one_plan_source_and_a_height(tmp_path, eyes_csv):
    path = room_file(tmp_path / "room.ifc")
    out = str(tmp_path / "iso.csv")
    with pytest.raises(SystemExit):
        main(["isovist", "--ifc", str(path), "--points", str(eyes_csv), "--out", out])
    with pytest.raises(SystemExit):
        main(["isovist", "--points", str(eyes_csv), "--out", out])
    walls = tmp_path / "walls.json"
    walls.write_text(json.dumps([{"start": [0, 0], "end": [1, 0]}]))
    with pytest.raises(SystemExit):
        main(["isovist", "--plan", str(walls), "--ifc", str(path), "--height", "1.2",
              "--points", str(eyes_csv), "--out", out])
    with pytest.raises(SystemExit, match="--height"):
        main(["isovist", "--plan", str(walls), "--height", "1.2", "--points", str(eyes_csv),
              "--out", out])


def test_volume_command_reads_ifc(tmp_path, eyes_csv):
    pytest.importorskip("open3d")
    path = room_file(tmp_path / "room.ifc", slabs=True)
    out = tmp_path / "vol.csv"
    main(["volume", "--ifc", str(path), "--ifc-classes", "walls", "IfcSlab", "--eye-height",
          "1.6", "--points", str(eyes_csv), "--n-rays", "65536", "--out", str(out)])
    df = pd.read_csv(out)
    assert df.volume.to_numpy() == pytest.approx(np.full(3, H * INNER_AREA), rel=1e-3)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["inputs"]["ifc"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["parameters"]["ifc_classes"] == ["walls", "IfcSlab"]


# --- the reader itself ----------------------------------------------------------------------------

def enclosed_volume(vertices, triangles):
    """Volume enclosed by closed triangle surfaces, by the divergence theorem."""
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6


@pytest.mark.parametrize("unit", ["m", "mm"])
def test_read_ifc_returns_the_triangles_of_the_elements_and_the_storey_elevation(tmp_path, unit):
    from pysovist.ifc import read_ifc

    path = room_file(tmp_path / "room.ifc", unit=unit, elevation=3.0, table=True)
    vertices, triangles, elevation = read_ifc(path, storey="Ground")
    assert elevation == pytest.approx(3.0, rel=1e-12)
    assert read_ifc(path)[2] == 0.0  # without a storey
    assert vertices.shape[1] == 3 and triangles.shape[1] == 3
    assert triangles.min() == 0 and triangles.max() == len(vertices) - 1
    # Closed form: four wall boxes of thickness T and height H around the inner 5.6 m x 3.6 m.
    walls = (2 * OUTER[0] + 2 * (OUTER[1] - 2 * T)) * T * H
    np.testing.assert_allclose(vertices.min(axis=0), [0, 0, 3.0], atol=1e-9)
    np.testing.assert_allclose(vertices.max(axis=0), [*OUTER, 3.0 + H], atol=1e-9)
    assert enclosed_volume(vertices, triangles) == pytest.approx(walls, rel=1e-12)
    x0, y0, x1, y1 = TABLE
    table = (x1 - x0) * (y1 - y0) * TABLE_HEIGHT
    v, t, _ = read_ifc(path, classes="furnished")
    assert enclosed_volume(v, t) == pytest.approx(walls + table, rel=1e-12)


def test_read_ifc_keeps_the_order_of_the_elements_in_the_file(row_of_boxes):
    from pysovist.ifc import read_ifc

    vertices, triangles, _ = read_ifc(row_of_boxes, classes=ROW)
    box = np.floor(vertices[:, 0] / 2).astype(int)  # box i, written i-th, stands at x = 2i
    assert (np.diff(box) >= 0).all() and sorted(set(box)) == list(range(len(ROW)))
    assert (box[triangles] == box[triangles[:, :1]]).all()  # no triangle joins two elements


def test_openings_are_cut_and_doors_close_them(tmp_path):
    import ifcopenshell.api.feature

    from pysovist.ifc import read_ifc

    m = Model()
    st = m.storey("Ground", 0.0)
    m.room(st)
    south = m.f.by_type("IfcWall")[0]
    width, height, x = 1.0, 2.1, 2.0  # a door opening in the south wall, from x = 2 to 3
    opening = api.root.create_entity(m.f, ifc_class="IfcOpeningElement")
    rep = api.geometry.add_wall_representation(m.f, context=m.body, length=width, height=height,
                                               thickness=T + 0.2)
    api.geometry.assign_representation(m.f, product=opening, representation=rep)
    api.geometry.edit_object_placement(m.f, product=opening,
                                       matrix=quarter_turn(0, x, -0.1, 0.0))
    ifcopenshell.api.feature.add_feature(m.f, feature=opening, element=south)
    door = m.box(st, "IfcDoor", width, T, height, at=(x, 0.0, 0.0))
    ifcopenshell.api.feature.add_filling(m.f, opening=opening, element=door)
    path = m.write(tmp_path / "door.ifc")
    # Closed form: the walls lose the volume of the opening.
    walls = (2 * OUTER[0] + 2 * (OUTER[1] - 2 * T)) * T * H
    assert enclosed_volume(*read_ifc(path)[:2]) == pytest.approx(walls - width * T * height,
                                                                 rel=1e-12)
    # The presets hold no doors: the eye sees out through the opening, and the isovist is open.
    seen = isovist(Plan.from_ifc(path, 1.2), EYES[0])
    assert seen.flags.unbounded and seen.area == math.inf
    # Above the opening, and with the doors, the room is closed again.
    assert isovist(Plan.from_ifc(path, 2.5), EYES[0]).area == pytest.approx(INNER_AREA, rel=1e-9)
    closed = Plan.from_ifc(path, 1.2, classes=("walls", "IfcDoor"))
    for eye in EYES:
        assert isovist(closed, eye).area == pytest.approx(INNER_AREA, rel=1e-9)


def test_a_cut_at_the_top_of_an_element_leaves_it_out(tmp_path):
    # The rule of the BIM models page: an element from z0 to z1 is cut for z0 <= z < z1.
    path = room_file(tmp_path / "room.ifc", table=True)
    at_top = Plan.from_ifc(path, TABLE_HEIGHT, classes="furnished")
    below = Plan.from_ifc(path, TABLE_HEIGHT - 1e-6, classes="furnished")
    assert isovist(at_top, EYES[0]).area == pytest.approx(INNER_AREA, rel=1e-9)
    assert isovist(below, EYES[0]).area == pytest.approx(
        isovist(inner_plan(table=True), EYES[0]).area, rel=1e-9)
    # walls standing on the floor are cut at height 0, and not at their top
    assert isovist(Plan.from_ifc(path, 0.0), EYES[0]).area == pytest.approx(INNER_AREA, rel=1e-9)
    assert len(Plan.from_ifc(path, H)) == 0


def test_a_storey_name_held_by_two_storeys_needs_its_globalid(tmp_path):
    m = Model()
    first = m.storey("Level", 0.0)
    m.room(first)
    second = m.storey("Level", H)
    m.room(second)
    path = m.write(tmp_path / "twins.ifc")
    with pytest.raises(ValueError, match="2 storeys are named 'Level': give a GlobalId"):
        Plan.from_ifc(path, 1.2, storey="Level")
    upper = Plan.from_ifc(path, 1.2, storey=second.GlobalId)
    np.testing.assert_array_equal(upper.segments, Plan.from_ifc(path, H + 1.2).segments)


def test_a_storey_without_placement_or_elevation_stands_at_zero(tmp_path):
    m = Model()
    st = m.storey("Ground", 0.0, placed=False)
    api.attribute.edit_attributes(m.f, product=st, attributes={"Elevation": None})
    m.room(st)
    path = m.write(tmp_path / "room.ifc")
    from pysovist.ifc import read_ifc

    assert read_ifc(path, storey="Ground")[2] == 0.0


def test_parts_are_read_at_any_depth(tmp_path):
    # A curtain wall without a body aggregates an assembly without a body, which aggregates a
    # plate: the plate is read with the curtain wall.
    m = Model()
    st = m.storey("Ground", 0.0)
    cw = api.root.create_entity(m.f, ifc_class="IfcCurtainWall")
    api.spatial.assign_container(m.f, relating_structure=st, products=[cw])
    api.geometry.edit_object_placement(m.f, product=cw)
    assembly = api.root.create_entity(m.f, ifc_class="IfcElementAssembly")
    api.aggregate.assign_object(m.f, relating_object=cw, products=[assembly])
    api.geometry.edit_object_placement(m.f, product=assembly)
    m.box(st, "IfcPlate", 1.0, 0.1, 2.0, at=(3.0, 0, 0), parent=assembly)
    path = m.write(tmp_path / "cw.ifc")
    lo, hi = Mesh.from_ifc(path, classes="IfcCurtainWall").bounds
    np.testing.assert_allclose([lo, hi], [[3.0, 0, 0], [4.0, 0.1, 2.0]], atol=1e-9)


def test_an_empty_class_list_reads_nothing(tmp_path):
    path = room_file(tmp_path / "room.ifc")
    with pytest.raises(ValueError, match="no geometry"):
        Mesh.from_ifc(path, classes=[])


def test_a_map_conversion_is_not_applied(tmp_path):
    import ifcopenshell.api.georeference as georeference

    plain = room_file(tmp_path / "plain.ifc")
    m = Model()
    m.room(m.storey("Ground", 0.0))
    georeference.add_georeferencing(m.f)
    georeference.edit_georeferencing(m.f, coordinate_operation={
        "Eastings": 2.6e6, "Northings": 1.2e6, "OrthogonalHeight": 400.0})
    geo = m.write(tmp_path / "geo.ifc")
    assert "2600000." in geo.read_text()
    np.testing.assert_array_equal(Mesh.from_ifc(geo).vertices, Mesh.from_ifc(plain).vertices)


# --- command line on IFC models -------------------------------------------------------------------

@pytest.fixture
def two_storeys(tmp_path):
    m = Model()
    m.room(m.storey("Ground", 0.0), table=True)
    m.room(m.storey("Level 1", H), table=True, slabs=True)
    return m.write(tmp_path / "two_storeys.ifc")


def test_isovist_command_reads_one_storey_bit_for_bit(two_storeys, tmp_path, eyes_csv):
    out = tmp_path / "iso.csv"
    main(["isovist", "--ifc", str(two_storeys), "--height", "0.5", "--ifc-storey", "Level 1",
          "--ifc-classes", "furnished", "--points", str(eyes_csv), "--out", str(out)])
    plan = Plan.from_ifc(two_storeys, 0.5, classes="furnished", storey="Level 1")
    ref = isovist_field(plan, pd.read_csv(eyes_csv, float_precision="round_trip").to_numpy())
    pd.testing.assert_frame_equal(pd.read_csv(out, float_precision="round_trip"), ref,
                                  check_exact=True, check_dtype=False)
    assert json.loads(out.with_suffix(".json").read_text())["parameters"]["ifc_storey"] == \
        "Level 1"


def test_volume_command_reads_one_storey_bit_for_bit(two_storeys, tmp_path, eyes_csv):
    pytest.importorskip("open3d")
    from pysovist import view_volume_field

    out = tmp_path / "vol.csv"
    main(["volume", "--ifc", str(two_storeys), "--ifc-storey", "Level 1", "--ifc-classes",
          "furnished", "IfcSlab", "--eye-height", str(H + 1.6), "--points", str(eyes_csv),
          "--n-rays", "4096", "--out", str(out)])
    mesh = Mesh.from_ifc(two_storeys, classes=("furnished", "IfcSlab"), storey="Level 1")
    eyes = pd.read_csv(eyes_csv, float_precision="round_trip").assign(z=H + 1.6).to_numpy()
    ref = view_volume_field(mesh, eyes, n_rays=4096)
    pd.testing.assert_frame_equal(pd.read_csv(out, float_precision="round_trip"), ref,
                                  check_exact=True, check_dtype=False)


def test_isovist_command_refuses_a_cut_that_meets_no_element(two_storeys, tmp_path, eyes_csv):
    # A height in millimetres cuts above the model, and gave a table of unbounded rows.
    out = tmp_path / "iso.csv"
    with pytest.raises(SystemExit) as e:
        main(["isovist", "--ifc", str(two_storeys), "--height", "1200", "--ifc-storey",
              "Level 1", "--points", str(eyes_csv), "--out", str(out)])
    assert "meets no element" in e.value.code
    assert "span 0 to 3 m above the storey 'Level 1'" in e.value.code, e.value.code
    assert not out.exists()


def test_volume_command_refuses_an_eye_outside_the_model_in_height(two_storeys, tmp_path,
                                                                   eyes_csv):
    # --eye-height is the absolute z. 1.6 m taken as a height above Level 1 lies below its
    # elements, and gave escaping rays.
    pytest.importorskip("open3d")
    out = tmp_path / "vol.csv"
    with pytest.raises(SystemExit) as e:
        main(["volume", "--ifc", str(two_storeys), "--ifc-storey", "Level 1", "--eye-height",
              "1.6", "--points", str(eyes_csv), "--n-rays", "1024", "--out", str(out)])
    msg = e.value.code
    assert "rows 1, 2, 3" in msg and "z = 3 to 6 m" in msg, msg
    assert "the storey 'Level 1' stands at z = 3 m: an eye 1.6 m above it has z = 4.6" in msg
    assert not out.exists()
