"""BIM models from IFC files, read through IfcOpenShell.

The body geometry of the chosen elements is triangulated by IfcOpenShell in
world coordinates and converted to metres, whatever the file's length unit.
Openings are cut from the elements that host them, as IfcOpenShell does by
default. An element of a chosen class brings the parts it aggregates, such as
the plates and members of a curtain wall.

ifcopenshell is imported on first use, so ``import pysovist`` works without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_WALLS = ("IfcWall", "IfcColumn", "IfcCurtainWall")

PRESETS: dict[str, tuple[str, ...]] = {
    "walls": _WALLS,
    "furnished": (*_WALLS, "IfcFurniture", "IfcFurnishingElement"),
}
"""Named class lists. Each class includes its subtypes, and a class that the file's schema
lacks (IfcFurniture in IFC2X3) is left out."""


def read_ifc(path, *, classes="walls", storey=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Triangles of the chosen elements of an IFC file, and the elevation of the storey.

    Parameters
    ----------
    path : str or os.PathLike
        IFC file.
    classes : str or sequence of str
        A preset name of ``PRESETS``, an IFC class, or a sequence of both.
        Each class includes its subtypes.
    storey : str, optional
        Name or GlobalId of an IfcBuildingStorey. Only the elements in that
        storey are read. By default all elements are read.

    Returns
    -------
    vertices : ndarray of shape (n, 3)
        World coordinates in metres.
    triangles : ndarray of shape (m, 3)
        Vertex indices, element by element in the order of the file.
    elevation : float
        Height of the storey in metres: the z of its placement in world
        coordinates, or its Elevation attribute when it has no placement.
        0 without a storey.
    """
    ifcopenshell = _import()
    f = ifcopenshell.open(str(Path(path)))
    elements = {}
    for name in _class_names(f, classes):
        elements.update((e.id(), e) for e in f.by_type(name))
    elevation = 0.0
    if storey is not None:
        st = _find_storey(f, storey)
        elevation = _elevation(f, st)
        inside = {e.id() for e in ifcopenshell.util.element.get_decomposition(st)}
        elements = {k: e for k, e in elements.items() if k in inside}
    for e in list(elements.values()):
        elements.update((p.id(), p) for p in _parts(e))
    pieces = _triangulate(f, list(elements.values()))
    if not pieces:
        where = "" if storey is None else f" in storey {storey!r}"
        raise ValueError(f"no geometry for the classes {classes!r}{where} in {path}")
    verts, tris, n = [], [], 0
    for k in sorted(pieces):
        v, t = pieces[k]
        verts.append(v)
        tris.append(t + n)
        n += len(v)
    return np.concatenate(verts), np.concatenate(tris), elevation


def _import():
    try:
        import ifcopenshell
        import ifcopenshell.geom
        import ifcopenshell.util.element
        import ifcopenshell.util.placement
        import ifcopenshell.util.shape
        import ifcopenshell.util.unit
    except ImportError as e:  # pragma: no cover
        raise ImportError("IFC files need ifcopenshell: pip install 'pysovist[ifc]'") from e
    return ifcopenshell


def _class_names(f, classes) -> list[str]:
    """IFC classes of ``classes``, presets expanded. Preset classes the schema lacks are left
    out, and any other name the schema lacks is an error."""
    out = []
    for name in [classes] if isinstance(classes, str) else list(classes):
        if name in PRESETS:
            out += [c for c in PRESETS[name] if _declared(f, c)]
        elif _declared(f, name):
            out.append(name)
        else:
            raise ValueError(f"{name!r} is neither a preset ({', '.join(PRESETS)}) "
                             f"nor a class of {f.schema}")
    return out


def _declared(f, name: str) -> bool:
    try:
        f.by_type(name)
    except RuntimeError:
        return False
    return True


def _find_storey(f, storey: str):
    storeys = f.by_type("IfcBuildingStorey")
    found = [s for s in storeys if storey in (s.Name, s.GlobalId)]
    if len(found) == 1:
        return found[0]
    names = ", ".join(repr(s.Name) for s in storeys)
    if found:
        raise ValueError(f"{len(found)} storeys are named {storey!r}: give a GlobalId")
    raise ValueError(f"no storey named {storey!r}, the storeys are {names}")


def _elevation(f, storey) -> float:
    import ifcopenshell.util.placement
    import ifcopenshell.util.unit

    scale = ifcopenshell.util.unit.calculate_unit_scale(f)  # metres per file length unit
    if storey.ObjectPlacement is not None:
        z = ifcopenshell.util.placement.get_local_placement(storey.ObjectPlacement)[2, 3]
    else:
        z = storey.Elevation or 0.0
    return float(z) * scale


def _parts(element):
    """The parts an element aggregates, at any depth (not its openings or their fillings)."""
    out = []
    for rel in getattr(element, "IsDecomposedBy", ()):
        if rel.is_a("IfcRelAggregates"):
            for part in rel.RelatedObjects:
                out += [part, *_parts(part)]
    return out


def _triangulate(f, elements) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Vertices and triangles of each element with a body, keyed by its id in the file."""
    import ifcopenshell.geom
    import ifcopenshell.util.shape

    if not elements:
        return {}
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(settings, f, os.cpu_count() or 1, include=elements)
    out = {}
    if not it.initialize():
        return out
    while True:
        shape = it.get()
        g = shape.geometry
        out[shape.id] = (np.array(ifcopenshell.util.shape.get_vertices(g), dtype=float),
                         np.array(ifcopenshell.util.shape.get_faces(g), dtype=np.int64))
        if not it.next():
            return out
