"""Every example of the documentation runs, and gives the numbers its page states.

Each page of ``docs/`` and ``README.md`` is run as its readers run it: its Python blocks in order,
as one script, in an empty directory. Its shell blocks that call ``pysovist`` then run in the same
directory. A block that reads a file of the reader's own (a scan, a BIM model) is not run: it is
reported as skipped under the name of that file, after the test has checked that the names it uses
are defined by the blocks before it.

The numbers are read from the page, never typed here, so the test fails when a page and the code
drift apart:

- A comment after an expression, or a comment line right below it, that starts with a value
  (``# 70.0, exact``, ``# [10, 6, 10]``, ``# (240, 360): ...``, ``# sky 0.1979, ground 0.2111``)
  states that value. A comment after an assignment states the value of the name it assigns, or
  its length when the name holds an array and the comment a whole number (``# 400 points``).
- A table printed right below a block whose last expression is a table, with the same columns,
  states that table.
- The sentences listed in ``SENTENCES`` state the values computed after their block.
- The sentences of the verification page listed in ``VERIFICATION`` state tolerances and counts,
  which are read from the source of the tests that assert them, and the error messages it quotes
  are those the library raises.

A value matches to the digits written: ``70.0`` holds for 69.95 to 70.05, a whole number written
for a count holds exactly, and ``13.416...`` means that the value starts with these digits.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pysovist

pytest.importorskip("numba")
pytest.importorskip("open3d")

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["README.md", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md"))]
# Files that a reader brings: a block that reads one of them is skipped under its name.
OWN_FILES = {"scan.las": "a laser scan", "scan.pts": "a laser scan", "model.ifc": "a BIM model",
             "unity_volumes.csv": "the volumes of a Unity run"}


# --- pages and blocks -----------------------------------------------------------------------------

@dataclass
class Block:
    page: str
    lang: str
    number: int  # 1 for the first Python block of the page, and so on; 0 for shell blocks
    line: int  # line of the first line of code, from 1
    end: int  # line of the closing fence
    source: str

    @property
    def name(self) -> str:
        return f"{self.page} block {self.number} (line {self.line})"


def page_blocks(page: str) -> list[Block]:
    lines = (ROOT / page).read_text(encoding="utf-8").splitlines()
    out, i, n = [], 0, 0
    while i < len(lines):
        m = re.match(r"^```(\w*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        j = i + 1
        while not lines[j].startswith("```"):
            j += 1
        lang = m.group(1)
        if lang == "python":
            n += 1
        out.append(Block(page, lang, n if lang == "python" else 0, i + 2, j + 1,
                         "\n".join(lines[i + 1:j])))
        i = j + 1
    return out


def own_file(block: Block) -> str | None:
    """The files of the reader's own that a Python block reads, if any."""
    names = [node.value for node in ast.walk(ast.parse(block.source))
             if isinstance(node, ast.Constant) and node.value in OWN_FILES]
    return " and ".join(dict.fromkeys(names)) or None


def free_names(source: str) -> set[str]:
    """Names a block reads before or without assigning them."""
    tree = ast.parse(source)
    stored = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Store)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            stored |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            stored.add(node.name)
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
              and isinstance(n.ctx, ast.Load)}
    return loaded - stored - set(dir(builtins))


# --- values written in the pages ------------------------------------------------------------------

NUM = r"-?\d+(?:\.\d*)?(?:e-?\d+)?(?:\.\.\.)?"


@dataclass(frozen=True)
class Num:
    text: str  # as written, for example "70.0", "8." or "13.416..."


def leading_value(comment: str):
    """The value a comment starts with, or None."""
    text = comment.strip()
    if not text:
        return None
    if text[0] in "[({":
        depth = 0
        for i, c in enumerate(text):
            depth += (c in "[({") - (c in "])}")
            if depth == 0:
                return parse_literal(text[:i + 1])
        return None
    if m := re.match(r"(True|False)\b", text):
        return m.group(1) == "True"
    if m := re.match(rf"([a-z]+) ({NUM})((?:, [a-z]+ {NUM})+)\b", text):  # sky 0.19, ground 0.21
        pairs = re.findall(rf"([a-z]+) ({NUM})", text[:m.end()])
        return {k: Num(v) for k, v in pairs}
    if m := re.match(rf"{NUM}(?:, {NUM})+(?=$|:)", text):  # 15.5, 0, 8.5: ...
        return [Num(t) for t in m.group(0).split(", ")]
    if m := re.match(rf"{NUM}(?=$|[\s:,;)])", text):  # 47.416..., 34 m of wall
        return Num(m.group(0))
    return None


def parse_literal(text: str):
    """A Python literal with every number kept as written."""
    def walk(node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return [walk(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {ast.literal_eval(k): walk(v) for k, v in zip(node.keys, node.values,
                                                                  strict=True)}
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Constant) or (isinstance(node, ast.UnaryOp)
                                              and isinstance(node.op, ast.USub)):
            return Num(ast.get_source_segment(text, node))
        raise ValueError(f"not a value: {ast.dump(node)}")

    return walk(ast.parse(text, mode="eval").body)


def number_problem(written: str, value) -> str | None:
    """None when ``value`` agrees with the number ``written`` to the digits written."""
    truncated = written.endswith("...")
    text = written[:-3] if truncated else written
    m = re.fullmatch(r"(-?)(\d+)(?:\.(\d*))?(?:e(-?\d+))?", text)
    assert m, f"cannot read the number {written!r}"
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        return f"{written} is stated, the value is {value!r}"
    if "." not in text and "e" not in text and isinstance(value, (int, np.integer)):
        return None if int(text) == value else f"{written} is stated, the value is {value}"
    v, lit = float(value), float(text)
    step = 10.0 ** (int(m.group(4) or 0) - len(m.group(3) or ""))
    tiny = 1e-12 * max(abs(v), 1.0)
    if truncated:
        ok = (math.copysign(1, v) == math.copysign(1, lit)
              and abs(lit) - tiny <= abs(v) < abs(lit) + step + tiny)
    else:
        ok = abs(v - lit) <= step / 2 + tiny
    return None if ok else f"{written} is stated, the value is {v!r}"


def problems(expected, value) -> list[str]:
    """Where ``value`` differs from the ``expected`` value written in a page."""
    if isinstance(value, pd.Series):
        value = value.to_numpy()
    if isinstance(expected, dict):
        out = []
        for key, e in expected.items():
            keys = [k for k in value if k == key or str(k).startswith(f"{key}_")]
            if len(keys) != 1:
                out.append(f"{key!r} names no single key of {list(value)}")
            else:
                out += problems(e, value[keys[0]])
        return out
    if isinstance(expected, bool):
        return [] if value is expected or value == expected else [f"{expected} is stated, "
                                                                   f"the value is {value!r}"]
    if isinstance(expected, list):
        shape = np.shape(np.asarray(expected, dtype=object))
        arr = np.asarray(value, dtype=object)
        if arr.shape != shape:
            return [f"a value of shape {shape} is stated, the value has shape {arr.shape}: "
                    f"{value!r}"]
        flat = np.asarray(expected, dtype=object).ravel()
        return [p for e, v in zip(flat, arr.ravel(), strict=True) for p in problems(e, v)]
    if isinstance(expected, Num):
        if isinstance(value, (np.ndarray, list, tuple, pd.DataFrame)):
            value = len(value)  # "# 400 points" after an array
        p = number_problem(expected.text, value)
        return [] if p is None else [p]
    raise TypeError(expected)


@dataclass
class Claim:
    statement: int  # index of the top-level statement in the block
    line: int  # line in the page
    expected: object


def comment_claims(block: Block) -> list[Claim]:
    """The values that the comments of a block state, by top-level statement."""
    tree = ast.parse(block.source)
    comments = {}
    for tok in tokenize.generate_tokens(io.StringIO(block.source).readline):
        if tok.type == tokenize.COMMENT:
            comments[tok.start[0]] = (tok.string[1:], tok.line.strip().startswith("#"))
    claims = []
    for k, node in enumerate(tree.body):
        if not isinstance(node, (ast.Expr, ast.Assign)):
            continue
        if isinstance(node, ast.Assign) and not (len(node.targets) == 1
                                                 and isinstance(node.targets[0], ast.Name)):
            continue
        found = comments.get(node.end_lineno)
        if found is None or found[1]:  # no comment after the code: try the line below
            below = comments.get(node.end_lineno + 1)
            found = below if below is not None and below[1] else None
            line = node.end_lineno + 1
        else:
            line = node.end_lineno
        if found is None:
            continue
        expected = leading_value(found[0])
        if expected is not None:
            claims.append(Claim(k, block.line + line - 1, expected))
    return claims


def markdown_table(page: str, after_line: int) -> tuple[list[str], list[list[str]]] | None:
    """Header and rows of a table that starts within two lines after ``after_line``."""
    lines = (ROOT / page).read_text(encoding="utf-8").splitlines()
    for i in range(after_line, min(after_line + 2, len(lines))):
        if lines[i].startswith("|"):
            rows = []
            for line in lines[i:]:
                if not line.startswith("|"):
                    break
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            return rows[0], rows[2:]
    return None


# --- sentences of the pages -----------------------------------------------------------------------
#
# Each entry: a page, the Python block after which it holds, a pattern that finds the numbers in
# the page, and the expressions, evaluated in the page's namespace, that give them. An expression
# that starts with "assert " is a statement of the page that is true or false.

SENTENCES = [
    # quick start
    ("docs/quickstart.md", 1, r"hides a shadow of (\d+) m²", ["100 - 4 - iso.area"]),
    ("docs/quickstart.md", 1, r"occlusivity is \\\(2\\sqrt\{45\} = ([\d.]+)\\\)",
     ["iso.metrics['occlusivity']"]),
    ("docs/quickstart.md", 1, r"# 47\.416\.\.\., (\d+) m of wall plus two shadow edges",
     ["iso.perimeter - iso.metrics['occlusivity']"]),
    ("docs/quickstart.md", 1, r"holds all (\d+) metrics", ["len(iso.metrics)"]),
    ("docs/quickstart.md", 2, r"An observer inside the pillar sees the pillar's (\d+) m² interior",
     ["pysovist.isovist(plan, (0.0, 0.0)).area"]),
    ("docs/quickstart.md", 2, r"drop the (\d+) inside the pillar",
     ["len(plan.grid(0.5)) - len(pts)"]),
    ("docs/quickstart.md", 3, r"# 174\.98, against ([\d.]+) m x (\d+) m² = (\d+) m³",
     ["mesh.bounds[1][2] - mesh.bounds[0][2]", "iso.area", "2.5 * iso.area"]),
    ("docs/quickstart.md", 3, r"# 0\.0, every ray hits a surface", ["assert vv.hit.all()"]),
    ("docs/quickstart.md", 3, r"the residual of ([\d.e-]+) is the error",
     ["1 - vv.volume / (2.5 * iso.area)", "_mesh_volume := vv.volume"]),
    ("docs/quickstart.md", 4, r"to the ([\d.e-]+) error of 65,536 directions",
     ["assert abs(vv.volume / (2.7 * pysovist.isovist(closed, (1.5, 1.2)).area) - 1) < 2e-4"]),
    ("docs/quickstart.md", 5, r"the cloud sees ([\d.]+) m³ where the mesh sees ([\d.]+) m³",
     ["vv.volume", "_mesh_volume"]),
    ("docs/quickstart.md", 5, r"samples the floor, the ceiling and the walls of the same room "
     r"every (\d+) cm", ["100 * s"]),
    # conventions
    ("docs/conventions.md", 1, r"# True: ([\d.]+) m seals the surfaces", ["cloud.radius"]),
    # metric definitions
    ("docs/metrics.md", 1, r"The surface area falls (\d+) % short at 65,536 rays",
     ["100 * (1 - m['surface_area'] / (2 * 70 + 3 * (34 + 2 * math.sqrt(45))))"]),
    ("docs/metrics.md", 3, r"# 69\.686, against the exact ([\d.]+)", ["iso.area"]),
    ("docs/metrics.md", 3, r"# 47\.233, against the exact ([\d.]+)", ["iso.perimeter"]),
    # obstacle visibility
    ("docs/obstacle-visibility.md", 1, r"The visible lengths add up to (\d+) m",
     ["vis.visible_length.sum()"]),
    ("docs/obstacle-visibility.md", 2, r"# 6\.93, which is (\d+) √(\d+)",
     ["2", "assert segment_visibility(walls, (2.0, 5.0), max_distance=4).visible_length[3] "
           "== pytest.approx(2 * math.sqrt(12), rel=1e-12)"]),
    ("docs/obstacle-visibility.md", 3, r"only while the pedestrian passes the kiosk, at (\d+) "
     r"of the (\d+) viewpoints", ["len(path) - table.seen_by[1]", "len(path)"]),
    ("docs/obstacle-visibility.md", 3, r"each outer front is hidden from (\d+) viewpoints",
     ["len(path) - table.seen_by[0]", "assert table.seen_by[2] == table.seen_by[0]"]),
    ("docs/obstacle-visibility.md", 4, r"# 15\.5, 0, 8\.5: at x = (\d+)", ["path[14, 0]"]),
    # DeCodingSpaces
    ("docs/decodingspaces.md", 1, r"the exact drift is sqrt\(13\) = ([\d.]+) m",
     ["pysovist.isovist(pysovist.Plan(walls), (2, 3)).metrics['drift']"]),
    ("docs/decodingspaces.md", 1, r"# (one) row per viewpoint", ["assert len(table) == 2"]),
    # directed view volumes
    ("docs/directed-volumes.md", 1, r"# 43028 of the (\d+) directions", ["1 << 18"]),
    # plan tools
    ("docs/plan-tools.md", 8, r"the boundary in the scene, (shape) \(k, 3\)",
     ["assert ring.ndim == 2 and ring.shape[1] == 3"]),
]


def check_sentences(page: str, number: int, ns: dict, text: str) -> list[str]:
    out = []
    for p, n, pattern, exprs in SENTENCES:
        if (p, n) != (page, number):
            continue
        # the expressions see the page's names and a few modules, without adding them to the page
        env = {"math": math, "np": np, "pytest": pytest, "pysovist": pysovist, **ns}
        m = re.search(pattern, text)
        if m is None:
            out.append(f"{page}: the sentence {pattern!r} is not in the page")
            continue
        groups = iter(m.groups())
        for expr in exprs:
            if expr.startswith("assert "):
                try:
                    exec(expr, env)
                except AssertionError:
                    out.append(f"{page}: {m.group(0)!r} is false ({expr})")
                next(groups, None)
                continue
            if ":=" in expr:  # keep a value for a later sentence
                name, value = (s.strip() for s in expr.split(":="))
                ns[name] = eval(value, env)
                continue
            value = eval(expr, env)
            out += [f"{page}: {m.group(0)!r}: {q}" for q in problems(Num(next(groups)), value)]
    return out


def street_table(ns: dict, text: str) -> list[str]:
    """The table of the cumulative visibility along a street, which names its rows in words."""
    page = "docs/obstacle-visibility.md"
    start = text.index("| Segment | Length (m) | Seen from |")
    header, rows = markdown_table(page, text[:start].count("\n"))
    table = ns["table"]
    out = []
    for row in rows:
        i = int(row[0].split(",")[0])
        seen, of = row[2].split(" of ")
        for written, value in ((row[1], table.length[i]), (seen, table.seen_by[i]),
                               (of, table.attrs["viewpoints"]),
                               (row[3], table.visible_length_sum[i]),
                               (row[4], table.visible_fraction_mean[i])):
            out += [f"{page}: street table, segment {i}: {q}" for q in problems(Num(written),
                                                                                 value)]
    if len(rows) != len(table):
        out.append(f"{page}: the street table has {len(rows)} rows, the value {len(table)}")
    return out


# Tables that a page prints in its own layout, checked after the block that computes them.
TABLES = {("docs/obstacle-visibility.md", 3): street_table}


def printed_tables(block: Block, value) -> list[str]:
    """A table printed below a block whose last expression is a table with the same columns."""
    if not isinstance(value, pd.DataFrame):
        return []
    table = markdown_table(block.page, block.end)
    if table is None or table[0] != [str(c) for c in value.columns]:
        return []
    header, rows = table
    if len(rows) != len(value):
        return [f"{block.name}: the printed table has {len(rows)} rows, the value {len(value)}"]
    return [f"{block.name}: printed row {i + 1}, {c}: {q}"
            for i, row in enumerate(rows) for c, cell in zip(header, row, strict=True)
            for q in problems(Num(cell), value[c].iloc[i])]


# --- running a page -------------------------------------------------------------------------------

SYSTEM_ENV = ("PATH", "HOME", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "USERPROFILE",
              "APPDATA", "LOCALAPPDATA")


def run_shell(block: Block, cwd: Path) -> list[str]:
    """Run the pysovist commands of a shell block, without those that read a reader's file.

    The lines run from Python, so no POSIX shell is needed: ``export`` sets variables, with
    ``$PWD`` the working folder, and ``pysovist`` runs the command line of this checkout. They
    stop at the first command that fails, as under ``set -e``.
    """
    logical = block.source.replace("\\\n", " ").splitlines()
    keep = [c for c in logical if re.match(r"\s*(pysovist|export)\b", c)
            and not any(f in c for f in OWN_FILES)]
    if not any(c.lstrip().startswith("pysovist") for c in keep):
        return []
    env = {k: os.environ[k] for k in SYSTEM_ENV if k in os.environ}
    env["PYTHONPATH"] = str(ROOT / "src")
    for line in keep:
        words = [w.replace("$PWD", str(cwd)) for w in shlex.split(line, comments=True)]
        if any("$" in w for w in words):
            return [f"{block.page}: the shell block at line {block.line} uses a variable other "
                    f"than $PWD: {line.strip()}"]
        if words[0] == "export":
            env.update(w.split("=", 1) for w in words[1:])
            continue
        r = subprocess.run([sys.executable, "-m", "pysovist.cli", *words[1:]], cwd=cwd, env=env,
                           capture_output=True, text=True, timeout=600)
        if r.returncode:
            return [f"{block.page}: the shell block at line {block.line} failed:\n"
                    f"{r.stderr[-2000:]}"]
    return []


def run_page(page: str, cwd: Path) -> tuple[list[str], list[str]]:
    """Run a page as one script. Returns its problems and the names of the blocks it skipped."""
    text = (ROOT / page).read_text(encoding="utf-8")
    ns = {"__name__": "__main__"}
    found, skipped = [], []
    for block in page_blocks(page):
        if block.lang in ("bash", "sh", ""):
            found += run_shell(block, cwd)
            continue
        if block.lang != "python":
            continue
        need = own_file(block)
        if need is not None:
            compile(block.source, block.name, "exec")
            missing = sorted(n for n in free_names(block.source) if n not in ns)
            if missing:
                found.append(f"{block.name} reads {need} and uses {missing}, which no block "
                             "before it defines")
            skipped.append(f"{block.name}: needs {need}, files of the reader's own")
            continue
        claims = {c.statement: c for c in comment_claims(block)}
        tree = ast.parse(block.source)
        value = None
        for k, node in enumerate(tree.body):
            try:
                if isinstance(node, ast.Expr):
                    value = eval(compile(ast.Expression(node.value), block.name, "eval"), ns)
                else:
                    exec(compile(ast.Module([node], []), block.name, "exec"), ns)
                    value = ns[node.targets[0].id] if k in claims else None
            except Exception as e:  # the rest of the page builds on this block
                found.append(f"{block.name}, line {block.line + node.lineno - 1}: "
                             f"{type(e).__name__}: {e}")
                return found, skipped
            if k in claims:
                found += [f"{block.name}, line {claims[k].line}: {q}"
                          for q in problems(claims[k].expected, value)]
        found += printed_tables(block, value)
        if (page, block.number) in TABLES:
            found += TABLES[page, block.number](ns, text)
        found += check_sentences(page, block.number, ns, text)
    return found, skipped


def run_page_as_a_script(page: str, cwd: Path) -> list[str]:
    """Run a page in a new interpreter, as a reader runs a script, and return its problems."""
    result = cwd / "problems.json"
    code = ("import json, sys; from pathlib import Path; "
            "from tests.test_docs_examples import run_page; "
            "json.dump(run_page(sys.argv[1], Path.cwd())[0], "
            "open(sys.argv[2], 'w', encoding='utf-8'))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    r = subprocess.run([sys.executable, "-c", code, page, str(result)], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=1200)
    if r.returncode or not result.is_file():
        return [f"{page}: the script stopped:\n{r.stderr[-3000:]}"]
    return json.loads(result.read_text(encoding="utf-8"))


@pytest.mark.parametrize("page", PAGES)
def test_page_examples_run_and_give_the_stated_numbers(page, tmp_path, monkeypatch):
    if any(b.lang == "python" for b in page_blocks(page)):
        found = run_page_as_a_script(page, tmp_path)
    else:  # nothing runs in Python: its shell blocks, if any, run from here
        monkeypatch.chdir(tmp_path)
        found = run_page(page, tmp_path)[0]
    assert not found, "\n".join(found)


OWN_FILE_BLOCKS = [b for page in PAGES for b in page_blocks(page)
                   if b.lang == "python" and own_file(b)]


@pytest.mark.parametrize("block", OWN_FILE_BLOCKS, ids=[b.name for b in OWN_FILE_BLOCKS])
def test_a_block_that_reads_a_file_of_the_readers_own_is_skipped_by_name(block):
    # The page test checks that the names it uses are defined by the blocks before it.
    compile(block.source, block.name, "exec")
    pytest.skip(f"needs {own_file(block)}, files of the reader's own")


def test_the_quick_start_commands_write_their_tables_and_records(tmp_path):
    # The quick start says that every run writes a JSON record next to its CSV, that the
    # columns are those of isovist_field or view_volume_field, and where ODTP mode writes.
    from pysovist import Plan, isovist_field

    found = run_page_as_a_script("docs/quickstart.md", tmp_path)
    assert not found, "\n".join(found)
    iso = pd.read_csv(tmp_path / "isovists.csv")
    walls = pd.read_json(tmp_path / "walls.json")
    plan = Plan([[a, b] for a, b in zip(walls.start, walls.end, strict=True)])
    ref = isovist_field(plan, pd.read_csv(tmp_path / "points.csv").to_numpy())
    assert list(iso.columns) == list(ref.columns)
    assert iso.area.iloc[0] == pytest.approx(70.0, rel=1e-12)  # the observer of the 2D example
    for name in ("isovists", "volumes", "output/pysovist_isovist"):
        assert (tmp_path / f"{name}.json").is_file()
    vol = pd.read_csv(tmp_path / "volumes.csv")
    assert list(vol.columns[:4]) == ["x", "y", "z", "volume"] and (vol.z == 1.7).all()


# --- the verification page ------------------------------------------------------------------------
#
# Each entry: a pattern in the verification page whose groups are numbers, and for each group an
# expression that gives the number from the test that asserts it. src(file, name, pattern) reads
# the first group of pattern in the source of a top-level name of a test file, decorators
# included, and T(module) imports a test module. An expression that starts with "assert " is a
# statement about the tests and takes no group.

def source_of(file: str, name: str) -> str:
    """Source of a top-level function, class or assignment of a test file, with decorators."""
    text = (ROOT / "tests" / file).read_text(encoding="utf-8")
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name) or (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
            return "\n".join(lines[start - 1:node.end_lineno])
    raise LookupError(f"{name} is not a top-level name of {file}")


def src(file: str, name: str, pattern: str):
    m = re.search(pattern, source_of(file, name))
    if m is None:
        raise LookupError(f"{pattern!r} is not in {file}::{name}")
    return eval(m.group(1), {"math": math})


def T(module: str):
    import importlib

    return importlib.import_module(f"tests.{module}")


def default_rtol_of_frame_comparison(file: str, name: str):
    """The relative tolerance with which pandas compares two tables of floats by default, when
    the test compares tables without a tolerance of its own: the power of ten r such that a
    relative difference of 0.99 r passes and one of 1.01 r fails."""
    assert "assert_frame_equal" in source_of(file, name)
    assert "rtol" not in source_of(file, name) and "check_exact" not in source_of(file, name)
    one = pd.DataFrame({"v": [1.0]})

    def passes(d):
        try:
            pd.testing.assert_frame_equal(one, one * (1 + d))
        except AssertionError:
            return False
        return True

    return next(r for r in 10.0 ** -np.arange(3, 13) if passes(0.99 * r) and not passes(1.01 * r))


V = "test_visibility2d.py"
R2 = "test_rays2d.py"
P = "test_paths.py"
F = "test_flood_grid.py"
VE = "test_vertical.py"
DS = "test_compat_decodingspaces_open.py"
DR = "test_volume3d_directed.py"
D3 = "test_compat_decodingspaces3d.py"
REF = "test_volume3d_references.py"
S3 = "test_volume3d_shape.py"
NUMBER = r"([\d.e-]+)"

VERIFICATION = [
    # 3D shape metrics
    (r"gives its 108 m² to ([\d.e-]+) at ([\d,]+) rays",
     [rf"src('{S3}', 'test_box_seen_from_an_interior_eye', r'approx\(108.0, rel={NUMBER}\)')",
      rf"src('{S3}', 'test_box_seen_from_an_interior_eye', r'n_rays=(\d+)')"]),
    (r"the occluding area equals the height times the 2D occlusivity to ([\d.e-]+)\. The surface "
     r"area equals the floor and the ceiling plus the height times the 2D perimeter to "
     r"([\d.e-]+)\.",
     [f"src('{S3}', 'test_prism_room_against_the_exact_2d_isovist', "
      rf"r'occlusivity\"\], rel={NUMBER}\)')",
      f"src('{S3}', 'test_prism_room_against_the_exact_2d_isovist', "
      rf"r'perimeter\"\], rel={NUMBER}\)')"]),
    (r"the convex deficiency equals the 2D one to ([\d.e-]+)\.",
     [f"src('{S3}', 'test_prism_room_against_the_exact_2d_isovist', "
      rf"r'convex_deficiency\"\], abs={NUMBER}\)')"]),
    (r"the occluding area of these rooms lies within (\d+) %, ([\d.]+) %, ([\d.]+) % and "
     r"([\d.]+) % of the exact faces",
     [f"100 * src('{S3}', 'test_occluding_area_converges_with_the_ray_count', "
      rf"r'{n}: ([\d.]+)')" for n in (4096, 16384, 65536, 262144)]),
    (r"with a log-log slope between (-[\d.]+) and (-[\d.]+), and by less than ([\d.]+) % at "
     r"262,144 rays",
     [rf"src('{S3}', 'test_surface_area_converges_with_the_ray_count', r'(-[\d.]+) < slope')",
      rf"src('{S3}', 'test_surface_area_converges_with_the_ray_count', r'slope < (-[\d.]+)')",
      f"100 * src('{S3}', 'test_surface_area_converges_with_the_ray_count', "
      rf"r'-err\[-1\] < {NUMBER},')"]),
    (r"an occluding area of exactly 0 at ([\d,]+), ([\d,]+) and ([\d,]+) rays",
     [rf"src('{S3}', 'test_a_corridor_has_no_occluding_face', r'\[{a}(\d+){b}\]')"
      for a, b in ((r"", r", \d+, \d+"), (r"\d+, ", r", \d+"), (r"\d+, \d+, ", r""))]),
    # obstacle visibility
    (r"\*\*The visible part of every wall agrees with closed forms to ([\d.e-]+)\.\*\*",
     [f"src('{V}', 'approx', r'rel={NUMBER}')"]),
    (r"square rooms with up to (\d+) interior walls, which may cross, the visible lengths add up "
     r"to the part of the isovist perimeter that lies on walls, to a relative ([\d.e-]+)\.",
     [rf"src('{V}', 'room_with_free_walls', r'st.integers\(1, (\d+)\)')",
      f"src('{V}', 'test_visible_length_is_the_wall_part_of_the_isovist_perimeter', "
      rf"r'approx\(on_walls, rel={NUMBER},')"]),
    (r"computed with shapely, to ([\d.e-]+)\. A shift of up to ([\d,]+) m along each axis and a "
     r"scaling by a factor from ([\d.]+) to (\d+) change the lengths as they change the scene, to "
     r"([\d.e-]+)\. Each property is checked on up to (\d+) random scenes\.",
     [rf"src('{V}', 'test_matches_shadow_subtraction', r'approx\(seen, rel={NUMBER},')",
      rf"src('{V}', 'test_moving_and_scaling_the_scene', r'st.floats\(-1e3, ([\de]+)\)')",
      rf"src('{V}', 'test_moving_and_scaling_the_scene', r'st.floats\(([\d.]+), 100\)')",
      rf"src('{V}', 'test_moving_and_scaling_the_scene', r'st.floats\(0.01, (\d+)\)')",
      f"src('{V}', 'test_moving_and_scaling_the_scene', r'rel={NUMBER},')",
      rf"src('{V}', 'SETTINGS', r'max_examples=(\d+)')",
      f"assert all('@SETTINGS' in source_of('{V}', t) for t in ("
      "'test_visible_length_is_the_wall_part_of_the_isovist_perimeter', "
      "'test_matches_shadow_subtraction', 'test_moving_and_scaling_the_scene'))"]),
    (r"single `segment_visibility` calls to ([\d.e-]+), and two processes give the table of one",
     [f"src('{V}', 'approx', r'rel={NUMBER}')",
      f"assert 'approx(singles.sum(axis=0))' in source_of('{V}', "
      "'test_field_sums_and_means_over_the_path')"]),
    (r"With ([\d,]+) even rays, .* on up to (\d+) random rooms\.",
     [rf"src('{V}', 'test_ray_hits_converge_to_the_visible_angle', r'n = ([\d_]+)')",
      rf"src('{V}', 'test_ray_hits_converge_to_the_visible_angle', r'max_examples=(\d+)')"]),
    # plan tools
    (r"rays along the axes and a diagonal end at their closed-form distances, to a relative "
     r"([\d.e-]+), and rays through the corners of regular polygons of (\d+), (\d+), (\d+) and "
     r"(\d+) sides end on the corners, to ([\d.e-]+)\.",
     [rf"src('{R2}', 'test_rays_into_a_square_room_hit_the_expected_walls', r'rel={NUMBER}\)')",
      *[f"src('{R2}', 'test_rays_through_the_corners_of_a_closed_polygon_never_escape', "
        rf"r'\[(?:\d+, ){{{k}}}(\d+)')" for k in range(4)],
      f"src('{R2}', 'test_rays_through_the_corners_of_a_closed_polygon_never_escape', "
      rf"r'rel={NUMBER}\)')"]),
    (r"square rooms with up to (\d+) interior walls, the distance of every ray equals the depth of "
     r"the exact isovist at the same angle to ([\d.e-]+), and its hit point lies on the wall it "
     r"reports, on up to (\d+) random scenes\.",
     [rf"src('{R2}', 'scene_and_rays', r'st.integers\(0, (\d+)\)')",
      f"src('{R2}', 'test_ray_depth_equals_the_exact_isovist_depth', r'rel={NUMBER},')",
      f"src('{R2}', 'test_hit_points_lie_on_the_reported_wall_within_range', "
      r"r'max_examples=(\d+)')"]),
    (r"the areas and perimeters of the rows equal their closed forms to ([\d.e-]+)\.",
     [rf"src('{P}', 'test_isovists_along_a_corridor_see_the_rest_of_it', r'rel={NUMBER}\)')"]),
    (r"exactly, on up to (\d+) random paths\. On up to (\d+) random polylines the samples lie on "
     r"the path at their arc length, to ([\d.e-]+)\.",
     [f"src('{P}', 'test_each_row_is_the_isovist_looking_along_the_tangent', "
      r"r'max_examples=(\d+)')",
      rf"src('{P}', 'test_samples_lie_on_the_path_at_their_arc_length', r'max_examples=(\d+)')",
      f"src('{P}', 'test_samples_lie_on_the_path_at_their_arc_length', "
      rf"r'starts\[k\]\), abs={NUMBER}\)')"]),
    (r"the rows equal those of `isovist_field` at the same points, to the relative ([\d.e-]+) of "
     r"the table comparison of pandas\.",
     [f"default_rtol_of_frame_comparison('{P}', "
      "'test_a_full_field_of_view_equals_the_isovist_field')"]),
    (r"On up to (\d+) random scenes with up to (\d+) walls, spacings from ([\d.]+) to ([\d.]+) m",
     [f"src('{F}', 'test_fill_equals_a_breadth_first_fill_with_geos_distances', "
      r"r'max_examples=(\d+)')",
      rf"src('{F}', 'free_walls', r'st.integers\(0, (\d+)\)')",
      rf"src('{F}', 'free_walls', r'spacing = draw\(st.floats\(([\d.]+),')",
      rf"src('{F}', 'free_walls', r'spacing = draw\(st.floats\([\d.]+, ([\d.]+)\)')"]),
    (r"An L-shaped room fills its (\d+) cells of (\d+) m",
     [f"src('{F}', 'test_l_shaped_room_fills_its_sixty_cells_and_not_the_notch', "
      r"r'len\(centres\) == (\d+)')",
      f"src('{F}', 'test_l_shaped_room_fills_its_sixty_cells_and_not_the_notch', "
      r"r'flood_grid\(([\d.]+),')"]),
    (r"on up to (\d+) random polygons of (\d+) to (\d+) sides\.",
     [rf"src('{F}', 'test_a_closed_polygon_never_leaks', r'max_examples=(\d+)')",
      rf"src('{F}', 'test_a_closed_polygon_never_leaks', r'st.integers\((\d+), \d+\)')",
      rf"src('{F}', 'test_a_closed_polygon_never_leaks', r'st.integers\(\d+, (\d+)\)')"]),
    (r"along, across and diagonally, to ([\d.e-]+), and a table",
     [f"src('{VE}', 'test_box_room_seen_from_inside_is_the_rectangle_of_its_depth_and_height', "
      rf"r'rel={NUMBER}\)')"]),
    (r"It equals, to ([\d.e-]+), the horizontal isovist of the building turned",
     [f"src('{VE}', 'test_rotating_the_building_as_in_the_tutorial_gives_the_same_isovist', "
      rf"r'rel={NUMBER}\)')"]),
    (r"first walls ahead and behind in the plan, to ([\d.e-]+), for up to (\d+) random eyes",
     [f"src('{VE}', 'test_extruded_plan_gives_the_rectangle_of_the_two_plan_rays', "
      rf"r'rel={NUMBER}\)')",
      f"src('{VE}', 'test_extruded_plan_gives_the_rectangle_of_the_two_plan_rays', "
      r"r'max_examples=(\d+)')"]),
    # DeCodingSpaces 2D layer
    (r"The (\d+) properties agree to a relative ([\d.e-]+) \(absolute ([\d.e-]+) near zero\)\.",
     ["len(__import__('pysovist.compat.decodingspaces', fromlist=['x']).DS_PROPERTY_NAMES)",
      f"src('{DS}', 'test_properties_match_the_plugin', r'rtol={NUMBER},')",
      rf"src('{DS}', 'test_properties_match_the_plugin', r'atol={NUMBER}\)')"]),
    (r"of the same rank to ([\d.e-]+) rad, and its length agrees to a relative ([\d.e-]+)\.",
     [rf"src('{DS}', 'test_rays_match_the_plugin_one_by_one', r'max\(\) < {NUMBER}')",
      f"src('{DS}', 'test_rays_match_the_plugin_one_by_one', r'rtol={NUMBER},')"]),
    (r"The ring has the area of the plugin's Isovist Polygon to ([\d.e-]+), and the hits per line "
     r"are equal\.",
     [rf"src('{DS}', 'test_polygon_matches_the_plugin', r'rel={NUMBER}\)')",
      f"assert 'assert_array_equal(hits, ov)' in source_of('{DS}', "
      "'test_hits_per_line_match_the_plugin')"]),
    (r"the areas agree to ([\d.e-]+), and to ([\d.e-]+) where the plugin's areas carry six "
     r"decimals\.",
     [r"src('test_compat_decodingspaces.py', 'test_hospital_areas_at_24_ray_counts', "
      f"r'rtol={NUMBER},')",
      r"src('test_compat_decodingspaces.py', 'test_hospital_areas_across_the_fallback', "
      f"r'rtol={NUMBER},')"]),
    # 3D references
    (r"end at the closed-form distances to its faces, to a relative ([\d.e-]+), with and without",
     [rf"src('{REF}', 'test_mesh_cast_gives_the_ray_box_distances', r'rtol={NUMBER}\)')"]),
    (r"computed with scipy, to ([\d.e-]+) and ([\d.e-]+)\.",
     [f"src('{REF}', 'test_depth_statistics_from_the_centre_of_a_cube', "
      rf"r'approx\(mean, rel={NUMBER}\)')",
      f"src('{REF}', 'test_depth_statistics_from_the_centre_of_a_cube', "
      rf"r'mean \* mean\), rel={NUMBER}\)')"]),
    (r"of a faceted sphere of radius (\d+) m is \d+ m to ([\d.e-]+)\.",
     [rf"src('{REF}', 'test_equivalent_radius_of_a_sphere_is_its_radius', r'radius=([\d.]+),')",
      f"src('{REF}', 'test_equivalent_radius_of_a_sphere_is_its_radius', "
      rf"r'approx\(2.0, rel={NUMBER}\)')"]),
    (r"(\d+) runs of ([\d,]+) directions each lie within (\d+) standard errors",
     [f"src('{REF}', 'test_random_directions_estimate_the_cube_volume_without_bias', "
      r"r'range\((\d+)\)')",
      f"src('{REF}', 'test_random_directions_estimate_the_cube_volume_without_bias', "
      r"r'n = (\d+)')",
      f"src('{REF}', 'test_random_directions_estimate_the_cube_volume_without_bias', "
      r"r'< (\d+) \* se')"]),
    (r"gives the volume of a cube from (\d+) eyes to ([\d.e-]+), on one thread, on three and on "
     r"all cores\.",
     [rf"len(re.findall(r'\(-?[\d.]+, -?[\d.]+, -?[\d.]+\)', source_of('{REF}', "
      "'test_field_gives_the_cube_volume_from_every_eye')))",
      rf"src('{REF}', 'test_field_gives_the_cube_volume_from_every_eye', r'rtol={NUMBER}\)')",
      f"assert '[-1, 1, 3]' in source_of('{REF}', "
      "'test_field_gives_the_cube_volume_from_every_eye')"]),
    # directed view volumes and view factors
    (r"by adaptive quadrature in azimuth, to ([\d.e-]+)\. Over (\d+) fields, from (\d+) x (\d+) "
     r"degrees to a full turn (\d+) degrees tall",
     [rf"src('{DR}', 'box_wedge_volume', r'epsrel={NUMBER}\)')",
      "len(T('test_volume3d_directed').FIELDS)",
      rf"src('{DR}', 'FIELDS', r'math.radians\((\d+)\), math.radians\(20\)')",
      rf"src('{DR}', 'FIELDS', r'math.radians\(30\), math.radians\((\d+)\)')",
      rf"src('{DR}', 'FIELDS', r'2 \* math.pi, math.radians\((\d+)\)')"]),
    (r"an angle grid of (\d+) x (\d+) cells agrees to ([\d.e-]+), and 2\^(\d+) Fibonacci "
     r"directions to ([\d.e-]+)\.",
     [f"src('{DR}', 'test_box_room_grid_volume_converges_to_the_exact_wedge', "
      r"r'angle_grid\((\d+),')",
      f"src('{DR}', 'test_box_room_grid_volume_converges_to_the_exact_wedge', "
      r"r'angle_grid\(\d+, (\d+),')",
      f"src('{DR}', 'test_box_room_grid_volume_converges_to_the_exact_wedge', "
      rf"r'rel={NUMBER}\)')",
      f"src('{DR}', 'test_box_room_fibonacci_volume_converges_to_the_exact_wedge', "
      r"r'n_rays=1 << (\d+)')",
      f"src('{DR}', 'test_box_room_fibonacci_volume_converges_to_the_exact_wedge', "
      rf"r'rel={NUMBER}\)')"]),
    (r"sum to the solid angle of its field to ([\d.e-]+), and to \\\(4\\pi\\\) over the sphere to "
     r"([\d.e-]+)\. Eight octants add up to the full view volume to ([\d.e-]+)\.",
     [rf"src('{DR}', 'test_angle_grid_weights_are_exact_cell_solid_angles', r'rel={NUMBER}\)')",
      rf"src('{DR}', 'test_full_angle_grid_covers_the_sphere', r'rel={NUMBER}\)')",
      rf"src('{DR}', 'test_octants_partition_the_full_view', r'rel={NUMBER}\)')"]),
    (r"to an absolute ([\d.e-]+), with 2\^(\d+) Fibonacci directions and with an angle grid of "
     r"(\d+) x (\d+) cells\. The three factors add up to (\d) to ([\d.e-]+)\.",
     [rf"src('{DR}', 'test_view_factors_match_exact_solid_angles', r'abs={NUMBER}\)')",
      rf"src('{DR}', 'test_view_factors_match_exact_solid_angles', r'n_rays=1 << (\d+)')",
      rf"src('{DR}', 'test_view_factors_match_exact_solid_angles', r'angle_grid\((\d+),')",
      rf"src('{DR}', 'test_view_factors_match_exact_solid_angles', r'angle_grid\(\d+, (\d+)\)')",
      f"src('{DR}', 'test_view_factors_match_exact_solid_angles', "
      r"r'sum\(f.values\(\)\) == pytest.approx\(([\d.]+),')",
      f"src('{DR}', 'test_view_factors_match_exact_solid_angles', "
      rf"r'sum\(f.values\(\)\) == pytest.approx\(1.0, abs={NUMBER}\)')"]),
    (r"equal single `view_volume` calls to ([\d.e-]+)\.\n",
     [f"src('{DR}', 'test_directed_field_rows_match_single_volumes', r'rel={NUMBER},')"]),
    # DeCodingSpaces Isovist3D layer
    (r"Each ray points in the direction of the component's ray to ([\d.e-]+), its length agrees "
     r"to a relative ([\d.e-]+), and its hit point to ([\d.e-]+) of its length\.",
     [rf"src('{D3}', 'test_ray_directions_and_order', r'atol={NUMBER}\)')",
      rf"src('{D3}', 'test_ray_lengths_and_hit_vectors', r'rtol={NUMBER}\)')",
      rf"src('{D3}', 'test_ray_lengths_and_hit_vectors', r'err <= {NUMBER} \* lengths')"]),
    (r"The volumes of the measured views agree to a relative ([\d.e-]+), or an absolute "
     r"([\d.e-]+) for a grid that encloses nothing\. They agree to ([\d.e-]+) for a closed room "
     r"and an extruded hospital plan up to (\d+) x (\d+) rays\. From there to ([\d,]+) x "
     r"([\d,]+) rays they agree to ([\d.e-]+),",
     [f"src('{D3}', 'test_volume', r'rel={NUMBER},')",
      rf"src('{D3}', 'test_volume', r'abs={NUMBER}\)')",
      rf"src('{D3}', 'test_hospital_volumes', r'rel={NUMBER}\)')",
      "max(r[0] for r in T('test_compat_decodingspaces3d').ROOM_VOLUMES "
      "+ T('test_compat_decodingspaces3d').HOSPITAL_VOLUMES)",
      "max(r[1] for r in T('test_compat_decodingspaces3d').ROOM_VOLUMES "
      "+ T('test_compat_decodingspaces3d').HOSPITAL_VOLUMES)",
      "max(r['HPr'] for r in T('test_compat_decodingspaces3d').LARGE)",
      "max(r['VPr'] for r in T('test_compat_decodingspaces3d').LARGE)",
      rf"src('{D3}', 'test_large_counts', r'LOOSE.get\(key, {NUMBER}\)')",
      rf"assert src('{D3}', 'test_closed_room_volumes', r'rel={NUMBER}\)') == 5e-6"]),
    (r"(\d+) x (\d+) rays from the centre of the closed room give a prism of ([\d.]+) m³, to "
     r"([\d.e-]+)\.",
     [rf"src('{D3}', 'test_volume_is_the_polyhedron_through_the_hit_points', r'A (\d+) x')",
      rf"src('{D3}', 'test_volume_is_the_polyhedron_through_the_hit_points', r'A \d+ x (\d+)')",
      f"src('{D3}', 'test_volume_is_the_polyhedron_through_the_hit_points', "
      r"r'approx\((3 \* 3 \* 6.75),')",
      f"src('{D3}', 'test_volume_is_the_polyhedron_through_the_hit_points', "
      rf"r'6.75, rel={NUMBER}\)')"]),
    (r"the exact volume is (\d) m times the exact isovist area, ([\d.]+) m³\. The component's "
     r"volumes lie ([\d.]+) %, ([\d.]+) % and ([\d.]+) % below it at (\d+) x (\d+), (\d+) x "
     r"(\d+) and ([\d,]+) x (\d+) rays, to ([\d.]+) %\.",
     [f"src('{D3}', 'test_hospital_exact_volume_is_height_times_isovist_area', "
      r"r'assert ([\d.]+) \* area')",
      "T('test_compat_decodingspaces3d').HOSPITAL_EXACT",
      *[f"-100 * src('{D3}', 'test_volume_does_not_converge_with_the_ray_count', "
        rf"r'\[(?:-[\d.]+, ){{{k}}}(-[\d.]+)')" for k in range(3)],
      *[x for k in range(3) for x in (
          f"src('{D3}', 'test_volume_does_not_converge_with_the_ray_count', "
          rf"r'for hpr, vpr in \((?:\(\d+, \d+\), ){{{k}}}\((\d+),')",
          f"src('{D3}', 'test_volume_does_not_converge_with_the_ray_count', "
          rf"r'for hpr, vpr in \((?:\(\d+, \d+\), ){{{k}}}\(\d+, (\d+)\)')")],
      f"100 * src('{D3}', 'test_volume_does_not_converge_with_the_ray_count', "
      rf"r'atol={NUMBER}\)')"]),
]


def test_the_verification_page_states_what_the_tests_assert():
    text = (ROOT / "docs" / "verification.md").read_text(encoding="utf-8")
    env = {"src": src, "T": T, "source_of": source_of, "re": re,
           "default_rtol_of_frame_comparison": default_rtol_of_frame_comparison}
    found = []
    for pattern, exprs in VERIFICATION:
        m = re.search(pattern, text)
        if m is None:
            found.append(f"the sentence {pattern!r} is not in the page")
            continue
        groups = iter(m.groups())
        for expr in exprs:
            if expr.startswith("assert "):
                try:
                    exec(expr, env)
                except AssertionError:
                    found.append(f"{m.group(0)[:60]!r}: {expr} is false")
                continue
            written = next(groups).replace(",", "")
            found += [f"{m.group(0)[:60]!r}: {q} ({expr})"
                      for q in problems(Num(written), eval(expr, env))]
        assert next(groups, None) is None, f"{pattern!r} has a group without an expression"
    assert not found, "\n".join(found)


def test_the_messages_the_verification_page_quotes_are_those_raised():
    text = (ROOT / "docs" / "verification.md").read_text(encoding="utf-8")
    section = text[text.index("## Checked inputs"):text.index("## Command line and documentation")]
    quoted = re.findall(r'raises "([^"]+)"', section)
    square = [[(0, 0), (10, 0)], [(10, 0), (10, 10)], [(10, 10), (0, 10)], [(0, 10), (0, 0)]]
    calls = [lambda: pysovist.isovist(square, (5, 5), max_distance=-1),
             lambda: pysovist.isovist(square, (5, 5), fov=120),
             lambda: pysovist.PointCloud([(0.0, 0.0, 0.0)], radius=float("nan")),
             lambda: pysovist.Mesh(np.eye(4)[:, :3], [(0, 1, 2, 3)])]
    assert len(quoted) == len(calls)
    for message, call in zip(quoted, calls, strict=True):
        with pytest.raises(ValueError) as e:
            call()
        assert str(e.value) == message


@pytest.mark.parametrize("page", ["README.md", "docs/cite.md"])
def test_the_software_reference_cites_the_version_of_the_package(page):
    text = (ROOT / page).read_text(encoding="utf-8")
    bibtex = re.search(r"@software\{pysovist,.*?\n\}", text, re.S)
    assert f"version = {{{pysovist.__version__}}}" in re.sub(r" +", " ", bibtex.group(0))


def test_the_citation_file_cites_the_version_of_the_package():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^version: (\S+)$", text, re.M).group(1) == pysovist.__version__


# --- the harness itself ---------------------------------------------------------------------------

@pytest.mark.parametrize("written, value, ok", [
    ("70.0", 70.04, True), ("70.0", 70.06, False), ("174.98", 174.984283, True),
    ("13.416...", 13.41640786, True), ("13.416...", 13.4159, False),
    ("24.926...", 24.926470588, True), ("8.", 8.3, True), ("8.", 8.6, False),
    ("384", 384, True), ("384", 385, False), ("9e-5", 8.98e-5, True), ("9e-5", 9.6e-5, False),
    ("0.0", 0.0, True), ("0.0", 0.06, False), ("-1", -1, True), ("373.8", 373.75, True),
])
def test_numbers_match_to_the_digits_written(written, value, ok):
    assert (number_problem(written, value) is None) == ok


def test_comment_values_are_read_with_their_digits():
    assert leading_value(" 70.0, exact") == Num("70.0")
    assert leading_value(" 13.416..., the two edges") == Num("13.416...")
    assert leading_value(" 400 points, 0.5 m apart") == Num("400")
    assert leading_value(" 47.416..., 34 m of wall") == Num("47.416...")
    assert leading_value(" 15.5, 0, 8.5: at x = 29") == [Num("15.5"), Num("0"), Num("8.5")]
    assert leading_value(" [[-1], [3]]: the first ray") == [[Num("-1")], [Num("3")]]
    assert leading_value(" {'median': 0.04, 'p90': 0.04}") == {"median": Num("0.04"),
                                                                "p90": Num("0.04")}
    assert leading_value(" sky 0.1979, ground 0.2111, object 0.5910") == {
        "sky": Num("0.1979"), "ground": Num("0.2111"), "object": Num("0.5910")}
    assert leading_value(" True: 0.04 m seals the surfaces") is True
    for text in (" room, then pillar", " drop the 16 inside", " the boundary, shape (k, 3)"):
        assert leading_value(text) is None
    assert problems([[Num("8."), Num("7.")]], np.array([[8.0, 7.2]])) == []
    assert problems([Num("1"), Num("2")], np.array([1.0, 2.0, 3.0]))  # a shape mismatch
    assert problems({"sky": Num("0.20")}, {"sky_factor": 0.2}) == []


def test_every_python_block_that_reads_a_file_names_one_the_repository_does_not_hold():
    for page in PAGES:
        for block in page_blocks(page):
            if block.lang == "python" and own_file(block):
                for name in own_file(block).split(" and "):
                    assert not list(ROOT.rglob(name)), block.name
