# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static import boundaries for all three distributions."""
from __future__ import annotations

import ast
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
PACKAGES = REPO / "packages"

LAYERS = {
    0: {"core", "ax25"},
    1: {"brubru", "shrike", "kestrel", "besra", "sabir", "lanner", "bateleur",
        "macagua", "merlin"},
    2: {"station", "monitor"},
    3: {"contact", "winlink"},
    4: {"bbs", "operator", "cli", "__main__"},
}

CREANCE_MODEM_CARVE_OUT = "creance/monitor/runners"
PRIVATE_ROOTS = {"tools", "working", "tests"} | {
    p.stem for p in (REPO / "tools").glob("*")
    if p.suffix == ".py" or p.is_dir() and p.name != "__pycache__"
}
ALLOWLIST = frozenset("""creance.monitor.runners.p4_runner -> corpuslib
hfmodem.besra.arq.modem -> hfmodem.besra.host.modem_core
hfmodem.besra.arq.modem -> hfmodem.besra.host.protocol
hfmodem.besra.arq.session -> hfmodem.besra.host.protocol
hfmodem.besra.host.run_server -> hfmodem.winlink
hfmodem.besra.host.run_server -> hfmodem.winlink.message
hfmodem.besra.sim.echo -> hfmodem.besra.host.modem_core
hfmodem.contact.varac -> hfmodem.bbs.frontends.varac
hfmodem.core.config -> hfmodem.winlink.session
hfmodem.kestrel.arq.modem -> hfmodem.kestrel.host.modem_core
hfmodem.kestrel.arq.run_pair -> hfmodem.kestrel.host.server
hfmodem.kestrel.host.vara_modem -> hfmodem.contact.broadcast
hfmodem.kestrel.rx.broadcast -> hfmodem.contact.broadcast
hfmodem.lanner.qso -> hfmodem.station.qsolog
hfmodem.macagua.chain -> hfmodem.contact.audio
hfmodem.macagua.phonetics -> hfmodem.contact.calls
hfmodem.macagua.script -> hfmodem.contact.facts
hfmodem.macagua.speech -> hfmodem.contact.language
hfmodem.merlin.modem -> hfmodem.contact.audio
hfmodem.merlin.modem -> hfmodem.contact.calls
hfmodem.merlin.modem -> hfmodem.contact.frame
hfmodem.merlin.modem -> hfmodem.monitor.cw
hfmodem.merlin.script -> hfmodem.contact.calls
hfmodem.merlin.script -> hfmodem.contact.facts
hfmodem.operator.bootstrap -> consolelib.onboarding
hfmodem.sabir.sim.m4 -> hfmodem.sabir.host
hfmodem.sabir.sim.m4 -> hfmodem.sabir.host.hostlink
hfmodem.sabir.sim.m4 -> hfmodem.sabir.host.messages
hfmodem.shrike.onair -> hfmodem.winlink
hfmodem.shrike.onair -> hfmodem.winlink.session
hfmodem.shrike.p3progress -> hfmodem.winlink.session
hfmodem.shrike.ptc -> hfmodem.winlink
hfmodem.shrike.traffic -> hfmodem.winlink
hfmodem.station.mail -> hfmodem.winlink""".splitlines())


def _module(path, root):
    parts = path.relative_to(root.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _targets(source, module, is_package, modules):
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            target = resolve_name(name, package) if node.level else name
            for alias in node.names:
                candidate = target + "." + alias.name
                yield candidate if candidate in modules else target


def _import_graph(roots):
    modules = {_module(path, root): path for root in roots for path in _sources(root)}
    edges = set()
    for module, path in modules.items():
        if "tests" in module.split("."):
            continue
        for target in _targets(path.read_text(encoding="utf-8"), module,
                               path.name == "__init__.py", modules):
            if target != module:
                edges.add((module, target))
    return edges


def _layer(module):
    parts = module.split(".")
    if len(parts) == 1:
        return 4
    for level, names in LAYERS.items():
        if parts[1] in names:
            if level == 1 and ("host" in parts[2:] or parts[-1] in {
                    "station_io", "station_runtime"}):
                return 2
            return level
    raise AssertionError(f"Assign a layer to {module}")


def _violation(source, target):
    src, dst = source.split("."), target.split(".")
    if dst[0] in PRIVATE_ROOTS or "tests" in dst:
        return True
    if src[0] == "hfhost":
        return dst[0] != "hfhost" and dst[0] not in sys.stdlib_module_names
    if src[0] == "creance":
        return dst[0] == "hfmodem" and not source.startswith("creance.monitor.runners.")
    if dst[0] == "creance":
        return True
    if dst[0] == "hfhost":
        return False
    if dst[0] != "hfmodem":
        return False
    source_layer, target_layer = _layer(source), _layer(target)
    if source_layer < target_layer:
        return True
    families = LAYERS[1]
    return (len(src) > 1 and len(dst) > 1 and src[1] in families
            and dst[1] in families and src[1] != dst[1])


def test_import_layers():
    roots = [PACKAGES / name / name for name in ("hfmodem", "hfhost", "creance")]
    for root in roots:
        if root.name == "hfmodem":
            for path in _sources(root):
                if "tests" not in path.relative_to(root).parts:
                    _layer(_module(path, root))
    violations = {f"{source} -> {target}" for source, target in _import_graph(roots)
                  if _violation(source, target)}
    allowed = ALLOWLIST
    assert not violations - allowed, "New import violations:\n" + "\n".join(sorted(violations - allowed))
    assert not allowed - violations, "Remove resolved allowances:\n" + "\n".join(sorted(allowed - violations))


@pytest.mark.parametrize("module, source, target", [
    ("hfmodem.core.example", "import os, hfmodem.contact.frame", "hfmodem.contact.frame"),
    ("hfmodem.core.example", "from ..contact import frame", "hfmodem.contact.frame"),
    ("hfmodem.core.example", "from hfmodem import contact", "hfmodem.contact"),
    ("hfmodem.core.example", "from .. import contact", "hfmodem.contact"),
    ("hfmodem.core.example", "def f():\n    from ..contact.frame import Over", "hfmodem.contact.frame"),
    ("hfmodem.core.example", "if TYPE_CHECKING:\n    import hfmodem.station", "hfmodem.station"),
    ("hfmodem.core.example", "from ..contact import *", "hfmodem.contact"),
    ("hfmodem.core.regulatory.example", "from ...contact.frame import Over", "hfmodem.contact.frame"),
    ("hfmodem.shrike.host.example", "from ...contact import frame", "hfmodem.contact.frame"),
    ("hfmodem.shrike.example", "from ..kestrel import phy", "hfmodem.kestrel.phy"),
    ("hfmodem.ax25.example", "from ..station import arbiter", "hfmodem.station.arbiter"),
    ("hfmodem.contact.example", "from ..bbs import service", "hfmodem.bbs.service"),
    ("hfmodem.station.example", "from ..contact import frame", "hfmodem.contact.frame"),
    ("hfmodem.bbs.example", "import hfmodem.tests.fixture", "hfmodem.tests.fixture"),
    ("hfmodem.bbs.example", "from ..tests import fixture", "hfmodem.tests.fixture"),
    ("hfmodem.bbs.example", "import tools.consolelib", "tools.consolelib"),
    ("hfmodem.bbs.example", "import consolelib", "consolelib"),
    ("hfhost.example", "from hfmodem.core import config", "hfmodem.core.config"),
    ("creance.example", "from hfmodem.core import config", "hfmodem.core.config"),
    ("creance.monitor.runners.example", "import tools.private", "tools.private"),
])
def test_new_import_violation_is_rejected(module, source, target, tmp_path, monkeypatch):
    from hfmodem.tests.gates import test_import_direction as gate

    for name, content in ((module, source), (target, "")):
        if name.split(".")[0] not in {"hfmodem", "hfhost", "creance"}:
            continue
        path = tmp_path / name.split(".")[0] / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(gate, "PACKAGES", tmp_path)
    monkeypatch.setattr(gate, "ALLOWLIST", set())
    with pytest.raises(AssertionError, match="New import violations"):
        gate.test_import_layers()
    monkeypatch.setattr(gate, "ALLOWLIST", {f"{module} -> {target}"})
    gate.test_import_layers()
    path = tmp_path / module.split(".")[0] / (module.replace(".", "/") + ".py")
    path.write_text("")
    with pytest.raises(AssertionError, match="Remove resolved allowances"):
        gate.test_import_layers()


@pytest.mark.parametrize("source, target", [
    ("hfmodem.shrike.phy", "hfmodem.core.crc"),
    ("hfmodem.shrike.host.server", "hfmodem.shrike.phy"),
    ("hfmodem.station.service", "hfmodem.shrike.station_runtime"),
    ("hfmodem.contact.runtime", "hfmodem.station.client"),
    ("hfmodem.bbs.service", "hfmodem.contact.runtime"),
    ("creance.monitor.runners.example", "hfmodem.shrike.phy"),
    ("creance.conformance", "hfhost.link"),
    ("hfmodem.core.config", "hfhost.config"),
    ("hfhost.link", "socket"),
])
def test_intended_import_direction(source, target):
    assert not _violation(source, target)


def test_package_initializer_resolves_relative_and_lazy_imports(tmp_path):
    root = tmp_path / "hfmodem"
    source = root / "core/regulatory/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("def f():\n    from ...contact import frame, runtime\n")
    for name in ("frame", "runtime"):
        target = root / f"contact/{name}.py"
        target.parent.mkdir(exist_ok=True)
        target.write_text("")
    assert _import_graph([root]) == {
        ("hfmodem.core.regulatory", "hfmodem.contact.frame"),
        ("hfmodem.core.regulatory", "hfmodem.contact.runtime"),
    }


def test_new_package_needs_a_layer():
    with pytest.raises(AssertionError, match="Assign a layer"):
        _layer("hfmodem.new_package.nested.module")


def _imports(path: Path) -> set[str]:
    """Top-level module names this file imports, absolute imports only."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _sources(pkg: Path) -> list[Path]:
    return sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts)


def test_hfhost_imports_nothing_but_the_standard_library():
    banned = {"hfmodem", "creance"}
    offences = []
    for f in _sources(PACKAGES / "hfhost" / "hfhost"):
        for name in _imports(f):
            if name in banned or not (name in sys.stdlib_module_names or name == "hfhost"):
                offences.append(f"{f.relative_to(REPO)} imports {name}")
    assert not offences, (
        "hfhost has to run beside a commercial modem on a machine with no numpy:\n  "
        + "\n  ".join(offences))


def _operator_resource(rel, value):
    return (rel == 'hfmodem/operator/bootstrap.py'
            and value in ('tools', 'tools/console.py', 'tools/desk/face.py'))


@pytest.mark.parametrize('value, accepted', [
    ('tools', True), ('tools/console.py', True), ('tools/desk/face.py', True),
    ('tools/desk/other.py', False), ('tools/private.py', False),
])
def test_bootstrap_resource_allowance_is_exact(value, accepted, tmp_path, monkeypatch):
    source = tmp_path / 'hfmodem/hfmodem/operator/bootstrap.py'
    source.parent.mkdir(parents=True)
    source.write_text(f'path = {value!r}\n')
    from hfmodem.tests.gates import test_import_direction as gate
    monkeypatch.setattr(gate, 'PACKAGES', tmp_path)
    monkeypatch.setattr(gate, 'REPO', tmp_path)
    if accepted:
        test_shipping_code_never_reaches_outside_its_distribution('hfmodem')
    else:
        with pytest.raises(AssertionError, match='names'):
            test_shipping_code_never_reaches_outside_its_distribution('hfmodem')


def _capture_output_component(rel, node, parents):
    if rel != 'hfmodem/core/capture_identity.py' or node.value != 'working':
        return False
    parent = parents.get(node)
    if not (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)
            and parent.right is node and isinstance(parent.left, ast.Name)
            and parent.left.id == 'repo'):
        return False
    while parent is not None and not isinstance(parent, ast.FunctionDef):
        parent = parents.get(parent)
    return parent is not None and parent.name == 'allocate_arm'


@pytest.mark.parametrize('expression, accepted', [
    ('repo / "working" / basename', True),
    ('Path("working")', False), ('sys.path.insert(0, "working")', False),
    ('repo / "tools" / basename', False),
])
def test_capture_output_exception_cannot_hide_private_inputs(expression, accepted):
    tree = ast.parse('def allocate_arm(repo):\n    return ' + expression)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))
    assert _capture_output_component('hfmodem/core/capture_identity.py', node, parents) is accepted
    assert not _capture_output_component('hfmodem/core/other.py', node, parents)


@pytest.mark.parametrize("dist", ["hfmodem", "hfhost", "creance"])
def test_shipping_code_never_reaches_outside_its_distribution(dist):
    """Private archive paths must never be runtime dependencies. The installed
    operator bootstrap resolves its reviewed tools bundle, checked byte-for-byte
    by the wheel gate. Tests may use optional archive fixtures."""
    pkg = PACKAGES / dist / dist
    offences = []
    for f in _sources(pkg):
        rel = f.relative_to(PACKAGES / dist).as_posix()
        if "tests" in f.relative_to(pkg).parts or rel.startswith(CREANCE_MODEM_CARVE_OUT):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            # Any string literal naming either tree, not the quoted form of the
            # word: an audit inserted `sys.path.insert(0, "/…/hfmodem/working/vara")`
            # and the substring check walked past it, because the path contains
            # `working` but not `"working"`.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parts = node.value.replace("\\", "/").split("/")
                # The installed console includes these exact reviewed resources;
                # packaging gates compare every bundled path and byte to its source.
                operator_resource = _operator_resource(rel, node.value)
                if ("working" in parts or "tools" in parts) and not operator_resource and not _capture_output_component(rel, node, parents):
                    offences.append(f"{f.relative_to(REPO)} names {node.value!r}")
            # And a bare import of either, which needs no sys.path line at all.
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in ("working", "tools"):
                        offences.append(f"{f.relative_to(REPO)} imports {a.name}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] in ("working", "tools"):
                    offences.append(f"{f.relative_to(REPO)} imports {node.module}")
    assert not offences, (
        "an unbundled path ships a distribution that installs "
        "and then fails on first use:\n  " + "\n  ".join(sorted(set(offences))))


#: The seven trees this was merged from. They are kept — they hold everything
#: `git mv` could not carry — but nothing here may point at them.
LEGACY_REPOS = ("pactor-open", "vara-open", "ardop-open")


def test_nothing_shipping_points_at_the_repositories_this_was_merged_from():
    """The originals are archives, not dependencies.

    Two of these were live when the rule was written, not merely stale prose:
    `shrike/ota.py` defaulted its oracle runner to a path under `pactor-open/`,
    and `perf/txbench.py` put `ardop-open/` on `sys.path`. A reference that still
    resolves on the machine where the merge happened is the worst kind — it works
    here and nowhere else, and it silently reads code that is no longer the code.

    `working/` is exempt: it *is* the record of those trees, and rewriting its
    references would falsify it.
    """
    roots = [PACKAGES / d / d for d in ("hfmodem", "hfhost", "creance")]
    roots += [REPO / "tools", REPO / "perf", REPO / "publish",
              REPO / "examples", REPO / "docs"]
    offences = []
    for root in roots:
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix not in (".py", ".md", ".toml", ".sh", ""):
                continue
            if "__pycache__" in f.parts or f.name == Path(__file__).name:
                continue        # this file names them in order to forbid them
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for legacy in LEGACY_REPOS:
                if legacy in text:
                    offences.append(f"{f.relative_to(REPO)} names {legacy}")
    assert not offences, (
        "the merged-from repositories are archives, not dependencies:\n  "
        + "\n  ".join(sorted(set(offences))))


def test_station_client_imports_only_stdlib_and_wire():
    from hfmodem.tests.gates.test_station_client_stdlib import STATION, import_offences

    assert not [offence for name in ('client.py', 'wire.py')
                for offence in import_offences(STATION / name)]
