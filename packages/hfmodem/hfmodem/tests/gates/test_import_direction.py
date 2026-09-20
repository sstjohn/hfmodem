# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Which way imports are allowed to point, asserted rather than remembered.

Four rules, and each one is load-bearing for a different reason:

  * **hfhost imports nothing from here, and nothing outside the standard
    library.** It has to run on a machine with a commercial modem and no numpy,
    and its independence from these implementations is the entire reason
    creance's grading means anything — a client written from the same source as
    the server grades nothing. This is checked against the built wheel too
    (`test_packaging.py`); here it is checked against the tree, because a
    violation should fail the moment it is written rather than at build time.

  * **creance imports no modem**, with one carve-out named below.

  * **No protocol imports another protocol.** They are four normative answers to
    the same question, not four layers.

  * **Shipping code imports nothing from `working/` or `tools/`.** Neither is in
    any wheel, so an import that crosses that line produces a distribution that
    installs and then fails on first use.

The carve-out: `creance/monitor/runners/` holds the processes the monitor spawns
— decode adapters that turn audio into detection JSON, and the PortAudio capture
that turns a sound card into that audio. Those necessarily import a decoder or
the modems' device table. They are not on the conformance-grading path — grading
goes through `hfhost.Link` over a socket — so the independence argument is
untouched. Naming
the exception here is the point: an unnamed exception is indistinguishable from a
rule nobody enforces.

One piece of debt the carve-out currently covers, recorded so it is not mistaken
for a design: `kestrel_runner.py` reads its burst segmenter out of
`tools/vara_monitor.py`, so it needs a source tree and cannot run from an
installed wheel. The segmenter belongs in the package.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
PACKAGES = REPO / "packages"

PROTOCOLS = {"shrike", "kestrel", "besra", "sabir"}

# creance/monitor/runners/ — decode adapters, not the grading path.
CREANCE_MODEM_CARVE_OUT = "creance/monitor/runners"


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
    banned = {"hfmodem", "creance"} | PROTOCOLS
    offences = []
    for f in _sources(PACKAGES / "hfhost" / "hfhost"):
        for name in _imports(f):
            if name in banned or not (name in sys.stdlib_module_names or name == "hfhost"):
                offences.append(f"{f.relative_to(REPO)} imports {name}")
    assert not offences, (
        "hfhost has to run beside a commercial modem on a machine with no numpy:\n  "
        + "\n  ".join(offences))


def test_creance_imports_no_modem_outside_the_runners():
    offences = []
    for f in _sources(PACKAGES / "creance" / "creance"):
        rel = f.relative_to(PACKAGES / "creance").as_posix()
        if rel.startswith(CREANCE_MODEM_CARVE_OUT):
            continue
        for name in _imports(f):
            if name in PROTOCOLS or name == "hfmodem":
                offences.append(f"{rel} imports {name}")
    assert not offences, (
        "creance grades a modem through its host interface, not by importing it. "
        f"The only exception is {CREANCE_MODEM_CARVE_OUT}/:\n  " + "\n  ".join(offences))


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_no_protocol_imports_another(protocol):
    """Absolute *and* relative.

    A text search for `hfmodem.<other>` misses `from ...kestrel.coding.crc import
    crc16_genibus` in a sabir module, which imports perfectly well — an audit
    constructed exactly that and the gate stayed green. The risk is new: before the
    merge these were separate distributions and a relative cross-import could not
    be written at all.
    """
    others = PROTOCOLS - {protocol}
    root = PACKAGES / "hfmodem" / "hfmodem" / protocol
    pkg_parts = ("hfmodem", protocol)
    offences = []
    for f in _sources(root):
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        here = list(f.relative_to(PACKAGES / "hfmodem" / "hfmodem").parts[1:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    target = (node.module or "").split(".")
                else:
                    # `from ...x import y` climbs `level - 1` packages from here.
                    base = list(pkg_parts) + here
                    up = node.level - 1
                    base = base[:len(base) - up] if up <= len(base) else []
                    target = base + (node.module or "").split(".")
            elif isinstance(node, ast.Import):
                target = node.names[0].name.split(".")
            else:
                continue
            for other in others:
                if other in target and "hfmodem" in target:
                    offences.append(f"{f.relative_to(REPO)} reaches {other}")
    assert not offences, (
        "four protocols, four normative answers — not four layers:\n  "
        + "\n  ".join(sorted(set(offences))))


def test_core_imports_no_protocol():
    """`core` is what is true of the radio rather than of a waveform. The moment
    it knows a protocol's name, every protocol inherits that protocol's choices."""
    core = PACKAGES / "hfmodem" / "hfmodem" / "core"
    if not core.is_dir():
        pytest.skip("core/ is not extracted yet")
    offences = [
        f"{f.relative_to(REPO)} names hfmodem.{p}"
        for f in _sources(core) for p in PROTOCOLS
        if f"hfmodem.{p}" in f.read_text(encoding="utf-8")
    ]
    assert not offences, "\n  ".join(offences)


@pytest.mark.parametrize("dist", ["hfmodem", "hfhost", "creance"])
def test_shipping_code_never_reaches_outside_its_distribution(dist):
    """`working/` is the archive and `tools/` are station instruments; neither is
    in any wheel. Tests may read fixtures from the archive — that is what it is
    for — so this covers the package, not the suite."""
    pkg = PACKAGES / dist / dist
    offences = []
    for f in _sources(pkg):
        rel = f.relative_to(PACKAGES / dist).as_posix()
        if "tests" in f.relative_to(pkg).parts or rel.startswith(CREANCE_MODEM_CARVE_OUT):
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            # Any string literal naming either tree, not the quoted form of the
            # word: an audit inserted `sys.path.insert(0, "/…/hfmodem/working/vara")`
            # and the substring check walked past it, because the path contains
            # `working` but not `"working"`.
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parts = node.value.replace("\\", "/").split("/")
                if "working" in parts or "tools" in parts:
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
        "neither tree is in any wheel, so this ships a distribution that installs "
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
