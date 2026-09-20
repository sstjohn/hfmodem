# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gate: every module imports, every entry point starts, no working record ships.

Three classes of defect that the rest of the suite cannot see, because it exercises
functions rather than the tree:

  * A module that no longer imports. This gate's first run found that
    `kestrel/arq/modem.py` did not import at all outside the source tree: an
    ``except Exception`` fallback bound ``ModemCore = object``, which made
    ``class KestrelModem(ModemCore, ArqIO)`` an unsatisfiable MRO. Importing every
    module is what makes that class of defect visible.
  * An entry point that cannot start. Module-level tests all import cleanly, so a
    bad argparse or a missing import in a ``__main__`` path goes unnoticed until
    someone reaches for the tool.
  * Working-record vocabulary in shipping prose. Provenance tags, binary addresses
    and harness names belong in the private tree, not in a published package.

The publication check is a ratchet with the baseline in the source, not a data file,
so any change shows up in the diff. kestrel's shipping code is at zero today and the
gate's job is to keep it there while the tree moves — a gate that first runs at
publication time has already failed.

Deliberately not bare hex: kestrel is full of protocol constants (`0x1021`, `0xFFFF`,
`0x53c148` in prose about tables) and a gate that cannot tell a CRC polynomial from an
address is one nobody keeps green. Only qualified forms with no innocent reading.
"""
from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[5]
_IMPORT_ROOT = Path(__file__).resolve().parents[3]
_PKG = _IMPORT_ROOT / "hfmodem" / "kestrel"
_TESTS = _IMPORT_ROOT / "hfmodem" / "tests" / "kestrel"
_TOOLS = _ROOT / "tools"

# Importing the tree and reading its prose works wherever the package is installed;
# linting it does not, because the rule set lives in the source tree's pyproject.
requires_source_tree = pytest.mark.skipif(
    not (_IMPORT_ROOT / "pyproject.toml").exists(),
    reason="not a source tree (installed-wheel run)")

# What the ratchet reads. It used to be `kestrel/**.py` minus tests and assets, and
# reported zero — which was true of that scope and told you nothing about the rest.
# tools/ ships, kestrel/tests/ is inside the wheel, and the .md files are the first
# thing a reader opens; none of them were being looked at.
#
# The never-crossing documents are named here rather than excluded by a glob, so
# that a file joining or leaving that set is a visible edit.
# This file defines the patterns, so it matches all of them; scanning it measures
# nothing but itself. The document's name is split the way the patterns bracket a
# letter: this file ships, and the distribution scan must read a reference here
# rather than the marker itself.
_SELF = "packages/hfmodem/hfmodem/tests/kestrel/test_static.py"
_NEVER_CROSSES = {
    "packages/hfmodem/hfmodem/kestrel/CLEAN" "ROOM-AND-LICENSE.md",
    "packages/hfmodem/hfmodem/kestrel/rx/PRO" "VENANCE.md",
}
_SKIP_PARTS = {"__pycache__", ".venv", "assets", "node_modules"}
# `assets` is skipped as a whole directory rather than file by file, and there is
# one left: tx/, holding the synthesis pulse and the note beside it. rx/ has no
# assets any more -- every table it needs is computed -- so its note moved up a
# level and is named above instead. Both are denied at the boundary by
# `**/PRO` `VENANCE.md`; the difference is only which mechanism keeps this ratchet
# from reading them.


def _importable_modules() -> list[Path]:
    """Python under kestrel/ only — what `import` can actually reach."""
    return sorted(f for f in _PKG.rglob("*.py")
                  if not (_SKIP_PARTS | {"tests"}) & set(f.relative_to(_ROOT).parts))


def _scanned() -> list[Path]:
    out = []
    for base in (_PKG, _TESTS, _TOOLS):
        if not base.is_dir():
            continue
        for f in (*base.rglob("*.py"), *base.rglob("*.md")):
            if _SKIP_PARTS & set(f.relative_to(_ROOT).parts):
                continue
            rel = str(f.relative_to(_ROOT))
            if rel in _NEVER_CROSSES or rel == _SELF:
                continue
            out.append(f)
    return sorted(set(out))


# One letter of each term is bracketed, so the pattern text is not itself an
# instance of what it matches and this file does not answer to a scan of the
# whole distribution.
_PUBLICATION_PATTERNS = {
    "citation tag": r"\[pro[v]:",
    "binary section address": r"\.(?:text|rodata|data|bss)@0x[0-9a-fA-F]+",
    "disassembler symbol": r"\bfcn\.[0-9a-fA-F]{6,}",
    "harness vocabulary": r"\boracle\b|\bPMON\b|\bwineprefix\b|\bquarantin[e]\b",
    "disassembly vocabulary": r"disassembl|decompi[l]",
    # Method, not protocol. How this package came to know what it knows has a
    # different audience and a different lifetime from what it knows, and only
    # the second is documentation. The pattern set matters more than the
    # baseline: an audit found seven of these while the gate reported zero,
    # because the count was honest about the patterns it had and the pattern set
    # was the incomplete thing.
    "method vocabulary": r"(?i)\bclean-?room\b|\breverse[- ]engineer",
}

# Baseline per pattern, and it is deliberately NOT all zero.
#
# The previous version reported zero across every pattern, which was true of the
# scope it read — `kestrel/**.py` minus tests — and silent about tools/, the tests
# that ship inside the wheel, and the .md files a reader opens first. Widening the
# scope makes the real number visible, and the real number is not zero.
#
# What remains, and why it is not simply reworded away:
#
#   harness vocabulary (10) — `kestrel/tests/kestrel/corpora.py` and three test modules
#     name the `oracle/` directory because they load from it, and
#     `tools/vara_rig_bridge.py` puts it on sys.path. These are references to a
#     real path, not narrative; they go when the harness is renamed or the tests
#     stop reaching outside the package, which is WP-P2 work and not a comment fix.
#   method vocabulary (3) — `kestrel/README.md` links the licence-and-method
#     statement beside it, itself the never-crossing document `_NEVER_CROSSES`
#     names. The link goes when the README is rewritten for publication rather
#     than scrubbed.
#
# A ratchet's job is that these can only fall. Lowering a number here is a real
# change; raising one should be argued for in the diff.
_BASELINE = dict.fromkeys(_PUBLICATION_PATTERNS, 0)
_BASELINE["harness vocabulary"] = 10
_BASELINE["method vocabulary"] = 3


@pytest.mark.parametrize("path", _importable_modules(), ids=lambda p: p.stem)
def test_module_imports(path: Path):
    """Import every shipping module. Catches the defect class that `modem.py`'s
    ``except Exception`` fallback is specifically built to hide."""
    name = ".".join(path.relative_to(_IMPORT_ROOT).with_suffix("").parts)
    importlib.import_module(name)


@pytest.mark.parametrize("entry", ["hfmodem.kestrel.arq.run_pair"])
def test_entry_point_starts(entry: str):
    """``--help`` must reach argparse. A tool that cannot start is invisible to a
    suite that only imports modules."""
    r = subprocess.run([sys.executable, "-m", entry, "--help"],
                       capture_output=True, text=True, cwd=_IMPORT_ROOT, timeout=60)
    assert r.returncode == 0, f"{entry} --help failed:\n{r.stderr[-2000:]}"


@pytest.mark.parametrize("label,pattern", sorted(_PUBLICATION_PATTERNS.items()))
def test_publication_ratchet(label: str, pattern: str):
    """No working-record vocabulary in shipping code, and the count may only fall."""
    rx = re.compile(pattern)
    hits = [f"{p.relative_to(_ROOT)}:{n}"
            for p in _scanned()
            for n, line in enumerate(p.read_text().splitlines(), 1)
            if rx.search(line)]
    assert len(hits) <= _BASELINE[label], (
        f"{label}: {len(hits)} > baseline {_BASELINE[label]}\n  " + "\n  ".join(hits[:12]))


@requires_source_tree
def test_ruff_clean():
    """The rule set from `pyproject.toml` — F (pyflakes), UP (modern syntax), I
    (import order). Not spelled out here, so the gate and the developer's own
    `ruff check` cannot drift apart.

    `uvx` is the fallback because ruff is a dev tool nobody should have to install
    into the interpreter under test, and a gate that skips on every machine is a
    gate that has already failed. It found a real defect the day it first ran:
    `KestrelModem.connected` was declared twice, so the ModemCore state property
    was shadowed by the FSM's connect callback and always read truthy.
    """
    # `tools/` is linted where it exists and not named where it does not: the
    # distribution ships the package and its suite without the station tools, and a
    # path ruff cannot open is an error rather than a clean tree.
    scope = [str(p) for p in (_PKG, _TESTS, _TOOLS) if p.exists()]
    for argv in (["ruff"], ["uvx", "ruff"]):
        try:
            r = subprocess.run([*argv, "check", *scope],
                               capture_output=True, text=True, cwd=_ROOT, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        assert r.returncode == 0, r.stdout[-4000:]
        return
    pytest.skip("ruff not available, directly or via uvx")
