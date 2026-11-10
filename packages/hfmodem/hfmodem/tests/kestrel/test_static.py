# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gate: every module imports and every entry point starts."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hfmodem.tests import hermetic

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

_SKIP_PARTS = {"__pycache__", ".venv", "assets", "node_modules"}


def _importable_modules() -> list[Path]:
    """Python under kestrel/ only — what `import` can actually reach."""
    return sorted(f for f in _PKG.rglob("*.py")
                  if not (_SKIP_PARTS | {"tests"}) & set(f.relative_to(_ROOT).parts))


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
        hermetic.guard.admit([*argv, "check", *scope], os.environ)
        try:
            r = subprocess.run([*argv, "check", *scope],
                               capture_output=True, text=True, cwd=_ROOT, timeout=300)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        assert r.returncode == 0, r.stdout[-4000:]
        return
    pytest.skip("ruff not available, directly or via uvx")
