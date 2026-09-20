# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gate: pyflakes-class checks over besra and its tests.

This exists because a `NameError` (an undefined name on a code path no test
executed) once shipped in the monitor CLI. Ruff's F rules catch that class —
undefined names, unused imports/vars — without having to run every path. Runs
`ruff` if it is installed (it is a `dev` dependency); skips cleanly otherwise so
the suite stays runnable on a bare checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[2]
_REPO = _PKG.parents[2]


def _ruff() -> str | None:
    for cand in (Path(sys.prefix) / "bin" / "ruff", _REPO / ".venv" / "bin" / "ruff"):
        if cand.exists():
            return str(cand)
    return shutil.which("ruff")


def test_no_pyflakes_errors():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff not installed (pip install -e '.[dev]')")
    result = subprocess.run(
        [ruff, "check", "besra", "tests/besra", "--select", "F"],
        cwd=_PKG, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
