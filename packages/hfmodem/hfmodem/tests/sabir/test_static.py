# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gates: undefined names and every CLI starts."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hfmodem.tests import hermetic

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG.parents[2]
PACKAGE = "sabir"
DOCS = ROOT / "docs" / "protocols" / PACKAGE


def _sources():
    yield from sorted((PKG / PACKAGE).rglob("*.py"))
    yield from sorted(DOCS.glob("*.md"))


def _ruff() -> str | None:
    for cand in (Path(sys.prefix) / "bin" / "ruff", ROOT / ".venv" / "bin" / "ruff"):
        if cand.exists():
            return str(cand)
    return shutil.which("ruff")


def test_no_pyflakes_errors():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff not installed")
    argv = [ruff, "check", PACKAGE, f"tests/{PACKAGE}", "--select", "F"]
    hermetic.guard.admit(argv, os.environ)
    r = subprocess.run(argv, cwd=PKG, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("argv", [
    ["-m", "hfmodem.sabir.host.run_server", "--help"],
    ["-m", "hfmodem.sabir.monitor", "--help"],
    ["-m", "hfmodem.sabir.offair", "--help"],
    ["-m", "hfmodem.sabir.offair", "presence", "--help"],
])
def test_entry_point_starts(argv):
    """Importing cleanly is not the same as being able to run."""
    r = subprocess.run([sys.executable, *argv], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (f"{' '.join(argv)} exited {r.returncode}\n"
                               f"{r.stderr[-800:]}")


# Products named in the tree must be named in NOTICE. Trademark hygiene rots
# quietly: a new comparative mention lands, and the non-affiliation statement
# silently stops covering it.
MARKS = ("VARA", "PACTOR", "ARDOP", "Winlink", "VarAC")


def test_notice_covers_every_product_named():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    body = "\n".join(p.read_text(encoding="utf-8") for p in _sources())
    named = {m for m in MARKS if re.search(rf"\b{m}\b", body)}
    missing = sorted(m for m in named if not re.search(rf"\b{m}\b", notice))
    assert not missing, f"named in the tree but not in NOTICE: {missing}"
