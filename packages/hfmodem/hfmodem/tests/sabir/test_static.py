# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gates: publication markers, undefined names, and every CLI starts.

Three classes of defect, one harness, because standing up a second one invites
them to drift apart.

1. **Publication markers.** Shipped prose documents the protocol; how anyone
   came to know it is a working record with a different audience and a different
   lifetime. Keeping the two apart is a tested property of the tree, checked on
   every run, rather than a scrub someone performs later against a tree that has
   been accreting markers for months.

2. **Undefined names** (ruff's F rules). Catches the `NameError` on a path no
   test executes.

3. **Every CLI entry point starts.** A module that imports cleanly can still be
   unable to run; both classes have reached sibling trees.

Scoped to the package tree and `docs/`. Working records are supposed to contain
this material.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG.parents[2]
PACKAGE = "sabir"
DOCS = ROOT / "docs" / "protocols" / PACKAGE

# Vocabulary describing how a protocol was learned rather than what it is. That
# is a working record: different audience, different lifetime, stays private.
#
# One letter of each term is bracketed, so the pattern text is not itself an
# instance of what it matches and this file does not answer to a scan of the
# whole distribution.
MARKERS = re.compile(
    r"\[pro[v]:|\bPMON\b|\boracle\b|\bdecompi[l]|\bdisassembl|\bclean[- ]?room\b"
    r"|\bquarantin[e]\b|reverse[- ]engineer|\bIDA Pro\b|\bGhidra\b",
    re.IGNORECASE)

# Deliberately NOT "any hex literal": a protocol is full of legitimate wide
# constants -- `host/cbor.py`'s `0x100000000` is a CBOR width bound. What leaks
# is an address inside a citation tag, so match that shape and nothing else.
PROV_HEX = re.compile(
    r"\.(?:text|rodata|data|bss)\s*@\s*0x[0-9a-fA-F]+|\bfcn\.0x[0-9a-fA-F]+"
    r"|\[pro[v]:[^\]]*0x[0-9a-fA-F]+", re.IGNORECASE)


def _sources():
    yield from sorted((PKG / PACKAGE).rglob("*.py"))
    yield from sorted(DOCS.glob("*.md"))


def _scan(pattern) -> list[str]:
    hits = []
    for path in _sources():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = pattern.search(line)
            if m:
                hits.append(f"{path.relative_to(ROOT)}:{i}: "
                            f"{m.group(0)!r} in {line.strip()[:72]}")
    return hits


def test_no_publication_markers():
    hits = _scan(MARKERS)
    assert not hits, "publication markers in the shipped tree:\n" + "\n".join(hits)


def test_no_provenance_addresses():
    hits = _scan(PROV_HEX)
    assert not hits, "binary addresses in provenance context:\n" + "\n".join(hits)


def _ruff() -> str | None:
    for cand in (Path(sys.prefix) / "bin" / "ruff", ROOT / ".venv" / "bin" / "ruff"):
        if cand.exists():
            return str(cand)
    return shutil.which("ruff")


def test_no_pyflakes_errors():
    ruff = _ruff()
    if ruff is None:
        pytest.skip("ruff not installed")
    r = subprocess.run([ruff, "check", PACKAGE, f"tests/{PACKAGE}",
                        "--select", "F"],
                       cwd=PKG, capture_output=True, text=True)
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
