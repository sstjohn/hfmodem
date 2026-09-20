# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""No working record in the shipped package tree.

`creance/` describes what the harness measures and how it measures it. How
anyone came to know a protocol is a different document with a different
audience, and the two mix easily once a module has been edited enough times.
This gate runs on every suite so the separation is a property of the tree rather
than a scrub someone performs later against months of accreted markers.

Scoped to the package. `prompts/`, `future-work/`, `tests/` and `examples/` are
working record: they are supposed to carry this vocabulary, and they are outside
the scanned tree — which is why this file, which must name every forbidden term
to look for it, does not trip itself.

There is no allowlist. The tree is clean today; keeping it clean is cheaper than
maintaining a list of blessed exceptions, and an allowlist that grows is how a
gate like this stops meaning anything.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "creance"

# One letter of each term is bracketed, so the pattern text is not itself an
# instance of what it matches and this file does not answer to a scan of the
# whole distribution.
PATTERNS = {
    "citation tags": r"\[pro[v]:",
    # 4+ digits, which no protocol constant in this tree needs: every hex
    # literal here is a one-byte frame code (`0x11`) or a byte mask (`0xFF`).
    # The narrower section-qualified form the modem packages need, which admits a
    # wide constant so long as it is not an address, buys nothing at this width.
    "binary offsets": r"0x[0-9a-fA-F]{4,}",
    "reference-decoder vocabulary": r"PMON|\boracle\b",
    "tooling names": r"\bwine\b|\bquarantin[e]\b",
    "disassembly vocabulary": r"decompi[l]|disassembl",
    "method vocabulary": r"clean-?room",
}


def _hits(pattern: str) -> list[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    found = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                found.append(f"{path.relative_to(PACKAGE.parent)}:{n}: {line.strip()}")
    return found


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_package_is_free_of_working_record(name):
    hits = _hits(PATTERNS[name])
    assert not hits, f"{name} in shipped package:\n  " + "\n  ".join(hits)
