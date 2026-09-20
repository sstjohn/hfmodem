# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The `besra/` tree documents the protocol, not how anyone came to know it.

Shipped prose has a different audience and a different lifetime from a working
note, and the two are easy to mix once a module has been edited a few dozen
times. This gate asserts that `besra/` carries none of the second kind: method
vocabulary, capture and analysis tooling names, or a binary address inside a
citation tag.

Two things it deliberately does *not* flag:

  * **Descriptive protocol usage.** "ARDOP", and cross-references to sibling
    modes, are normative or descriptive and are supposed to ship. ARDOP is not
    even a trademark. A gate that deleted those would delete correct text.
  * **Bare hex.** `0x8810`, `0x11D`, `0xC6` are besra's own protocol constants.
    Only an address *in a citation context* is a marker, so the hex rule is
    narrow. A gate that flags correct work is a gate people switch off.
"""

from __future__ import annotations

import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[2] / "besra"

#: Method vocabulary. Whole-word where a fragment would false-match (an "oracle"
#: pattern must not fire on "oracular"); ARDOP and spec citations are absent by
#: construction and are not markers.
#:
#: One letter of each term is bracketed, so the pattern text is not itself an
#: instance of what it matches and this file does not answer to a scan of the
#: whole distribution.
_MARKERS = [
    re.compile(r"reverse[- ]?engineer", re.I),
    re.compile(r"clean[- ]?room", re.I),
    re.compile(r"\bdecompi[l]", re.I),
    re.compile(r"\bdisassembl", re.I),
    re.compile(r"\bquarantin[e]\b", re.I),
    re.compile(r"\bPMON\b"),
    re.compile(r"\boracle\b", re.I),
    re.compile(r"\[pro[v]:"),
    # An address inside a citation tag — not a bare protocol constant.
    re.compile(r"fcn\.0x[0-9a-fA-F]+"),
    re.compile(r"\.(?:text|rodata|data|bss)@0x[0-9a-fA-F]+"),
]


def test_besra_tree_carries_no_provenance():
    offenders = []
    for py in sorted(_PKG.rglob("*.py")):
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for pat in _MARKERS:
                if pat.search(line):
                    rel = py.relative_to(_PKG.parent)
                    offenders.append(f"{rel}:{n}: {line.strip()}  [{pat.pattern}]")
    assert not offenders, (
        "method vocabulary must not ship; state the fact, not how it was "
        "learned:\n" + "\n".join(offenders))
