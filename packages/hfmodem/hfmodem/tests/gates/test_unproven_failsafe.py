# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One answer, in every file, to whether the keying line falls when its owner dies.

For weeks the tree asserted both. This station's operating log counted it as the
fourth of four ways a transmission comes down — "the line is held by the open port, so
losing the process drops the key by itself" — and `docs/ONAIR-READINESS.md` and
`shrike/ota.py` both said "RTS drops when it exits or is killed. The direction is
the safe one", while `core/ptt.py`, `core/rig.py`, `docs/STATION.md`,
`tools/lib/attempts.sh` and `hfhost/supervisor.py` said in as many words that it
is **unmeasured on this hardware**. A reader could not tell which the code
believed, because it said both, and the comfortable one is the one nothing
supports: `HUPCL` is set, which is necessary and may not be sufficient, and the
CP2105's behaviour on last close has never been put to the test.

It is not an untestable claim — a serial line's state after a process dies is
observable with a second adapter, a jumper from its CTS to this one's RTS, and no
radio at all. `docs/STATION.md` carries the procedure. Until somebody runs it, the
tree may not write the sentence down.

The one time the question was put in anger it answered no. On 2026-08-04 a session
keying RTS was stopped from outside and kept transmitting on 10.1 MHz until the
operator powered the radio down. That is not the experiment — `ota.Rig` keys
through a child `rigctl`, so the process holding the port was not the one that
died, and hamlib clears `HUPCL` on the ports it opens — but it is the only
evidence on file, and it does not point the comfortable way.

WHAT THIS REFUSES, and it is narrow on purpose. A sentence that says the line
drops, or falls, or is dropped, when a process exits or dies or is killed, and
that carries no qualification anywhere in its paragraph. WHAT IT LEAVES ALONE: the
same sentence hedged (*may*, *if*, *unmeasured*, *unproven*, *hope*, *not known*),
which is the true version and belongs in the documents; anything about a CAT PTT,
which latches at the rig and genuinely does survive the process; and the
`HUPCL`-clearing note about hamlib, which is a statement about a port configured
by someone else.

And it holds the honest sentences in place too. Deleting "unmeasured on this
hardware" from `core/ptt.py` would satisfy every check above while making the tree
say less than it knows, so the three files that carry the finding are named and
must go on carrying it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]

#: Everything shipped that has ever had an opinion about this: the package, the
#: launchers they run under, and the documents an operator reads before a slot.
#: `working/` is deliberately out — a field note dated to the night it was written
#: is a record and not a claim, and a gate that fires on those is one people
#: switch off.
SEARCHED = [REPO / "packages", REPO / "tools", REPO / "docs",
            REPO / "README.md", REPO / "ARCHITECTURE.md"]
SUFFIXES = {".py", ".sh", ".md"}
#: `packages/*/build/lib` holds a copy of the package from whenever a wheel was
#: last made. It ships nothing and nobody reads it, so a claim corrected in the
#: source and still standing there is neither a finding nor a thing to fix.
_NOT_SOURCE = {"build", "dist", "__pycache__", ".venv", ".pytest_cache"}

#: The claim, as the four files that made it actually spelled it.
_CLAIM = re.compile(
    r"\b(?:line|RTS|DTR|key)\b[^.]{0,80}?\b(?:drops?|falls?|dropped|lowered)\b"
    r"[^.]{0,80}?\b(?:process|it)\b[^.]{0,40}?\b(?:dies|died|exits?|killed|death)\b"
    r"|\b(?:losing|lose)\s+the\s+\w*\s*process[^.]{0,60}?\b(?:drops?|unkeys?)\b"
    r"|\bfalls?\s+when\s+the\s+port\s+closes\b",
    re.IGNORECASE)

#: The finding, in the words the tree uses for it. Anywhere in the paragraph is
#: enough — a paragraph that says "unmeasured" and then quotes the sentence it is
#: correcting is the repair, not the offence. `HUPCL` counts: naming the bit is
#: naming the condition the claim rests on.
_HEDGED_PARA = re.compile(
    r"\bunmeasured\b|\bunproven\b|\b(?:not\s+(?:been\s+)?|never\s+)measured\b|"
    r"\bhope\b|\bHUPCL\b|\bnot\s+known\b|\bis\s+not\s+claimed\b|"
    r"\bnot\s+a\s+mechanism\b",
    re.IGNORECASE)
#: The ordinary hedges, which have to be IN THE SENTENCE and not merely in the
#: paragraph. That log is where the distinction was earned: the sentence
#: claiming the fail-safe outright sat in a bulleted list whose next bullet said
#: "a watchdog thread force-unkeys **if** a playback hangs", and a paragraph-wide
#: `if` spared the one claim in the tree this gate was written for.
_HEDGED_SENTENCE = re.compile(
    r"\bmay\b|\bmight\b|\bcan\b|\bcould\b|\bif\b|\bunless\b|\bwould\s+not\b",
    re.IGNORECASE)

#: The finding itself, and where it is kept. Removing it from any of these is the
#: other way to make the tree quieter than the evidence.
CARRIERS = {
    Path("packages/hfmodem/hfmodem/core/ptt.py"): "UNPROVEN on this hardware",
    Path("packages/hfmodem/hfmodem/core/rig.py"): "UNPROVEN on this\n   hardware",
    Path("docs/STATION.md"): "has not been measured on this hardware",
}


def _files():
    for root in SEARCHED:
        if root.is_file():
            yield root
        elif root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.suffix in SUFFIXES and _NOT_SOURCE.isdisjoint(p.parts):
                    yield p


#: What ends a sentence in the documents this reads. A bullet counts, and so does
#: a semicolon: the claim there lived in a `;`-separated list, and bounding on
#: full stops alone reached three bullets past it to borrow an `if`.
_BOUND = re.compile(r"[.;]|\n\s*(?:[-*]|\d+\.)\s")


def _sentence(para: str, start: int, end: int) -> str:
    """The claim with its own sentence around it and no more."""
    lo = max((m.end() for m in _BOUND.finditer(para, 0, start)), default=0)
    after = _BOUND.search(para, end)
    return para[lo:after.start() if after else len(para)]


def _paragraphs(text: str):
    at = 0
    for para in re.split(r"\n\s*\n", text):
        yield at, para
        at += para.count("\n") + 2


def test_nothing_shipped_says_the_keying_line_falls_when_its_owner_dies():
    said = []
    for path in _files():
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if path.name == Path(__file__).name:
            continue
        for line0, para in _paragraphs(text):
            for m in _CLAIM.finditer(para):
                if _HEDGED_PARA.search(para) or _HEDGED_SENTENCE.search(
                        _sentence(para, m.start(), m.end())):
                    continue
                line = line0 + para[:m.start()].count("\n") + 1
                said.append(f"{path.relative_to(REPO)}:{line}\n    {m.group(0).strip()}")
    assert not said, (
        "the fail-safe is asserted again, and nothing has measured it:\n\n"
        + "\n\n".join(said)
        + "\n\nSay what is true instead -- the line MAY fall when the port closes, "
          "HUPCL is set for it, and this adapter has never been asked. "
          "docs/STATION.md carries the procedure, and it needs no transmitter.")


@pytest.mark.parametrize("path,finding", CARRIERS.items(), ids=lambda v: str(v)[:40])
def test_the_files_that_know_it_is_unmeasured_go_on_saying_so(path, finding):
    """The other direction. Quietly dropping the finding satisfies every check
    above and leaves the reader with nothing, which is how this started."""
    assert finding in (REPO / path).read_text(), (
        f"{path} no longer records that deassert-on-process-death is unmeasured "
        "on this hardware -- and it still is")
