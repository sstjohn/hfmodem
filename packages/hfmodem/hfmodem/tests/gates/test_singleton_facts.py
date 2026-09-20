# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Facts about this station that must have exactly one definition, ratcheted.

Each has already cost real time by being written down more than once:

  * **The codec input gain.** The measured working point for this station's
    FT-891 into its C-Media dongle is 0.040. It was wrong in four places at once
    — 0.51, 0.118, 0.50 and 0.18 — and one of those re-set the wrong value during
    a session handover. A hot input does not present as an error; it presents as
    a working radio that decodes nothing. Across 617 captures taken under the old
    settings, 39% peaked at 0 dBFS and 12% clipped more than 1% of their samples.

  * **The transmit drive.** The peak every keyed burst leaves at. Four modems
    share one interface and one radio and wrote it four ways — shrike 0.6,
    kestrel 0.8, sabir 0.1, and besra not at all, which put up to 1.9% of a
    4FSK.2000.600 frame past the rail. 18 dB between them, on a knob the operator
    sets once against a meter; and `[audio] tx_drive`, which is where an operator
    would go to change it, reached none of them.

  * **The dial offset.** A published channel names its centre; the dial is that
    minus 1500 Hz. Getting it wrong is silent and kills both directions, and
    every modem here has had it wrong at some point.

  * **The QSY tolerance.** How far the dial may sit from what was asked for and
    still be that channel. Four transmit paths each wrote it at the readback
    comparison itself and one of them said 10 against the other three's measured
    20 — silently stricter on the same radio, with nothing to say which was
    meant. Nobody had to be wrong for that to be a defect; two numbers for one
    fact is the defect.

**The gate keys on the named setting, never on the literal**, because both
numbers are overloaded in this tree and a literal match would fire on unrelated
correct code:

  * `0.04` is also the FT-891's PTT settle time (`RIGS["ft891"]["settle"]`, and
    the cycle-budget arithmetic in shrike's tests), and appears incidentally as a
    duration elsewhere.
  * `1500` is also the audio passband centre — a modulation constant — in
    `shrike/spec.py`, `shrike/p2rx.py`, `sabir/phy/preamble.py`,
    `sabir/floor/beacon.py` and `besra/dsp/templates.py`. That it equals the dial
    offset is a property of the USB-dial convention, not a shared constant. Do
    not collapse them.

This is a ratchet rather than a pass/fail assertion, following the form that let
the publication gate land on a tree that did not yet satisfy it: the counts below
are what is true today, they may only go down, and the failure message says which
direction moved. `core/levels.py` and `core/band.py` are where each ends up.

**The prose ships too, and it drifts on its own.** `docs/protocols/sabir/ONAIR.md`
crosses the publication boundary and told an operator `--gain` "defaults to 0.1,
deliberately quiet" for eleven days after the drive was unified at 0.6 — a
procedure that would have put the beacon 15 dB under every other emission from
this station, into a one-way test whose whole verdict is whether a WebSDR heard
it. Collapsing the four levels fixed the code and left the sentence, so the last
test here reads the shipped documents and holds a stated drive default to the
one number, keyed on the flag names rather than on the literal for the same
reason the scans above are: `drive 0.10` in the ATU section is a tune carrier,
set below the data drive deliberately.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from hfmodem.core import levels

REPO = Path(__file__).resolve().parents[5]
SEARCH = [REPO / "packages" / "hfmodem" / "hfmodem", REPO / "tools"]

#: name = value, where the name says which fact it is
# A module-level assignment of the working point. Deliberately NOT a match on the
# bare literal: 0.04 is also the FT-891's PTT settle time and appears as an
# incidental duration, and a gate that fires on those is a gate people disable.
# Keyed on the name, and any case. The miss that mattered was a lowercase
# dataclass field default — `input_gain: float = 0.040` — inside this gate's own
# scan path, which an ALL-CAPS pattern could not see. Matching the bare literal
# instead is not the fix: 0.04 is also the FT-891's PTT settle and a filter
# constant in p1rx, and a gate that fires on those is a gate people switch off.
# An assignment whose right-hand side is a reference is an alias and costs nothing.
_GAIN = re.compile(
    r"^\s*(?!#)\w*(?:gain|GAIN|working_point|WORKING_POINT)\w*"
    r"(?:\s*:[^=]+)?\s*=\s*0?\.0\d+\b", re.M)
_GAIN_RETURN = re.compile(r"return\s+0\.040\b")
# The same shape for the transmit side, and the same rule: a name bound to a
# number defines the drive, a name bound to `levels.TX_DRIVE` aliases it. The
# second form the four modems actually used was an argparse default -- kestrel's
# `--amplitude` carried 0.8 there and nothing else in the tree knew -- so a flag
# named for the drive counts as a definition too. `--tune-drive` and `tune_atu`'s
# amplitude are not this fact: a steady carrier for an ATU is set well below the
# data drive on purpose, and collapsing them would key a tune tone at the peak a
# burst leaves at.
_TX_DRIVE = re.compile(
    r"^\s*(?!#)\w*(?:TX_DRIVE|tx_drive|TX_GAIN)\w*"
    r"(?:\s*:[^=]+)?\s*=\s*0?\.\d+\b", re.M)
_DRIVE_FLAG = re.compile(
    r'add_argument\(\s*"--(?:tx-)?(?:drive|amplitude|gain)"[^)]*?default\s*=\s*0?\.\d',
    re.S)
#: Neither of these sets the peak a keyed burst leaves at, and both carry a flag
#: named like one. `station_id` writes a WAV to listen to and keys nothing.
#: `vara_rig_bridge`'s --tx-gain is a linear scalar on a continuous pass-through
#: of Wine VARA's own output, where there is no burst to normalise and no known
#: peak to normalise against — a different quantity, not a second copy of this one.
_NOT_THE_DRIVE = frozenset({"tools/station_id.py", "tools/vara_rig_bridge.py"})
# `= 1500` is a definition; `= band.DIAL_OFFSET_HZ` is an alias and costs nothing.
_DIAL_OFFSET = re.compile(
    r"^\s*(?!#)(DIAL_OFFSET_HZ|CENTRE_OFFSET_HZ|CENTER_OFFSET_HZ)\s*=\s*\d", re.M)
_CENTRE_TO_DIAL = re.compile(r"^def cent(?:er|re)_to_dial\b", re.M)
# A named assignment is one way to define this; a bare number on the readback
# comparison itself is the other, and is how three of the four sites wrote it.
# `> band.QSY_TOLERANCE_HZ` is a reference and costs nothing.
_QSY_TOLERANCE = re.compile(r"^\s*(?!#)\w*QSY_TOLERANCE\w*\s*=\s*\d", re.M)
_QSY_LITERAL = re.compile(r"abs\(.*dial.*\)\s*>\s*\d")

# What is true today. Lower these in the same commit that removes a definition.
BASELINE = {
    "codec input gain": 1,      # core/levels.py
    "transmit drive": 1,        # core/levels.py
    "dial offset": 1,           # core/band.py
    "QSY tolerance": 1,         # core/band.py
    "centre_to_dial": 1,        # shrike/ota.py — folds into core/band on the shrike pass
}


def _sources() -> list[Path]:
    return sorted(p for root in SEARCH for p in root.rglob("*.py")
                  if "__pycache__" not in p.parts and "tests" not in p.parts)


def _count(pattern: re.Pattern, extra: re.Pattern | None = None,
           skip: frozenset[str] = frozenset()) -> list[str]:
    hits = []
    for f in _sources():
        rel = str(f.relative_to(REPO))
        if rel in skip:
            continue
        text = f.read_text(encoding="utf-8")
        if pattern.search(text) or (extra and extra.search(text)):
            hits.append(rel)
    return hits


@pytest.mark.parametrize("fact,pattern,extra,skip", [
    ("codec input gain", _GAIN, _GAIN_RETURN, frozenset()),
    ("transmit drive", _TX_DRIVE, _DRIVE_FLAG, _NOT_THE_DRIVE),
    ("dial offset", _DIAL_OFFSET, None, frozenset()),
    ("QSY tolerance", _QSY_TOLERANCE, _QSY_LITERAL, frozenset()),
    ("centre_to_dial", _CENTRE_TO_DIAL, None, frozenset()),
])
def test_ratchet(fact, pattern, extra, skip):
    hits = _count(pattern, extra, skip)
    baseline = BASELINE[fact]
    assert len(hits) <= baseline, (
        f"{fact}: {len(hits)} definitions, baseline {baseline}. "
        f"A second place to write this down is how it goes wrong:\n  " + "\n  ".join(hits))
    if len(hits) < baseline:
        pytest.fail(
            f"{fact} is down to {len(hits)} (baseline {baseline}) — lower the baseline "
            f"in this file, in the commit that did it. Remaining:\n  " + "\n  ".join(hits))


# A drive default written out in prose: one of the flags that names this fact,
# then `default` and a number, with no sentence end between them. That bound is
# what keeps it off `--tx-drive 0.6` in a list of flags introduced by "left at
# its default", where each number belongs to the flag beside it, and off the ATU
# section's `drive 0.10`, a tune carrier set below the data drive on purpose.
_DOC_DRIVE = re.compile(
    r"(?:--gain|--tx-drive|--amplitude|tx_drive)[^.]{0,160}?"
    r"defaults?[^.]{0,120}?(\d*\.\d+)", re.I)


def _documents() -> list[Path]:
    return sorted([REPO / "README.md", *(REPO / "docs").rglob("*.md")])


def test_no_shipped_document_states_a_drive_of_its_own():
    # Unwrapped first: these documents are hard-wrapped at 80 columns, and the
    # sentence that shipped the wrong number had the flag on one line and its
    # default on the next.
    wrong = [(p.relative_to(REPO), m)
             for p in _documents() if p.exists()
             for m in _DOC_DRIVE.findall(
                 re.sub(r"\s+", " ", p.read_text(encoding="utf-8")))
             if float(m) != levels.TX_DRIVE]
    assert not wrong, (
        f"a shipped procedure states a transmit drive that is not "
        f"{levels.TX_DRIVE}:\n  " +
        "\n  ".join(f"{p}: {m}" for p, m in wrong) +
        "\nThe operator follows the document, not core/levels.py.")
