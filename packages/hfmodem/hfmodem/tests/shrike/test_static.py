# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gate: pyflakes-class checks, entry points, and the publication ratchet.

Three classes of defect kept reaching the tree because nothing looked for them:

  * undefined names (F821) -- a NameError that only fires on the branch nobody
    exercised. One shipped in rxfront's P1-data path and only surfaced when a
    real capture happened to reach it.
  * entry points that cannot run at all. The module-level tests all import
    cleanly, so a broken `python -m hfmodem.shrike.<tool>` -- a bad import, an argparse
    mistake -- goes unnoticed until someone reaches for the tool on the air.
  * working-record material in shipping prose. Shipped text documents the
    protocol; citation tags, binary addresses and harness vocabulary describe how
    someone came to know it, which is a different document with a different
    audience.

The third one is a RATCHET rather than a pass/fail. Each category has a baseline;
exceeding it fails, and a count below its baseline is reported so the baseline can
be tightened in the same edit. The numbers live in the source rather than a data
file, so a change to one shows up in the diff. Every category but one reads zero,
so the mechanism is a floor: a term that comes back fails on the run that brings
it back.

It reads the shipped SUITE as well as the package. The suites cross the
publication boundary -- `publish/manifest.toml` includes them, on the grounds
that a modem whose conformance nobody can re-run is a claim rather than a result
-- and a reader who opens a test meets the same prose under the same
expectations as one who opens a module. Scoping this to `shrike/` alone measured
half of what ships and reported it as the whole.

The ratchet shape is what the tree needs rather than a preference: between two
consecutive readings taken hours apart during unrelated DSP work, citation tags
went 56 -> 58 and `print(` 80 -> 82. Prose accretes while attention is elsewhere,
so the count has to be checked continuously.

Run:  python -m hfmodem.tests.shrike.test_static
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PKG = Path(__file__).resolve().parents[2]
SRC = PKG / "shrike"
SUITE = PKG / "tests" / "shrike"
DOCS = ROOT / "docs"
"""Shipped prose. It crosses the publication boundary exactly as the package does,
and it was NOT scanned until 2026-08-01 -- a gate looking at two of the three
trees that ship. The omission cost nothing only because it was found before the
push; two markers had already settled in `protocols/pactor/`."""
PY = sys.executable

# Files this ratchet does not read, each for its own reason.
#
# The two `*_oracle.py` gates run an external decoder and quote its transcripts
# back. A reader cannot re-run either without being told what to install, so the
# name is those files' SUBJECT rather than residue in them. Reading them would
# force a baseline high enough to hide anything that arrived afterwards, which is
# the opposite of what a floor is for.
#
# This file states the patterns, so its text matches several of them; reading it
# would measure nothing but itself.
UNSCANNED = {"test_p1_oracle.py", "test_p2_oracle.py", "test_p3_oracle.py",
             Path(__file__).name}

# Patterns whose presence in a shipped file would be working record rather than
# protocol. Deliberately NOT bare hex: 43 `0x4...` literals in shrike/ are
# protocol constants -- CRC polynomials, control-signal codewords, tone masks --
# and a gate that cannot tell those from an address is a gate nobody can keep
# green. The qualified forms below have no innocent reading.
#
# The separator between a qualifier and its address is a character class rather
# than a literal, and the symbol form accepts a zero-padded address with no `0x`
# prefix. One citation gets typed with an `@`, with a space and with a colon
# depending on which tool printed it, and a rule that knows a single spelling
# reports zero while the rest cross. `publish/build.py` carries the same widening;
# `tests/gates/test_manifest.py` holds a live fixture for each spelling, built at
# run time so that naming them does not put them in the distribution.
#
# One letter of each term is bracketed, so the pattern text is not itself an
# instance of what it matches and this file does not answer to a scan of the
# whole distribution.
PUBLICATION_PATTERNS = {
    "citation tags": r"\[pro[v]:",
    "binary section addresses": r"\.(?:text|rodata|data|bss)[\s@.:]0x[0-9a-fA-F]{4,}",
    "disassembler symbol names": r"\bfcn[\s@.:_]+(?:0x)?[0-9a-fA-F]{4,}",
    "binary: citations": r"binary:",
    "reference decoder named": r"PM[O]N",
    "harness vocabulary": r"\bora[c]le\b",
    "disassembly vocabulary": r"disassembl|decompi[l]",
    "method vocabulary": r"clean-?room",
    "tooling names": r"\bgdb\b|\bunicorn\b|\bwine\b|\bquarantin[e]\b",
}
# `emulat*` is deliberately absent: shrike emulates an SCS PTC-IIIusb, which is
# the product it presents itself as, and the same word describes an instruction
# emulator. One word, two meanings, no gate.
PUBLICATION_BASELINE = {
    "citation tags": 0,
    "binary section addresses": 0,
    "disassembler symbol names": 0,
    "binary: citations": 0,
    "reference decoder named": 0,
    # The one non-zero, and it is a PATH rather than prose: `test_rx.py` names the
    # directory holding the soft-raster vectors it loads, under `working/`, which
    # never crosses. A reader with the harness needs the real path and a reader
    # without it sees the module skip. It reaches zero when that directory is
    # renamed, which is a move across hundreds of files and not a comment fix.
    "harness vocabulary": 1,
    "disassembly vocabulary": 0,
    "method vocabulary": 0,
    "tooling names": 0,
}
# Every operator-facing CLI. --help exercises import + argparse without
# touching a radio, an external decoder or the network. (p2rx/analysis scripts
# take file paths rather than flags, so they are not argparse CLIs, not listed.)
ENTRY_POINTS = ["monitor", "live", "onair", "ota", "beacon", "ptc"]

_SKIPPED = "SKIPPED "

# A suite whose recording does not cross the publication boundary says so with
# `pytest.skip(..., allow_module_level=True)`, which raises out of a bare
# `import` and is indistinguishable from a broken module. The distribution is
# where that happens, so a probe that cannot tell them apart reports the
# publication boundary working as a static-gate failure.
_IMPORT_PROBE = """\
import importlib, sys
import pytest
try:
    importlib.import_module(sys.argv[1])
except pytest.skip.Exception as skipped:
    print("SKIPPED", skipped)
"""

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok &= passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))


def main() -> int:
    # In the monorepo an absent ruff means the gate is not running anywhere, which
    # is the state this file exists to prevent, so it is a failure and not a skip.
    # The distribution is the other tree: ruff is not in its documented install and
    # `tools/` -- which only a checkout has -- is how the two are told apart.
    ruff = ROOT / ".venv" / "bin" / "ruff"
    if ruff.exists():
        r = subprocess.run([str(ruff), "check", "--select", "F", str(SRC), str(SUITE)],
                           cwd=ROOT, capture_output=True, text=True)
        check("no pyflakes findings (F: undefined names, dead imports/vars)",
              r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else "")
    elif (ROOT / "tools").is_dir():
        check("ruff is installed", False, f"{ruff} absent -- the lint gate cannot run")
    else:
        print(f"  [SKIP] pyflakes findings: no ruff at {ruff} -- "
              f"`{ROOT}/.venv/bin/pip install ruff` turns this check on")

    for mod in ENTRY_POINTS:
        r = subprocess.run([PY, "-m", f"hfmodem.shrike.{mod}", "--help"],
                           cwd=ROOT, capture_output=True, text=True, timeout=90)
        check(f"python -m hfmodem.shrike.{mod} --help runs", r.returncode == 0,
              (r.stderr.strip().splitlines() or [""])[-1][:90])

    for t in sorted(SUITE.glob("test_*.py")):
        r = subprocess.run([PY, "-c", _IMPORT_PROBE, f"hfmodem.tests.shrike.{t.stem}"],
                           cwd=ROOT, capture_output=True, text=True, timeout=90)
        if r.stdout.startswith(_SKIPPED):
            print(f"  [SKIP] hfmodem.tests.shrike.{t.stem} imports: "
                  + r.stdout[len(_SKIPPED):].strip())
            continue
        check(f"hfmodem.tests.shrike.{t.stem} imports", r.returncode == 0,
              (r.stderr.strip().splitlines() or [""])[-1][:90])

    publication_ratchet()
    event_kinds_declared()
    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


def event_kinds_declared() -> None:
    """`rxfront.EVENT_KINDS` must list every kind rxfront actually emits.

    A consumer that maps kinds to its own vocabulary breaks on an undeclared one --
    that has already happened once downstream, discarding a whole capture's later
    detections. The declaration is only useful if it cannot drift from the code.
    """
    from hfmodem.shrike import rxfront
    src = (SRC / "rxfront.py").read_text()
    emitted = set(re.findall(r'Event\([^,]+,\s*"([a-z0-9]+)"', src))
    emitted |= set(re.findall(r'kind="([a-z0-9]+)"', src))
    undeclared = sorted(emitted - set(rxfront.EVENT_KINDS))
    check("every emitted Event kind is in rxfront.EVENT_KINDS", not undeclared,
          f"undeclared: {undeclared}" if undeclared else f"{len(emitted)} kinds")

    # ...and every CONSUMER must handle all of them. Declaring the vocabulary is
    # only half the contract: shrike.monitor indexed its display map with [] and
    # died with a KeyError on the first p1reply, discarding every later detection
    # in that recording, on 10 of 23 corpus fixtures -- while the check above was
    # green. That is the same defect this project had already reported downstream
    # in another repo, so the gate has to cover consumption, not just declaration.
    from hfmodem.shrike import monitor
    unhandled = sorted(set(rxfront.EVENT_KINDS) - set(monitor._TAG))
    check("shrike.monitor renders every declared Event kind", not unhandled,
          f"missing from _TAG: {unhandled}" if unhandled else f"{len(monitor._TAG)} kinds")


def publication_ratchet() -> None:
    """Working-record material must not grow, in the package or in its suite.

    `analysis/`, the reference-decoder sandbox and the provenance sidecar are
    SUPPOSED to carry this material, and none of them crosses the publication
    boundary. The three trees read here DO cross, so all three are held to the
    same bar; the files that have to name an external decoder to be runnable at
    all are named in UNSCANNED, as exemptions rather than findings that they are
    clean.

    `docs/` was outside this scan until 2026-08-01 and had accumulated two
    markers. A gate that reads two of the three trees that ship is the same
    failure as a gate that reads none of them, just slower to notice.
    """
    text = {p: p.read_text(errors="replace")
            for p in sorted((*SRC.glob("*.py"), *SUITE.glob("*.py"),
                             *DOCS.rglob("*.md")))
            if p.name not in UNSCANNED}
    slack = []
    for name, pattern in PUBLICATION_PATTERNS.items():
        rx = re.compile(pattern, re.IGNORECASE if "vocab" in name else 0)
        hits = sum(len(rx.findall(t)) for t in text.values())
        base = PUBLICATION_BASELINE.get(name, 0)
        if hits > base:
            # EVERY file holding a hit, not the largest one. The counts are
            # usually 1 apiece, so `max` picked whichever the dict order put
            # first -- and named the baseline's own file for a marker somebody
            # else had just added, which is the reading that costs a search.
            found = sorted(((len(rx.findall(t)), p.name) for p, t in text.items()
                            if rx.search(t)), key=lambda kv: (-kv[0], kv[1]))
            check(f"publication: {name} <= {base}", False,
                  f"{hits} now (+{hits - base}) in "
                  + ", ".join(f"{n} x{k}" if k > 1 else n for k, n in found))
        else:
            check(f"publication: {name} <= {base}", True,
                  f"{hits}" + (f" -- baseline can drop to {hits}" if hits < base else ""))
            if hits < base:
                slack.append((name, hits))
    if slack:
        print("     ratchet has slack; tighten PUBLICATION_BASELINE in the same "
              "commit that removed the markers:")
        for name, hits in slack:
            print(f"       {name!r}: {hits},")


def test_main() -> None:
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
