# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Static gate: pyflakes checks, module imports and entry points.

Run: python -m hfmodem.tests.shrike.test_static
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from hfmodem.tests.evidence import require_input

ROOT = Path(__file__).resolve().parents[5]
PKG = Path(__file__).resolve().parents[2]
SRC = PKG / "shrike"
SUITE = PKG / "tests" / "shrike"
PY = sys.executable

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


def _ruff_check(ruff: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(ruff), "check", "--select", "F", str(SRC), str(SUITE)],
                          cwd=ROOT, capture_output=True, text=True)


def main(*, include_lint: bool = True) -> int:
    # In the monorepo an absent ruff means the gate is not running anywhere, which
    # is the state this file exists to prevent, so it is a failure and not a skip.
    # The distribution is the other tree: ruff is not in its documented install and
    # `tools/` -- which only a checkout has -- is how the two are told apart.
    if include_lint:
        ruff = ROOT / ".venv" / "bin" / "ruff"
        if ruff.exists():
            r = _ruff_check(ruff)
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


def test_ruff() -> None:
    ruff = require_input(ROOT / ".venv/bin/ruff")
    r = _ruff_check(ruff)
    assert r.returncode == 0, r.stdout + r.stderr


def test_main() -> None:
    assert main(include_lint=False) == 0


if __name__ == "__main__":
    sys.exit(main())
