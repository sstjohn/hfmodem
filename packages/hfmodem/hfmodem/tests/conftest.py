# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Marker gating, so an absent dependency is reported rather than assumed.

The rule this file exists to enforce: **a test that cannot run must say so.** A
suite that quietly omits a test looks identical to a suite where it passed, and
this project has already been bitten by that — a conformance selftest returned 0
with an entire modem's stages skipped, and a lint gate passed for weeks by
shelling out to a tool that was not installed.

So nothing here silently deselects. Every gate produces a visible skip carrying
the environment variable that would turn it on.

  corpus     HFMODEM_CORPUS   the off-air recordings
  ardopcf    ARDOPCF          a built ardopcf to cross-decode against
  hardware   HFMODEM_AUDIO    a sound card, or HFMODEM_RIG for a transmitter
             or the one variable a test names, hardware(opt_in=...)
  wheel      -                always runs; builds a real wheel, a few seconds

`hardware` covers tests that need the machine to themselves as much as ones that
need a device. `tests/shrike/test_duplex.py` measures PTT timing against a live
audio loopback with a 5 ms jitter budget; under a full-suite run it fails on
scheduling noise, having nothing to do with the code under test.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from hfmodem.tests.hermetic import captures_untouched, station_kept_out  # noqa: F401  (autouse fixtures)

_GATES = {
    "corpus": ("HFMODEM_CORPUS", "the off-air corpus"),
    "ardopcf": ("ARDOPCF", "a built ardopcf binary"),
}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if (item.get_closest_marker("wall_clock") is None
            or os.environ.get("HFMODEM_QUALIFICATION_PRIORITY_ADVISED") == "1"):
        return
    try:
        nice = os.getpriority(os.PRIO_PROCESS, 0)
    except (AttributeError, OSError):
        nice = None
    if nice is None or nice > 0:
        reporter = item.config.pluginmanager.getplugin("terminalreporter")
        message = (f"WARNING: wall_clock {item.nodeid}: actual nice={nice}; "
                   "relaunch the split coordinator from a normal-priority shell (nice 0). "
                   "Continuing without elevation.")
        capture = item.config.pluginmanager.getplugin("capturemanager")
        from contextlib import nullcontext
        with capture.global_and_fixture_disabled() if capture else nullcontext():
            if reporter is not None:
                reporter.write_line(message, yellow=True)
            else:
                import sys
                print(message, file=sys.stderr)


#: Pins that extract an earlier commit's adapter with `git archive`, which only a
#: checkout holding that history can do.
_ARCHIVED = {"release043": "e597c9aca"}


def _in_history(commit: str) -> bool:
    from hfmodem.tests import hermetic
    from hfmodem.tests.evidence import TREE

    argv = ["git", "cat-file", "-e", f"{commit}^{{commit}}"]
    if hermetic.guard is not None:
        hermetic.guard.admit(argv, os.environ)
    return subprocess.run(argv, cwd=TREE, capture_output=True).returncode == 0


#: Pins in files that stay as written whose expectation the receiver design
#: overrules; each fails, strictly, for the stated reason.
_OVERRULED = {
    "kestrel/test_vara_levels_arq.py::test_an_over_our_nak_draws_again_repacked_reaches_the_host_once"
    "[resend2- has 1400 daily minutes remaini daily minutes]":
        "DESIGN §5/§6: the resend b' daily mi' and its short terminal b'nutes' both "
        "match inside the replayed bytes and are dropped, and the final answer ends "
        "the replay check, so the host gets b' has 1400 daily minutes remaini' "
        "without the expected second b' daily minutes'",
}


def pytest_collection_modifyitems(config, items):
    audio = os.environ.get("HFMODEM_AUDIO")
    rig = os.environ.get("HFMODEM_RIG")
    wanted = {f for item in items for f in getattr(item, "fixturenames", ()) if f in _ARCHIVED}
    absent = {f for f in wanted if not _in_history(_ARCHIVED[f])}
    for item in items:
        for node, reason in _OVERRULED.items():
            if item.nodeid.endswith("/" + node):
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
        for name in absent.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.skip(
                reason=f"needs git history holding {_ARCHIVED[name]} (source checkout only)"))
        for name, (var, what) in _GATES.items():
            if name in item.keywords and not os.environ.get(var):
                item.add_marker(pytest.mark.skip(
                    reason=f"set {var} to point at {what}"))
        hardware = item.get_closest_marker("hardware")
        if hardware is None:
            continue
        # A test that names its own variable runs on that alone: the real keying
        # line is not the sound card, and arming one must not arm the other.
        own = hardware.kwargs.get("opt_in")
        if own and not os.environ.get(own):
            item.add_marker(pytest.mark.skip(reason=f"set {own} to run this against the station"))
        elif not own and not (audio or rig):
            item.add_marker(pytest.mark.skip(
                reason="set HFMODEM_AUDIO=1 (sound card, and the machine to "
                       "itself) or HFMODEM_RIG=1 (transmitter)"))


@pytest.fixture
def tools(monkeypatch):
    """`tools/` importable, for a test of a station instrument only a checkout has."""
    from hfmodem.tests.evidence import TREE

    path = TREE / "tools"
    if not path.is_dir():
        pytest.skip("tools/ not present (installed-wheel run)")
    monkeypatch.syspath_prepend(str(path))
    return path
