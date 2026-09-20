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
  wheel      -                always runs; builds a real wheel, a few seconds

`hardware` covers tests that need the machine to themselves as much as ones that
need a device. `tests/shrike/test_duplex.py` measures PTT timing against a live
audio loopback with a 5 ms jitter budget; under a full-suite run it fails on
scheduling noise, having nothing to do with the code under test.
"""
from __future__ import annotations

import os

import pytest

_GATES = {
    "corpus": ("HFMODEM_CORPUS", "the off-air corpus"),
    "ardopcf": ("ARDOPCF", "a built ardopcf binary"),
}


def pytest_collection_modifyitems(config, items):
    audio = os.environ.get("HFMODEM_AUDIO")
    rig = os.environ.get("HFMODEM_RIG")
    for item in items:
        for name, (var, what) in _GATES.items():
            if name in item.keywords and not os.environ.get(var):
                item.add_marker(pytest.mark.skip(
                    reason=f"set {var} to point at {what}"))
        if "hardware" in item.keywords and not (audio or rig):
            item.add_marker(pytest.mark.skip(
                reason="set HFMODEM_AUDIO=1 (sound card, and the machine to "
                       "itself) or HFMODEM_RIG=1 (transmitter)"))
