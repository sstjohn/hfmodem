# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The bridge hears a peer still holding its FSK answer, faded or clipped.

09-28 80 m, inverted callsign marker: WS8EOC and KB5LZK repeated robust answer
entry 0 in every bridge slot. A Kiwi witness and a fixed-phase read take all of
them at 0 errors, and the operator heard each one. The bridge's envelope-placed
read logged most of them as silence: one tone faded out, and at WS8EOC our own
receive chain clipped the answer into its harmonics. Fixture cut by
tools/fixtures/cut_robust_held_answer_0928.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p1rx, session
from hfmodem.shrike.robust import Bridge

FS = 48000
HERE = Path(__file__).parent / "fixtures" / "robust-held-0928"
META = json.loads((HERE / "metadata.json").read_text()) if (HERE / "metadata.json").is_file() else {}
requires_recording = pytest.mark.skipif(
    not META, reason="fixtures/robust-held-0928/metadata.json is not included in this distribution")


def _slots(name):
    arm = META[name]
    audio = session.load_wav(str(HERE / f"{name}.wav"), FS).astype(float)
    n = round(arm["window_s"] * FS)
    lead = round(arm["window_lead_s"] * FS)
    for i, row in enumerate(arm["windows"]):
        window = audio[i * n:(i + 1) * n]
        # The window sits `lead` past its slot's boundary on the arm's clock.
        boundary = row["slot"] * 60000
        yield row, window, boundary + lead, boundary


def _bridge(name):
    arm = META[name]
    bridge = Bridge("W9SSJ", arm["answer_phase"], offset_hz=arm["offset_hz"])
    bridge.fsk_sense = arm["answer_sense"]
    bridge.fsk_slot = arm["answer_slot"]
    return bridge


@requires_recording
@pytest.mark.parametrize("name", sorted(META) or ["missing-recordings"])
def test_every_held_answer_is_read_as_fsk_entry_0(name):
    bridge = _bridge(name)
    reports = [bridge.read(window, origin, boundary, row["slot"])[1]
               for row, window, origin, boundary in _slots(name)
               if row["kind"] == "bridge"]
    assert sum(r.startswith("FSK entry 0") for r in reports) == 10, reports
    assert [h["form"] for h in bridge.heard] == ["FSK"] * 10
    # Heard, so never silent, and nothing restarts the P2 acquisition.
    assert bridge.silent == 0


@requires_recording
@pytest.mark.parametrize("name", sorted(META) or ["missing-recordings"])
def test_the_pre_answer_slots_read_nothing(name):
    arm = META[name]
    for row, window, origin, boundary in _slots(name):
        if row["kind"] != "pre-answer":
            continue
        sense = arm["answer_sense"] ^ ((row["slot"] - arm["answer_slot"]) & 1)
        at = boundary + arm["answer_phase"] - origin
        assert p1rx.read_held_answer(window, at, FS, sense=sense) is None, row
        assert p1rx.read_held_answer(window, at, FS) is None, row


@requires_recording
def test_a_held_answer_is_read_only_in_its_slot_polarity():
    arm = META["ws8eoc-80"]
    row, window, origin, boundary = next(_slots("ws8eoc-80"))
    sense = arm["answer_sense"] ^ ((row["slot"] - arm["answer_slot"]) & 1)
    at = boundary + arm["answer_phase"] - origin
    assert p1rx.read_held_answer(window, at, FS, sense=sense).errors == 0
    assert p1rx.read_held_answer(window, at, FS, sense=1 - sense) is None


def test_noise_is_not_a_held_answer():
    rng = np.random.default_rng(7)
    hits = sum(p1rx.read_held_answer(rng.standard_normal(round(.4 * FS)) * .05,
                                     round(.1 * FS), FS) is not None
               for _ in range(200))
    assert hits == 0
