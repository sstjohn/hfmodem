# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The frame a station keys when its send queue has drained, against real tones.

Ground truth: two bidirectional real-VARA <-> real-VARA BW2300 sessions with
DIFFERENT callsign pairs (W9SSJ/W1AW 2026-08-14, K5ABC/N0DX 2026-08-15), 126
bytes each way, each station's transmissions recorded on its own cable so which
one keyed a burst is a property of the tape. The tones below are what the modems
emitted, read with the parity-alternating run behind the ``(74,)`` preamble.

Two pairs is the test, not a nicety. Each burst pins one 24-bit generator state
out of 2**24, but the seeding map is many-to-one and either session ALONE leaves
eleven ``(SEED_OFF, PREADV)`` pairs standing for the same tones; only the
intersection is a singleton.
"""
from __future__ import annotations

import pytest

from hfmodem.kestrel.vara import vara_frames as VF

# (called callsign, initiator's frame, responder's frame)
_SESSIONS = [
    ("W1AW", [68, 65, 40, 37, 80, 89, 64, 57, 50, 63, 66, 35, 34, 73, 70, 85,
              68, 91, 32, 57, 46, 35, 60, 53, 46, 83, 46, 89, 64, 75, 36],
     [36, 97, 32, 71, 44, 89, 82, 35, 78, 47, 96, 57, 46, 59, 66, 37, 50, 81,
      80, 39, 98, 79, 42, 65, 32, 69, 90, 35, 54, 65, 54]),
    ("N0DX", [86, 65, 60, 85, 74, 45, 40, 39, 78, 97, 48, 61, 38, 35, 74, 35,
              98, 39, 68, 47, 72, 85, 42, 61, 88, 49, 56, 61, 48, 33, 68],
     [70, 29, 36, 79, 32, 67, 48, 63, 80, 79, 30, 59, 84, 61, 42, 77, 32, 39,
      68, 63, 64, 59, 86, 49, 68, 67, 58, 37, 58, 39, 58]),
]


@pytest.mark.parametrize("called,initiator,responder", _SESSIONS)
def test_the_generator_reproduces_both_drained_frames(called, initiator,
                                                      responder):
    assert VF.payload_bins(called, VF.SESSION_DRAINED) == initiator
    assert VF.payload_bins(called, VF.SESSION_DRAINED_RESPONDER) == responder


def test_the_two_drained_frames_are_distinct_and_called_keyed():
    """One frame keyed to whoever sent it would make the two identical; they are
    not, and the recordings admit no sender- or peer-keyed reading at all."""
    for kind in (VF.SESSION_DRAINED, VF.SESSION_DRAINED_RESPONDER):
        assert kind.keyed_by == "called"
    a = VF.payload_bins("W1AW", VF.SESSION_DRAINED)
    b = VF.payload_bins("W1AW", VF.SESSION_DRAINED_RESPONDER)
    assert sum(1 for u, v in zip(a, b) if u == v) <= 6


def test_a_drained_frame_is_not_any_other_session_frame():
    """It shares SEED_OFF with the over-response and the idle-response but not
    their pre-advance, so nothing in the family can be mistaken for it."""
    for called in ("W1AW", "N0DX", "W9SSJ"):
        mine = {k: VF.payload_bins(called, k)
                for k in (VF.SESSION_DRAINED, VF.SESSION_DRAINED_RESPONDER)}
        for kind in VF.BURSTS.values():
            if kind in mine or kind.n_payload != 31:
                continue
            other = VF.payload_bins(called, kind)
            for tones in mine.values():
                assert not VF.recognize(tones, called, kind)
                assert sum(1 for u, v in zip(tones, other) if u == v) <= 6
