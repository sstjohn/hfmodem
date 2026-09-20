# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 2026-08-03 KE8LVA connect failure, kept from coming back.

Eleven ConReq2000M went out; KE8LVA answered six with ConAck2000 carrying the
expected session id 0x51, headers reading crisp 8.3–9.4 at their true +6 Hz
CFO — and every reply died at the unsolicited bare-control floor (9.5), so the
connect failed with the answers on tape
(``logs/onair/20260803T151721Z-besra-7102000.wav``). The fix is the
expected-session lane: a session in progress tells the demodulator the id it is
party to, and a bare control matching that id is admitted at the calibrated 8.0
floor instead.

The fixture is 1.8 s of that recording around the cycle-9 reply, through the
live 48→12 kHz resampler: the tail of our own leaked ConReq, the ~0.11 s
post-TX dead gap, then the ConAck with ~70 ms of its 240 ms leader clipped —
the exact acquisition geometry of a live ARQ turnaround.

Exposure of the lane, measured the same way the floors were calibrated: zero
bare-control candidates at crisp ≥ 8.0 in 3.6M scanned positions of off-air
noise (60.5 s of the same recording); busy non-ARDOP channels reach the floor
only when the misread session also matches the one expected (5 of 256 possible
ids on the worst rf-corpus capture, one phantom DATANAK each — a spurious
retransmit at worst), and never with no session in progress.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from hfmodem.besra import crc
from hfmodem.besra.phy.demodulator import Demodulator

_FIXTURE = Path(__file__).parent / "fixtures" / "offair_ke8lva_conack2000.wav"
_SESSION = crc.session_id("W9SSJ", "KE8LVA")        # 0x51 on the air that day


def _audio() -> np.ndarray:
    with wave.open(str(_FIXTURE)) as w:
        assert w.getframerate() == 12000
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def test_the_recorded_session_id_is_ours():
    """The fixture's provenance in one line: the id the lane keys on is exactly
    what both ends derive from the callsigns of that connect."""
    assert _SESSION == 0x51


def test_expected_session_recovers_the_offair_conack():
    frames = Demodulator(expect_session=lambda: _SESSION).decode(_audio())
    hit = [f for f in frames if f.ok and f.name == "ConAck2000"]
    assert len(hit) == 1, f"expected the ConAck, got {[(f.name, f.ok) for f in frames]}"
    assert hit[0].session_id == _SESSION
    assert hit[0].conack_timing_ms == 210               # ~the 240 ms leader KE8LVA heard


def test_unsolicited_scan_still_rejects_it():
    """Without a session in progress the strict floor stands: the same audio
    yields nothing, exactly as the monitor and a listening station should."""
    assert Demodulator().decode(_audio()) == []


def test_wrong_expectation_rejects_it():
    """The lane is the id match, not a relaxation: expecting a *different*
    session leaves the reply below the unsolicited floor."""
    assert Demodulator(expect_session=lambda: 0xAE).decode(_audio()) == []
