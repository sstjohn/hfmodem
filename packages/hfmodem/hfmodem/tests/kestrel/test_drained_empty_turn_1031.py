# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A callee's drained frame hands the turn to a caller with nothing queued.

2026-10-31, BW2300, a VARA HF station as callee:
this station read the callee's 555-byte delivery down to L3 and answered its
close (control byte 0x82). The callee's host wrote a 2000-byte reply as that
close was delivered; the callee keyed session-drained-responder (288/807, 31/31)
behind our answer, and again behind each of our four 11-symbol re-keys,
10.6 s apart; here it starts 0.4 s past our last sample. With nothing of our
own queued the frame was declined every time, the reply never went out, and
the link closed after 83 s without progress.

A VARA HF caller answers that frame with its frame-1 poll 0.24 s behind it
(a VARA HF BW500 link, at 55.586 s), and the callee answers the
poll. Above the handback level the same handover is the 11-symbol grant, which
was already held and polled; the controls below pin that and the other
single-delivery turns as they were.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from .receive_answer_helpers import key_answer
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

CALLER, CALLED, BW = "WI9YZA", "W9SSJ", "2300"
FS = MK.FS
#: Our answer's last sample to the drained frame's first, as the callee keyed it.
_TURNAROUND = 0.4


class IO:
    def __init__(self):
        self.log_lines, self.sent, self.at = [], [], []
        self.now = 0.0

    def key(self, on): ...

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))
        self.at.append(self.now)

    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, p): ...
    def log(self, m): self.log_lines.append(m)


def _after_the_close(queued: int = 0):
    """The link at 59.0 s: the callee's L3 close answered, its release owed."""
    io = IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw=BW)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    hs._txq = [b"x" * 89] * queued
    hs.tx_level = 3
    key_answer(hs, last=True, owes_release=False)
    assert hs._answer_owed == VA._OWED_RELEASE and len(io.sent) == 1
    return hs, io


def _peer(burst: np.ndarray, trail: float = 1.0) -> np.ndarray:
    return np.concatenate([np.zeros(int(_TURNAROUND * FS)), burst, np.zeros(int(trail * FS))])


def _drained(called: str = CALLED) -> np.ndarray:
    return _peer(MK.synth_burst(called, VF.for_bw(VF.SESSION_DRAINED_RESPONDER, BW)))


def _drive(hs, io, x, block: float = 0.125) -> None:
    n = int(block * FS)
    for a in range(0, len(x), n):
        io.now = (a + n) / FS
        hs.on_rx_stream(x[a:a + n])


def _poll() -> np.ndarray:
    return MK.synth_tone_pairs(VF.control_bursts(CALLER, BW, frame=1)[0])


def test_the_recorded_close_takes_the_empty_turn_and_polls():
    hs, io = _after_the_close()
    _drive(hs, io, _drained())
    assert any("rx session-drained-responder" in m for m in io.log_lines), io.log_lines
    assert hs.turn == VA._TURN_OURS and hs._answer_owed is None
    assert len(io.sent) == 2 and np.array_equal(io.sent[1], _poll()), io.log_lines
    last = _TURNAROUND + VA._span(VF.for_bw(VF.SESSION_DRAINED_RESPONDER, BW)) / FS
    assert abs(io.at[1] - last) <= VA._GRANT_FRESH_S
    # The link's next 10.6 s: the idle tick must not re-key the release ladder.
    hs.idle_keepalive()
    assert not any("no release in" in m or "never released" in m for m in io.log_lines)


def test_a_queue_still_goes_out_on_the_drained_frame():
    hs, io = _after_the_close(queued=2)
    _drive(hs, io, _drained())
    assert hs.turn == VA._TURN_OURS and len(io.sent) == 2
    assert len(io.sent[1]) > 4 * FS and len(hs._txq) == 1
    assert any(m.startswith("tx DATA over") for m in io.log_lines), io.log_lines


@pytest.mark.parametrize("queued", [0, 1])
def test_the_11_symbol_grant_above_the_handback_level_is_unchanged(queued):
    hs, io = _after_the_close(queued)
    hs.tx_level = 6
    _drive(hs, io, _peer(MK.synth_tone_pairs(VF.turn_grant(CALLED, BW))))
    assert hs.turn == VA._TURN_OURS and len(io.sent) == 2, io.log_lines
    if queued:
        assert len(io.sent[1]) > 4 * FS
    else:
        assert np.array_equal(io.sent[1], _poll())


def test_the_17_symbol_release_still_goes_straight_back():
    hs, io = _after_the_close()
    hs._took_responder_release()
    assert hs.turn == VA._TURN_PEER and hs.idle_poll_due_in() is None
    assert any("turn released to" in m for m in io.log_lines), io.log_lines


def test_a_foreign_or_unowed_drained_frame_moves_nothing():
    hs, io = _after_the_close()
    _drive(hs, io, _drained("N0XYZ"))
    assert hs.turn == VA._TURN_PEER and len(io.sent) == 1, io.log_lines

    hs, io = _after_the_close()
    hs._answer_owed = None
    _drive(hs, io, _drained())
    assert hs.turn == VA._TURN_PEER and len(io.sent) == 1, io.log_lines
