# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The drained-responder handover is a state, not a bandwidth.

`_drained_hands_over` was BW500-only. K0SI on 40 m keyed the 32-symbol
drained-responder at BW2300 twice on 2026-09-16 — 67.3 and 76.7 s, 31/31 tones,
0.12 s after our final ACK — and the run named "no release in cadence" and
re-keyed. The same state test and freshness bound apply at every session
bandwidth now, and so does the stream route: at 2300/2750 the energy gate that
feeds `on_rx_audio` is what missed both handovers, so `_stream_answer` carries
its own arm for them.

That recording is lost. Verified K0SI 09-20 01:35Z excerpts now pin the same
route and state at source 64.615 and 84.769 s (31/32 and 31/31 tones).
The replay stops 4 s after the handover. Its absence checks therefore cannot
cover the original failure's ladder rung at +8.6 s or turn request at +20.6 s.
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import turns_evidence as E


class IO:
    def __init__(self):
        self.log_lines, self.sent = [], []
        self.now = 0.0                     # tape clock, set by the live driver
        self.at: list[float] = []

    def key(self, on): ...

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))
        self.at.append(self.now)

    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, p): ...
    def log(self, m): self.log_lines.append(m)


def _handover(bw="2300"):
    io = IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", "K0SI", "W9SSJ"
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    hs._answer_owed = VA._OWED_RELEASE
    hs._txq = [VF.SESSION_DRAINED.name.encode()]  # a queued block, contents irrelevant
    hs.tx_level = 3
    return hs, io


def test_the_handover_no_longer_reads_as_bw500_only():
    # A synthesised drained-responder taken at BW2300 with the handover state set.
    kind = VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "2300")
    burst = np.concatenate([MK.synth_burst("K0SI", kind), np.zeros(int(0.3 * MK.FS))])
    hs, _ = _handover()
    assert hs._drained_hands_over(burst)
    assert 0 <= hs._grant_held <= VA._GRANT_FRESH_S


def test_empty_queue_or_no_owed_release_declines_it():
    kind = VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "2300")
    burst = np.concatenate([MK.synth_burst("K0SI", kind), np.zeros(int(0.3 * MK.FS))])
    hs, _ = _handover()
    hs._txq, hs._tx_pending = [], None
    assert not hs._drained_hands_over(burst)
    hs, _ = _handover()
    hs._answer_owed = None
    assert not hs._drained_hands_over(burst)


def test_a_foreign_drained_frame_is_not_the_handover():
    kind = VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "2300")
    burst = np.concatenate([MK.synth_burst("N0XYZ", kind), np.zeros(int(0.3 * MK.FS))])
    hs, _ = _handover()
    assert not hs._drained_hands_over(burst)


@pytest.mark.parametrize("name,start", [
    ("k0si-0920-drained-first", 0.615),
    ("k0si-0920-drained-second", 0.569),
])
def test_k0si_40m_drained_hands_over_and_takes_the_turn(name, start):
    x = E.samples(name, tracked=True)
    kind = VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "2300")
    span = VA._span(kind)
    a = int((start - 0.334) * MK.FS)
    seg = x[a:a + span + int(0.6 * MK.FS)]
    hs, io = _handover()
    assert hs._drained_hands_over(seg), "the BW2300 handover was declined"
    hs._took_drained_handover()
    assert hs.turn == VA._TURN_OURS
    assert len(io.sent) == 1                       # our queued over went out
    assert any("drained-responder" in m and "taking the turn" in m
               for m in io.log_lines)


#: Verified 09-20 substitute: source 64.521 and 64.615 s, minus clip start 64 s.
_ACK_UNKEY = 0.521
_HANDOVER = 0.615


def _drive(hs, io, x, t0, t1, block=0.125):
    """The tape through `on_rx_stream` the way the receive path feeds it."""
    n = int(block * MK.FS)
    for a in range(int(t0 * MK.FS), int(t1 * MK.FS), n):
        io.now = (a + n) / MK.FS
        hs.on_rx_stream(x[a:a + n])


def test_the_stream_route_reads_the_handover_the_energy_gate_missed():
    """The route, not just the recogniser. `_drained_hands_over` returns True on
    this audio and reached nothing live: the bracket route is the only thing that
    ran it, and the gate a gateway's short burst cannot open never bracketed one.

    The historical failure was a ladder rung and a turn-request instead of DATA
    [working/onair-0915-2131]. The 09-20 substitute preserves the incoming
    handover. The replay stops 4 s after the handover, so the later successful
    exchange and the original +8.6/+20.6 s absence claims are outside its window.
    """
    x = E.samples("k0si-0920-drained-first", tracked=True)
    last_symbol = _HANDOVER + VA._span(
        VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "2300")) / MK.FS
    hs, io = _handover()
    # The substitute actually sent DATA and got an answer. Stop before that
    # later exchange: unlike the lost failed arm, it legitimately keys again.
    _drive(hs, io, x, _ACK_UNKEY, _HANDOVER + 4.0)

    assert hs.turn == VA._TURN_OURS, "the handover reached nothing on the stream"
    assert any("drained-responder" in m and "taking the turn" in m
               for m in io.log_lines), io.log_lines
    assert 0 <= hs._grant_held <= VA._GRANT_FRESH_S
    assert not hs._txq and hs._answer_owed is None

    assert len(io.sent) == 1, f"{len(io.sent)} bursts for one handover"
    assert io.at[0] >= last_symbol, (
        f"keyed {last_symbol - io.at[0]:.3f} s inside the handover")
    assert any("tx DATA over" in m for m in io.log_lines), io.log_lines
    assert not any("re-keying the 11-symbol control burst" in m
                   for m in io.log_lines), io.log_lines
    assert not any("asking for the turn" in m for m in io.log_lines), io.log_lines
