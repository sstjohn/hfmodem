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
"""
import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora


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


@corpora.requires_onair_drained_handover_2300
@pytest.mark.parametrize("start", [67.334, 76.734])
def test_k0si_40m_drained_hands_over_and_takes_the_turn(start):
    x = corpora.wav_mono(corpora.ONAIR_DRAINED_HANDOVER_2300)
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


#: Where the tape's first handover sits. Our final acknowledgement's last sample
#: is at 67.24 s and K0SI keys the 32-symbol handover 0.09 s behind it, so that
#: unkey is both the instant the release falls due and the first sample the
#: receive path delivers  [working/onair-0915-2131].
_ACK_UNKEY = 67.24
_HANDOVER = 67.334


def _drive(hs, io, x, t0, t1, block=0.125):
    """The tape through `on_rx_stream` the way the receive path feeds it."""
    n = int(block * MK.FS)
    for a in range(int(t0 * MK.FS), int(t1 * MK.FS), n):
        io.now = (a + n) / MK.FS
        hs.on_rx_stream(x[a:a + n])


@corpora.requires_onair_drained_handover_2300
def test_the_stream_route_reads_the_handover_the_energy_gate_missed():
    """The route, not just the recogniser. `_drained_hands_over` returns True on
    this audio and reached nothing live: the bracket route is the only thing that
    ran it, and the gate a gateway's short burst cannot open never bracketed one.

    What the live run did instead is on the tape — a ladder rung at 75.95 s and a
    turn-request at 87.98 s, with the handover keyed at 67.334 and again at
    76.734  [working/onair-0915-2131].
    """
    x = corpora.wav_mono(corpora.ONAIR_DRAINED_HANDOVER_2300)
    last_symbol = _HANDOVER + VA._span(
        VF.for_bw(VF.SESSION_DRAINED_RESPONDER, "2300")) / MK.FS
    hs, io = _handover()
    _drive(hs, io, x, _ACK_UNKEY, 80.0)

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
