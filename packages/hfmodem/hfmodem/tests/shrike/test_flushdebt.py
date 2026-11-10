# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The end-of-cycle flush decodes the cycle, not an empty buffer.

`_SessionRx.flush` is the one decode a keyed cycle budgets for: it runs in the
0.96 s of dead time behind our own carrier, over the audio the listen window and
the bridge collected in front of the key. Two other places state that as
fact -- `live.RollingRx.hold` ("the audio is still in the stream for the flush a
moment later, which is the decode the slot was budgeted for") and `onair.py`'s
hold loop ("the flush still re-decodes the tail behind our carrier").

It was not fact. `RadioTx._tx` declared our carrier to the receiver with
`sessrx.skip(dur)` the instant the key came down, and `RollingRx.skip` empties
the buffer as well as advancing the clock -- three statements ahead of the flush
that wanted it. So on every keyed cycle the flush ran on nothing, and the audio
it was budgeted to read reached the session's WAV and no decoder at all: the
capture path never stops, and this one did.

It costs most on the receiving side of a held link, which is what is exercised
here. An IRS listens through the peer's whole 0.96 s packet and FEEDS none of it
-- the rolling regime costs 140 ms inside the gap its acknowledgement has to key
in, measured on `onair-0803-225309` -- so every slice is held, and the flush is
the only decoder that audio was ever going to reach.

The clock is the other half. Dropping the buffer was never the point of `skip`;
stepping the clock over time nobody heard is, so that the audio either side of a
transmission is not decoded in one window and timestamped as if the burst had not
happened. Both are asserted below.

Run:  python -m pytest hfmodem/tests/shrike/test_flushdebt.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, rxfront
from hfmodem.shrike.arq import IRS
from hfmodem.shrike.ptc import PtcHost

FS = rxfront.FS
CARRIER_S = 0.96          # our own transmission, one short cycle's packet
CHUNK = 12000             # 0.25 s, the listen loop's slice


class _Seam:
    """A transmit seam that keys nothing. The FSM needs one to reach CONNECTED."""

    def attach(self, host): pass
    def connect_burst(self, *a, **k): pass
    def send_cs(self, *a, **k): pass
    def send_p1_cs(self, *a, **k): pass
    def send_p1_packet(self, *a, **k): pass
    def send_packet(self, *a, **k): pass
    def pump(self): pass
    def cycle(self): pass


def _irs() -> PtcHost:
    host = PtcHost(peer=_Seam(), mycall="W9SSJ")
    host.arq.on_host_listen(True)
    host.arq.role, host.arq.dxcall = IRS, "WS8EOC"
    host.arq._enter_connected()
    return host


def _receive_window() -> np.ndarray:
    """A control signal in quiet air, the length of a peer's transmission."""
    cs = np.asarray(pactor1.control_signal(pactor1.CS_ACK_A, repeats=3),
                    np.float32)
    pad = int(0.30 * FS)
    audio = np.concatenate([np.zeros(pad, np.float32), cs,
                            np.zeros(pad, np.float32)])
    rng = np.random.default_rng(7)
    return audio + rng.normal(0, 0.002, audio.size).astype(np.float32)


def _hold_cycle(rx: onair._SessionRx, audio: np.ndarray) -> None:
    """One receiving hold cycle: hold the window, transmit, flush behind it."""
    for i in range(0, audio.size, CHUNK):
        rx.bridge(audio[i:i + CHUNK])
    rx.skip(CARRIER_S)
    rx.flush()


def test_the_flush_reads_the_window_the_cycle_collected() -> None:
    rx = onair._SessionRx(_irs(), tag="TEST")
    audio = _receive_window()
    _hold_cycle(rx, audio)
    assert rx.count, ("the flush decoded nothing: the cycle's audio was dropped "
                      "by skip() before the decode it was collected for")


def test_the_clock_still_steps_over_our_own_carrier() -> None:
    rx = onair._SessionRx(_irs(), tag="TEST")
    audio = _receive_window()
    _hold_cycle(rx, audio)
    # The window, then the burst. A clock that stopped short would date the next
    # cycle's events inside this one and splice the two sides of a transmission
    # into one decode window.
    assert rx.rx.t0 == pytest.approx(audio.size / FS + CARRIER_S, abs=1e-3)
    assert len(rx.rx.buf) == 0


def test_a_second_cycle_starts_where_the_first_one_ended() -> None:
    rx = onair._SessionRx(_irs(), tag="TEST")
    audio = _receive_window()
    _hold_cycle(rx, audio)
    _hold_cycle(rx, audio)
    assert rx.rx.t0 == pytest.approx(2 * (audio.size / FS + CARRIER_S), abs=1e-3)
    # Every event this session delivered belongs to a window it actually heard,
    # never to the seconds we spent transmitting.
    assert rx.count >= 2
