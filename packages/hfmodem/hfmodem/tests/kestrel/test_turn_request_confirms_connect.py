# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A peer asking for the channel has told us the link is up.

`session-turn-request-responder` exists only inside a session, and a station with
a queue and no channel is what a gateway is the instant it answers a call. A
stock caller reads it that way: keyed into the turnaround behind its link setup
it confirmed the connect 5 times in 10, against 5 in 11 for the connected-ack
itself, 0 in 9 for the release keyed into the same slot, and 0 in 1 for an empty
slot — a bench of 2026-08-30, where the control arm keying nothing left the
caller sending link setups to its own timeout.

This build ran the frame's only route in CONNECTED, so AJ4GU spent four keyings
on 2026-08-29 — 72.857, 96.618, 106.667 and 111.688 s, up to 32 of 32 tones —
telling a station that answered by resending link setups and then dropping back
to connect requests at a gateway that had already connected it.

The turnaround holds one burst, so what goes out here is the answer the frame
asks for and not the step-6 confirm: keying the confirm would have the peer
answering that while we key the release on top of the answer.
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _IO

MYCALL, CALLED = "W9SSJ", "AJ4GU"
ASK = VF.SESSION_TURN_REQUEST_RESPONDER


class _ConnectIO(_IO):
    def __init__(self):
        super().__init__()
        self.up: list[tuple] = []

    def connected(self, *a):
        self.up.append(a)


def _awaiting_ack(io=None):
    """An initiator that keyed its link setup and is waiting for the ack — the
    state AJ4GU found this station in, four times."""
    io = io or _ConnectIO()
    hs = VA.VaraStationHandshake([MYCALL], io)
    hs.role, hs.called, hs.caller = "initiator", CALLED, MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTING, VA._I_LINKSETUP_SENT
    return hs, io


def _ask(callsign: str = MYCALL) -> np.ndarray:
    """The turn-request as it reaches the state machine off the segmenter: the
    frame, with the pre-roll and hangover a bracket keeps around it."""
    x = MK.synth_burst(callsign, ASK)
    return np.concatenate([np.zeros(2 * MK.HOP), x, np.zeros(3 * MK.HOP)])


def test_the_ask_brings_the_link_up():
    hs, io = _awaiting_ack()
    hs.on_rx_audio(_ask())
    assert hs.state is VA.VaraState.CONNECTED
    assert io.up == [(MYCALL, CALLED, "2300")]


def test_the_ask_is_answered_with_the_release_and_nothing_else():
    """One burst in the turnaround, and it is the one the peer asked for: with
    nothing of ours queued the channel is the gateway's for the asking."""
    hs, io = _awaiting_ack()
    hs.on_rx_audio(_ask())
    assert io.keys == 1
    assert f"tx {VF.SESSION_TURN_RELEASE.name}" in " ".join(io.msgs)
    assert VF.SESSION_CONFIRM.name not in " ".join(io.msgs)


def test_an_ask_keyed_to_another_station_confirms_nothing():
    """The control the reading needs: it is this frame keyed to us, not any burst
    arriving in that window, and a station whose seed does not match ours leaves
    the connect exactly where it was."""
    hs, io = _awaiting_ack()
    hs.on_rx_audio(_ask("KB3AC-10"))
    assert hs.state is VA.VaraState.CONNECTING
    assert io.keys == 0


def test_silence_in_the_window_confirms_nothing():
    hs, io = _awaiting_ack()
    hs.on_rx_audio(np.zeros(len(_ask())))
    assert hs.state is VA.VaraState.CONNECTING
    assert io.keys == 0


def test_the_connect_still_needs_the_link_setup_behind_it():
    """A gateway that has not yet had our link setup cannot have connected us, so
    the ask is not read as a confirmation before step 4 has gone out."""
    hs, io = _awaiting_ack()
    hs.step = VA._I_CR_SENT
    hs.on_rx_audio(_ask())
    assert hs.state is VA.VaraState.CONNECTING
    assert io.keys == 0


def test_the_same_frame_still_works_once_connected():
    """The route that already existed is untouched: the ask arriving on a live
    link is answered the same way."""
    io = _ConnectIO()
    hs = VA.VaraStationHandshake([MYCALL], io)
    hs.role, hs.called, hs.caller = "initiator", CALLED, MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.on_rx_audio(_ask())
    assert io.keys == 1
    assert f"tx {VF.SESSION_TURN_RELEASE.name}" in " ".join(io.msgs)
