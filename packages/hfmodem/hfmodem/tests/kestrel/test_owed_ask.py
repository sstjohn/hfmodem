# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A turn-request the host owes while the peer is keying goes out in the peer's
next turnaround, and only there.

`send` keys nothing across a burst the gate has open, because a request keyed
into the middle of the peer's frame is heard by nobody. What it used to do
instead was leave the ask to the idle cadence, and on the bench of 2026-09-03
that put the request 6.3 s behind an unkey, into a gap that was already the
peer's again: a stock responder idling between our overs keys every 3.4 s and
listens for 1.96 s, and the request itself is 1.37 s. The gap that fits it is
the one that opens as the burst under which the ask was owed closes, and that
instant is when the gate hands the bracket back.
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.arq import phy as _phy
from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF

_MYCALL, _CALLED = "W9SSJ", "KC9GHZ"


class _IO(VA.VaraIO):
    def __init__(self):
        self.receiving = False
        self.keys = 0
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []

    def key(self, on): self.keys += bool(on)

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def pending(self): ...

    def connected(self, *a): ...

    def data(self, payload): ...

    def log(self, msg): self.msgs.append(msg)


def _connected():
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _owed(hs, io, payload=b"x" * 10):
    io.receiving = True
    hs.send(payload)
    assert io.keys == 0, "keyed across the peer's burst"
    assert hs.turn == VA._TURN_PEER
    io.receiving = False


def _their_over(text: bytes, index: int) -> np.ndarray:
    return OF.data_over_tx(_phy.vara_body(text, _MYCALL), over=index)


def _requests(io) -> list[str]:
    return [m for m in io.msgs if m.startswith("tx session-turn-request")]


def test_the_owed_ask_goes_out_when_the_peers_idle_burst_closes():
    hs, io = _connected()
    _owed(hs, io)
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    assert len(_requests(io)) == 1, io.msgs
    assert hs.turn == VA._TURN_ASKED
    assert io.keys == 1


def test_a_burst_nothing_names_still_opens_the_turnaround():
    hs, io = _connected()
    _owed(hs, io)
    hs.on_rx_audio(np.random.default_rng(3).standard_normal(int(1.4 * MK.FS)) * 0.1)
    assert any("not an MFSK handshake burst" in m for m in io.msgs), io.msgs
    assert len(_requests(io)) == 1, io.msgs


def test_the_ask_is_owed_once():
    hs, io = _connected()
    _owed(hs, io)
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    hs.turn = VA._TURN_PEER          # the request was lost and the turn handed back
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    assert len(_requests(io)) == 1, "asked again on a burst with nothing owed"


def test_a_gate_still_open_at_the_bracket_holds_the_ask():
    hs, io = _connected()
    _owed(hs, io)
    io.receiving = True              # the next burst opened before this one was read
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    assert not _requests(io), io.msgs
    io.receiving = False
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    assert len(_requests(io)) == 1, io.msgs


def test_an_over_under_the_owed_ask_is_answered_and_not_also_asked_into():
    """One burst per turnaround, and the over's own answer is what goes in it.

    Every over of the peer's is acknowledged, the last one included: the release
    that hands us the channel follows that acknowledgement, and a turn-request
    keyed in its place buys the ask and loses the release
    [see `_answer_data_over`]. So the ask outlives the delivery and goes out in
    the peer's own cadence behind it  [see `_reack_release`].
    """
    hs, io = _connected()
    _owed(hs, io)
    hs.on_rx_audio(_their_over(b"y" * _phy.payload_size("2300"), 1))
    assert io.keys == 1 and not _requests(io), io.msgs
    assert hs.turn == VA._TURN_PEER
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    assert not _requests(io), "asked mid-delivery on an ask the over had already settled"
    hs.on_rx_audio(_their_over(b"end", 2))
    assert not _requests(io), "asked in place of acknowledging the last over"
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    hs.on_rx_audio(MK.synth_burst(_CALLED, VF.SESSION_RESPONDER_IDLE))
    assert len(_requests(io)) == 1, io.msgs
    assert hs.turn == VA._TURN_ASKED


def test_the_cadence_still_carries_the_ask_when_no_bracket_ever_closes():
    hs, io = _connected()
    _owed(hs, io)
    hs.idle_keepalive()
    assert len(_requests(io)) == 1, io.msgs
    assert hs.turn == VA._TURN_ASKED
