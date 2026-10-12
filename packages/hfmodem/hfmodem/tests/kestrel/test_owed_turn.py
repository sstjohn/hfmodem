# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A caller with DATA queued whose ack of the peer's last over was lost asks
for the turn at every poll the peer keys.

K5FIT, 40 m BW500, 2026-09-29 01:14Z: we read its L1 resend `FS Y\\r` with 556 B
queued and keyed our 11-symbol ack; K5FIT missed it and polled with 288/745. The
second rung of the release ladder asked while the 745's tail was still being
read, the ask was deferred and charged as spent, keepalive-a went out instead and
cleared it, and that loop ran fourteen times to the mail timeout.

Two stock 4.9s, 2026-09-29 (bench-745owed m4, m8, m12 at BW500, f4 at BW2300; d3
to d10 without a resend): the callee polls 289/361 then 288/745, or 8-symbol
idx11, 289/331 then 288/683, and the caller answers every one with the
turn-request 0.08-0.14 s behind it, twelve times with no cap. The callee
releases with 288/807 to the first one it hears.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_idle_poll import (  # noqa: F401  (autouse clock)
    CALLED, CALLER, FS, _IO, _KEEPALIVE_S, _Clock, _feed, _noise, _station, clock)

_REPLY = bytes(range(256)) * 2 + bytes(44)      # 556 B, as queued on air


class _HostIO(_IO):
    """The mail client answering into the delivery, and a gate that is still
    open on the peer's burst when the stream names it."""

    def __init__(self, reply: bytes):
        super().__init__()
        self.reply, self.hs, self.receiving = reply, None, False

    def data(self, payload):
        super().data(payload)
        reply, self.reply = self.reply, b""
        if reply:
            self.hs.send(reply)


def _caller(bw: str, reply: bytes = _REPLY):
    hs, _ = _station(bw)
    io = hs.io = _HostIO(reply)
    io.hs = hs
    hs.turn = VA._TURN_PEER
    return hs, io


def _burst(kind: VF.BurstKind, bw: str) -> np.ndarray:
    return MK.synth_burst(CALLED, VF.for_bw(kind, bw))


def _named(io, kind: VF.BurstKind) -> int:
    return sum(m.startswith(f"tx {kind.name} keyed") for m in io.msgs)


def _asks(io) -> int:
    return _named(io, VF.SESSION_TURN_REQUEST)


class _Driver:
    """kestrel_connect's mail loop: the keepalive 10 s after the link last moved."""

    def __init__(self, hs):
        self.hs, self.seen = hs, (hs.progress, hs.idle_keyed)
        self.next = _Clock.t + _KEEPALIVE_S

    def quiet(self, seconds: float) -> None:
        hs = self.hs
        for _ in range(int(seconds * FS) // 960):
            _feed(hs, _noise(960 / FS))
            if (hs.progress, hs.idle_keyed) != self.seen:
                self.seen, self.next = (hs.progress, hs.idle_keyed), _Clock.t + _KEEPALIVE_S
            elif _Clock.t >= self.next:
                hs.idle_keepalive()
                self.next = _Clock.t + _KEEPALIVE_S


def _poll(hs, io, drv, kind: VF.BurstKind, cadence: float) -> None:
    """One poll of the callee's, the gate still open on it as the stream names
    it  [K5FIT's 745 was named +0.045 s behind its last symbol]."""
    io.receiving = True
    _feed(hs, _burst(kind, hs.bw))
    _feed(hs, _noise(0.3))
    io.receiving = False
    drv.quiet(cadence - 0.3 - len(_burst(kind, hs.bw)) / FS)


def _l1_close(payload: bytes) -> np.ndarray:
    """A short over at BW500 level 1, record 0, closing the delivery."""
    body = phy.vara_body(payload, CALLER, body_len=phy.body_size("500", 0), level=0)
    return VA.OF.data_over_tx(body, level=0, bw="500",
                              lead=TX.VB6Rnd(VF.lead_state(CALLER, False, "500")))


def _k5fit():
    """A high speed capable BW500 caller reads the L1 resend `FS Y\\r`, with
    556 B queued by the host inside the delivery, and acks it."""
    hs, io = _caller("500")
    hs.high_speed = hs._link_high_speed = True
    hs._expect_level, hs._rx_read = 1, (1, 30.0)
    hs._lv_owed = hs._lv_prompt = True
    hs._key(False)
    _feed(hs, _noise(0.3))
    io.receiving = True
    _feed(hs, _l1_close(b"FS Y\r"))
    io.receiving = False
    _feed(hs, _noise(0.4))
    assert io.host == [b"FS Y\r"], io.msgs
    assert hs._txq and hs._answer_owed == VA._OWED_RELEASE, io.msgs
    return hs, io


def test_our_ack_with_data_queued_is_stocks_s852_idx0():
    """Stock's caller acks the callee's last over with 11-symbol s852 idx0 with
    data queued (bench m4 26.39 s, d3 17.96 s), and so do we; our own tap of the
    K5FIT arm reads s852 idx0 11/11 at 187.56, 238.59 and 242.41 s."""
    hs, io = _k5fit()
    want = MK.synth_tone_pairs(VF.CONNECTED_ACK_PREAMBLE
                               + VF.answer_tail(CALLER, 852, "500", 0))
    assert np.array_equal(io.sent[-1], want), io.msgs[-3:]


def test_k5fit_every_745_draws_the_turn_request_and_the_release_draws_the_over():
    """The arm as flown: K5FIT's polls were 745s, each named while the gate was
    still open on it."""
    hs, io = _k5fit()
    drv = _Driver(hs)
    for n in range(1, 13):
        _poll(hs, io, drv, VF.SESSION_RESPONDER_OVER_IDLE, 3.58)
        assert _asks(io) == n, io.msgs[-6:]
        assert _named(io, VF.SESSION_KEEPALIVE_A) == 0, io.msgs[-6:]
        assert hs._txq
    assert hs.state is VA.VaraState.CONNECTED and hs.turn == VA._TURN_ASKED
    assert _named(io, VF.SESSION_OVER_NAK) == 0
    assert not any("re-keying the 11-symbol control burst" in m for m in io.msgs)

    _feed(hs, _burst(VF.SESSION_DRAINED_RESPONDER, "500"))
    _feed(hs, _noise(0.5))
    assert hs.turn == VA._TURN_OURS, io.msgs[-4:]
    assert any(m.startswith("tx DATA over") for m in io.msgs), io.msgs[-4:]


def test_the_short_over_idle_ahead_of_the_745s_draws_the_ask():
    """m4: 289/361 2.08 s behind the ack, answered 0.1 s later with the ask."""
    hs, io = _k5fit()
    drv = _Driver(hs)
    _poll(hs, io, drv, VF.SESSION_RESPONDER_OVER_IDLE_SHORT, 2.45)
    assert _asks(io) == 1, io.msgs[-4:]
    _poll(hs, io, drv, VF.SESSION_RESPONDER_OVER_IDLE, 3.13)
    assert _asks(io) == 2, io.msgs[-4:]


def test_the_683_poll_draws_the_turn_request_too():
    """d3: without a resend the callee polls with 8-symbol idx11, 289/331 and
    then 288/683 every 3.13 s, and the caller asks at each idle."""
    hs, io = _k5fit()
    drv = _Driver(hs)
    idx11 = MK.synth_tone_pairs((VF.OVER_CONTINUE_LEAD,)
                                + VF.answer_tail(CALLED, 290, "500", 11))
    _feed(hs, idx11)
    drv.quiet(1.7)
    _poll(hs, io, drv, VF.SESSION_RESPONDER_IDLE_SHORT, 2.47)
    asked = _asks(io)
    assert asked >= 1, io.msgs[-4:]
    for n in range(1, 13):
        _poll(hs, io, drv, VF.SESSION_RESPONDER_IDLE, 3.13)
        assert _asks(io) == asked + n, io.msgs[-6:]
    assert _named(io, VF.SESSION_OVER_NAK) == 0
    assert _named(io, VF.SESSION_KEEPALIVE_A) == 0
    assert hs.state is VA.VaraState.CONNECTED


def test_bw2300_every_745_draws_the_turn_request():
    """f4, the same at BW2300."""
    hs, io = _caller("2300")
    io.receiving = True
    hs.send(_REPLY)
    io.receiving = False
    hs._key_over_answer(last=True, owes_release=False)
    assert hs._answer_owed == VA._OWED_RELEASE and hs._txq
    drv = _Driver(hs)
    for n in range(1, 13):
        _poll(hs, io, drv, VF.SESSION_RESPONDER_OVER_IDLE, 3.13)
        assert _asks(io) == n, io.msgs[-6:]
    assert _named(io, VF.SESSION_KEEPALIVE_A) == 0
    _feed(hs, _burst(VF.SESSION_DRAINED_RESPONDER, "2300"))
    _feed(hs, _noise(0.5))
    assert hs.turn == VA._TURN_OURS, io.msgs[-4:]


@pytest.mark.parametrize("kind, answer", [
    (VF.SESSION_RESPONDER_OVER_IDLE, VF.SESSION_KEEPALIVE_A),
    (VF.SESSION_RESPONDER_IDLE, VF.SESSION_OVER_NAK)])
def test_nothing_queued_keeps_the_2026_09_16_answers(kind, answer):
    hs, io = _caller("500", reply=b"")
    drv = _Driver(hs)
    for n in range(1, 4):
        _poll(hs, io, drv, kind, 3.13)
        assert _named(io, answer) == n, io.msgs[-4:]
    assert _asks(io) == 0
