# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The responder polls after our release, and the poll is answered in its turnaround.

A stock VARA HF 4.9.0 responder handed the channel before its host has queued
anything keys its 0.48 s control burst every 2.3 s for the next 60 s, reply or
no reply, and this station named none of 27. The reply comes once a keepalive
of ours lands whole in the gap behind one poll — the responder stops polling,
asks for the turn 5.5 s later, and our release answers that — 5 of 5 on
2026-09-03. A release keyed at the poll instead restarts the polling. So the
poll is named, the cadence's burst is keyed into its turnaround, and the cadence
clock restarts on it.

Nothing here opens a device, and nothing keys a radio.
"""
from __future__ import annotations

import numpy as np

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

_MYCALL, _CALLED = "W9SSJ", "W1AW"


class _IO(VA.VaraIO):
    def __init__(self):
        self.msgs: list[str] = []
        self.sent: list[np.ndarray] = []

    def key(self, on): ...

    def tx(self, samples): self.sent.append(np.asarray(samples, float))

    def pending(self): ...

    def connected(self, *a): ...

    def log(self, msg): self.msgs.append(msg)


def _connected():
    io = _IO()
    hs = VA.VaraStationHandshake([_MYCALL], io, bw="2300")
    hs.role, hs.called, hs.caller = "initiator", _CALLED, _MYCALL
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _peer_control() -> np.ndarray:
    return MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)


def _drained():
    """A one-over delivery of ours, answered, and the release that follows."""
    hs, io = _connected()
    hs.send(b"FF\r")
    hs.on_rx_audio(_peer_control())          # granted; the over is keyed
    hs.on_rx_audio(_peer_control())          # answered; nothing left, released
    assert hs.turn == VA._TURN_PEER and not hs._txq
    assert _releases(io) == 1, io.msgs
    return hs, io


def _releases(io) -> int:
    return sum("tx session-turn-release keyed" in m for m in io.msgs)


def _keepalives(io) -> int:
    return sum("tx session-keepalive" in m for m in io.msgs)


def test_the_peers_poll_after_our_release_is_answered_in_its_turnaround():
    hs, io = _drained()
    hs.on_rx_audio(_peer_control())
    assert any("control-burst poll" in m for m in io.msgs), io.msgs
    assert _keepalives(io) == 1 and _releases(io) == 1, io.msgs
    assert hs.turn == VA._TURN_PEER


def test_the_answer_is_the_cadences_burst_keyed_early():
    """Charged to the budget as a cadence burst is, and counted for the clock
    that drives the cadence, so the loop does not key another on top of it."""
    hs, io = _drained()
    before = (hs.progress, hs._since_progress, hs.idle_keyed)
    hs.on_rx_audio(_peer_control())
    assert hs.progress == before[0]
    assert hs._since_progress == before[1] + 1
    assert hs.idle_keyed == before[2] + 1


def test_one_answer_per_cadence_and_not_one_per_poll():
    hs, io = _drained()
    for _ in range(2 * VA._POLLS_PER_ANSWER):
        hs.on_rx_audio(_peer_control())
    assert _keepalives(io) == 2, io.msgs


def test_a_polling_peer_is_closed_by_the_give_up_budget():
    hs, io = _drained()
    for _ in range(200):
        hs.on_rx_audio(_peer_control())
        if hs.state is not VA.VaraState.CONNECTED:
            break
    assert hs.state is not VA.VaraState.CONNECTED
    assert _keepalives(io) == VA._MAX_WITHOUT_PROGRESS


def test_the_poll_count_starts_again_at_every_release():
    hs, io = _drained()
    hs.on_rx_audio(_peer_control())
    hs._release_turn()
    hs.on_rx_audio(_peer_control())
    assert _keepalives(io) == 2, io.msgs


def test_a_control_burst_before_any_release_is_not_a_poll():
    hs, io = _connected()
    hs.on_rx_audio(_peer_control())
    assert not any("control-burst poll" in m for m in io.msgs), io.msgs
    assert _keepalives(io) == 0


def test_a_poll_with_something_queued_is_not_answered():
    hs, io = _drained()
    hs._txq.append(b"x")
    hs.on_rx_audio(_peer_control())
    assert not any("control-burst poll" in m for m in io.msgs), io.msgs
    assert _keepalives(io) == 0


def test_the_peer_spending_the_channel_ends_the_polling_state():
    hs, io = _drained()
    hs._progressed()
    hs.on_rx_audio(_peer_control())
    assert not any("control-burst poll" in m for m in io.msgs), io.msgs
