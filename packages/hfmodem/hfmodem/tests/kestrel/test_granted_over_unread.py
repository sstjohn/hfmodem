# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A caller that granted the turn to a peer with traffic never asks for it back
while that peer's over is unread.

VA2OPS, 20 m BW500 CHAT ON, 2026-11-08 20:07Z: its frame 0 answered our level-6
close, we keyed caller-turn-grant ("VA2OPS has traffic"), and its first over went
unread. Its session-responder-idle was named twice and answered with nothing;
our host's chat line then keyed the short turn-request, VA2OPS answered with
session-drained-responder, and the block it had sent never reached the host.
A VARA HF receiver never asks while an over of the peer's is owed: the sender
takes the ask as that over's acknowledgement (2026-10-21).

Every burst here is generated. Nothing opens a device or keys a radio.
"""
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from hfmodem.tests.kestrel.test_idle_poll import _feed, _noise, clock  # noqa: F401
from hfmodem.tests.kestrel.test_owed_turn import _Driver, _caller, _named, _poll


def _asks(io) -> int:
    return (_named(io, VF.SESSION_TURN_REQUEST)
            + _named(io, VF.SESSION_TURN_REQUEST_SHORT))


def _granted(site: str):
    hs, io = _caller("500", reply=b"")
    hs.chat = True
    hs._start_compression("OFF")
    if site == "caller-grant":
        hs.turn = VA._TURN_OURS
        hs._grant_held_turn()
        assert any(m.startswith("tx caller-turn-grant") for m in io.msgs), io.msgs
    elif site == "final-ask":
        # Frame 0 answering our last over below the 11-symbol handback level.
        hs.turn = VA._TURN_OURS
        hs._queue_host_payload(b"previous delivery")
        assert hs._tx_data_over()
        hs._took_control_burst(ask=True)
        assert any(m.startswith("turn released to") for m in io.msgs), io.msgs
    else:
        hs._released = True
        hs._took_peer_control(0)
        assert any("it has traffic" in m for m in io.msgs), io.msgs
    assert hs.turn == VA._TURN_PEER
    _feed(hs, _noise(1.0))
    return hs, io


@pytest.mark.parametrize("site", ["caller-grant", "final-ask", "peer-control-grant"])
def test_host_text_behind_a_traffic_grant_keys_no_turn_request(site):
    hs, io = _granted(site)
    hs.send(b"AI: does this work?\n")
    assert _asks(io) == 0, io.msgs[-4:]
    assert hs._ask_owed and hs._txq


def test_a_refused_traffic_release_keeps_its_traffic_on_the_retry():
    hs, io = _caller("500", reply=b"")
    hs.chat, hs.turn = True, VA._TURN_OURS
    hs._start_compression("OFF")
    hs._queue_host_payload(b"previous delivery")
    assert hs._tx_data_over()
    io.tx_went_out = lambda: False
    hs._took_control_burst(ask=True)
    assert hs._release_owed and not hs._granted_over_unread(), io.msgs[-3:]
    io.tx_went_out = lambda: True
    hs.idle_keepalive()
    assert not hs._release_owed and hs._granted_over_unread(), io.msgs[-3:]
    _feed(hs, _noise(1.0))
    hs.send(b"AI: does this work?\n")
    assert _asks(io) == 0, io.msgs[-4:]


def test_the_peers_idle_behind_a_traffic_grant_draws_the_nak_not_the_ask():
    hs, io = _granted("caller-grant")
    io.receiving = True
    hs.send(b"AI: does this work?\n")
    io.receiving = False
    _poll(hs, io, _Driver(hs), VF.SESSION_RESPONDER_IDLE, 3.13)
    assert _asks(io) == 0, io.msgs[-6:]
    assert _named(io, VF.SESSION_OVER_NAK) >= 1, io.msgs[-6:]
    assert hs._ask_owed and hs._txq
    assert hs.state is VA.VaraState.CONNECTED


def test_reading_the_granted_over_frees_the_ask():
    hs, io = _granted("caller-grant")
    assert hs._host_ask_blocked() and not hs._owes_turn()
    hs._peer_over += 1
    assert not hs._granted_over_unread() and not hs._host_ask_blocked()


def test_a_plain_release_still_leaves_the_ask_free():
    """A grant alone proves nothing unread: the peer may have nothing to send."""
    hs, io = _caller("500", reply=b"")
    hs.chat, hs.turn = True, VA._TURN_OURS
    assert hs._release_turn()
    assert hs.turn == VA._TURN_PEER
    assert not hs._granted_over_unread() and not hs._host_ask_blocked()
