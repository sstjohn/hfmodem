# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_mfsk as MK
from .test_nak_repacketization import pending, unconfirmed


@pytest.mark.parametrize("bw", ["2300", "2750"])
@pytest.mark.parametrize("closing", [False, True])
def test_late_host_writes_preserve_recovery_and_the_next_delivery(bw, closing, monkeypatch):
    hs, io, data = pending(bw, 94)
    hs.send(b"next host write")
    hs._took_responder_nak()
    delivered = bytearray()
    if closing:
        while hs._tx_recovery_levels:
            body = hs._tx_pending[0]
            delivered.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
            hs._took_over_continue()
    hs.send(b"later host write")
    assert hs._txq[-2:] == [b"next host write", b"later host write"]
    assert bytes(delivered) + unconfirmed(hs) == data + b"next host writelater host write"
    while hs._tx_recovery_levels:
        hs._took_over_continue()
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    kinds = []
    io.tx_metadata = lambda **metadata: kinds.append(metadata["packet_kind"])
    hs._took_control_burst()
    assert kinds == ["session-turn-release"]
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert hs._txq == [b"next host write", b"later host write"] and hs._ask_owed
    now[0] += .294
    hs.idle_keepalive()
    assert kinds == ["session-turn-release"]
    now[0] += .002
    hs.idle_keepalive()
    assert kinds == ["session-turn-release", "session-turn-request"]
    assert hs._tx_pending is None and hs.turn == VA._TURN_ASKED
    assert not hs._ask_owed
    hs._turn_granted((), said="session-drained-responder")
    assert kinds[-1] == "data"
    body, _, level = hs._tx_pending
    assert level == phy.base_level(bw)
    assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) == b"next host write"
    assert hs._txq == [b"later host write"]
    hs._took_control_burst()
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert hs._txq == [b"later host write"] and hs._ask_owed


@pytest.mark.parametrize("bw,answer", [("2300", "query"), ("2750", "query"),
                                       ("500", "request")])
def test_alternative_final_answer_restores_normal_host_joining(bw, answer):
    hs, io, _ = pending(bw, 94)
    hs._took_responder_nak()
    while hs._tx_recovery_levels:
        hs._took_over_continue()
    assert phy.over_is_last(hs._tx_pending[0], hs.caller)
    hs._stream_owns_turn_request = True
    if answer == "query":
        assert hs._query_final_answer()
        hs._final_query_samples = VA._SESSION_NEED + MK.HOP
    assert hs._took_turn_request(raw=True)
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    io.receiving = True
    hs.send(b"new")
    hs.send(b" joined")
    assert hs._txq == [b"new joined"]
