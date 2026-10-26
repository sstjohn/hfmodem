# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA
from .test_data_over_gate import _connected


@pytest.mark.parametrize("bw", ["500", "2300"])
def test_answered_repacketized_joined_close_releases_before_late_write(monkeypatch, bw):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    hs, io = _connected(bw=bw)
    kinds = []
    io.tx_metadata = lambda **metadata: kinds.append(metadata["packet_kind"])
    io.receiving = True
    hs.send(b"a" * 24)
    hs.send(b"b" * 16)
    assert hs._txq == [b"a" * 24 + b"b" * 16]
    io.receiving = False
    hs._turn_granted((), said="session-drained-responder")
    original = hs._tx_pending
    assert phy.over_is_last(original[0], hs.caller)
    assert phy.vara_payload(original[0], caller=hs.caller,
                            body_len=len(original[0])) == b"a" * 24 + b"b" * 16
    hs._retry_data_over()
    hs._took_responder_nak()
    body, over, level = hs._tx_pending
    assert over == original[1] and level < original[2]
    assert len(body) < len(original[0]) and hs._tx_recovery_levels
    assert hs._tx_recovering
    hs.send(b"late host write")
    assert hs._txq[-1] == b"late host write"
    delivered = bytearray()
    while True:
        body = hs._tx_pending[0]
        delivered.extend(phy.vara_payload(body, caller=hs.caller, body_len=len(body)))
        if phy.over_is_last(body, hs.caller):
            break
        hs._took_over_continue()
    assert bytes(delivered) == b"a" * 24 + b"b" * 16
    assert hs.pending_bytes == len(phy.vara_payload(
        body, caller=hs.caller, body_len=len(body))) + len(b"late host write")
    mark = len(kinds)
    hs._took_control_burst()
    release = "session-drained" if bw == "500" else "session-turn-release"
    assert kinds[mark:] == [release]
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert hs._txq == [b"late host write"] and hs._ask_owed
    assert hs.pending_bytes == len(b"late host write") and not hs._tx_recovering
    now[0] += .294
    hs.idle_keepalive()
    assert kinds[mark:] == [release]
    now[0] += .002
    hs.idle_keepalive()
    assert kinds[mark:] == [release, "session-turn-request"]
    assert hs._tx_pending is None and hs.turn == VA._TURN_ASKED
    assert not hs._ask_owed
    hs._turn_granted((), said="session-drained-responder")
    assert kinds[mark:] == [release, "session-turn-request", "data"]
    body = hs._tx_pending[0]
    assert (phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            + b"".join(hs._txq)) == b"late host write"
