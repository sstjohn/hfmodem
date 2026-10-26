# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from types import SimpleNamespace
import numpy as np
import pytest
from hfmodem.contact import slots as S
from hfmodem.contact.broadcast import Broadcast
from hfmodem.kestrel.rx import broadcast as B, varahf500 as R


def test_off_channel_ui_routes_as_broadcast_without_decoding_or_executing_info():
    message = Broadcast(1, "K1ABC", "ALL", b"\xff<Q:75>")
    monitor = object.__new__(S.SlotMonitor)
    monitor.mycall = "W9SSJ"
    monitor._coverage = lambda *a: []
    monitor._lit = lambda *a, **k: None
    heard = []
    monitor.heard = lambda *args: heard.append(args)
    ch = SimpleNamespace(
        slot=1,
        home=False,
        idles=SimpleNamespace(feed=lambda *a: []),
        high=SimpleNamespace(feed=lambda y: None),
        cq=SimpleNamespace(feed=lambda y: [message]),
        requests=SimpleNamespace(push=lambda y: []),
        ident=SimpleNamespace(push=lambda y: []),
        frames=SimpleNamespace(feed=lambda *a, **k: [], overflows=0, expired=0),
    )
    assert monitor._read(ch, SimpleNamespace(caller="", called="", keyed=""), np.empty(0)) == [
        ("broadcast", 1, message)
    ]
    assert heard == [("K1ABC", None)]


@pytest.mark.parametrize("info", [b"A", b"UI1 TEST", b"x" * 25, b"x" * 77])
def test_broadcast_packet_does_not_create_passive_qso_or_gibberish(info):
    level, frames = B.packetize(Broadcast(1, "K1ABC", "ALL", info))
    frame = R.FrameResult(0, (), 0, 1.0, True, frames[0], level=level)
    monitor = object.__new__(S.SlotMonitor)
    monitor._lit = lambda *a, **k: None
    monitor._link = lambda *a: (_ for _ in ()).throw(AssertionError("UI is not a QSO"))
    assert monitor._frame(SimpleNamespace(), SimpleNamespace(), frame) == []
