# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Host controls must reach the real ARDOP session and rendered samples."""
import numpy as np
import pytest

from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.arq import session as S
from hfmodem.besra.host.server import HostServer
from hfmodem.besra.host.protocol import ArdopState
from hfmodem.tests.besra.test_arq_session import Pair


def setup():
    modem = BesraModem()
    host = HostServer(modem, control_port=0, data_port=0, quiet=True)
    events = []
    host._send_cmd = events.append
    modem.start(host)
    return modem, host, events


def test_codec_cycles_lane_without_disabling_other_station_resources():
    modem, host, events = setup()
    host._dispatch("MYCALL W9SSJ")
    host._dispatch("RADIOCTRL FALSE")
    host._dispatch("CODEC FALSE")
    assert modem.state == ArdopState.OFFLINE
    assert "CODEC now FALSE" in events
    host._dispatch("ARQCALL K7ABC 5")
    assert events[-1].startswith("FAULT")
    host._dispatch("CODEC TRUE")
    assert modem.state == ArdopState.DISC
    assert modem._session.mycall == "W9SSJ"
    host._dispatch("RADIOCTRL TRUE")
    assert events[-1].startswith("FAULT")
    assert host._cfg["RADIOCTRL"] == "FALSE"


def test_rejected_codec_stop_does_not_store_offline_state():
    modem, host, events = setup()
    modem.transmit(b"pending")
    host._dispatch("CODEC FALSE")
    assert events[-1].startswith("FAULT")
    assert host._cfg["CODEC"] == "TRUE"
    assert modem.queued == 7
    host._dispatch("INITIALIZE")
    assert modem.queued == 0
    assert modem._session.confirmed_total == 0
    assert modem._session.discarded_total == 7


def test_drive_level_changes_rendered_output(monkeypatch):
    modem, host, events = setup()
    captured = []
    modem.audio_out = captured.append
    monkeypatch.setattr("hfmodem.besra.arq.modem.modulator.render_frame",
                        lambda *args, **kw: np.array([-10000, 0, 10000], dtype="<i2"))
    host._dispatch("DRIVELEVEL 25")
    modem.send(S.IDLE, b"", 1)
    assert captured[-1].tolist() == [-2500, 0, 2500]
    host._dispatch("DRIVELEVEL 101")
    assert host._cfg["DRIVELEVEL"] == "25"
    assert events[-1].startswith("FAULT")


def test_autobreak_and_timeout_reach_active_session():
    modem, host, _ = setup()
    host._dispatch("AUTOBREAK FALSE")
    host._dispatch("ARQTIMEOUT 120")
    assert not modem._session.autobreak
    assert modem._session._timeout_s == 120


def test_manual_break_works_with_autobreak_disabled():
    pair = Pair()
    pair.connect()
    pair.responder.autobreak = False
    pair.responder.queue_data(b"answer")
    assert not any(ft == S.BREAK for _, ft in pair.relay.log)
    pair.responder.request_break()
    assert any(ft == S.BREAK for _, ft in pair.relay.log)


def test_graceful_close_waits_for_final_peer_ack():
    pair = Pair()
    pair.connect()
    payload = bytes(range(256)) + b"short final block"
    pair.caller.queue_data(payload)
    pair.caller.disconnect()
    assert not any(ft == S.DISC for _, ft in pair.relay.log)
    assert pair.caller.confirmed_total == 0
    with pytest.raises(ValueError, match="closing"):
        pair.caller.queue_data(b"late write")
    for _ in range(100):
        pair.pump()
        pair.tick(.5)
        if pair.caller.state == ArdopState.DISC:
            break
    assert pair.obs_b.received == payload
    assert pair.caller.confirmed_total == len(payload)
    assert pair.caller.discarded_total == 0
    assert pair.caller.state == ArdopState.DISC
