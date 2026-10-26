# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pat's gate-3 writes and closing-over answer, 2026-10-18."""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara import vara_ofdm as OF
from hfmodem.tests import evidence
from hfmodem.tests.kestrel import corpora


HANDSHAKE = (b";FW: W9SSJ\r[Pat-1.0.0-B2FHMG$]\r;PR: 00000000\r"
             b"; W1AW DE W9SSJ (EN63bc)\r")
PROPOSAL = b"FC EM OZC0UQQPX80Z 639 499 0\r"
CHECKSUM = b"F> 43\r"


class IO(VA.VaraIO):
    def __init__(self):
        self.kinds, self.msgs = [], []

    def key(self, on): pass
    def tx(self, samples): pass
    def pending(self): pass
    def connected(self, *args): pass
    def data(self, payload): pass
    def log(self, message): self.msgs.append(message)
    def tx_metadata(self, **metadata): self.kinds.append(metadata["packet_kind"])


@pytest.mark.parametrize("recorded", [False, True], ids=["events", "cable"])
@pytest.mark.parametrize("bw,record,late", [
    ("500", 7, False), ("2300", 5, False), ("500", 7, True),
])
def test_gate3_closing_answer_releases_before_more_host_data(monkeypatch, bw, record, late, recorded):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    answer = None
    if recorded:
        start, end = (30.58, 31.06) if bw == "500" else (24.21, 24.69)
        path = (evidence.WORKING / "varac-p2/gates/gate3" / f"main-bw{bw}"
                / "cables" / ("pat-main__b2a.wav" if bw == "500"
                              else "pat-main-2300__b2a.wav"))
        x = corpora.wav_mono(evidence.require_input(path))
        answer = x[int((start - .30) * MK.FS):int((end + .25) * MK.FS)]
        answer /= np.abs(answer).max()

    io = IO()
    modem = VaraHostModem(io, bandwidth=bw)
    hs = modem.station
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_PEER
    hs._link_high_speed, hs._rec500 = True, record
    hs._lead = VB6Rnd(VF.lead_state("W9SSJ", False, bw))

    def write(payload):
        modem.transmit(payload)
        modem.poll()

    assert tuple(map(len, (HANDSHAKE, PROPOSAL, CHECKSUM))) == (70, 29, 6)
    write(HANDSHAKE)
    if not late:
        write(PROPOSAL)
    mark = len(io.kinds)
    hs._turn_granted(np.zeros(0), said="session-drained-responder")
    pending = hs._tx_pending
    if late:
        write(PROPOSAL)
        assert hs._tx_pending is pending
    assert phy.over_is_last(pending[0], hs.caller)
    sent = HANDSHAKE if late else HANDSHAKE + PROPOSAL
    assert phy.vara_payload(pending[0], caller=hs.caller,
                            body_len=len(pending[0])) == sent
    if recorded:
        hs.on_rx_audio(answer)
    else:
        hs._took_control_burst()

    expected = ["data", "session-turn-release"]
    assert io.kinds[mark:] == expected, io.msgs
    assert hs._tx_pending is None
    assert hs.pending_bytes == (len(PROPOSAL) if late else 0)
    assert b"".join(hs._txq) == (PROPOSAL if late else b"")
    assert any(f"record {record} ({len(sent)} payload bytes" in m for m in io.msgs)

    write(CHECKSUM)
    assert io.kinds[mark:] == expected
    assert hs.turn == VA._TURN_PEER and hs._ask_owed
    now[0] += (VA._RELEASE_ANSWER_WAIT_S if bw == "500" else VA._RELEASE_ASK_GAP_S) + .001
    hs.on_rx_stream(np.zeros(MK.FS // 100))
    assert io.kinds[mark:] == ["data", "session-turn-release", "session-turn-request"]
    assert b"".join(hs._txq) == (PROPOSAL if late else b"") + CHECKSUM
    assert hs._tx_pending is None and hs.turn == VA._TURN_ASKED
    hs._turn_granted(np.zeros(0), said="session-drained-responder")
    assert io.kinds[-1] == "data"
    body = hs._tx_pending[0]
    assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) == (
        (PROPOSAL if late else b"") + CHECKSUM)
    hs._took_control_burst()
    assert hs.pending_bytes == 0 and hs.turn == VA._TURN_PEER


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_empty_write_preserves_the_queued_close_of_an_exact_full_over(bw):
    hs = VA.VaraStationHandshake(["W9SSJ"], IO(), bw=bw)
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs.send(b"x" * hs._tx_payload_size())
    pending = hs._tx_pending
    assert not phy.over_is_last(pending[0], hs.caller)
    assert hs._txq == [b""]
    hs.send(b"")
    assert hs._tx_pending is pending and hs._txq == [b""]
    hs._took_control_burst(continue_reply=True)
    assert phy.over_is_last(hs._tx_pending[0], hs.caller)
    hs._took_control_burst()
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER


@pytest.mark.parametrize("bw", ["500", "2300"])
@pytest.mark.parametrize("full_overs", [1, 3])
def test_late_writes_preserve_the_announced_delivery(bw, full_overs):
    hs = VA.VaraStationHandshake(["W9SSJ"], IO(), bw=bw)
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    n = hs._tx_payload_size()
    hs.send(b"a" * (n * full_overs + 10))
    pending = hs._tx_pending
    announced = list(hs._txq)
    assert pending[0][-1] == VA._frame_field(full_overs - 1, False)
    hs.send(b"b" * 60)
    hs.send(b"c" * 5)
    later = b"b" * 60 + b"c" * 5
    queued = [later[i:i + n] for i in range(0, len(later), n)]
    assert hs._tx_pending is pending
    assert hs._txq == announced + queued
    for remaining in range(full_overs - 1, -1, -1):
        hs._took_control_burst(continue_reply=True)
        body, _, level = hs._tx_pending
        if remaining:
            assert body[-1] == VA._frame_field(remaining - 1, False)
        else:
            assert level == phy.close_level(bw)
            assert phy.over_is_last(body, hs.caller)
            assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) == b"a" * 10
    hs._took_control_burst()
    assert hs._txq == queued
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER
    assert hs._ask_owed


@pytest.mark.parametrize("bw", ["500", "2300"])
@pytest.mark.parametrize("boundary", [b"", b"a" * 10])
@pytest.mark.parametrize("guard", ["full-keyed", "pending"])
def test_announced_prefix_survives_between_overs_and_with_only_pending(bw, boundary, guard):
    io = IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    n = hs._tx_payload_size()
    hs.send(b"a" * (2 * n) + boundary)
    if guard == "full-keyed":
        io._refused = True
        hs._took_control_burst(continue_reply=True)
        assert hs._tx_pending is None and hs._full_keyed
    else:
        hs._full_keyed = 0
        assert not phy.over_is_last(hs._tx_pending[0], hs.caller)
    prefix = list(hs._txq)
    hs.send(b"b" * 20)
    hs.send(b"c" * 5)
    assert hs._txq == prefix + [b"b" * 20 + b"c" * 5]


@pytest.mark.parametrize("bw,record,release", [
    ("2300", 3, "session-turn-release"),
    ("500", 4, "session-turn-release"),
    ("500", 0, "session-drained"),
])
@pytest.mark.parametrize("route", ["stream", "bracket", "idle", "write"])
def test_queued_write_waits_for_the_stock_release_gap(monkeypatch, bw, record, release, route):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    io = IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs._rec500 = record
    hs.send(b"first")
    hs.send(b"second")
    hs._took_control_burst()
    assert io.kinds == ["data", release]
    assert hs.turn == VA._TURN_PEER and hs._ask_owed

    def poll():
        if route == "stream":
            hs.on_rx_stream(np.zeros(MK.FS // 100))
        elif route == "bracket":
            hs.on_rx_audio(np.zeros(0))
        elif route == "idle":
            hs.idle_keepalive()
        else:
            hs.send(b"more")

    # Behind the BW500 17-symbol release the ask waits for stock's answer,
    # here never heard, for _RELEASE_ANSWER_WAIT_S.
    gap = (VA._RELEASE_ANSWER_WAIT_S if (bw, release) == ("500", "session-turn-release")
           else VA._RELEASE_ASK_GAP_S)
    now[0] += gap - .001
    poll()
    assert io.kinds == ["data", release]
    now[0] += .002
    poll()
    assert io.kinds == ["data", release, "session-turn-request"]
    assert hs.turn == VA._TURN_ASKED and not hs._ask_owed


@pytest.mark.parametrize("bw", ["500", "2300"])
@pytest.mark.parametrize("peer_delay", [.05, .10])
def test_peer_over_after_release_is_handled_before_the_owed_ask(monkeypatch, bw, peer_delay):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    io = IO()
    received = []
    io.data = received.append
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step, hs.turn = VA.VaraState.CONNECTED, VA._I_CONNECTED, VA._TURN_OURS
    hs.send(b"first")
    hs.send(b"second")
    hs._took_control_burst()
    assert io.kinds == ["data", "session-turn-release"]
    now[0] += peer_delay
    io.receiving = True
    assert not hs._ask_for_turn()
    hs.idle_keepalive()
    assert io.kinds == ["data", "session-turn-release"]
    now[0] += .5
    level = phy.base_level(bw)
    body = phy.vara_body(b"peer reply", hs.caller, body_len=phy.body_size(bw, level))
    peer = OF.data_over_tx(body, over=1, bw=bw, level=level)
    now[0] += len(peer) / MK.FS
    io.receiving = False
    if bw == "500":
        hs._level_read = phy.LevelRead(phy.host_level(bw, level), (body,), located=1)
    hs.on_rx_audio(peer)
    hs._level_read = None
    assert received == [b"peer reply"], io.msgs
    assert hs.turn == VA._TURN_PEER and hs._txq == [b"second"]
    assert hs._answer_owed == VA._OWED_RELEASE
    assert "session-turn-request" not in io.kinds
    now[0] += 1.0
    hs.on_rx_audio(MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_RESPONDER_OVER_IDLE, bw)))
    assert io.kinds[-1] == "session-turn-request"
    assert hs.turn == VA._TURN_ASKED and hs._tx_pending is None


@pytest.mark.parametrize("after_release", [False, True], ids=["host-write", "release"])
def test_owed_ask_with_stuck_receive_gate_spends_idle_budget(monkeypatch, after_release):
    now = [100.0]
    monkeypatch.setattr(VA.time, "monotonic", lambda: now[0])
    io = IO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2300")
    hs.role, hs.caller, hs.called = "initiator", "W9SSJ", "W1AW"
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_OURS if after_release else VA._TURN_PEER
    if after_release:
        hs.send(b"first")
        hs.send(b"second")
        hs._took_control_burst()
        assert hs._ask_not_before == pytest.approx(100.295)
    io.receiving = True
    if not after_release:
        hs.send(b"second")
    assert hs._ask_owed and hs._txq == [b"second"]
    mark = len(io.kinds)
    if after_release:
        now[0] = 100.294
        hs.idle_keepalive()
        assert len(io.kinds) == mark
        assert hs._since_progress == 0
    now[0] = 100.296
    hs.idle_keepalive()
    assert io.kinds[mark:] == ["session-turn-request"]
    assert hs.turn == VA._TURN_ASKED and not hs._ask_owed
    assert hs._asked == 1 and hs._since_progress == 1
    for _ in range(VA._MAX_WITHOUT_PROGRESS + 1):
        now[0] += 12.0
        hs.idle_keepalive()
        if hs.state is not VA.VaraState.CONNECTED:
            break
    assert hs.state is VA.VaraState.DISCONNECTED, io.msgs
    assert hs._since_progress == VA._MAX_WITHOUT_PROGRESS
    assert io.receiving
