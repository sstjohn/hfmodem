# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A station handed the turn with nothing queued polls, as stock does.

KD7UHR, 40 m BW500, 2026-09-28 14:11Z: its L6 over (132 B, field 0x81) and L7
short close (66 B, 0x82) were read and answered, and its control burst handed us
the turn unasked. Our queue was empty, and we keyed session-drained at once and
again every ~11.6 s; KD7UHR answered none of it, CW-IDed and dropped.

Two stock 4.9s on the Wine bench, 2026-09-28, BW500 and BW2300: the station
handed the turn keys its own 11-symbol control burst, frame 1, 0.57 s after the
grant and every 12.06 s after that; the peer answers each with
session-idle-response; after seven polls the holder keys the disconnect.
Session-drained is keyed once, by a station whose own delivery has just drained.

Every burst here is generated. Nothing opens a device or keys a radio.
"""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varatone
from hfmodem.kestrel.tx import varahf2300_tx as TX, varaofdm_tx, varatone_tx
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

CALLER, CALLED = "W9SSJ", "KD7UHR"
FS = MK.FS
_BLOCK = 960
_KEEPALIVE_S = 10.0


class _Clock:
    t = 1000.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.t


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    _Clock.t = 1000.0
    monkeypatch.setattr(VA, "time", _Clock)


class _IO(VA.VaraIO):
    def __init__(self):
        self.msgs, self.sent, self.at, self.host = [], [], [], []

    def key(self, on): ...

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))
        self.at.append(_Clock.t)

    def tx_went_out(self): return True
    def pending(self): ...
    def connected(self, *a): ...
    def data(self, payload): self.host.append(bytes(payload))
    def log(self, msg): self.msgs.append(msg)


def _station(bw: str):
    io = _IO()
    hs = VA.VaraStationHandshake([CALLER], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", CALLED, CALLER
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    return hs, io


def _noise(seconds: float) -> np.ndarray:
    return np.random.default_rng(int(_Clock.t)).standard_normal(int(seconds * FS)) * 1e-4


def _feed(hs, x) -> None:
    for a in range(0, len(x), _BLOCK):
        _Clock.t += len(x[a:a + _BLOCK]) / FS
        hs.on_rx_stream(x[a:a + _BLOCK])


def _over(payload: bytes, host: int, field: int) -> np.ndarray:
    row = varatone.ROWS.get((500, host))
    frame = (varatone_tx.build_frame(payload, field, row, caller=CALLER) if row else
             varaofdm_tx.build_frame(payload, field, 500, host, caller=CALLER))
    return VA.OF.data_over_tx(frame[:-2], level=phy.level_key("500", host), bw="500",
                              lead=TX.VB6Rnd(4107106))


def _grant(bw: str) -> np.ndarray:
    return MK.synth_tone_pairs(VF.turn_grant(CALLED, bw))


#: Stock's idle poll: seven of them, the first gap 11.15 s and the rest 12.06 s
#: (bench-idle500 r1b, r2a, r4a).
_POLLS, _GAPS = 7, [11.15] + [12.06] * 5


def _poll(bw: str, role: int = 0) -> np.ndarray:
    seed = (852, 1080)[role]
    return MK.synth_tone_pairs(VF.CONNECTED_ACK_PREAMBLE + VF.answer_tail(CALLER, seed, bw, 1))


def _is_poll(hs, x) -> bool:
    return np.array_equal(x, _poll(hs.bw, hs.role == "responder"))


def _kd7uhr():
    """The arm to the grant: a high speed capable caller reads the L6 over and the
    L7 close, answers both, and is handed the turn. Returns the grant's first
    sample on the clock."""
    hs, io = _station("500")
    hs.high_speed = hs._link_high_speed = True
    hs.turn = VA._TURN_PEER
    hs._expect_level, hs._rx_read = 6, (6, 30.0)
    hs._lv_owed = hs._lv_prompt = True
    hs._key(False)
    _feed(hs, _noise(0.3))
    _feed(hs, _over(bytes(range(132)), 6, 0x81))
    _feed(hs, _noise(1.5))
    _feed(hs, _over(bytes(range(66)), 7, 0x82))
    _feed(hs, _noise(0.4))
    assert io.host == [bytes(range(132)), bytes(range(66))], io.msgs
    assert hs._answer_owed == VA._OWED_RELEASE, io.msgs
    return hs, io, _handed(hs, "500")


def _handed(hs, bw: str) -> float:
    at = _Clock.t
    _feed(hs, _grant(bw))
    _feed(hs, _noise(0.5))
    assert hs.turn == VA._TURN_OURS, hs.io.msgs
    return at


def _hold(hs, io, seconds: float, answer: np.ndarray | None = None) -> None:
    """The band quiet but for ``answer`` 0.08 s behind each poll of ours, and
    the driver as kestrel_connect's mail loop runs it: the poll on its own
    deadline, the keepalive 10 s after the link last moved."""
    seen, next_ka = (hs.progress, hs.idle_keyed), _Clock.t + _KEEPALIVE_S
    end, queued = _Clock.t + seconds, np.zeros(0)
    while _Clock.t < end and hs.state is VA.VaraState.CONNECTED:
        block, queued = (queued[:_BLOCK], queued[_BLOCK:]) if len(queued) else (
            _noise(_BLOCK / FS), queued)
        _feed(hs, block)
        sent = len(io.sent)
        due = getattr(hs, "idle_poll_due_in", lambda: None)()
        if due is not None and due <= 0:
            hs.idle_keepalive()
        elif (hs.progress, hs.idle_keyed) == seen and _Clock.t >= next_ka:
            hs.idle_keepalive()
        if (hs.progress, hs.idle_keyed) != seen:
            seen, next_ka = (hs.progress, hs.idle_keyed), _Clock.t + _KEEPALIVE_S
        if answer is not None and len(io.sent) > sent and _is_poll(hs, io.sent[-1]):
            queued = np.concatenate([np.zeros(int(0.08 * FS)), answer])


def _polls(hs, io) -> list[float]:
    return [t for x, t in zip(io.sent, io.at) if _is_poll(hs, x)]


def _drained(io) -> int:
    return sum("tx session-drained keyed" in m for m in io.msgs)


def _closed(io) -> bool:
    return any("tx session-disconnect-request keyed" in m for m in io.msgs)


def _idle_response(bw: str) -> np.ndarray:
    return MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_IDLE_RESPONSE, bw))


def test_kd7uhr_the_turn_handed_over_empty_is_held_and_polled():
    hs, io, granted = _kd7uhr()
    first = len(io.sent) - 1
    assert _is_poll(hs, io.sent[first]), io.msgs[-4:]
    assert 0.47 <= io.at[first] - granted <= 0.75
    assert _drained(io) == 0, io.msgs

    _hold(hs, io, 100.0)
    polls = _polls(hs, io)
    assert len(polls) == _POLLS, io.msgs
    assert np.allclose(np.diff(polls), _GAPS, atol=0.05), np.diff(polls)
    assert _drained(io) == 0, io.msgs
    assert _closed(io) and hs.state is VA.VaraState.DISCONNECTED
    assert not any("idle checks without protocol progress" in m for m in io.msgs)
    assert 83.0 <= io.at[-1] - granted <= 86.0, io.at[-1] - granted


def test_kd7uhr_an_answered_poll_is_progress_and_the_budget_holds():
    hs, io, granted = _kd7uhr()
    answer = _idle_response("500")
    _hold(hs, io, 70.0, answer)
    assert hs.state is VA.VaraState.CONNECTED
    assert sum("rx session-idle-response" in m for m in io.msgs) >= 5, io.msgs
    assert hs._since_progress == 0
    _hold(hs, io, 30.0, answer)
    assert len(_polls(hs, io)) == _POLLS and _closed(io), io.msgs
    assert _drained(io) == 0


def _ask(hs) -> np.ndarray:
    return MK.synth_tone_pairs(VF.control_bursts(CALLER, hs.bw)[0])


def _drained_responder(bw: str) -> np.ndarray:
    return MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_DRAINED_RESPONDER, bw))


def test_host_data_arriving_mid_poll_asks_with_frame_0_and_delivers_on_the_drained_answer():
    """Stock, bench-idlepoll-verify s2 (2026-09-29): data queued between polls went
    out as control frame 0 0.16 s after the queue; the callee answered 288/807
    0.58 s later and the over followed. Keyed straight, our over went unread and
    the callee timed out (q1, q2)."""
    hs, io, _ = _kd7uhr()
    hs._lead = TX.VB6Rnd(VF.lead_state(CALLER, False, "500"))
    _hold(hs, io, 20.0, _idle_response("500"))
    assert len(_polls(hs, io)) == 2
    n = len(io.sent)
    hs.send(b"FC EM 12345_KD7UHR 123 99 0\r")
    assert len(io.sent) == n + 1 and np.array_equal(io.sent[-1], _ask(hs)), io.msgs[-3:]
    assert not any(m.startswith("tx DATA over") for m in io.msgs)
    _feed(hs, _noise(0.1))
    _feed(hs, _drained_responder("500"))
    _feed(hs, _noise(0.5))
    assert any(m.startswith("tx DATA over") for m in io.msgs), io.msgs[-4:]
    assert hs.idle_poll_due_in() is None
    _hold(hs, io, 15.0)
    assert len(_polls(hs, io)) == 2, io.msgs


def test_an_unanswered_ask_from_a_held_turn_repeats_on_the_poll_cadence():
    hs, io, _ = _kd7uhr()
    _hold(hs, io, 13.0)
    n = len(io.sent)
    hs.send(b"FF\r")
    _hold(hs, io, 30.0)
    asks = [t for x, t in zip(io.sent[n:], io.at[n:]) if np.array_equal(x, _ask(hs))]
    assert len(asks) == 3 and np.allclose(np.diff(asks), _GAPS[1], atol=0.05), io.msgs
    assert not any(m.startswith("tx DATA over") for m in io.msgs)
    _hold(hs, io, 60.0)
    assert _closed(io)


def test_bw2300_grant_is_held_and_polled_the_same_way():
    hs, io = _station("2300")
    hs.turn = VA._TURN_PEER
    hs.tx_level = 3
    hs._key_over_answer(last=True, owes_release=False)
    granted = _handed(hs, "2300")
    # Keyed as soon as the stream names the grant, which at BW2300 is a scan later.
    assert _is_poll(hs, io.sent[-1])
    assert 0.47 <= io.at[-1] - granted <= 0.47 + VA._GRANT_FRESH_S
    _hold(hs, io, 100.0)
    polls = _polls(hs, io)
    assert len(polls) == _POLLS
    assert np.allclose(np.diff(polls), _GAPS, atol=0.05)
    assert _closed(io) and _drained(io) == 0
    assert not any("session-turn-release keyed" in m for m in io.msgs), io.msgs


@pytest.mark.parametrize("took", ["_took_responder_release", "_took_drained_handover"])
def test_an_empty_17_or_32_symbol_handover_still_goes_straight_back(took):
    """Only the 11-symbol grant, behind a drained peer's short close, is held and
    polled. The 17-symbol release of 2026-08-29 followed a full greeting over
    broken mid-word (KB5LZK, N5TW), and its on-air tests pin the release
    [test_unasked_handover]."""
    hs, io = _station("500")
    hs.turn = VA._TURN_PEER
    getattr(hs, took)()
    assert hs.turn == VA._TURN_PEER and hs.idle_poll_due_in() is None
    assert not any(_is_poll(hs, x) for x in io.sent)


def test_a_peer_turn_request_mid_poll_takes_the_turn_as_before():
    hs, io, _ = _kd7uhr()
    hs._took_turn_request()
    assert hs.turn == VA._TURN_PEER and hs.idle_poll_due_in() is None


def test_the_peers_answer_to_a_poll_is_not_the_answer_to_an_over():
    """The first poll drew a control burst on the bench, not the idle response."""
    hs, io, _ = _kd7uhr()
    n = len(io.sent)
    hs._took_control_burst()
    assert len(io.sent) == n and hs.turn == VA._TURN_OURS and _drained(io) == 0


def _delivered_and_drained():
    """Our delivery drains and we hand over with session-drained, once."""
    hs, io = _station("500")
    hs.turn = VA._TURN_OURS
    hs._rec500 = phy.level_key("500", 3)
    hs.send(b"FF\r")
    assert hs._tx_pending is not None
    hs._took_control_burst()
    assert _drained(io) == 1 and hs.turn == VA._TURN_PEER, io.msgs
    return hs, io


def _peer_polls(hs, io, n: int) -> None:
    """The callee holding the turn empty: its control stream, 1080 frame 1, every
    12.06 s, and the driver's keepalive 10 s after the link last moved."""
    peer_poll = MK.synth_tone_pairs(VF.control_bursts(CALLER, "500", frame=1)[1])
    seen, next_ka = (hs.progress, hs.idle_keyed), _Clock.t + _KEEPALIVE_S
    for _ in range(n):
        _feed(hs, peer_poll)
        for _ in range(int(11.57 * FS) // _BLOCK):
            _feed(hs, _noise(_BLOCK / FS))
            if (hs.progress, hs.idle_keyed) != seen:
                seen, next_ka = (hs.progress, hs.idle_keyed), _Clock.t + _KEEPALIVE_S
            elif _Clock.t >= next_ka:
                hs.idle_keepalive()
                next_ka = _Clock.t + _KEEPALIVE_S
        assert hs.state is VA.VaraState.CONNECTED, io.msgs[-4:]


def _keepalives(io, kind) -> int:
    return sum(f"tx {kind.name} keyed" in m for m in io.msgs)


def test_the_reverse_role_answers_every_poll_with_keepalive_b_and_waits_for_the_close():
    """Stock caller, bench r3a/r3b 2026-09-28: the callee polls seven times, 12.06 s
    apart; the caller answers each with keepalive-B and never closes; the callee
    closes with 288/125 at ~84 s."""
    hs, io = _delivered_and_drained()
    _peer_polls(hs, io, _POLLS)
    assert _keepalives(io, VF.SESSION_KEEPALIVE_B) == _POLLS, io.msgs
    assert _keepalives(io, VF.SESSION_KEEPALIVE_A) == 0
    assert _drained(io) == 1 and not _closed(io), io.msgs
    _feed(hs, MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_DISCONNECT_REQ_RESPONDER, "500")))
    _feed(hs, _noise(0.5))
    assert hs.state is VA.VaraState.DISCONNECTED, io.msgs[-3:]


def test_a_peer_that_stops_polling_is_still_closed_by_the_budget():
    hs, io = _delivered_and_drained()
    _peer_polls(hs, io, 2)
    for _ in range(VA._MAX_WITHOUT_PROGRESS + 2):
        hs.idle_keepalive()
    assert hs.state is not VA.VaraState.CONNECTED, io.msgs[-3:]


def test_the_release_below_the_base_is_the_drained_frame_and_above_it_the_release():
    for record, frame in ((2, VF.SESSION_DRAINED), (4, VF.SESSION_TURN_RELEASE),
                          (6, VF.SESSION_TURN_RELEASE)):
        hs, io = _station("500")
        hs.turn, hs._rec500 = VA._TURN_OURS, record
        assert hs._release_turn()
        assert io.msgs[0].startswith(f"tx {frame.name} keyed"), (record, io.msgs)


def test_the_poll_is_frame_one_of_our_roles_control_stream():
    for bw in ("500", "2300", "2750"):
        for role, seed in (("initiator", 852), ("responder", 1080)):
            burst = VF.control_bursts(CALLER, bw, frame=1)[role == "responder"]
            tail = burst[len(VF.CONNECTED_ACK_PREAMBLE):]
            assert tail == VF.answer_tail(CALLER, seed, bw, 1)


def test_pending_bytes_counts_an_over_awaiting_its_answer():
    """``vara_payload``'s second argument is the length: the callsign passed
    there raised a TypeError in the host adapter for any pending over."""
    hs, _ = _station("500")
    hs.turn = VA._TURN_OURS
    hs.send(b"FF\r")
    assert hs._tx_pending is not None and not hs._txq
    assert hs.pending_bytes == len(b"FF\r")


def _confirmed(bw: str = "500", heard_db: float | None = None):
    """The link just up: our confirm keyed, the turn the callee's."""
    hs, io = _station(bw)
    hs.state, hs.turn = VA.VaraState.CONNECTING, VA._TURN_PEER
    if heard_db is not None:
        hs._link_high_speed, hs._heard_db = True, heard_db
    hs._connected()
    assert any("tx session-confirm" in m for m in io.msgs), io.msgs
    return hs, io


def _confirm_32(hs) -> np.ndarray:
    confirm = next(k for k, lv in VA._CONFIRM_START.items() if lv == hs._confirm_start)
    kind = replace(VF.SESSION_KEEPALIVE_A, preadv=1 + 62 * ((confirm.preadv - 1) // 30))
    return MK.synth_burst(CALLED, VF.for_bw(kind, hs.bw))


def _callee(hs, frame: int) -> None:
    _feed(hs, _noise(0.1))
    _feed(hs, MK.synth_tone_pairs(VF.control_bursts(CALLER, hs.bw, frame)[1]))
    _feed(hs, _noise(0.6))


@pytest.mark.parametrize("bw,heard", [("500", None), ("500", 30.0), ("2300", None)])
def test_the_callees_ack_of_our_confirm_draws_the_32_symbol_confirm(bw, heard):
    """Stock's caller answers the callee's control burst behind its confirm with
    the same state at 32 symbols, 1550/373 behind 1551/181, 0.57 s after it
    (bench-idle500 r1b, r3a, r4a; bench-idlepoll-verify s2, s3). Without it the
    callee asked for the turn and never reached its poll (p1, p2)."""
    hs, io = _confirmed(bw, heard)
    n = len(io.sent)
    _callee(hs, 0)
    assert len(io.sent) == n + 1 and np.array_equal(io.sent[-1], _confirm_32(hs)), io.msgs[-3:]
    if heard:
        assert hs._confirm_start > 4 and not np.array_equal(
            io.sent[-1], MK.synth_burst(CALLED, VF.for_bw(VF.SESSION_KEEPALIVE_A, bw)))


def test_the_empty_callee_then_polls_and_each_poll_draws_keepalive_b():
    """bench-idlepoll-verify s3: confirm, ack, 1550/373, then the callee's poll
    every 12.06 s, each answered with keepalive-B."""
    hs, io = _confirmed()
    _callee(hs, 0)
    for _ in range(3):
        _callee(hs, 1)
    assert sum("tx session-keepalive-b keyed" in m for m in io.msgs) == 3, io.msgs
    assert not any("tx session-turn-release keyed" in m for m in io.msgs)


def test_data_queued_at_connect_asks_in_answer_to_the_callees_poll():
    """Stock caller with data, bench-idle500 r3a: confirm, ack, 1550/373, the
    callee's poll, 850/1 in answer, the callee's 11-symbol grant, the over. Ours
    asked across the callee's ack instead, and the two deadlocked (r3a, r3c)."""
    hs, io = _confirmed()
    hs.send(b"FF\r")
    assert not any("tx session-turn-request keyed" in m for m in io.msgs), io.msgs
    _callee(hs, 0)
    _callee(hs, 1)
    assert io.msgs[-1].startswith("tx session-turn-request keyed") or any(
        "tx session-turn-request keyed" in m for m in io.msgs[-2:]), io.msgs[-3:]
    assert hs.turn == VA._TURN_ASKED
    _feed(hs, MK.synth_tone_pairs(VF.turn_grant(CALLED, "500")))
    _feed(hs, _noise(0.5))
    assert any(m.startswith("tx DATA over") for m in io.msgs), io.msgs[-3:]
