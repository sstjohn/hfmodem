# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Frame-level loopback of besra's ARQ session engine.

Two `ArqSession`s (caller W9SSJ, responder K7ABC/listen) are cross-wired by an
in-memory relay: each session's `Transport.send` drops a real ARDOP frame into a
FIFO the harness drains into the other session's `on_receive`. A shared fake clock
is advanced by hand. No DSP, no radio, no wall-clock — the whole ARQ choreography
(connect handshake, stop-and-wait data with quality ACK/NAK, BREAK turnover,
DISC/END teardown, idle timeout) is exercised at the byte-frame layer.
"""

from __future__ import annotations

import logging
from collections import deque

import pytest

from hfmodem.besra.crc import session_id
from hfmodem.besra.host import protocol as P
from hfmodem.besra.arq.session import (ArqSession, BREAK, CONACK, DISC, END,
                                       _DATA_LADDER, _dataack_for, _datanak_for,
                                       _is_data, peer_quality,
                                       _CONNECT_TAIL_S)
from hfmodem.besra.frame import frame as F
from hfmodem.besra.phy.modulator import DEFAULT_LEADER_MS


class Relay:
    """A half-duplex ether: frames queue FIFO and are delivered on `pump`. It can
    corrupt the next data frame (deliver it ``ok=False``) for the NAK path."""

    def __init__(self) -> None:
        self.q: deque = deque()
        self.corrupt_next_data = False
        self.log: list[tuple[str, int]] = []   # (sender_call, frame_type)

    def pump(self, limit: int = 1000) -> None:
        steps = 0
        while self.q and steps < limit:
            dest, ft, payload, sid = self.q.popleft()
            ok = True
            if _is_data(ft) and self.corrupt_next_data:
                ok = False
                self.corrupt_next_data = False
            dest.on_receive(ft, payload, sid, ok)
            steps += 1
        assert steps < limit, "relay did not settle — probable frame loop"


class Wire:
    """One session's Transport, delivering into its peer's inbox."""

    def __init__(self, relay: Relay, sender_call: str) -> None:
        self._relay = relay
        self._call = sender_call
        self.dest: ArqSession | None = None

    def send(self, frame_type: int, payload: bytes, session_id: int) -> float:
        self._relay.log.append((self._call, frame_type))
        self._relay.q.append((self.dest, frame_type, bytes(payload), session_id))
        return 0.0                     # synchronous relay — no real transmit time


class Recorder:
    """Captures every Observer callback for assertions.

    `notices` keeps the host-visible connect notifications in order, since the
    documented contract is as much about sequence (PENDING, TARGET, CONNECTED)
    as about the individual lines."""

    def __init__(self) -> None:
        self.states: list[str] = []
        self.connects: list[tuple[str, int]] = []
        self.disconnects = 0
        self.received = bytearray()
        self.buffers: list[int] = []
        self.statuses: list[str] = []
        self.notices: list[str] = []

    def newstate(self, state: str) -> None:
        self.states.append(state)

    def connected(self, remote: str, bw: int) -> None:
        self.connects.append((remote, bw))
        self.notices.append(f"CONNECTED {remote} {bw}")

    def disconnected(self) -> None:
        self.disconnects += 1

    def data_received(self, kind: str, blob: bytes) -> None:
        assert kind == "ARQ"
        self.received += blob

    def buffer(self, nbytes: int) -> None:
        self.buffers.append(nbytes)

    def pending(self, cancel: bool = False) -> None:
        self.notices.append("CANCELPENDING" if cancel else "PENDING")

    def target(self, call: str) -> None:
        self.notices.append(f"TARGET {call}")

    def status(self, text: str) -> None:
        self.statuses.append(text)


class Pair:
    """A wired caller+responder on one relay and one fake clock."""

    def __init__(self, *, bandwidth: int = 500, timeout_s: float = 90.0) -> None:
        self.relay = Relay()
        self.obs_a = Recorder()
        self.obs_b = Recorder()
        wire_a = Wire(self.relay, "W9SSJ")
        wire_b = Wire(self.relay, "K7ABC")
        self.caller = ArqSession("W9SSJ", wire_a, self.obs_a,
                                 bandwidth=bandwidth, timeout_s=timeout_s)
        self.responder = ArqSession("K7ABC", wire_b, self.obs_b,
                                    bandwidth=bandwidth, timeout_s=timeout_s,
                                    listen=True)
        wire_a.dest = self.responder
        wire_b.dest = self.caller
        self.now = 0.0

    def tick(self, dt: float = 0.0) -> None:
        self.now += dt
        self.caller.tick(self.now)
        self.responder.tick(self.now)

    def pump(self) -> None:
        self.relay.pump()

    def connect(self) -> None:
        self.caller.connect("K7ABC")
        self.pump()


def test_connect_handshake_brings_both_up():
    pair = Pair()
    pair.connect()

    assert pair.caller.state == P.ArdopState.ISS
    assert pair.responder.state == P.ArdopState.IRS
    assert pair.caller.connected and pair.responder.connected
    assert pair.obs_a.connects == [("K7ABC", 500)]
    assert pair.obs_b.connects == [("W9SSJ", 500)]
    assert pair.caller._session == pair.responder._session != 0

    # The wire sequence: ConReq → ConAck → ConAck → DataAck.
    kinds = [ft for _, ft in pair.relay.log]
    from hfmodem.besra.arq import session as S
    assert kinds[0] in S.CONREQ_MAX.values()
    assert S._is_conack(kinds[1]) and S._is_conack(kinds[2])
    assert S._is_dataack(kinds[3])


def test_the_conack_states_the_leader_this_end_actually_sends():
    """A ConAck's three bytes are the received leader length in tens of ms, and
    besra measures no leader — it states a nominal one. The nominal is honest only
    because it is the leader besra itself transmits: to a besra peer it is exactly
    what a measurement would have returned, and to anyone else it is the station
    describing its own signal. Written as a bare 24 it was tied to nothing and read
    green at 0 and at 255, both of which the wire carries happily — 0 ms of leader
    is a frame nothing can acquire, 2.55 s of it is longer than the ARQ turnaround.

    A wrong number here does not break a live link: ardopcf's
    ``CalculateOptimumLeader`` (ARQ.c) has its whole body commented out and
    ``intCalcLeader`` is taken from the station's own ``LeaderLength``, so a real
    peer logs this field rather than timing against it. What it costs is a false
    statement about our signal, on every connect we answer.
    """
    pair = Pair()
    pair.caller.connect("K7ABC")
    # Hand the ConReq over rather than pumping, so the ConAck is still on the
    # relay with its payload attached (the log keeps frame types only).
    dest, ft, payload, sid = pair.relay.q.popleft()
    dest.on_receive(ft, payload, sid, True)

    _dest, conack, timing, _sid = pair.relay.q[-1]
    assert conack == CONACK[500]
    assert DEFAULT_LEADER_MS % 10 == 0, "the field is tens of ms, not ms"
    assert timing == bytes([DEFAULT_LEADER_MS // 10]) * 3


def test_expected_session_tracks_the_session_lifecycle():
    """What the demodulator's expected-session lane keys on: None while
    disconnected, the derived id from the moment a connect is initiated or
    answered, None again after teardown."""
    pair = Pair()
    assert pair.caller.expected_session is None
    assert pair.responder.expected_session is None

    pair.connect()
    from hfmodem.besra import crc
    sid = crc.session_id("W9SSJ", "K7ABC")
    assert pair.caller.expected_session == sid
    assert pair.responder.expected_session == sid

    pair.caller.abort()
    assert pair.caller.expected_session is None


def test_data_flows_iss_to_irs_with_ack():
    pair = Pair()
    pair.connect()

    pair.caller.queue_data(b"hello winlink")
    pair.pump()

    assert bytes(pair.obs_b.received) == b"hello winlink"
    assert pair.obs_a.buffers[-1] == 0          # TX queue drained on ACK
    from hfmodem.besra.arq import session as S
    assert any(S._is_dataack(ft) for who, ft in pair.relay.log if who == "K7ABC")


def test_nak_triggers_retransmit_then_succeeds():
    pair = Pair()
    pair.connect()

    pair.relay.corrupt_next_data = True         # first data frame decodes bad
    pair.caller.queue_data(b"retry me")
    pair.pump()

    from hfmodem.besra.arq import session as S
    naks = [ft for who, ft in pair.relay.log if who == "K7ABC" and S._is_datanak(ft)]
    data = [ft for who, ft in pair.relay.log if who == "W9SSJ" and S._is_data(ft)]
    assert naks, "IRS must NAK the corrupted frame"
    assert len(data) >= 2, "ISS must retransmit after the NAK"
    assert bytes(pair.obs_b.received) == b"retry me"
    assert pair.obs_a.buffers[-1] == 0


def test_break_turnover_reverses_the_link():
    pair = Pair()
    pair.connect()

    pair.responder.queue_data(b"reverse path")
    pair.pump()

    assert bytes(pair.obs_a.received) == b"reverse path"
    assert pair.responder.state == P.ArdopState.ISS
    assert pair.caller.state == P.ArdopState.IRS
    from hfmodem.besra.arq import session as S
    assert any(ft == S.BREAK for _, ft in pair.relay.log)


def test_graceful_disconnect_disc_end():
    """The close costs one DISC repeat: an IRS peer answers the second copy, never
    the first (`ArqSession._corroborated`)."""
    pair = Pair()
    pair.connect()

    pair.caller.disconnect()
    pair.pump()
    assert not pair.obs_b.disconnects, "a single DISC ended the session"
    for _ in range(6):
        pair.tick(0.5)
        pair.pump()

    assert pair.caller.state == P.ArdopState.DISC
    assert pair.responder.state == P.ArdopState.DISC
    assert pair.obs_a.disconnects == 1
    assert pair.obs_b.disconnects == 1
    from hfmodem.besra.arq import session as S
    assert any(ft == S.DISC for _, ft in pair.relay.log)
    assert any(ft == S.END for _, ft in pair.relay.log)


def test_idle_timeout_auto_disconnects():
    pair = Pair(timeout_s=30.0)
    pair.connect()
    assert pair.caller.connected

    pair.tick(60.0)                             # advance well past the timeout
    pair.pump()

    assert pair.caller.state == P.ArdopState.DISC
    assert pair.responder.state == P.ArdopState.DISC
    assert pair.obs_a.disconnects >= 1
    assert pair.obs_b.disconnects >= 1
    assert any("Timeout" in s for s in pair.obs_a.statuses)


def test_conreq_repeat_anchors_to_tx_end():
    """The connect-request repeat is scheduled from when the ConReq *finishes*
    transmitting, not when it starts — matching ardopcf's dttNextPlay (set after
    playback). Anchoring at send-start left only ~0.25 s of a 2.0 s interval to
    hear a 0.70 s ConAck behind a 1.75 s ConReq: the on-air "not waiting for a
    reply between transmissions" bug."""
    class _DurTx:
        def send(self, ft, payload, sid) -> float:
            return 1.75                         # a ConReq500 is ~1.75 s on air

    s = ArqSession("W9SSJ", _DurTx(), Recorder(), bandwidth=500)
    s.tick(10.0)
    s.connect("KC9GHZ")
    assert s._next_conreq == 10.0 + 1.75 + s._connect_interval
    # a zero-length transmit (the synchronous-air case) reduces to the old timing
    s2 = ArqSession("W9SSJ", type("Z", (), {"send": lambda *a: 0.0})(), Recorder(), bandwidth=500)
    s2.tick(5.0); s2.connect("KC9GHZ")
    assert s2._next_conreq == 5.0 + s2._connect_interval


class _Calls:
    """A caller-only Transport that logs what was keyed and charges air time for it."""

    def __init__(self, dur: float = 1.75) -> None:
        self.sent: list[int] = []
        self._dur = dur

    def send(self, ft, payload, sid) -> float:
        self.sent.append(ft)
        return self._dur


def _called_out(repeats: int = 1) -> tuple[ArqSession, _Calls, Recorder, float]:
    """A caller whose ConReq budget has just run out, and the clock it ran out at."""
    tx, obs = _Calls(), Recorder()
    s = ArqSession("W9SSJ", tx, obs, bandwidth=500)
    s.tick(0.0)
    s.connect("KY4RY", repeats=repeats)
    now = 0.0
    for _ in range(repeats + 1):
        now += 1.75 + s._connect_interval
        s.tick(now)
    return s, tx, obs, now


def test_the_receive_window_outlives_the_calling():
    """A gateway that answers late is still answered.

    Every failed connect of 2026-08-14/15 recorded exactly 22.6 s and ended on our
    own last ConReq, because the budget running out tore the session down — which
    retires `expected_session`, drops the host link and stops the capture. So a
    slow gateway could not be heard live *or* found in the recording afterwards,
    and the one KY4RY answer of that evening cleared the edge by 0.5 s. `_CONNECT_TAIL_S`
    sizes the tail off the peer's own answering cycle; here it only has to outlive
    the 2.0 s window that was.
    """
    s, tx, obs, ran_out = _called_out()
    keyed = len(tx.sent)

    s.tick(ran_out + 10.0)                          # five times the old window
    assert len(tx.sent) == keyed, "the tail keyed something"
    assert s.state == P.ArdopState.ISS
    assert s.expected_session is not None, "the demodulator's corroboration is gone"

    s.on_receive(CONACK[500], bytes([24]) * 3, s.expected_session, True)
    assert tx.sent[-1] == CONACK[500]               # the confirming ConAck
    s.on_receive(0xFF, b"", s.expected_session, True)
    assert s.connected and obs.connects == [("KY4RY", 500)]


def test_the_connect_verdict_waits_for_the_tail_and_does_not_pre_announce_it():
    """The verdict lands at the tail's end, and what is said before then must not
    read as one: a mail run ends on any STATUS carrying "FAILED"
    (`host.run_server._mail_observer`), and an early one would close the link and
    the capture at the exact instant this exists to keep them open."""
    s, tx, obs, ran_out = _called_out()

    assert not any("FAILED" in t for t in obs.statuses)
    s.tick(ran_out + _CONNECT_TAIL_S - 0.1)
    assert s.state == P.ArdopState.ISS
    assert not any("FAILED" in t for t in obs.statuses)

    s.tick(ran_out + _CONNECT_TAIL_S)
    assert s.state == P.ArdopState.DISC
    assert obs.statuses[-1] == "CONNECT TO KY4RY FAILED!"


def test_the_conack_leg_keeps_its_repeats_and_gets_the_same_tail():
    """The confirming-ConAck leg spends the same budget it always did — ten
    repeats — and then listens rather than quitting on its own last burst."""
    tx, obs = _Calls(0.70), Recorder()
    s = ArqSession("W9SSJ", tx, obs, bandwidth=500)
    s.tick(0.0)
    s.connect("KY4RY")
    s.on_receive(CONACK[500], bytes([24]) * 3, s.expected_session, True)
    conacks = tx.sent.count(CONACK[500])

    now = 0.0
    for _ in range(11):
        now += 0.70 + s._connect_interval
        s.tick(now)
    assert tx.sent.count(CONACK[500]) == conacks + 10
    assert s.state == P.ArdopState.ISS

    s.tick(now + _CONNECT_TAIL_S)
    assert s.state == P.ArdopState.DISC


def test_inbound_connect_announces_pending_then_target_before_connected():
    """The notification order a listening host is entitled to (spec §4, §5): a
    heard ConReq pauses scanning, TARGET names which of our calls was dialled,
    and only then does the link come up. Pat's listener keys on TARGET arriving
    ahead of CONNECTED."""
    pair = Pair()
    pair.connect()

    assert pair.obs_b.notices == ["PENDING", "TARGET K7ABC", "CONNECTED W9SSJ 500"]


def test_connect_for_another_station_cancels_the_pending():
    """A ConReq addressed elsewhere resumes the host's scan through
    CANCELPENDING — its own notification, not a line of STATUS prose."""
    from hfmodem.besra.arq import session as S
    from hfmodem.besra.frame import callsign

    pair = Pair()
    payload = callsign.pack_callsign("W9SSJ") + callsign.pack_callsign("N0ELS")
    pair.responder.on_receive(S.CONREQ_MAX[500], payload, 0xFF, True)

    assert pair.obs_b.notices == ["PENDING", "CANCELPENDING"]
    assert pair.obs_b.statuses == []
    assert pair.responder.state == P.ArdopState.DISC


def test_purge_empties_the_queue_without_ending_the_session():
    """PURGEBUFFER has no session effect (spec §3.1): the queue empties, the link
    stays up. The frame already in flight survives — stop-and-wait clears it on
    the peer's ACK, and dropping it here would skip past undelivered bytes."""
    pair = Pair()
    pair.connect()
    pair.caller.queue_data(b"K" * 4000)
    in_flight = pair.caller._in_process
    assert in_flight > 0

    pair.caller.purge()

    assert pair.caller.queued == in_flight
    assert pair.caller.connected and pair.caller.state == P.ArdopState.ISS
    assert pair.obs_a.disconnects == 0
    assert pair.obs_a.buffers[-1] == in_flight

    # The outstanding frame still completes; nothing is queued behind it.
    pair.pump()
    assert pair.caller.queued == 0
    assert bytes(pair.obs_b.received) == b"K" * in_flight


# -- mail-shaped conversations -------------------------------------------
#
# A mail layer answers the data it receives from inside the delivery callback:
# read a proposal, write the response, hand the link over — many times per
# session. These tests drive that shape, which no single-over test reaches.

def _answer_with(obs, session, script: dict[bytes, bytes]) -> None:
    """Wrap an observer so received data is answered from inside `data_received`
    — the moment a real host writes, and the moment the session is least ready
    for it (mid-delivery, possibly still IRS_FROM_ISS)."""
    orig = obs.data_received

    def answering(kind: str, blob: bytes) -> None:
        orig(kind, blob)
        reply = script.pop(bytes(blob), None)
        if reply is not None:
            session.queue_data(reply)

    obs.data_received = answering


def test_reply_queued_inside_the_delivery_callback_is_not_stranded():
    """Three overs, each written in response to the one just read. The second
    reply is queued while the caller is still IRS_FROM_ISS — the sub-state a
    station holds on the first frame after conceding the link — and must still
    turn the link around and arrive exactly once."""
    pair = Pair()
    pair.connect()
    _answer_with(pair.obs_b, pair.responder, {b"over-1": b"over-2"})
    _answer_with(pair.obs_a, pair.caller, {b"over-2": b"over-3"})

    pair.caller.queue_data(b"over-1")
    pair.pump()
    for _ in range(30):
        pair.tick(1.0)
        pair.pump()

    assert bytes(pair.obs_a.received) == b"over-2"
    assert bytes(pair.obs_b.received) == b"over-1over-3"
    assert pair.caller.queued == 0 and pair.responder.queued == 0


def test_ack_precedes_break_when_the_reply_is_queued_in_callback():
    """The ACK for a delivered frame goes out before the BREAK that answers it.
    BREAK first makes the old ISS concede with its frame unacknowledged, and
    stop-and-wait replays those bytes as duplicates on a later over."""
    pair = Pair()
    pair.connect()
    _answer_with(pair.obs_b, pair.responder, {b"ping": b"pong"})

    pair.caller.queue_data(b"ping")
    pair.pump()

    from hfmodem.besra.arq import session as S
    log = pair.relay.log
    i_data = next(i for i, (who, ft) in enumerate(log)
                  if who == "W9SSJ" and _is_data(ft))
    after = [ft for who, ft in log[i_data + 1:] if who == "K7ABC"]
    i_break = after.index(S.BREAK)
    assert any(S._is_dataack(ft) for ft in after[:i_break]), after


def test_reply_after_a_reversal_turns_the_link_again():
    """A host that answers after the delivery completes (not inside the
    callback) still gets the link back on an already-reversed session."""
    pair = Pair()
    pair.connect()
    pair.responder.queue_data(b"seize")             # responder takes the link
    pair.pump()
    assert bytes(pair.obs_a.received) == b"seize"
    assert pair.caller.state == P.ArdopState.IRS

    pair.caller.queue_data(b"late reply")           # host speaks; ISS is idle
    pair.pump()
    for _ in range(20):
        pair.tick(1.5)
        pair.pump()

    assert bytes(pair.obs_b.received) == b"late reply"


def test_bytes_queued_mid_handover_ride_the_idle_chirp():
    """The threaded-host race: bytes land in the instant between conceding the
    link and the new ISS's first frame (sub-state IRS_FROM_ISS). If the new ISS
    then only chirps IDLE, the pending bytes must still seize the link."""
    from hfmodem.besra.arq.session import _Sub
    pair = Pair()
    pair.connect()
    pair.responder._sub = _Sub.IRS_FROM_ISS         # as if mid-handover
    pair.responder.queue_data(b"raced bytes")
    for _ in range(10):
        pair.tick(1.5)                              # caller chirps IDLE
        pair.pump()

    assert bytes(pair.obs_a.received) == b"raced bytes"


def test_break_repeats_until_the_concession_is_heard():
    """IRStoISS is half a turnover. If the peer's conceding ACK is lost, the
    BREAK repeats, and an IRS hearing a repeated BREAK re-ACKs — otherwise both
    ends sit silent until the session timeout."""
    from hfmodem.besra.arq import session as S
    pair = Pair()
    pair.connect()
    pair.responder.queue_data(b"seize")
    dest, ft, payload, sid = pair.relay.q.popleft()
    assert ft == S.BREAK
    dest.on_receive(ft, payload, sid, True)         # caller concedes...
    pair.relay.q.clear()                            # ...but its ACK is lost
    assert pair.responder.state == P.ArdopState.IRStoISS

    for _ in range(6):
        pair.tick(1.0)
        pair.pump()

    assert bytes(pair.obs_a.received) == b"seize"
    # The seizure completed: the responder took the link, sent, and (with an
    # empty queue) may already be idling — any connected sending state is a pass.
    assert pair.responder.state in (P.ArdopState.ISS, P.ArdopState.IDLE)


def _yield_and_lose_the_ack(pair) -> int:
    """Drive the responder into IRStoISS and swallow the caller's concession, so
    the caller sits as a conceded IRS hearing BREAKs it cannot answer. Returns the
    session id the peer's BREAKs carry."""
    from hfmodem.besra.arq import session as S
    pair.responder.queue_data(b"seize")
    dest, ft, payload, sid = pair.relay.q.popleft()
    assert ft == S.BREAK
    dest.on_receive(ft, payload, sid, True)
    pair.relay.q.clear()
    assert pair.caller.state == P.ArdopState.IRS
    return sid


def test_a_peer_that_never_takes_the_link_is_conceded_to_a_bounded_number_of_times():
    """The KY4RY stall of 2026-08-14: the gateway broke 16 times over 84 seconds
    and this end answered every one. Each answer refreshed the liveness clock, so
    the 90 s session timeout was pushed out of reach and nothing but the gateway
    giving up could end it — 84 s in which it repeated its whole greeting, and the
    mail parser above was handed two copies concatenated. The concession is
    budgeted: the ACKs stop, and the link is dropped for the host to retry."""
    from hfmodem.besra.arq import session as S
    pair = Pair()
    pair.connect()
    sid = _yield_and_lose_the_ack(pair)
    mark = len(pair.relay.log)

    for _ in range(40):                             # 80 s of BREAKs on the peer's clock
        pair.tick(2.0)
        pair.caller.on_receive(S.BREAK, b"", sid, True)
        pair.relay.q.clear()                        # every concession is lost too

    acks = [ft for who, ft in pair.relay.log[mark:]
            if who == "W9SSJ" and S._is_dataack(ft)]
    assert len(acks) == S._CONCESSION_BUDGET
    assert pair.caller.state == P.ArdopState.DISC
    assert pair.obs_a.disconnects == 1
    assert any("NOT COMPLETED" in s for s in pair.obs_a.statuses)


def test_a_concession_on_the_last_of_the_budget_still_turns_the_link_around():
    """The budget must not cost a peer that is merely losing ACKs. A seizure whose
    concession lands on the last try in budget hands the link over exactly as one
    answered first time does."""
    from hfmodem.besra.arq import session as S
    pair = Pair()
    pair.connect()
    sid = _yield_and_lose_the_ack(pair)

    for _ in range(S._CONCESSION_BUDGET - 1):
        pair.tick(2.0)
        pair.caller.on_receive(S.BREAK, b"", sid, True)
        pair.relay.q.clear()

    for _ in range(6):                              # the channel comes back
        pair.tick(2.0)
        pair.pump()

    assert bytes(pair.obs_a.received) == b"seize"
    assert pair.caller.connected
    assert pair.obs_a.disconnects == 0


def test_the_concession_budget_resets_when_the_peer_takes_the_link():
    """The count is consecutive concessions, not concessions ever: a peer that
    breaks, is heard, sends, and later breaks again gets the whole budget the
    second time — the budget's worth of BREAKs plus the one that only corroborates
    them, since the peer's data frame in between made every earlier BREAK stale."""
    from hfmodem.besra.arq import session as S
    pair = Pair()
    pair.connect()
    sid = _yield_and_lose_the_ack(pair)

    for _ in range(S._CONCESSION_BUDGET - 1):
        pair.tick(2.0)
        pair.caller.on_receive(S.BREAK, b"", sid, True)
        pair.relay.q.clear()
    for _ in range(6):
        pair.tick(2.0)
        pair.pump()
    assert bytes(pair.obs_a.received) == b"seize"

    mark = len(pair.relay.log)
    for _ in range(S._CONCESSION_BUDGET + 1):
        pair.tick(2.0)
        pair.caller.on_receive(S.BREAK, b"", sid, True)
        pair.relay.q.clear()

    acks = [ft for who, ft in pair.relay.log[mark:]
            if who == "W9SSJ" and S._is_dataack(ft)]
    assert len(acks) == S._CONCESSION_BUDGET
    assert pair.caller.connected


def test_disc_while_seizing_still_tears_down():
    """A peer may answer our BREAK with a disconnect (mail done, gateway
    leaving). IRStoISS answers DISC with END and tears down rather than
    repeating BREAK at a station that is gone."""
    pair = Pair()
    pair.connect()
    pair.responder.queue_data(b"unsent")
    pair.relay.q.clear()                            # the BREAK never arrives
    pair.caller.disconnect()
    pair.pump()

    assert pair.responder.state == P.ArdopState.DISC
    assert pair.obs_b.disconnects == 1


# -- session-id addressing (ardopcf decodes the header against the id it is
# party to, SoundInput.c ComputeDecodeDistance; besra decodes (type, session)
# jointly, so the equivalent filter is the ARQ layer's) ------------------------

def test_foreign_disc_after_a_failed_connect_is_not_answered():
    """Counterexample from the air (2026-08-05, 7103.5 kHz): after our connect
    to KC9GHZ failed, a third-party QSO's DISC (session 0x7C) was answered with
    an END bearing our stale session id — a transmission into someone else's
    teardown. A DISC that is not ours must not key the transmitter; one bearing
    our last session id still earns the rule-1.5 END."""
    pair = Pair()
    pair.caller.connect("K7ABC", repeats=1)         # nobody answers
    pair.relay.q.clear()
    for _ in range(4):
        pair.tick(2.5)
    pair.tick(_CONNECT_TAIL_S)                      # and the listening tail out
    assert pair.caller.state == P.ArdopState.DISC
    ours = pair.caller._session

    sent = len(pair.relay.log)
    pair.caller.on_receive(DISC, b"", 0x7C, True)   # someone else's teardown
    assert len(pair.relay.log) == sent

    pair.caller.on_receive(DISC, b"", ours, True)   # our old peer, END lost
    assert pair.relay.log[-1] == ("W9SSJ", END)


def test_a_session_that_has_never_connected_is_party_to_nobody():
    """The hole the rule-1.5 END left open until a session id existed to key it on.

    ``_session`` started at 0, which is not a spare value: `crc.session_id` remaps
    the reserved 0xFF onto 0x00, so zero is the *most* likely id a real call pair
    can hash to — 0.77% of them, measured over 200,000 random pairs against an
    even 0.39%. ``NS0A`` calling ``K5DAT-13`` is one of them, and this station has
    worked both. A besra that had not connected yet therefore answered their
    teardown with an END, which is exactly the transmission the filter above
    exists to stop.

    Which is why the sweep runs over every id rather than that one collision.
    Naming a single id showed only that the starting value was not *that* id: with
    ``_session`` starting at 0x2A this read green, and 0x2A is as ordinary a
    session as 0 is. So is 0xFF, which every ConReq, Ping and ID frame on the air
    carries and the demodulator hands up routinely. The property is that no byte a
    peer can stamp on a frame is a value this end holds before it has connected.
    """
    assert session_id("NS0A", "K5DAT-13") == 0, "the collision this test rests on"

    for wire_id in range(0x100):
        pair = Pair()
        assert pair.caller.state == P.ArdopState.DISC
        assert pair.caller.expected_session is None

        pair.caller.on_receive(DISC, b"", wire_id, True)

        assert pair.relay.log == [], (
            f"answered a teardown stamped {wire_id:#04x} between two other stations")


def test_foreign_conack_does_not_seize_the_caller():
    """A ConAck stamped with another session's id is another caller's grant.
    Consuming it would flip us into the confirming-ConAck leg with nobody on
    the other end (and take the session bandwidth from a stranger's frame)."""
    pair = Pair()
    pair.caller.connect("K7ABC")
    pair.relay.q.clear()                            # the responder never hears it

    sent = len(pair.relay.log)
    pair.caller.on_receive(CONACK[500], bytes([24]) * 3, 0x7C, True)
    assert len(pair.relay.log) == sent

    pair.caller.on_receive(CONACK[500], bytes([24]) * 3,
                           pair.caller.expected_session, True)
    assert pair.relay.log[-1] == ("W9SSJ", CONACK[500])


def test_foreign_dataack_does_not_clear_outstanding_data():
    """A DATAACK from another session must not delete bytes our peer has never
    acknowledged."""
    pair = Pair()
    pair.connect()
    pair.caller.queue_data(b"hold these")
    pair.relay.q.clear()                            # the DATA frame is lost
    queued = pair.caller.queued

    pair.caller.on_receive(0xEC, b"", 0x7C, True)   # a stranger's ACK
    assert pair.caller.queued == queued

    pair.caller.on_receive(0xEC, b"", pair.caller.expected_session, True)
    assert pair.caller.queued == 0


# -- transmit rate selection --------------------------------------------------

FSK_200, PSK_200S, PSK_200 = [base for base, _ in _DATA_LADDER[200]]


class Tap:
    """A transport that keeps what it is handed instead of relaying it, so a test
    can answer each frame with a quality of its own choosing."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, bytes]] = []

    def send(self, frame_type: int, payload: bytes, session_id: int) -> float:
        self.sent.append((frame_type, bytes(payload)))
        return 0.0

    @property
    def data(self) -> list[int]:
        return [ft for ft, _ in self.sent if _is_data(ft)]


class Iss:
    """A connected ISS whose peer is scripted frame by frame."""

    #: Long enough to clear the stale-ACK guard a retransmit arms (one turnaround),
    #: short enough that the 2.0 s repeat timer never puts a frame back on the air —
    #: so what the tap holds is what the protocol chose to send.
    STEP_S = 1.5

    def __init__(self, bandwidth: int = 200) -> None:
        self.tap = Tap()
        self.obs = Recorder()
        self.now = 0.0
        self.session = ArqSession("W9SSJ", self.tap, self.obs, bandwidth=bandwidth)
        self.session.connect("K7ABC")
        self.sid = self.session.expected_session
        self.session.on_receive(CONACK[bandwidth], bytes([24]) * 3, self.sid, True)
        self.session.on_receive(_dataack_for(100), b"", self.sid, True)
        self.tap.sent.clear()

    def _advance(self) -> None:
        self.now += self.STEP_S
        self.session.tick(self.now)

    def ack(self, quality: int) -> None:
        self._advance()
        self.session.on_receive(_dataack_for(quality), b"", self.sid, True)

    def nak(self, quality: int) -> None:
        self._advance()
        self.session.on_receive(_datanak_for(quality), b"", self.sid, True)


def test_the_peers_grade_of_our_transmission_is_readable_off_the_ack_type():
    """Q = 38 + 2·code, over both ACK and NAK ranges and nothing else. This is the
    whole of ARDOP's transmit-rate feedback, and it lived in a frame type the log
    printed as the bare word DATAACK for all 32 of its codes."""
    for q in range(38, 101, 2):
        assert peer_quality(_dataack_for(q)) == q
        assert peer_quality(_datanak_for(q)) == q
    assert peer_quality(BREAK) is None and peer_quality(FSK_200) is None


def test_the_200_hz_ladder_climbs_a_rung_on_every_two_good_acks():
    """4FSK.200.50S → 4PSK.200.100S → 4PSK.200.100, on the reference's rule: the
    exponential average of the peer's grades over the rung's threshold, and at
    least two ACKs since the last shift. WW2MI graded 32 of our frames on
    2026-08-23 at a median 96 and a minimum 84, every one of them clear of the
    first rung's bar of 82, while this end sat at 4.9 B/s for the whole session."""
    iss = Iss()
    iss.session.queue_data(b"x" * 300)
    for _ in range(5):
        iss.ack(96)

    assert iss.tap.data == [FSK_200, FSK_200 + 1, PSK_200S, PSK_200S + 1,
                            PSK_200, PSK_200 + 1]


def test_the_ladder_does_not_climb_for_a_tail_that_already_fits():
    """The reference's first brake: 40 bytes go in three 16-byte frames whatever
    the grades say, so there is no faster mode to reach for."""
    iss = Iss()
    iss.session.queue_data(b"x" * 40)
    for _ in range(3):
        iss.ack(100)

    assert set(iss.tap.data) == {FSK_200, FSK_200 + 1}
    assert iss.session.queued == 0


def _held(caplog) -> list[str]:
    """The reasons the ladder gave for staying where it is, in order."""
    return [r.getMessage() for r in caplog.records if "rate holds" in r.getMessage()]


def test_every_way_of_holding_a_rung_names_itself_in_the_record(caplog):
    """Four brakes and a bar all read the same from outside — a rung that did not
    move — and on 2026-08-26 the W6IDS logs held ten ACKs, one shift on one
    session and none on the next, with nothing to say which of the five was the
    reason. Each one states itself now, at the instant it decides."""
    caplog.set_level(logging.INFO, logger="hfmodem.besra.arq.session")

    iss = Iss()
    iss.session.queue_data(b"x" * 400)
    iss.ack(96)
    assert _held(caplog)[-1].endswith("2 ACKs needed to leave a rung")

    iss = Iss()
    iss.session.queue_data(b"x" * 400)
    iss.ack(80)
    iss.ack(80)
    assert _held(caplog)[-1].endswith("the bar is 82")

    iss = Iss()
    iss.session.queue_data(b"x" * 40)
    iss.ack(100)
    iss.ack(100)
    assert _held(caplog)[-1].endswith("the 8 B tail already fits one 16 B frame")

    iss = Iss()
    iss.session.queue_data(b"x" * 400)
    iss.ack(96)
    iss.ack(96)
    iss.nak(60)
    iss.ack(96)
    iss.ack(96)
    assert iss.session._rung == 0
    assert _held(caplog)[-1].endswith(
        "4PSK.200.100S failed when reached, 5 ACKs needed to retry it")

    iss = Iss()
    iss.session.queue_data(b"x" * 600)
    for _ in range(7):
        iss.ack(100)
    assert iss.session._rung == len(_DATA_LADDER[200]) - 1
    assert _held(caplog)[-1].endswith("no rung above it")


def test_the_w6ids_fetch_held_its_rung_on_the_tail_and_one_nak_put_it_there():
    """`~/ardop-day-08-w6ids.log` 08:40 and `~/ardop-day-12-w6ids.log` 10:26, two
    BW500 fetches to the same gateway forty minutes apart, each carrying the same
    four-frame B2F stint. Day 08 was graded 92 then 90 and climbed. Day 12 was
    graded 88, NAKed at 82, then 96 and 90 — a better average at the deciding ACK
    — and never left 4FSK.200.50S.

    The NAK is the whole difference, and not because it dropped a rung (there is
    none below the base): it zeroed the ACK count, so the two ACKs that earn a
    shift did not accrue until three of the four frames were gone. By then the
    tail brake was right — one frame left is nothing to climb for. On a fetch,
    where a stint is a handful of 16 B command lines, the window in which a shift
    can be both earned and worth taking is a single ACK wide.
    """
    def replay(grades) -> Iss:
        iss = Iss(bandwidth=500)
        iss.session.queue_data(b"x" * 52)
        for kind, q in grades:
            (iss.ack if kind == "ack" else iss.nak)(q)
        return iss

    day08 = replay((("ack", 92), ("ack", 90)))
    assert day08.session._rung == 1, "the day-08 stint climbed and this must too"

    day12 = replay((("ack", 88), ("nak", 82), ("ack", 96), ("ack", 90)))
    assert day12.session._avg_quality == 91 > _DATA_LADDER[500][0][1]
    assert day12.session._acks_at_rung == 2
    assert day12.session._rung == 0, "day 12 held its rung"
    assert day12.session.queued == 4


def test_a_nak_zeroes_the_acks_earned_at_the_rung_as_the_reference_does():
    """Asked after the two W6IDS fetches above: the ACK count zeroed at
    `_rx_iss`'s DATANAK is what pushed day 12's earn-point past three of its four
    frames, and on a stint that short it can decide the whole fetch. Whether it is
    ours to change is the question, and it is not — both references do it.

    `ARQ.c` ISSData, DataNAK branch: `intNAKctr++`, `ComputeQualityAvg`,
    `Gearshift_9()`, then `intACKctr = 0` — outside the `if (intShiftUpDn != 0)`
    that follows, so the reset lands on every NAK for a data frame and not only on
    the ones that shift a rung. `ArdopGearshift.RecordNak` in the C# reference is
    the same statement in the same order. Gearshift_9 clears both counters again
    wherever it does shift, up or down.

    So the counter is the reference's and the two ACKs it costs are its price.
    `_shift_down` used to be the place besra was genuinely stricter and is not any
    more; neither rule reaches day 12, whose NAK was at the base rung with nothing
    below it.
    """
    iss = Iss(bandwidth=500)
    iss.session.queue_data(b"x" * 400)
    iss.ack(96)
    assert iss.session._acks_at_rung == 1
    iss.nak(82)
    assert iss.session._acks_at_rung == 0
    assert iss.session._rung == 0, "the base rung has none below it to drop to"


def test_the_w6ids_fetch_says_in_the_log_which_brake_held_it(caplog):
    """The line the 10:26:23 record did not have."""
    caplog.set_level(logging.INFO, logger="hfmodem.besra.arq.session")
    iss = Iss(bandwidth=500)
    iss.session.queue_data(b"x" * 52)
    for kind, q in (("ack", 88), ("nak", 82), ("ack", 96), ("ack", 90)):
        (iss.ack if kind == "ack" else iss.nak)(q)

    assert _held(caplog)[-1] == (
        "rate holds 4FSK.200.50S (peer quality avg 91 over 2 ACKs): "
        "the 4 B tail already fits one 16 B frame")


def test_a_rung_that_failed_the_moment_we_reached_it_waits_five_acks():
    """The reference's second brake. Two ACKs took us up and a NAK brought us
    straight back; the next two ACKs must not spend the frame again."""
    iss = Iss()
    iss.session.queue_data(b"x" * 400)
    iss.ack(96)
    iss.ack(96)
    assert iss.session._rung == 1

    iss.nak(60)
    assert iss.session._rung == 0

    for _ in range(4):
        iss.ack(96)
    assert iss.session._rung == 0, "the failed rung was tried again on two ACKs"

    iss.ack(96)
    assert iss.session._rung == 1


def test_one_nak_above_the_base_mode_drops_a_rung():
    """`DownNAKS` is 1 on a rung that has never ACKed anything, and a rung two ACKs
    took us up to has not: the ACKs that earned the climb were the rung below's.
    The base rung has nowhere to go and just resends."""
    iss = Iss()
    iss.session.queue_data(b"x" * 400)
    iss.ack(96)
    iss.ack(96)
    assert iss.session._rung == 1

    iss.nak(70)
    assert iss.session._rung == 0
    iss.nak(70)
    assert iss.session._rung == 0
    assert iss.tap.data[-1] in (FSK_200, FSK_200 + 1)


def test_a_rung_that_has_carried_a_frame_takes_two_naks_to_leave():
    """`Gearshift_9` reads `DownNAKS` off `ModeHasWorked`: a rung that has ACKed
    something is left on the second consecutive NAK, not the first, and an ACK in
    between puts the count back to nothing. besra dropped on the first for its whole
    life and never once did it on the air — every inbound DATANAK this station has
    logged arrived at the base rung, and the whole record holds one rate change, a
    climb. What the divergence costs is a frame: scripted at BW500 over 600 B with
    one isolated NAK on a rung that had ACKed, the reference's rule delivers all
    600 B in twelve frames where this one delivered 496 in thirteen.
    """
    def climbed() -> Iss:
        iss = Iss()
        iss.session.queue_data(b"x" * 400)
        iss.ack(96)
        iss.ack(96)          # the climb; rung 1 has ACKed nothing yet
        iss.ack(96)          # now it has
        assert iss.session._rung == 1
        return iss

    iss = climbed()
    iss.nak(70)
    assert iss.session._rung == 1, "one NAK left a rung the path had been carrying"
    iss.nak(70)
    assert iss.session._rung == 0

    iss = climbed()
    iss.nak(70)
    iss.ack(96)
    iss.nak(70)
    assert iss.session._rung == 1, "the NAKs were not consecutive"


def test_a_nak_drops_a_rung_and_re_cuts_the_outstanding_frame_to_fit_it():
    """The 64-byte 4PSK.200.100 chunk does not fit the 16-byte mode below it, so
    the retransmit is re-cut from the queue head rather than replayed. Safe
    because bytes leave the queue only on an ACK — and the ACK that follows must
    delete the 16 the peer was actually sent, not the 64 it was not."""
    iss = Iss()
    iss.session.queue_data(bytes(range(256)))
    for _ in range(4):
        iss.ack(96)
    assert iss.session._rung == 2

    ft, big = iss.tap.sent[-1]
    assert ft == PSK_200
    assert len(big) == F.FRAMES[PSK_200].net_payload == 64

    queued = iss.session.queued
    iss.nak(60)

    ft, small = iss.tap.sent[-1]
    assert iss.session._rung == 1
    assert ft == PSK_200S
    assert len(small) == F.FRAMES[PSK_200S].net_payload == 16
    assert small == big[:16], "the re-cut has to start where the old chunk did"

    iss.ack(96)
    assert iss.session.queued == queued - 16


def test_the_ladder_survives_a_turnover_and_not_the_link():
    """The rung is a fact about one path on one link: the reference keeps it
    across a BREAK and resets it at the connect, and so does this."""
    pair = Pair(bandwidth=200)
    pair.connect()

    pair.caller.queue_data(b"x" * 300)
    pair.pump()
    assert pair.caller._rung == 2

    pair.responder.queue_data(b"y" * 300)
    pair.pump()
    assert pair.responder._rung == 2
    assert pair.caller._rung == 2, "a turnover is not a new path"

    pair.caller.abort()
    pair.responder.abort()
    pair.relay.q.clear()

    pair.caller.connect("K7ABC")
    pair.pump()
    assert pair.caller._rung == 0 and pair.responder._rung == 0


def test_the_grades_ww2mi_sent_would_have_taken_us_up_the_ladder():
    """Recovered from `logs/onair/20260823T030452Z-besra-7102100.wav` and
    `…031121Z-…`: 32 DATAACKs grading our 4FSK at 84-100, median 96, against one
    DATANAK at 80. Every one of them was discarded, and this end sent 4.9 B/s for
    the whole session while the gateway was talking to us at 4PSK.200.100.

    Replayed at that median the ladder tops out. Replayed at the worst grade of
    the 32 it stops one rung short, because 84 does not clear an 84 bar — the two
    go together, because the second is what a 200 Hz margin looks like and the
    reason the rungs above 4PSK.200.100 stay out of the table until a run measures
    them.
    """
    for grade, rung in ((96, 2), (84, 1)):
        iss = Iss()
        iss.session.queue_data(b"x" * 600)
        for _ in range(10):
            iss.ack(grade)
        assert iss.session._rung == rung, grade


@pytest.mark.parametrize("bw", (500, 1000, 2000))
def test_every_bandwidth_climbs_its_whole_ladder_on_the_same_machinery(bw):
    """The gateways this station has reason to call run at 500, 1000 and 2000 —
    AJ4GU, N3MEL, K4PAR-2, KY4RY — and not one of them at 200. A single-rung table
    at those widths is a gearshift that can never move, whatever the path is doing.

    Same averager, same two-ACK gate, same brakes: nothing in them reads a width,
    so the only thing a wider ladder needed was the rungs. Graded at 100 — clear of
    every threshold in every column — the climb ends on the top rung and stays.
    """
    iss = Iss(bandwidth=bw)
    ladder = _DATA_LADDER[bw]
    iss.session.queue_data(b"x" * (F.FRAMES[ladder[-1][0]].net_payload * 4))
    for _ in range(2 * len(ladder) + 2):
        iss.ack(100)

    assert iss.session._rung == len(ladder) - 1
    assert iss.tap.data[-1] in (ladder[-1][0], ladder[-1][0] + 1)


def test_a_nak_at_the_top_of_a_wide_ladder_walks_back_down_it():
    """Down one rung per pair of NAKs — every rung of a climb has ACKed, so every
    one of them is worth a second look — and the outstanding chunk re-cut to fit
    each: 512 bytes at 4PSK.2000.100 do not go into the 32 of the 4FSK.500.100S at
    the foot of the same ladder."""
    iss = Iss(bandwidth=2000)
    ladder = [base for base, _ in _DATA_LADDER[2000]]
    iss.session.queue_data(bytes((i * 7) & 0xFF for i in range(4096)))
    for _ in range(2 * len(ladder)):
        iss.ack(100)
    assert iss.session._rung == len(ladder) - 1

    for rung in range(len(ladder) - 2, -1, -1):
        iss.nak(60)
        iss.nak(60)                       # every rung of this climb has ACKed
        assert iss.session._rung == rung
        ft, chunk = iss.tap.sent[-1]
        assert ft in (ladder[rung], ladder[rung] + 1)
        assert len(chunk) <= F.FRAMES[ft].net_payload

    queued = iss.session.queued
    iss.ack(100)
    assert iss.session.queued == queued - len(chunk), \
        "the ACK deleted bytes the peer was never sent"
