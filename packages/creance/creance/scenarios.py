# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Scenario layer: CwpSession (CWP over one ModemClient) plus the scenario
registry with initiator and responder halves.

Initiator contract: initiate(session, name, params) is called only after the
CONNECTED notification — a modem is free to drop pre-CONNECT writes, and one
did until it was fixed, so a scenario never counts on the queue. It sends HELLO,
arbitrates the reply — HELLO_ACK, our own HELLO echoed back (sid match: the
kestrel-loopback self-test path, outcome echo_peer), accept:false (refused),
or silence/garbage (plain-peer policy) — then runs the initiate half.
DISCONNECT is the caller's job (initiator.run_session), for every scenario.

Responder contract: after the daemon has seen CONNECTED, built a CwpSession
and parsed HELLO (session.await_hello()), it calls respond(session, hello).
respond() sends the HELLO_ACK — accept:false with a reason in caps for a
scenario it cannot run, outcome refused — and executes the scenario's
responder half. Both HELLO and HELLO_ACK name their sender's sid and
callsign, so each end lands the other's on session.peer_sid/peer_call: the
join key between two sites' records, which share no clock. NotCwp from
await_hello() is the daemon's cue to call fallback_sink() instead.

Transcript hproto events follow the metrics recorder contract (metrics.py):
hello_ack* closes the handshake latency, end*/report* carry sha256/bytes/
dur_s (+ match where this side verified a sha itself), desync* marks a CWP
defect, which is also a conformance FAIL: the data channel is ARQ-reliable,
so framing damage is a real modem defect, and the session degrades to sink.
Payload writes are labelled with the payload generator; control frames carry
metrics.CONTROL_LABEL, which is what keeps them out of the drain rate.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

from . import hproto, payloads
from .link import BUFFER, STATS, Link
from .hproto import Desync, End, Frame, Hello, HelloAck, NotCwp, Report
from .metrics import CONTROL_LABEL, MIN_MEASURE_S
from hfhost.transcript import Transcript

DEFAULT_HIGH_WATER = 16 * 1024
DEFAULT_SIZE = "10k"
DEFAULT_PAYLOAD = "prbs9"
DEFAULT_REPORT_TIMEOUT_S = 60.0   # sized for >= 2 half-duplex turnarounds

_EMPTY_SHA = hashlib.sha256(b"").hexdigest()


class SessionCancelled(Exception):
    """The session's cancel event fired mid-primitive (abort/watchdog)."""


@dataclass
class ScenarioResult:
    outcome: str
    stats: dict = field(default_factory=dict)


def _payload_plan(params: dict) -> tuple[bytes, str, float | None]:
    size = payloads.size_bytes(params.get("size", DEFAULT_SIZE))
    name = str(params.get("payload", DEFAULT_PAYLOAD))
    duration = params.get("duration")
    return payloads.build(name, size), name, (float(duration) if duration else None)


def _chunks(blocks: Iterable[bytes]) -> Iterable[bytes]:
    for block in blocks:
        for i in range(0, len(block), hproto.MAX_PAYLOAD):
            yield block[i:i + hproto.MAX_PAYLOAD]


def _stream(block: bytes, duration: float | None) -> Iterable[bytes]:
    """The payload as frame-sized chunks, repeated until duration expires.

    The deadline is checked per chunk, not per block: at block granularity a
    --duration run overruns by minutes on a slow link and gets filed as
    aborted:watchdog.
    """
    chunks = list(_chunks([block]))
    if duration is None or not chunks:
        yield from chunks
        return
    deadline = time.monotonic() + duration
    while True:
        for chunk in chunks:
            if time.monotonic() >= deadline:
                return
            yield chunk


def _report_timeout(params: dict) -> float:
    return float(params.get("report_timeout_s", DEFAULT_REPORT_TIMEOUT_S))


class CwpSession:
    """One CWP conversation over an attached Link.

    Owns a Deframer fed from the client's epoch-guarded rx buffer, sha256
    accumulators for both directions, and TX pacing that caps the outstanding
    modem buffer at high_water by waiting on BUFFER notifications — dumping
    the whole payload into the modem's TCP queue at once would destroy the
    drain measurement.
    """

    def __init__(self, client: Link, transcript: Transcript, *,
                 sid: str, mycall: str,
                 cancel: threading.Event | None = None, epoch: int | None = None,
                 high_water: int = DEFAULT_HIGH_WATER,
                 recv_timeout_s: float = 60.0,
                 stall_timeout_s: float = 30.0) -> None:
        self.client = client
        self.transcript = transcript
        self.sid = sid
        self.mycall = mycall
        self.cancel = cancel if cancel is not None else threading.Event()
        self.epoch = client.epoch if epoch is None else epoch
        self.high_water = high_water
        self.recv_timeout_s = recv_timeout_s
        self.stall_timeout_s = stall_timeout_s

        self.sha_tx = hashlib.sha256()
        self.sha_rx = hashlib.sha256()
        self.payload_tx = 0
        self.payload_rx = 0
        self.t_first_data: float | None = None
        self.t_last_data: float | None = None
        self.t_rx_start: float | None = None
        self.peer_sid = ""
        self.peer_call = ""
        self.desync: Desync | None = None

        self._deframer = hproto.Deframer()
        self._pending: deque[Frame] = deque()

    @property
    def tx_sha256(self) -> str:
        return self.sha_tx.hexdigest()

    @property
    def rx_sha256(self) -> str:
        return self.sha_rx.hexdigest()

    def note(self, event: str, **meta) -> None:
        """Record a CWP event against this session (metrics.py reads these)."""
        self.transcript.hproto(self.client.name, event, **meta)

    # -- outbound ----------------------------------------------------------

    def _await_room(self, need: int) -> None:
        # Checked before the wait loop, not only inside it: when the modem
        # drains faster than we fill, the loop body never runs, and an
        # unchecked send path streams a whole payload past an abort — the
        # zombie writer then bleeds into the next session.
        if self.cancel.is_set():
            raise SessionCancelled(self.client.name)
        deadline = time.monotonic() + self.stall_timeout_s
        while self.client.queue_bytes + need > self.high_water:
            if self.cancel.is_set():
                raise SessionCancelled(self.client.name)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{self.client.name}: TX stalled at buffer high-water "
                    f"{self.high_water}")
            # Either dialect's queue-depth signal will do: the VARA one is a
            # BUFFER notification, the structured one a LinkStats sample.
            if self.client.wait_for(BUFFER, min(remaining, 0.25),
                                    cancel=self.cancel) is None:
                self.client.wait_for(STATS, min(remaining, 0.05),
                                     cancel=self.cancel)

    def send_hello(self, scenario: str, params: dict) -> None:
        blob = Hello(sid=self.sid, call=self.mycall, scenario=scenario,
                     params=params).pack()
        self.client.send(blob, label=CONTROL_LABEL)
        self.note("hello_sent", sid=self.sid, scenario=scenario, params=params)

    def send_hello_ack(self, accept: bool, caps: dict) -> None:
        self.client.send(HelloAck(accept=accept, caps=caps).pack(),
                              label=CONTROL_LABEL)
        self.note("hello_ack_sent", accept=accept, caps=caps)
        if accept:
            self.mark_rx_start()

    def send_end(self, sha256: str, nbytes: int, dur_s: float) -> None:
        self.client.send(End(sha256=sha256, bytes=nbytes, dur_s=dur_s).pack(),
                              label=CONTROL_LABEL)
        self.note("end_sent", sha256=sha256, bytes=nbytes, dur_s=dur_s)

    def send_report(self, sid: str, nbytes: int, sha256: str, dur_s: float) -> None:
        self.client.send(
            Report(sid=sid, bytes=nbytes, sha256=sha256, dur_s=dur_s).pack(),
            label=CONTROL_LABEL)
        self.note("report_sent", sid=sid, sha256=sha256, bytes=nbytes, dur_s=dur_s)

    def mark_rx_start(self) -> None:
        """Stamp the instant the peer was cleared to transmit (the completed
        HELLO_ACK). The receive leg is timed from here — not from the first
        frame we dequeue, which for a single-chunk payload lands *after* the
        whole transfer and would collapse the window to the END turnaround."""
        if self.t_rx_start is None:
            self.t_rx_start = time.monotonic()

    def send_data_frame(self, chunk: bytes, label: str = "") -> None:
        frame = hproto.pack(hproto.DATA, chunk)
        self._await_room(len(frame))
        self.client.send(frame, label=label)
        self.sha_tx.update(chunk)
        self.payload_tx += len(chunk)
        self.note("data_sent", len=len(chunk))

    def send_frames_paced(self, data: bytes | Iterable[bytes], label: str = "") -> int:
        """DATA-frame the payload in <=4 KiB chunks, pacing on high_water.
        Returns payload bytes sent."""
        sent = 0
        blocks = [data] if isinstance(data, (bytes, bytearray)) else data
        for chunk in _chunks(blocks):
            self.send_data_frame(chunk, label=label)
            sent += len(chunk)
        return sent

    def send_raw_paced(self, data: bytes | Iterable[bytes], label: str = "") -> int:
        """Unframed paced send — the blind path toward a plain peer."""
        sent = 0
        blocks = [data] if isinstance(data, (bytes, bytearray)) else data
        for chunk in _chunks(blocks):
            self._await_room(len(chunk))
            self.client.send(chunk, label=label)
            sent += len(chunk)
        return sent

    # -- inbound -----------------------------------------------------------

    def recv_frame(self, timeout: float | None = None) -> Frame | None:
        """Next CWP frame, or None on timeout. DATA frames are accumulated
        (sha/counters) and transcribed on the way out. Raises NotCwp/Desync
        from the deframer and SessionCancelled on the cancel event."""
        deadline = time.monotonic() + (self.recv_timeout_s if timeout is None
                                       else timeout)
        while True:
            if self._pending:
                frame = self._pending.popleft()
                if frame.type == hproto.DATA:
                    now = time.monotonic()
                    if self.t_first_data is None:
                        self.t_first_data = now
                    self.t_last_data = now
                    self.sha_rx.update(frame.payload)
                    self.payload_rx += len(frame.payload)
                    self.note("data_rx", len=len(frame.payload))
                return frame
            if self.desync is not None:
                raise self.desync
            if self.cancel.is_set():
                raise SessionCancelled(self.client.name)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            chunk = self.client.recv(None, timeout=min(remaining, 0.25),
                                          epoch=self.epoch, cancel=self.cancel)
            if not chunk:
                continue
            try:
                self._pending.extend(self._deframer.feed(chunk))
            except Desync as exc:
                # Frames that completed before the damage are still good — an
                # END in the same chunk must not turn into a phantom
                # end_missing. Record now, deliver those, raise once drained.
                self.record_desync(exc)
                self._pending.extend(exc.frames)
                if not self._pending:
                    raise

    def _budget(self, timeout: float | None) -> Callable[[], float]:
        """Absolute deadline for a multi-frame wait. Passing the same timeout
        to every recv_frame would restart the window per frame, so a peer
        trickling one frame per timeout could hold the session open forever."""
        deadline = time.monotonic() + (self.recv_timeout_s if timeout is None
                                       else timeout)
        return lambda: deadline - time.monotonic()

    def await_hello(self, timeout: float | None = None) -> Hello | None:
        """Responder entry point: the daemon calls this after CONNECTED, then
        hands the result to respond(). Raises NotCwp for the sink fallback."""
        left = self._budget(timeout)
        while (frame := self.recv_frame(left())) is not None:
            if frame.type == hproto.HELLO:
                h = Hello.parse(frame)
                self.note("hello_rx", sid=h.sid, call=h.call,
                          scenario=h.scenario, params=h.params)
                self.peer_sid, self.peer_call = h.sid, h.call
                return h
            self.note("unexpected_frame", name=frame.name)
        return None

    def parse_end(self, frame: Frame) -> End:
        end = End.parse(frame)
        self.note("end_rx", sha256=end.sha256, bytes=end.bytes,
                  dur_s=end.dur_s, match=(end.sha256 == self.rx_sha256))
        return end

    def parse_report(self, frame: Frame) -> Report:
        rep = Report.parse(frame)
        self.note("report_rx", sid=rep.sid, sha256=rep.sha256,
                  bytes=rep.bytes, dur_s=rep.dur_s,
                  match=(rep.sha256 == self.tx_sha256))
        return rep

    def await_end(self, timeout: float | None = None) -> End | None:
        """Consume (accumulating) until the peer's END arrives."""
        left = self._budget(timeout)
        while (frame := self.recv_frame(left())) is not None:
            if frame.type == hproto.END:
                return self.parse_end(frame)
            if frame.type != hproto.DATA:
                self.note("unexpected_frame", name=frame.name)
        return None

    def await_report(self, timeout: float) -> Report | None:
        left = self._budget(timeout)
        while (frame := self.recv_frame(left())) is not None:
            if frame.type == hproto.REPORT:
                return self.parse_report(frame)
            if frame.type != hproto.DATA:
                self.note("unexpected_frame", name=frame.name)
        return None

    # -- failure paths -----------------------------------------------------

    def record_desync(self, exc: Desync) -> None:
        """Idempotent: recv_frame records the defect as soon as it sees it, so
        the scenario's own handler must not double-file it."""
        if self.desync is not None:
            return
        self.desync = exc
        self.note("desync", reason=exc.reason, evidence=exc.evidence.hex())
        self.transcript.conf(self.client.name, "cwp_desync", "FAIL",
                             evidence=exc.reason)

    def drain_sink(self, *, idle_s: float = 1.0, max_s: float = 30.0,
                   initial: int = 0) -> int:
        """Count raw bytes until an idle window, cancel, or max_s elapses —
        the degraded tail of a desynced or never-CWP session."""
        total = initial
        deadline = time.monotonic() + max_s
        while time.monotonic() < deadline and not self.cancel.is_set():
            chunk = self.client.recv(None, timeout=idle_s,
                                          epoch=self.epoch, cancel=self.cancel)
            if not chunk:
                break
            total += len(chunk)
        return total


# -- scenario halves ----------------------------------------------------------

def _rx_duration(s: CwpSession) -> float:
    """Wall time of the receive leg, for the REPORT the far end will use as
    authoritative goodput: from the completed handshake (when the peer was
    cleared to send) to the arrival of the last payload byte.

    It therefore includes the peer's turnaround before it started
    transmitting — conservative on half-duplex HF, and the figure a user
    actually experiences. 0.0 means unmeasurable, which metrics reads as
    "no goodput figure": better a missing number than an absurd one.
    """
    start = s.t_rx_start if s.t_rx_start is not None else s.t_first_data
    if start is None or s.t_last_data is None:
        return 0.0
    dur = s.t_last_data - start
    return dur if dur >= MIN_MEASURE_S else 0.0


def _stall_evidence(s: CwpSession) -> dict:
    """The modem's own last word on the link, recorded when a wait expires."""
    try:
        tel = s.client.telemetry() or {}
    except Exception:
        return {}
    keep = ("queue_bytes", "throughput_bps", "gear", "rung", "harq_rounds",
            "rebuilds", "snr3k_db")
    out = {f"end_{k}": tel[k] for k in keep if k in tel}
    q = tel.get("queue_bytes")
    if isinstance(q, int):
        out["still_draining"] = q > 0
    return out


def _send_leg(s: CwpSession, params: dict) -> ScenarioResult:
    block, name, duration = _payload_plan(params)
    t0 = time.monotonic()
    s.send_frames_paced(_stream(block, duration), label=name)
    dur = time.monotonic() - t0
    sha = s.tx_sha256
    s.send_end(sha, s.payload_tx, dur)
    rep = s.await_report(_report_timeout(params))
    stats = {"bytes": s.payload_tx, "sha256": sha, "dur_s": dur}
    if rep is None:
        # Say *why* there was no report. A modem still draining and a peer that
        # has gone away produce the same silence, and the difference is the
        # whole diagnosis: at real HF rates a few KiB takes minutes, so a budget
        # sized for simulated time reads as a dead peer.
        stats.update(_stall_evidence(s))
        return ScenarioResult("report_missing", stats)
    stats.update(far_bytes=rep.bytes, far_dur_s=rep.dur_s)
    ok = rep.sha256 == sha and rep.bytes == s.payload_tx
    return ScenarioResult("ok" if ok else "failed:integrity", stats)


def _recv_leg(s: CwpSession, report_sid: str) -> ScenarioResult:
    end = s.await_end()
    if end is None:
        return ScenarioResult("failed:end_missing", {"bytes": s.payload_rx})
    dur = _rx_duration(s)
    sha = s.rx_sha256
    s.send_report(report_sid, s.payload_rx, sha, dur)
    ok = end.sha256 == sha and end.bytes == s.payload_rx
    return ScenarioResult("ok" if ok else "failed:integrity",
                          {"bytes": s.payload_rx, "sha256": sha, "dur_s": dur})


def _i_connect(s: CwpSession, params: dict) -> ScenarioResult:
    s.send_end(_EMPTY_SHA, 0, 0.0)
    rep = s.await_report(_report_timeout(params))
    if rep is None:
        return ScenarioResult("report_missing", {})
    return ScenarioResult("ok", {})


def _r_connect(s: CwpSession, hello: Hello) -> ScenarioResult:
    if s.await_end() is None:
        return ScenarioResult("failed:end_missing", {})
    s.send_report(hello.sid, 0, _EMPTY_SHA, 0.0)
    return ScenarioResult("ok", {})


def _i_unidir(s: CwpSession, params: dict) -> ScenarioResult:
    return _send_leg(s, params)


def _r_unidir(s: CwpSession, hello: Hello) -> ScenarioResult:
    return _recv_leg(s, hello.sid)


def _i_reverse(s: CwpSession, params: dict) -> ScenarioResult:
    return _recv_leg(s, s.sid)


def _r_reverse(s: CwpSession, hello: Hello) -> ScenarioResult:
    return _send_leg(s, hello.params)


def _bidir(s: CwpSession, params: dict, report_sid: str) -> ScenarioResult:
    tx_err: list[Exception] = []

    def tx() -> None:
        try:
            block, name, duration = _payload_plan(params)
            t0 = time.monotonic()
            s.send_frames_paced(_stream(block, duration), label=name)
            s.send_end(s.tx_sha256, s.payload_tx, time.monotonic() - t0)
        except Exception as exc:
            tx_err.append(exc)

    th = threading.Thread(target=tx, name=f"{s.client.name}-bidir-tx", daemon=True)
    th.start()
    try:
        end = s.await_end()
    finally:
        th.join(timeout=s.stall_timeout_s + 5.0)
        if th.is_alive():
            # A TX thread outliving the session would write the next session's
            # stream full of this one's payload; cancel it and say so.
            s.cancel.set()
            th.join(timeout=5.0)
    if th.is_alive():
        raise TimeoutError(f"{s.client.name}: bidir TX thread would not stop")
    if tx_err:
        raise tx_err[0]
    if end is None:
        return ScenarioResult("failed:end_missing", {"bytes_rx": s.payload_rx})
    dur = _rx_duration(s)
    sha_rx = s.rx_sha256
    s.send_report(report_sid, s.payload_rx, sha_rx, dur)
    rep = s.await_report(_report_timeout(params))
    stats = {"bytes_tx": s.payload_tx, "bytes_rx": s.payload_rx}
    if rep is None:
        return ScenarioResult("report_missing", stats)
    ok = end.sha256 == sha_rx and rep.sha256 == s.tx_sha256
    return ScenarioResult("ok" if ok else "failed:integrity", stats)


def _i_bidir(s: CwpSession, params: dict) -> ScenarioResult:
    return _bidir(s, params, s.sid)


def _r_bidir(s: CwpSession, hello: Hello) -> ScenarioResult:
    return _bidir(s, hello.params, hello.sid)


def _i_echo(s: CwpSession, params: dict) -> ScenarioResult:
    block, name, duration = _payload_plan(params)
    turnarounds: list[float] = []
    mismatches = 0
    t0 = time.monotonic()
    for chunk in _stream(block, duration):
        sent_at = time.monotonic()
        s.send_data_frame(chunk, label=name)
        frame = s.recv_frame()
        while frame is not None and frame.type != hproto.DATA:
            s.note("unexpected_frame", name=frame.name)
            frame = s.recv_frame()
        if frame is None:
            return ScenarioResult("failed:echo_timeout",
                                  {"chunks": len(turnarounds)})
        turnarounds.append(time.monotonic() - sent_at)
        if frame.payload != chunk:
            mismatches += 1
    sha = s.tx_sha256
    s.send_end(sha, s.payload_tx, time.monotonic() - t0)
    rep = s.await_report(_report_timeout(params))
    stats = {"chunks": len(turnarounds), "bytes": s.payload_tx,
             "mismatched_chunks": mismatches}
    if turnarounds:
        stats["turnaround_s"] = {"min": min(turnarounds),
                                 "mean": sum(turnarounds) / len(turnarounds),
                                 "max": max(turnarounds)}
    if rep is None:
        return ScenarioResult("report_missing", stats)
    ok = mismatches == 0 and rep.sha256 == sha
    return ScenarioResult("ok" if ok else "failed:integrity", stats)


def _r_echo(s: CwpSession, hello: Hello) -> ScenarioResult:
    while (frame := s.recv_frame()) is not None:
        if frame.type == hproto.DATA:
            s.send_data_frame(frame.payload, label="echo")
        elif frame.type == hproto.END:
            end = s.parse_end(frame)
            s.send_report(hello.sid, s.payload_rx, s.rx_sha256, _rx_duration(s))
            ok = end.sha256 == s.rx_sha256
            return ScenarioResult("ok" if ok else "failed:integrity",
                                  {"bytes": s.payload_rx})
        else:
            s.note("unexpected_frame", name=frame.name)
    return ScenarioResult("failed:end_missing", {"bytes": s.payload_rx})


def _i_sink(s: CwpSession, params: dict) -> ScenarioResult:
    block, name, duration = _payload_plan(params)
    s.send_frames_paced(_stream(block, duration), label=name)
    return ScenarioResult("ok", {"bytes": s.payload_tx})


def _r_sink(s: CwpSession, hello: Hello | None) -> ScenarioResult:
    idle = float(((hello.params if hello else None) or {}).get("idle_s", 5.0))
    n = s.drain_sink(idle_s=idle, max_s=s.recv_timeout_s)
    return ScenarioResult("ok", {"bytes": n})


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    initiate: Callable[[CwpSession, dict], ScenarioResult]
    respond: Callable[[CwpSession, Hello], ScenarioResult]


SCENARIOS: dict[str, Scenario] = {
    "connect": Scenario("connect", _i_connect, _r_connect),
    "unidir": Scenario("unidir", _i_unidir, _r_unidir),
    "reverse": Scenario("reverse", _i_reverse, _r_reverse),
    "bidir": Scenario("bidir", _i_bidir, _r_bidir),
    "echo": Scenario("echo", _i_echo, _r_echo),
    "sink": Scenario("sink", _i_sink, _r_sink),
}


# -- top-level halves ---------------------------------------------------------

def _echo_peer(s: CwpSession, params: dict) -> ScenarioResult:
    """Our own HELLO came back: the peer is a dumb echo (kestrel loopback).
    Run echo integrity against ourselves instead of misfiling as plain_peer."""
    s.note("echo_peer", sid=s.sid)
    block, name, _ = _payload_plan(params)
    t0 = time.monotonic()
    sent = s.send_frames_paced(block, label=name)
    deadline = time.monotonic() + s.recv_timeout_s
    while s.payload_rx < sent and time.monotonic() < deadline:
        if s.recv_frame(timeout=deadline - time.monotonic()) is None:
            break
    match = s.payload_rx == sent and s.rx_sha256 == s.tx_sha256
    s.note("end_local", sha256=s.tx_sha256, bytes=sent,
           dur_s=time.monotonic() - t0, match=match)
    s.transcript.conf(s.client.name, "echo_integrity",
                      "PASS" if match else "FAIL",
                      bytes_tx=sent, bytes_rx=s.payload_rx)
    return ScenarioResult("echo_peer", {"bytes": sent, "match": match})


def _plain_peer(s: CwpSession, params: dict, policy: str,
                buffered: bytes = b"") -> ScenarioResult:
    s.note("plain_peer", policy=policy, buffered=len(buffered))
    if policy == "blind":
        block, name, duration = _payload_plan(params)
        sent = s.send_raw_paced(_stream(block, duration), label=name)
        return ScenarioResult("plain_peer", {"policy": policy, "bytes": sent})
    return ScenarioResult("plain_peer", {"policy": policy})


def _guard_desync(session: CwpSession, result: ScenarioResult) -> ScenarioResult:
    """A half that completed on frames salvaged from a damaged feed still ran
    over a broken stream: the defect decides the outcome, not the salvage."""
    if session.desync is None or result.outcome.startswith("failed:cwp_desync"):
        return result
    return ScenarioResult("failed:cwp_desync",
                          dict(result.stats, desync=session.desync.reason))


def initiate(session: CwpSession, scenario: str, params: dict | None = None, *,
             on_plain_peer: str = "disconnect",
             ack_timeout_s: float = 30.0) -> ScenarioResult:
    """Initiator half. Call only after CONNECTED. Never raises for protocol
    outcomes — those come back in ScenarioResult per the outcome vocabulary."""
    params = dict(params or {})
    scn = SCENARIOS.get(scenario)
    if scn is None:
        raise ValueError(f"unknown scenario {scenario!r}")
    try:
        session.send_hello(scenario, params)
        deadline = time.monotonic() + ack_timeout_s
        while True:
            frame = session.recv_frame(timeout=deadline - time.monotonic())
            if frame is None:
                return _plain_peer(session, params, on_plain_peer)
            if frame.type == hproto.HELLO_ACK:
                ack = HelloAck.parse(frame)
                session.note("hello_ack_rx", accept=ack.accept, caps=ack.caps)
                session.peer_sid = str(ack.caps.get("sid") or "")
                session.peer_call = str(ack.caps.get("call") or "")
                if not ack.accept:
                    return ScenarioResult("refused", {"caps": ack.caps})
                session.mark_rx_start()
                return _guard_desync(session, scn.initiate(session, params))
            if frame.type == hproto.HELLO:
                h = Hello.parse(frame)
                session.note("hello_rx", sid=h.sid, call=h.call,
                             scenario=h.scenario, params=h.params)
                session.peer_sid, session.peer_call = h.sid, h.call
                if h.sid == session.sid:
                    return _echo_peer(session, params)
                # crossed initiators: both ends dialled at once
                return ScenarioResult("failed:crossed_hello", {"peer_sid": h.sid})
            session.note("unexpected_frame", name=frame.name)
    except NotCwp as exc:
        session.note("not_cwp", len=len(exc.buffered),
                     head=exc.buffered[:16].hex())
        return _plain_peer(session, params, on_plain_peer, buffered=exc.buffered)
    except Desync as exc:
        session.record_desync(exc)
        n = session.drain_sink()
        return ScenarioResult("failed:cwp_desync", {"sink_bytes": n})
    except SessionCancelled:
        return ScenarioResult("aborted:watchdog", {})
    except TimeoutError as exc:
        session.transcript.error(session.client.name, str(exc))
        return ScenarioResult("failed:tx_stall", {})


def respond(session: CwpSession, hello: Hello) -> ScenarioResult:
    """Responder half: the daemon calls this with the parsed HELLO from
    session.await_hello(). Sends the HELLO_ACK and runs the scenario.

    The ACK carries our own sid and callsign: that is how the initiator learns
    which of this site's records belongs to the exchange it just ran."""
    caps = {"sid": session.sid, "call": session.mycall,
            "scenarios": sorted(SCENARIOS)}
    scn = SCENARIOS.get(hello.scenario)
    if scn is None:
        caps["reason"] = f"unknown scenario {hello.scenario!r}"
        session.send_hello_ack(False, caps)
        return ScenarioResult("refused", {"scenario": hello.scenario})
    session.send_hello_ack(True, caps)
    try:
        return _guard_desync(session, scn.respond(session, hello))
    except Desync as exc:
        session.record_desync(exc)
        n = session.drain_sink()
        return ScenarioResult("failed:cwp_desync", {"sink_bytes": n})
    except SessionCancelled:
        return ScenarioResult("aborted:watchdog", {})
    except TimeoutError as exc:
        session.transcript.error(session.client.name, str(exc))
        return ScenarioResult("failed:tx_stall", {})


def fallback_sink(session: CwpSession, buffered: bytes = b"", *,
                  idle_s: float = 5.0) -> ScenarioResult:
    """The responder's no-magic path: await_hello() raised NotCwp (or timed
    out); count the plain peer's bytes until it goes quiet."""
    session.note("sink", reason="not_cwp", len=len(buffered))
    n = session.drain_sink(idle_s=idle_s, initial=len(buffered))
    return ScenarioResult("plain_peer", {"bytes": n})
