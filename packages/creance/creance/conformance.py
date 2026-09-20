# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Conformance probes for the VARA-dialect host interface, plus the golden
verdict table for today's M1 targets.

A probe interrogates a live modem through an attached ModemClient and returns
a Finding. PASS/FAIL judge grammar and choreography against spec §7 (as
amended by wire.SPEC_DELTAS), ABSENT marks documented-but-unobserved behavior,
EXTRA marks vocabulary the grammar does not know, WARN marks tolerated
oddities. ``creance selftest`` gates on deviations from GOLDEN, not on raw
FAIL/ABSENT, so it is meaningful today: an ABSENT entry is a standing statement
that the behavior is documented but not yet implemented, and it flips to PASS
-- with GOLDEN re-derived against the live target -- when the modem grows it.
kestrel-loopback now passes preconnect_queue; its bitrate_absent and sn_absent
are ABSENT because the loopback stopped fabricating channel state the real core
has no hook for. Sabir's bitrate/sn/pending entries flip when its ARQ wires
those notifications through.

GOLDEN was derived empirically by running the suites against live targets on
ephemeral ports — 2026-07-20 for the first two, 2026-07-28 for kestrel-pair:

  kestrel-loopback   -m hfmodem.kestrel.host.run_server --cmd-port E
                     --data-port E --iamalive-interval 2 --quiet,
                     quirks: iamalive_s=2, preconnect_queue=true
  kestrel-pair       -m hfmodem.kestrel.arq.run_pair --a-cmd E --a-data E
                     --b-cmd E --b-data E --quiet — two real KestrelModems
                     (ARQ FSM + VARA HF 500 burst codec) on one AudioChannel,
                     default quirks; single suite on A with B as armed peer,
                     then the paired suite
                     --profile clean --snr 90 --quiet, default quirks; same
                     two-suite order

Both pairs' IAMALIVE interval is fixed at 60 s (no CLI knob), so neither pair's
iamalive_cadence is judgeable: their suites end well inside 1.5 intervals and
both entries are ABSENT. tests/test_conformance_live.py reproduces all three
derivations and is the selftest gate.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from hfhost.client import ModemClient
from .config import ModemConfig, Quirks
from .payloads import prbs
from hfhost.transcript import Transcript
from hfhost.wire import NOTIFICATION, REPLY_OK, REPLY_WRONG, UNKNOWN, Line

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
ABSENT = "ABSENT"
EXTRA = "EXTRA"

VERDICTS = (PASS, FAIL, WARN, ABSENT, EXTRA)


@dataclass(frozen=True, slots=True)
class Finding:
    probe: str
    verdict: str
    evidence: tuple[str, ...] = ()
    detail: str = ""


ProbeResult = Finding | list[Finding]


@dataclass
class ProbeContext:
    """One attached modem under probe, with the knobs the probes time by.

    ``peer`` is an attached client at the far end of a linked pair; when set,
    session probes arm it (MYCALL dst + LISTEN ON) so CONNECT completes and
    delivered data is read there instead of at the loopback echo.
    """

    client: ModemClient
    mycall: str = "N0CRE"
    dst: str = "K7CRE"
    peer: ModemClient | None = None
    settle_s: float = 0.3
    reply_timeout_s: float = 5.0
    connect_timeout_s: float = 30.0
    drain_timeout_s: float = 30.0
    data_timeout_s: float = 30.0
    abort_timeout_s: float = 5.0
    payload_n: int = 512
    #: minimum passive-observation window; run_single sleeps out any remainder
    #: (None: judge on whatever the active probes took — never a stall)
    iamalive_window_s: float | None = None

    @property
    def cfg(self) -> ModemConfig:
        return self.client.cfg

    @property
    def quirks(self) -> Quirks:
        return self.client.cfg.quirks

    @property
    def transcript(self) -> Transcript:
        return self.client.transcript


# -- registry ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Probe:
    id: str
    mode: str                   # "single" | "passive" | "paired"
    fn: Callable


PROBES: dict[str, Probe] = {}   # insertion order == execution order


def _probe(pid: str, mode: str):
    def register(fn):
        PROBES[pid] = Probe(pid, mode, fn)
        return fn
    return register


# -- shared helpers ----------------------------------------------------------

def _payload(n: int) -> bytes:
    return prbs(n, 9)


def _reply(ctx: ProbeContext, text: str,
           client: ModemClient | None = None) -> Line | None:
    try:
        return (client or ctx.client).command(text, ctx.reply_timeout_s)
    except (TimeoutError, OSError, RuntimeError):
        return None


def _got(line: Line | None) -> str:
    if line is None:
        return "no reply"
    return "OK" if line.kind == REPLY_OK else "WRONG"


def _grammar(ctx: ProbeContext, pid: str, cases: list[tuple[str, str]],
             restore: tuple[str, ...] = (), detail_pass: str = "") -> Finding:
    """Send each command, compare its OK/WRONG against the expected kind."""
    evidence, bad = [], False
    for cmd, want in cases:
        got = _got(_reply(ctx, cmd))
        ok = got == want
        bad |= not ok
        evidence.append(f"{cmd!r} -> {got}" + ("" if ok else f" (expected {want})"))
    for cmd in restore:
        _reply(ctx, cmd)
    if bad:
        return Finding(pid, FAIL, tuple(evidence), "grammar violated")
    return Finding(pid, PASS, tuple(evidence), detail_pass or "grammar as specified")


def _wait(pred: Callable[[], bool], timeout: float, poll: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll)
    return pred()


def _arm_peer(ctx: ProbeContext) -> None:
    if ctx.peer is not None:
        ctx.peer.set_mycall(ctx.dst)
        ctx.peer.set_listen(True)


def _bump_all(ctx: ProbeContext) -> None:
    ctx.client.bump_epoch()
    if ctx.peer is not None:
        ctx.peer.bump_epoch()


def _establish(ctx: ProbeContext) -> bool:
    """CONNECT mycall->dst and wait for CONNECTED. On the kestrel loopback the
    modem fabricates the session; on a pair the armed peer answers.

    The request is retried inside the connect budget rather than graded on one
    attempt, because getting into a session is these probes' setup and not
    their subject. A core that has just aborted a session can refuse the next
    request until it is back to idle: on the kestrel pair, measured over six
    teardowns, the connect after an ABORT failed and the one after that
    succeeded, strictly alternating, and a 0.3 s pause was enough to clear it.
    Asking once therefore made every probe downstream of a teardown turn on how
    quickly the machine got round to asking again — buffer_cadence's verdict,
    and with it the length of the passive window iamalive_cadence is judged in.
    """
    _reply(ctx, f"MYCALL {ctx.mycall}")
    deadline = time.monotonic() + ctx.connect_timeout_s
    attempt_s = max(5.0, ctx.connect_timeout_s / 4)
    while True:
        _arm_peer(ctx)
        _bump_all(ctx)
        line = _reply(ctx, f"CONNECT {ctx.mycall} {ctx.dst}")
        if line is None or line.kind != REPLY_OK:
            return False
        remaining = deadline - time.monotonic()
        if _wait(lambda: ctx.client.connected, min(attempt_s, remaining)):
            return True
        if remaining <= attempt_s:
            return False
        _reply(ctx, "ABORT")        # retract the abandoned request first
        _wait(lambda: not ctx.client.connected, ctx.abort_timeout_s)


def _teardown(ctx: ProbeContext) -> None:
    """Best-effort return to idle: abort any live session, settle the peer,
    fence leftover data out of the next probe's reads."""
    if ctx.client.connected:
        _reply(ctx, "ABORT")
        _wait(lambda: not ctx.client.connected, ctx.abort_timeout_s)
    if ctx.peer is not None:
        _wait(lambda: not ctx.peer.connected, ctx.abort_timeout_s)
    _bump_all(ctx)


def _read_exact(client: ModemClient, n: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    got = b""
    while len(got) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        got += client.read_data(n - len(got), timeout=remaining)
    return got


def _drain_lines(q: queue.Queue) -> list[Line]:
    out = []
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return out
        if isinstance(item, Line):
            out.append(item)


# -- single-modem probes (active) --------------------------------------------

@_probe("reply_discipline", "single")
def probe_reply_discipline(ctx: ProbeContext) -> Finding:
    battery = (f"MYCALL {ctx.mycall}", f"COMPRESSION {ctx.cfg.compression}",
               "LISTEN OFF", "NOSUCHVERB")
    evidence, bad = [], []
    for cmd in battery:
        q = ctx.client.subscribe()
        try:
            try:
                ctx.client.command(cmd, ctx.reply_timeout_s)
            except TimeoutError:
                bad.append(f"{cmd!r} -> no reply")
                continue
            time.sleep(ctx.settle_s)
            replies = [l.raw.strip() for l in _drain_lines(q)
                       if l.kind in (REPLY_OK, REPLY_WRONG)]
            evidence.append(f"{cmd!r} -> {' + '.join(replies)}")
            if len(replies) != 1:
                bad.append(f"{cmd!r} drew {len(replies)} replies: {' + '.join(replies)}")
        finally:
            ctx.client.unsubscribe(q)
    if bad:
        return Finding("reply_discipline", FAIL, tuple(bad),
                       "every command must draw exactly one OK/WRONG")
    return Finding("reply_discipline", PASS, tuple(evidence),
                   "every command drew exactly one reply")


@_probe("unknown_verb_wrong", "single")
def probe_unknown_verb_wrong(ctx: ProbeContext) -> Finding:
    line = _reply(ctx, "XYZZY QUUX")
    if line is not None and line.kind == REPLY_WRONG:
        return Finding("unknown_verb_wrong", PASS, ("'XYZZY QUUX' -> WRONG",),
                       "unknown verbs are rejected")
    return Finding("unknown_verb_wrong", FAIL,
                   (f"'XYZZY QUUX' -> {_got(line)}",),
                   "unknown verb must draw WRONG")


@_probe("mycall_arity", "single")
def probe_mycall_arity(ctx: ProbeContext) -> Finding:
    return _grammar(ctx, "mycall_arity", [
        ("MYCALL T1AAA", "OK"),
        ("MYCALL T1AAA T2BBB T3CCC T4DDD T5EEE", "OK"),
        ("MYCALL", "WRONG"),
        ("MYCALL T1AAA T2BBB T3CCC T4DDD T5EEE T6FFF", "WRONG"),
    ], restore=(f"MYCALL {ctx.mycall}",),
        detail_pass="1-5 callsigns accepted, 0 and 6 rejected")


@_probe("bw_forms", "single")
def probe_bw_forms(ctx: ProbeContext) -> Finding:
    return _grammar(ctx, "bw_forms", [
        ("BW500", "OK"),
        ("BW 500", "OK"),
        ("BW1234", "WRONG"),
    ], restore=(f"BW{ctx.cfg.bandwidth}",),
        detail_pass="both BW500 and 'BW 500' accepted, bogus bandwidth rejected")


@_probe("compression_args", "single")
def probe_compression_args(ctx: ProbeContext) -> Finding:
    return _grammar(ctx, "compression_args", [
        ("COMPRESSION OFF", "OK"),
        ("COMPRESSION TEXT", "OK"),
        ("COMPRESSION FILES", "OK"),
        ("COMPRESSION", "WRONG"),
        ("COMPRESSION ZIP", "WRONG"),
    ], restore=(f"COMPRESSION {ctx.cfg.compression}",),
        detail_pass="OFF/TEXT/FILES accepted, bare and bogus rejected")


@_probe("listen_semantics", "single")
def probe_listen_semantics(ctx: ProbeContext) -> Finding:
    return _grammar(ctx, "listen_semantics", [
        ("LISTEN ON", "OK"),
        ("LISTEN OFF", "OK"),
        ("CHAT ON", "OK"),
        ("CHAT OFF", "OK"),
    ], restore=("LISTEN OFF",),
        detail_pass="LISTEN/CHAT toggles accepted (CHAT ON implies LISTEN ON "
                    "per spec §7.2; observable here only as OK)")


@_probe("version_shape", "single")
def probe_version_shape(ctx: ProbeContext) -> Finding:
    if not ctx.quirks.version_reply:
        return Finding("version_shape", ABSENT,
                       detail="quirks.version_reply=false: not probed")
    v = ctx.client.request_version(ctx.reply_timeout_s)
    if v:
        return Finding("version_shape", PASS, (f"VERSION {v}",),
                       "VERSION replied with a version string")
    return Finding("version_shape", FAIL,
                   detail="no VERSION reply despite quirks.version_reply=true")


@_probe("preconnect_queue", "single")
def probe_preconnect_queue(ctx: ProbeContext) -> Finding:
    if not ctx.quirks.preconnect_queue:
        return Finding("preconnect_queue", ABSENT,
                       detail="quirks.preconnect_queue=false (known spec §7.4 "
                              "divergence): not probed")
    payload = _payload(ctx.payload_n)
    _reply(ctx, f"MYCALL {ctx.mycall}")
    _arm_peer(ctx)
    _bump_all(ctx)
    try:
        ctx.client.send_data(payload, label="preconnect")
        line = _reply(ctx, f"CONNECT {ctx.mycall} {ctx.dst}")
        if line is None or line.kind != REPLY_OK or \
                not _wait(lambda: ctx.client.connected, ctx.connect_timeout_s):
            return Finding("preconnect_queue", FAIL,
                           detail=f"no CONNECTED within {ctx.connect_timeout_s:.0f}s; "
                                  "cannot judge pre-connect queueing")
        sink = ctx.peer or ctx.client
        got = _read_exact(sink, len(payload), ctx.data_timeout_s)
        evidence = (f"sent {len(payload)} B before CONNECT; "
                    f"{len(got)} B arrived within {ctx.data_timeout_s:.0f}s",)
        if len(got) >= len(payload):
            return Finding("preconnect_queue", PASS, evidence,
                           "pre-connect data buffered and flushed (spec §7.4)")
        return Finding("preconnect_queue", FAIL, evidence,
                       "pre-connect data was discarded, contradicting spec §7.4")
    finally:
        _teardown(ctx)


@_probe("buffer_cadence", "single")
def probe_buffer_cadence(ctx: ProbeContext) -> Finding:
    if not _establish(ctx):
        return Finding("buffer_cadence", FAIL,
                       detail=f"no CONNECTED within {ctx.connect_timeout_s:.0f}s; "
                              "cannot probe BUFFER cadence")
    q = ctx.client.subscribe()
    try:
        ctx.client.send_data(_payload(ctx.payload_n), label="buffer_cadence")
        seq: list[int] = []
        deadline = time.monotonic() + ctx.drain_timeout_s
        while time.monotonic() < deadline:
            try:
                item = q.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                break
            if isinstance(item, Line) and item.kind == NOTIFICATION \
                    and item.name == "BUFFER":
                seq.append(item.fields["n"])
                if item.fields["n"] == 0 and max(seq) > 0:
                    break
    finally:
        ctx.client.unsubscribe(q)
    evidence = ("BUFFER " + " -> ".join(map(str, seq)),) if seq else ()
    if not seq:
        return Finding("buffer_cadence", FAIL, (),
                       "no BUFFER notifications during a data session")
    if max(seq) == 0:
        return Finding("buffer_cadence", WARN, evidence,
                       "BUFFER never rose above 0 (drain outpaced observation?)")
    if seq[-1] != 0:
        return Finding("buffer_cadence", FAIL, evidence,
                       f"BUFFER never drained to 0 within {ctx.drain_timeout_s:.0f}s")
    # session deliberately left connected for disconnect_flush_order
    return Finding("buffer_cadence", PASS, evidence,
                   "BUFFER rose on enqueue and drained to 0")


@_probe("disconnect_flush_order", "single")
def probe_disconnect_flush_order(ctx: ProbeContext) -> Finding:
    if not ctx.client.connected and not _establish(ctx):
        return Finding("disconnect_flush_order", FAIL,
                       detail="no session; cannot probe disconnect flush order")
    q = ctx.client.subscribe()
    try:
        ctx.client.send_data(_payload(ctx.payload_n), label="disconnect_flush")
        _reply(ctx, "DISCONNECT")
        seen: list[str] = []
        disconnected = False
        deadline = time.monotonic() + ctx.drain_timeout_s
        while time.monotonic() < deadline and not disconnected:
            try:
                item = q.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                break
            if isinstance(item, Line) and item.name in ("BUFFER", "DISCONNECTED"):
                seen.append(item.raw.strip())
                disconnected = item.name == "DISCONNECTED"
    finally:
        ctx.client.unsubscribe(q)
        _teardown(ctx)
    evidence = (" -> ".join(seen),) if seen else ()
    pre_disc = seen[:seen.index("DISCONNECTED")] if disconnected else seen
    buffers = [int(s.split()[1]) for s in pre_disc if s.startswith("BUFFER")]
    if not disconnected:
        return Finding("disconnect_flush_order", FAIL, evidence,
                       f"no DISCONNECTED within {ctx.drain_timeout_s:.0f}s of DISCONNECT")
    if not buffers:
        return Finding("disconnect_flush_order", FAIL, evidence,
                       "no BUFFER flush observed before DISCONNECTED")
    if buffers[-1] != 0:
        return Finding("disconnect_flush_order", FAIL, evidence,
                       f"last BUFFER before DISCONNECTED was {buffers[-1]}, not 0")
    return Finding("disconnect_flush_order", PASS, evidence,
                   "BUFFER 0 preceded DISCONNECTED")


@_probe("abort_prompt", "single")
def probe_abort_prompt(ctx: ProbeContext) -> Finding:
    if not _establish(ctx):
        return Finding("abort_prompt", FAIL,
                       detail="no session; cannot probe ABORT promptness")
    try:
        ctx.client.send_data(_payload(ctx.payload_n), label="abort_prompt")
        t0 = time.monotonic()
        _reply(ctx, "ABORT")
        if _wait(lambda: not ctx.client.connected, ctx.abort_timeout_s):
            dt = time.monotonic() - t0
            return Finding("abort_prompt", PASS,
                           (f"ABORT -> DISCONNECTED in {dt:.2f}s",),
                           "ABORT dropped the session promptly, no flush wait")
        return Finding("abort_prompt", FAIL,
                       detail=f"still connected {ctx.abort_timeout_s:.0f}s after ABORT")
    finally:
        _teardown(ctx)


# -- single-modem probes (passive: evaluated over the whole run) -------------

def _lines_named(lines: list[tuple[float, Line]], name: str, kind: str) -> list[Line]:
    return [l for _, l in lines if l.kind == kind and l.name == name]


def _spec_notification_probe(pid: str, name: str):
    def fn(ctx: ProbeContext, lines: list[tuple[float, Line]],
           window_s: float) -> Finding:
        good = _lines_named(lines, name, NOTIFICATION)
        bad = _lines_named(lines, name, UNKNOWN)
        if bad:
            return Finding(pid, FAIL, tuple(l.raw.strip() for l in bad[:5]),
                           f"malformed {name} line(s)")
        if good:
            return Finding(pid, PASS, tuple(l.raw.strip() for l in good[:5]),
                           f"{name} emitted and well-formed")
        return Finding(pid, ABSENT,
                       detail=f"{name} documented in spec §7.3, never seen "
                              f"in {window_s:.1f}s (implemented by neither target today)")
    return fn


_probe("bitrate_absent", "passive")(_spec_notification_probe("bitrate_absent", "BITRATE"))
_probe("sn_absent", "passive")(_spec_notification_probe("sn_absent", "SN"))


@_probe("iamalive_cadence", "passive")
def probe_iamalive_cadence(ctx: ProbeContext, lines: list[tuple[float, Line]],
                           window_s: float) -> Finding:
    interval = ctx.quirks.iamalive_s
    if not interval:
        return Finding("iamalive_cadence", ABSENT,
                       detail="quirks.iamalive_s=0: cadence not judged")
    beats = [t for t, l in lines
             if l.kind == NOTIFICATION and l.name == "IAMALIVE"]
    if not beats:
        if window_s < 1.5 * interval:
            return Finding("iamalive_cadence", ABSENT,
                           detail=f"no IAMALIVE, but the {window_s:.1f}s window is "
                                  f"shorter than 1.5x the {interval:.0f}s interval — "
                                  "too short to judge")
        return Finding("iamalive_cadence", ABSENT,
                       detail=f"no IAMALIVE within {window_s:.1f}s "
                              f"(>= 1.5x the {interval:.0f}s interval)")
    # line times are relative to collection start ≈ attach, so the first
    # beat's offset is itself the attach->first gap
    rel = [beats[0]] + [b - a for a, b in zip(beats, beats[1:])]
    evidence = (f"{len(beats)} beat(s); gaps "
                + ", ".join(f"{g:.1f}s" for g in rel[:8]),)
    # an early first beat is fine — only lateness violates "flows from attach";
    # inter-beat gaps must hold the cadence within ±50%
    off = ([rel[0]] if rel[0] > interval * 1.5 else []) + \
        [g for g in rel[1:] if not interval * 0.5 <= g <= interval * 1.5]
    if off:
        return Finding("iamalive_cadence", WARN, evidence,
                       f"cadence outside ±50% of the {interval:.0f}s interval")
    return Finding("iamalive_cadence", PASS, evidence,
                   "IAMALIVE flows from attach at the configured cadence "
                   "(de-facto semantics; spec §7.3 says only while connected)")


@_probe("vocabulary_sweep", "passive")
def probe_vocabulary_sweep(ctx: ProbeContext, lines: list[tuple[float, Line]],
                           window_s: float) -> Finding:
    unknown = [l.raw.strip() for _, l in lines if l.kind == UNKNOWN]
    if unknown:
        return Finding("vocabulary_sweep", EXTRA, tuple(unknown[:8]),
                       f"{len(unknown)} modem->host line(s) outside the known vocabulary")
    return Finding("vocabulary_sweep", PASS,
                   detail=f"all {len(lines)} modem->host lines classified")


# -- paired probes -----------------------------------------------------------
# run_paired performs one shared choreography (connect A->B, transfer,
# disconnect) and each probe judges the collected evidence; the last probe is
# active and runs after the session is over.

@_probe("connected_fields", "paired")
def probe_connected_fields(ctx_a: ProbeContext, ctx_b: ProbeContext, ev: dict,
                           la: list[tuple[float, Line]],
                           lb: list[tuple[float, Line]]) -> Finding:
    if not ev["connected"]:
        return Finding("connected_fields", FAIL,
                       detail="no session established between the pair")
    ca, cb = ev["conn_a"], ev["conn_b"]
    raws = tuple(l.raw.strip() for _, l in la + lb
                 if l.kind == NOTIFICATION and l.name == "CONNECTED")
    bad = []
    if ca["src"] != ctx_a.mycall or ca["dst"] != ctx_b.mycall:
        bad.append(f"A saw src/dst {ca['src']}/{ca['dst']}, "
                   f"expected {ctx_a.mycall}/{ctx_b.mycall}")
    if cb["src"] != ctx_b.mycall or cb["dst"] != ctx_a.mycall:
        bad.append(f"B saw src/dst {cb['src']}/{cb['dst']}, "
                   f"expected {ctx_b.mycall}/{ctx_a.mycall}")
    for end, c in (("A", ca), ("B", cb)):
        if c["bw"] is not None and not c["bw"].isdigit():
            bad.append(f"{end} bw field {c['bw']!r} is not numeric")
    if bad:
        return Finding("connected_fields", FAIL, tuple(bad) + raws,
                       "CONNECTED src dst [bw] fields disagree")
    return Finding("connected_fields", PASS, raws,
                   "CONNECTED fields well-formed and callsigns agree at both ends")


@_probe("ptt_alternation", "paired")
def probe_ptt_alternation(ctx_a: ProbeContext, ctx_b: ProbeContext, ev: dict,
                          la: list[tuple[float, Line]],
                          lb: list[tuple[float, Line]]) -> Finding:
    evidence, bad, any_seen = [], [], False
    for end, lines in (("A", la), ("B", lb)):
        states = [l.fields["on"] for _, l in lines
                  if l.kind == NOTIFICATION and l.name == "PTT"]
        if not states:
            evidence.append(f"{end}: no PTT")
            continue
        any_seen = True
        evidence.append(f"{end}: " + " ".join("ON" if s else "OFF" for s in states))
        if states[0] is False:
            # the unkey of a burst that straddled the observation-window
            # start; attributable to a pre-window ON, so not a violation
            states = states[1:]
            evidence.append(f"{end}: leading OFF dropped (burst straddled window start)")
        if not states:
            continue
        if states[0] is not True:
            bad.append(f"{end}: first PTT was OFF")
        if any(a == b for a, b in zip(states, states[1:])):
            bad.append(f"{end}: PTT did not strictly alternate")
        if states[-1] is not False:
            bad.append(f"{end}: PTT left keyed (unbalanced)")
    if not any_seen:
        return Finding("ptt_alternation", ABSENT, tuple(evidence),
                       "no PTT notifications during the session")
    if bad:
        return Finding("ptt_alternation", FAIL, tuple(bad + evidence),
                       "PTT must strictly alternate ON/OFF and end unkeyed")
    return Finding("ptt_alternation", PASS, tuple(evidence),
                   "PTT strictly alternated and balanced at both ends")


@_probe("both_ends_disconnected", "paired")
def probe_both_ends_disconnected(ctx_a: ProbeContext, ctx_b: ProbeContext,
                                 ev: dict, la, lb) -> Finding:
    if not ev["connected"]:
        return Finding("both_ends_disconnected", FAIL,
                       detail="no session established between the pair")
    missing = [end for end, done in (("A", ev["a_disc"]), ("B", ev["b_disc"]))
               if not done]
    if missing:
        return Finding("both_ends_disconnected", FAIL,
                       (f"no DISCONNECTED at {'/'.join(missing)}",),
                       "both ends must see DISCONNECTED after a graceful close")
    return Finding("both_ends_disconnected", PASS,
                   detail="both ends saw DISCONNECTED")


@_probe("pending_before_connected", "paired")
def probe_pending_before_connected(ctx_a: ProbeContext, ctx_b: ProbeContext,
                                   ev: dict, la, lb) -> Finding:
    if not ev["connected"]:
        return Finding("pending_before_connected", FAIL,
                       detail="no session established between the pair")
    t_pending = next((t for t, l in lb
                      if l.kind == NOTIFICATION and l.name == "PENDING"), None)
    t_conn = next((t for t, l in lb
                   if l.kind == NOTIFICATION and l.name == "CONNECTED"), None)
    if t_pending is not None and t_conn is not None and t_pending < t_conn:
        return Finding("pending_before_connected", PASS,
                       (f"PENDING {t_conn - t_pending:.2f}s before CONNECTED",),
                       "callee announced PENDING before CONNECTED")
    if t_pending is not None:
        return Finding("pending_before_connected", WARN,
                       ("PENDING seen, but not before CONNECTED",),
                       "PENDING ordering is off")
    return Finding("pending_before_connected", ABSENT,
                   detail="no PENDING before CONNECTED — this is a paired probe "
                          "and sabir's ARQ never calls the hook; kestrel's "
                          "LoopbackModem no longer calls it at all. kestrel's "
                          "real core does emit it, on the responder and ahead "
                          "of CONNECTED, whenever the pair links")


@_probe("listen_off_side_effect_free", "paired")
def probe_listen_off_side_effect_free(ctx_a: ProbeContext, ctx_b: ProbeContext,
                                      ev: dict, la, lb) -> Finding:
    # Post-session, A is idle: LISTEN OFF to it must draw OK and disturb
    # neither end. (The full arbitration case — a third, non-winning modem —
    # needs three stations; this covers the idle-modem half of the claim.)
    qa = ctx_a.client.subscribe()
    qb = ctx_b.client.subscribe()
    try:
        line = _reply(ctx_a, "LISTEN OFF")
        time.sleep(ctx_a.settle_s)
        alarming = [l.raw.strip() for l in _drain_lines(qa) + _drain_lines(qb)
                    if l.name in ("DISCONNECTED", "CONNECTED", "CANCELPENDING")]
    finally:
        ctx_a.client.unsubscribe(qa)
        ctx_b.client.unsubscribe(qb)
    if line is None or line.kind != REPLY_OK:
        return Finding("listen_off_side_effect_free", FAIL,
                       (f"LISTEN OFF -> {_got(line)}",),
                       "LISTEN OFF to an idle modem must draw OK")
    if alarming:
        return Finding("listen_off_side_effect_free", FAIL, tuple(alarming),
                       "LISTEN OFF to an idle modem disturbed session state")
    return Finding("listen_off_side_effect_free", PASS,
                   ("LISTEN OFF -> OK; no session notifications followed",),
                   "LISTEN OFF to an idle modem is side-effect-free")


# -- runners -----------------------------------------------------------------

class _Collector:
    """Passively records every classified cmd line with its arrival time
    (seconds relative to collection start)."""

    def __init__(self, client: ModemClient) -> None:
        self.lines: list[tuple[float, Line]] = []
        self.t0 = time.monotonic()
        self._client = client
        self._q = client.subscribe()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"{client.name}-conf-collect")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(item, Line):
                self.lines.append((time.monotonic() - self.t0, item))

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._client.unsubscribe(self._q)
        return time.monotonic() - self.t0


def _flat(result) -> list[Finding]:
    return list(result) if isinstance(result, (list, tuple)) else [result]


def _guard(pid: str, fn: Callable, *args) -> list[Finding]:
    try:
        return _flat(fn(*args))
    except Exception as exc:            # a probe crash is a finding, not a crash
        return [Finding(pid, FAIL, detail=f"probe raised {exc!r}")]


def run_single(ctx: ProbeContext) -> list[Finding]:
    """Run the single-modem catalog against one attached modem. Probes run in
    registry order; the passive collectors span the whole run so IAMALIVE and
    vocabulary judgments never add a stall of their own."""
    collector = _Collector(ctx.client)
    findings: list[Finding] = []
    try:
        for probe in [p for p in PROBES.values() if p.mode == "single"]:
            findings += _guard(probe.id, probe.fn, ctx)
        if ctx.iamalive_window_s is not None:
            remaining = ctx.iamalive_window_s - (time.monotonic() - collector.t0)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        window = collector.stop()
    for probe in [p for p in PROBES.values() if p.mode == "passive"]:
        findings += _guard(probe.id, probe.fn, ctx, collector.lines, window)
    return findings


def run_paired(ctx_a: ProbeContext, ctx_b: ProbeContext) -> list[Finding]:
    """Run the paired catalog over two linked modems: one shared choreography
    (A connects to B, transfers, disconnects), then per-probe judgment."""
    a, b = ctx_a.client, ctx_b.client
    col_a, col_b = _Collector(a), _Collector(b)
    ev = {"connected": False, "conn_a": None, "conn_b": None,
          "delivered": 0, "a_disc": False, "b_disc": False}
    try:
        _reply(ctx_a, f"MYCALL {ctx_a.mycall}")
        _reply(ctx_b, f"MYCALL {ctx_b.mycall}")
        _reply(ctx_b, "LISTEN ON")
        a.bump_epoch()
        b.bump_epoch()
        line = _reply(ctx_a, f"CONNECT {ctx_a.mycall} {ctx_b.mycall}")
        if line is not None and line.kind == REPLY_OK and \
                _wait(lambda: a.connected and b.connected, ctx_a.connect_timeout_s):
            ev["connected"] = True
            ev["conn_a"], ev["conn_b"] = a.last_connected, b.last_connected
            payload = _payload(ctx_a.payload_n)
            a.send_data(payload, label="paired")
            ev["delivered"] = len(_read_exact(b, len(payload), ctx_a.data_timeout_s))
            _reply(ctx_a, "DISCONNECT")
            ev["a_disc"] = _wait(lambda: not a.connected, ctx_a.drain_timeout_s)
            ev["b_disc"] = _wait(lambda: not b.connected, ctx_a.drain_timeout_s)
        if a.connected or b.connected:      # never leave a wedged session
            _reply(ctx_a, "ABORT")
            _reply(ctx_b, "ABORT")
            _wait(lambda: not (a.connected or b.connected), ctx_a.abort_timeout_s)
    finally:
        col_a.stop()
        col_b.stop()
        a.bump_epoch()
        b.bump_epoch()
    findings: list[Finding] = []
    for probe in [p for p in PROBES.values() if p.mode == "paired"]:
        findings += _guard(probe.id, probe.fn, ctx_a, ctx_b, ev,
                           col_a.lines, col_b.lines)
    return findings


# -- golden verdict table ----------------------------------------------------
# Empirical, per the module docstring. Deviations — not raw FAIL/ABSENT —
# gate `creance selftest`; flip an entry, and re-derive it against the live
# target, when the modem gains a behavior marked ABSENT.

_SINGLE_COMMON = {
    "reply_discipline": PASS,
    "unknown_verb_wrong": PASS,
    "mycall_arity": PASS,
    "bw_forms": PASS,
    "compression_args": PASS,
    "listen_semantics": PASS,
    "version_shape": PASS,
    "buffer_cadence": PASS,
    "disconnect_flush_order": PASS,
    "abort_prompt": PASS,
    "bitrate_absent": ABSENT,       # spec §7.3, implemented by neither target
    "sn_absent": ABSENT,            # spec §7.3, implemented by neither target
    "vocabulary_sweep": PASS,
}

GOLDEN: dict[str, dict[str, str]] = {
    "kestrel-loopback": _SINGLE_COMMON | {
        "preconnect_queue": PASS,       # LoopbackModem buffers pre-connect writes,
                                        # flushes on CONNECT (spec §7.4 resolved)
        "iamalive_cadence": PASS,       # spawned with --iamalive-interval 2
    },                                  # bitrate_absent went back to ABSENT with
                                        # _SINGLE_COMMON when the loopback
                                        # stopped inventing BITRATE
    # The real ARQ core rather than the loopback that stands in for it above.
    # The two gaps this entry used to record are closed in the core. Pre-connect
    # host bytes are queued and flushed on CONNECTED instead of dropped; and
    # ABORT sends a disconnect burst instead of closing purely locally, backed
    # by a link-death timer for peers that never hear it — so one ABORT no
    # longer leaves the far end CONNECTED for good, refusing every later
    # connect-request and wedging every probe downstream of it. bitrate_absent
    # and sn_absent stay ABSENT from _SINGLE_COMMON because the core has no
    # hook for either.
    "kestrel-pair": _SINGLE_COMMON | {
        "preconnect_queue": PASS,
        "buffer_cadence": PASS,         # was FAIL, and the FAIL was ours: the
                                        # connect after preconnect_queue's ABORT
                                        # teardown was asked once, and the core
                                        # refuses that one request while it
                                        # returns to idle. _establish retries
                                        # inside its budget now, so this grades
                                        # the BUFFER cadence of the real ARQ
                                        # core, which is what it is named for
        "iamalive_cadence": ABSENT,     # 60 s fixed interval >> suite window,
                                        # exactly as for sabir below. Its PASS
                                        # was an artefact of the entry above:
                                        # the only thing that ever stretched
                                        # the window past 1.5 intervals was
                                        # buffer_cadence burning a 60 s connect
                                        # timeout, so the two flipped together
        "connected_fields": PASS,
        "ptt_alternation": PASS,        # connect-request retries alternate and
                                        # end unkeyed, session or no session
        "both_ends_disconnected": PASS,
        "pending_before_connected": PASS,
        "listen_off_side_effect_free": PASS,
    },
    # NO "shrike" ENTRY, deliberately. An earlier one graded shrike against
    # this catalog -- MYCALL, BW forms, COMPRESSION, LISTEN, BUFFER cadence --
    # on the premise that shrike would grow a VARA-style TCP server. That
    # premise was wrong and was dropped: Winlink reaches PACTOR over the SCS PTC
    # serial dialect, so shrike emulates a PTC-IIIusb over a pty and will never
    # emit any of these verbs. A table full of expectations an implementation
    # cannot meet is worse than no table, because it looks like coverage.
    #
    # Grading shrike needs a third dialect client (PTC CRC hostmode) that
    # creance does not have yet. Until it does, shrike is ungraded and says so.
}


def diff_golden(target: str, findings: Iterable[Finding],
                tables: dict[str, dict[str, str]] | None = None) -> list[str]:
    """Deviations of a findings set from the golden table (empty == selftest
    pass). Probes are compared by first finding per id. `tables` lets the
    structured catalog reuse this with its own golden entries."""
    expected = (GOLDEN if tables is None else tables)[target]
    got: dict[str, str] = {}
    for f in findings:
        got.setdefault(f.probe, f.verdict)
    devs = [f"{pid}: expected {want}, " +
            (f"got {got[pid]}" if pid in got else "not run")
            for pid, want in expected.items() if got.get(pid) != want]
    devs += [f"{pid}: unexpected probe (not in golden table)"
             for pid in got if pid not in expected]
    return devs


# -- rendering ---------------------------------------------------------------

def render(findings: Iterable[Finding]) -> str:
    findings = list(findings)
    if not findings:
        return "no findings"
    width = max(len(f.probe) for f in findings)
    out = []
    for f in findings:
        out.append(f"{f.verdict:<6} {f.probe:<{width}}  {f.detail}")
        for e in f.evidence:
            out.append(" " * 7 + "| " + e)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    out.append(f"{len(findings)} probes: "
               + ", ".join(f"{counts[v]} {v}" for v in VERDICTS if v in counts))
    return "\n".join(out)
