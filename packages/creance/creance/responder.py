# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The responder daemon: the autonomous end of the link.

One single-threaded event loop consumes a merged queue fed by every attached
modem (cmd lines via the client's pub/sub, attach/detach callbacks, timer
ticks). Scenario execution runs in a worker thread so the loop keeps
supervising — watchdog, arbitration, attach retries — while a session is live.

    INIT -> LISTEN_ALL -> ARBITRATING -> SESSION -> DRAINING -> LISTEN_ALL
                 ^             |
                 +-------------+  (pending set empties or times out)

Arbitration rules, from the design: PENDING is cheap and often spurious, so it
never disturbs another modem; only CONNECTED decides, and CONNECTED without a
preceding PENDING is the norm today (no modem emits PENDING yet). The winner
keeps LISTEN ON; every other modem is muted for the session's duration.

Wedge avoidance is the whole point of DRAINING: the worker gets a cancel event
that every blocking primitive honours, and LISTEN is never re-armed until the
worker is dead or fenced by an epoch bump — a zombie worker holding a stale
read would otherwise eat the next session's HELLO.

The unattended-site duties ride the same loop: a modem whose heartbeat has
stopped is recycled even though it is still attached (attached-but-deaf is the
failure nobody is there to see), status.json is rewritten every tick, and
retention runs between sessions on its own thread so a month of results never
blocks arbitration.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from . import scenarios
from hfhost.client import EpochFenced, ModemError
from .config import Config
from .hproto import Desync, NotCwp
from .metrics import compute
from .report import write_session
from .scenarios import CwpSession, ScenarioResult, SessionCancelled
from hfhost.supervisor import Supervisor
from hfhost.transcript import Transcript
from hfhost.transcript import read as read_transcript
from . import link as linkmod
from .link import Link, LinkEvent, open_link

INIT = "INIT"
LISTEN_ALL = "LISTEN_ALL"
ARBITRATING = "ARBITRATING"
SESSION = "SESSION"
DRAINING = "DRAINING"

EV_EVENT = "event"
EV_ATTACHED = "attached"
EV_DETACHED = "detached"
EV_TICK = "tick"
EV_DONE = "done"
EV_STOP = "stop"

RETENTION_MAX_AGE_S = 30 * 86400.0
RETENTION_MAX_BYTES = 2 * 1024 ** 3
TRANSCRIPT_CAP = 5 * 1024 ** 2           # responder.jsonl rotation, as modem logs
FANIN_DEPTH = 2000                       # per-modem cmd lines in flight


def _live_file(path: Path) -> bool:
    """Files a running daemon holds open. Never collectible however old; their
    rotated siblings (responder.jsonl.1, modem-x.log.1) are."""
    return (path.name in ("responder.jsonl", "status.json")
            or (path.name.startswith("modem-") and path.name.endswith(".log")))


def _entry(path: Path, session: bool) -> tuple[float, int, Path, bool]:
    if not session:
        st = path.stat()
        return (st.st_mtime, st.st_size, path, False)
    stats = [p.stat() for p in path.rglob("*") if p.is_file()]
    return (max((s.st_mtime for s in stats), default=path.stat().st_mtime),
            sum(s.st_size for s in stats), path, True)


def sweep_results(root: str | Path, *, max_age_s: float = RETENTION_MAX_AGE_S,
                  max_bytes: int = RETENTION_MAX_BYTES,
                  now: float | None = None) -> list[Path]:
    """Enforce retention caps on the results tree — a remote disk must not
    fill. Collects session directories (results/<site>/<date>/<sid>/) and the
    stray files that grow beside them: campaign-<ts>.txt reports, rotated
    daemon transcripts and modem logs. Drops everything past max_age_s, then
    the oldest remaining until the tree fits in max_bytes. The newest session
    and the files a live daemon holds open are never dropped. Returns what was
    removed."""
    root = Path(root)
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    entries: list[tuple[float, int, Path, bool]] = []
    for site in root.iterdir():
        if not site.is_dir():
            continue
        for child in site.iterdir():
            if child.is_file():
                if not _live_file(child):
                    entries.append(_entry(child, False))
            elif child.is_dir():
                entries += [_entry(p, p.is_dir()) for p in child.iterdir()
                            if p.is_dir() or not _live_file(p)]
    entries.sort()

    newest = max((e for e in entries if e[3]), default=None)
    live = [e for e in entries if e is not newest]
    removed = [e[2] for e in live if now - e[0] > max_age_s]
    live = [e for e in live if now - e[0] <= max_age_s]
    total = (newest[1] if newest else 0) + sum(e[1] for e in live)
    while total > max_bytes and live:
        mtime, size, path, _ = live.pop(0)
        total -= size
        removed.append(path)

    for path in removed:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    for parent in {p.parent for p in removed}:
        while parent != root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return removed


class _Fanin(queue.Queue):
    """A client subscriber queue that forwards straight to the merged loop
    queue — the client's cmd-reader thread hands the line over with no
    per-modem forwarder thread in between.

    Bounded by lines still undispatched: while the loop blocks in DRAINING
    (worker join, settle) every attached modem keeps talking, and an unbounded
    fan-in would let one chatty modem grow the loop queue without limit.
    Overflow is dropped and counted rather than blocking a modem's cmd reader,
    so put() ignores block/timeout by design."""

    def __init__(self, sink: queue.Queue, modem: str,
                 depth: int = FANIN_DEPTH) -> None:
        super().__init__()
        self._sink = sink
        self._modem = modem
        self._depth = depth
        self._lock = threading.Lock()
        self.inflight = 0
        self.dropped = 0
        self.hangups = 0

    def put(self, item, block=True, timeout=None) -> None:
        with self._lock:
            if getattr(item, "kind", None) == linkmod.DISCONNECTED:
                # counted here, where every event passes, because the loop is
                # not dispatching while it drains: a subscription taken out at
                # hangup time misses the DISCONNECTED that already landed.
                self.hangups += 1
            if self.inflight >= self._depth:
                self.dropped += 1
                return
            self.inflight += 1
        self._sink.put((EV_EVENT, self._modem, item))

    def dispatched(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)


class Responder:
    """Any-protocol listener daemon. start() brings the modems up, serve() /
    run_forever() runs the loop, stop() tears it down."""

    def __init__(self, config: Config, *, host: str = "127.0.0.1",
                 results_root: str | Path | None = None,
                 supervisor: Supervisor | None = None,
                 tick_s: float = 1.0,
                 retry_s: float = 5.0,
                 attach_timeout_s: float = 10.0,
                 join_timeout_s: float = 10.0,
                 settle_s: float = 5.0,
                 sink_idle_s: float = 5.0,
                 attach_fail_limit: int = 3) -> None:
        self.config = config
        self.host = host
        self.results_root = Path(results_root if results_root is not None
                                 else config.site.results_dir)
        self.tick_s = tick_s
        self.retry_s = retry_s
        self.attach_timeout_s = attach_timeout_s
        self.join_timeout_s = join_timeout_s
        self.settle_s = settle_s
        self.sink_idle_s = sink_idle_s
        self.attach_fail_limit = attach_fail_limit

        self.state = INIT
        self.sessions_completed = 0
        self.last_metrics = None
        self.started_at = time.monotonic()

        site_dir = self.results_root / config.site.name
        site_dir.mkdir(parents=True, exist_ok=True)
        self.transcript = Transcript(str(site_dir / "responder.jsonl"), "daemon")
        self.status_path = site_dir / "status.json"

        self._owns_supervisor = supervisor is None
        self.supervisor = (Supervisor(config, str(site_dir))
                           if supervisor is None else supervisor)

        self._events: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._started = False

        self._links: dict[str, Link] = {}
        self._fanin: dict[str, _Fanin] = {}
        self._held: set[str] = set()
        self._up: dict[str, bool] = {m.name: False for m in config.modems}
        self._respawn_base: dict[str, int] = {}
        self._next_retry: dict[str, float] = {}
        self._pending: dict[str, float] = {}
        self._sweep_seen: dict[str, float] = {}
        self._dropped: dict[str, int] = {}

        # transcripts whose last writer has to go away before they can close
        self._closing: list[tuple[Callable[[], bool], Transcript]] = []
        self._sweeper: threading.Thread | None = None
        self._last_sweep = float("-inf")

        self._winner: str | None = None
        self._sid: str | None = None
        self._session_dir: Path | None = None
        self._session_t: Transcript | None = None
        self._meta: dict = {}
        self._cancel = threading.Event()
        self._cancel_reason = "complete"
        self._worker: threading.Thread | None = None
        self._result: ScenarioResult | None = None
        self._deadline = 0.0
        self._hangups_at = 0

    # -- introspection -----------------------------------------------------

    @property
    def modem_states(self) -> dict[str, str]:
        return {name: ("up" if up else "down") for name, up in self._up.items()}

    @property
    def pending(self) -> dict[str, float]:
        return dict(self._pending)

    @property
    def winner(self) -> str | None:
        return self._winner if self.state in (SESSION, DRAINING) else None

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "Responder":
        """Idempotent INIT: bring every modem up, run with whatever subset
        answers, retry the rest on the timer."""
        if self._started:
            return self
        self._started = True
        for m in self.config.modems:
            self._bring_up(m.name)
        self.state = LISTEN_ALL
        self._ticker = threading.Thread(target=self._tick_loop,
                                        name="responder-tick", daemon=True)
        self._ticker.start()
        return self

    def serve(self, until=None, timeout: float | None = None) -> None:
        """Run the loop until stop(), until() turns true, or timeout."""
        self.start()
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._stop.is_set():
            if until is not None and until():
                return
            if self._check_watchdog():
                continue
            wait = self.tick_s
            if self.state == SESSION:
                # wake on the watchdog itself, not on the next tick: a coarse
                # tick would let a scenario's own timeout beat the ABORT
                wait = min(wait, max(self._deadline - time.monotonic(), 0.0))
            if deadline is not None:
                wait = min(wait, deadline - time.monotonic())
                if wait <= 0:
                    return
            try:
                event = self._events.get(timeout=wait)
            except queue.Empty:
                continue
            self._dispatch(event)

    def run_forever(self) -> None:
        self.serve()

    def stop(self) -> None:
        """Tear down. A session in flight is cancelled without a session.json;
        its transcript survives and metrics.from_files() reconstructs it."""
        self._stop.set()
        self._events.put((EV_STOP, None, None))
        self._cancel.set()
        if self._ticker is not None:
            self._ticker.join(timeout=self.tick_s + 1.0)
        if self._worker is not None:
            self._worker.join(timeout=self.join_timeout_s)
        if self._sweeper is not None:
            self._sweeper.join(timeout=self.tick_s + 1.0)
        for client in self._links.values():
            client.set_transcript(None)
            client.close()
        for name in sorted(self._held):
            self.supervisor.release(name)
        self._held.clear()
        if self._session_t is not None:
            self._session_t.close()
            self._session_t = None
        for _, t in self._closing:
            t.close()
        self._closing.clear()
        if self._owns_supervisor:
            self.supervisor.stop_all()
        self.transcript.close()

    # -- modem bring-up ----------------------------------------------------

    def _bring_up(self, name: str) -> bool:
        cfg = self.config.modem(name)
        client = self._links.get(name)
        if client is None:
            client = open_link(cfg, self.host, self.transcript,
                               attach_timeout_s=self.attach_timeout_s)
            self._fanin[name] = _Fanin(self._events, name)
            client.subscribe(self._fanin[name])
            client.configure(self.config.site.mycall)
            client.set_listen(True)
            self._links[name] = client
        if client.attached:
            self._up[name] = True
            return True
        self._next_retry[name] = time.monotonic() + self.retry_s
        try:
            if not self.supervisor.ensure(name):
                raise ModemError(f"{name}: not reachable")
            if name not in self._held:
                # one holder per modem for the daemon's life: retries and
                # recycles keep it, stop() drops it. Acquiring per attempt
                # would leave the process alive after shutdown.
                self.supervisor.acquire(name)
                self._held.add(name)
            client.attach()
        except (ModemError, OSError) as exc:
            self.transcript.error(name, f"bring-up failed: {exc}")
            self._up[name] = False
            self._escalate(name, client)
            return False
        self._respawn_base[name] = 0
        self._up[name] = True
        if cfg.quirks.version_reply:
            self._safe(name, "version", client.request_version)
        return True

    def _escalate(self, name: str, client: Link) -> None:
        """Repeated attach failures against a live spawned process mean the
        server is wedged (a half-attach jams their accept loop); only a
        respawn clears it. Externally-launched modems can't be helped."""
        base = self._respawn_base.get(name, 0)
        if client.attach_failures - base < self.attach_fail_limit:
            return
        proc = self.supervisor.process_for(name)
        if proc is None or not proc.alive:
            return
        self.transcript.note(name, "kill_and_respawn",
                             attach_failures=client.attach_failures)
        self._respawn_base[name] = client.attach_failures
        self._safe(name, "respawn", lambda: self.supervisor.kill_and_respawn(name))

    def _recycle(self, name: str, why: str = "idle_recycle") -> None:
        """Deaf-modem recycle: nothing heard for max_idle_recycle_s, or the
        modem's own IAMALIVE heartbeat stopped while it was still attached."""
        self.transcript.note(name, why)
        client = self._links.pop(name, None)
        self._fanin.pop(name, None)
        if client is not None:
            client.set_transcript(None)
            client.close()
        self._up[name] = False
        self._respawn_base[name] = 0
        self._safe(name, "respawn", lambda: self.supervisor.kill_and_respawn(name))
        self._bring_up(name)

    # -- event loop --------------------------------------------------------

    def _tick_loop(self) -> None:
        while not self._stop.wait(self.tick_s):
            self._events.put((EV_TICK, None, None))

    def _dispatch(self, event) -> None:
        kind, modem, payload = event
        try:
            if kind == EV_EVENT:
                fan = self._fanin.get(modem)
                if fan is not None:
                    fan.dispatched()
                if isinstance(payload, LinkEvent):
                    self._on_link_event(modem, payload)
            elif kind == EV_TICK:
                self._on_tick()
            elif kind == EV_ATTACHED:
                self._up[modem] = True
            elif kind == EV_DETACHED:
                self._on_detached(modem)
            elif kind == EV_DONE:
                self._on_done(modem, payload)
        except Exception as exc:      # a malformed frame is a finding, not a crash
            self._log(self.transcript, modem or "-", f"loop error on {kind}: {exc!r}")

    def _on_link_event(self, modem: str, ev: LinkEvent) -> None:  # noqa: C901
        if ev.kind == linkmod.PENDING:
            self._on_pending(modem)
        elif ev.kind == linkmod.CANCELPENDING:
            self._on_cancel_pending(modem)
        elif ev.kind == linkmod.CONNECTED:
            self._on_connected(modem, ev)
        elif ev.kind == linkmod.DISCONNECTED:
            self._on_disconnected(modem)
        elif ev.kind == linkmod.ATTACHED:
            self._up[modem] = True
        elif ev.kind == linkmod.DETACHED:
            self._on_detached(modem)

    def _on_pending(self, modem: str) -> None:
        self.supervisor.note_activity(modem)
        if self.state not in (LISTEN_ALL, ARBITRATING):
            self.transcript.note(modem, "pending_ignored", state=self.state)
            return
        self._pending.setdefault(modem, time.monotonic())
        self.state = ARBITRATING

    def _on_cancel_pending(self, modem: str) -> None:
        self._pending.pop(modem, None)
        if self.state == ARBITRATING and not self._pending:
            self.state = LISTEN_ALL

    def _on_connected(self, modem: str, ev: LinkEvent) -> None:
        self.supervisor.note_activity(modem)
        if self.state in (LISTEN_ALL, ARBITRATING):
            self._begin_session(modem, _describe(ev))
        elif modem != self._winner:
            # one audio channel should yield one handshake; two is a defect
            (self._session_t or self.transcript).conf(
                modem, "double_connect", "WARN",
                evidence=f"CONNECTED during {self.state} on {self._winner}")
            self._safe(modem, "abort", lambda: self._links[modem].abort())
        else:
            self.transcript.note(modem, "connected_repeat", state=self.state)

    def _on_disconnected(self, modem: str) -> None:
        if self.state == SESSION and modem == self._winner:
            self._end_session("disconnected")
        else:
            # kestrel emits spurious DISCONNECTED when idle; be idempotent
            self.transcript.note(modem, "disconnected_outside_session",
                                 state=self.state)

    def _on_detached(self, modem: str) -> None:
        self._up[modem] = False
        self._pending.pop(modem, None)
        if self.state == SESSION and modem == self._winner:
            self._end_session("detach")
        elif self.state == ARBITRATING and not self._pending:
            self.state = LISTEN_ALL

    def _on_done(self, modem: str, sid: str) -> None:
        if self.state == SESSION and sid == self._sid:
            self._end_session("complete")

    def _check_watchdog(self) -> bool:
        if self.state != SESSION or time.monotonic() < self._deadline:
            return False
        self._safe(self._winner, "abort",
                   lambda: self._links[self._winner].abort())
        self._end_session("watchdog")
        return True

    def _on_tick(self) -> None:
        now = time.monotonic()
        if self.state == ARBITRATING:
            timeout = self.config.responder.pending_timeout_s
            for modem, since in list(self._pending.items()):
                if now - since >= timeout:
                    self.transcript.note(modem, "pending_expired")
                    del self._pending[modem]
            if not self._pending:
                self.state = LISTEN_ALL
        idle = self.state in (LISTEN_ALL, ARBITRATING)
        for name in list(self._up):
            if not self._up[name]:
                if now >= self._next_retry.get(name, 0.0):
                    self._bring_up(name)
            elif not idle:
                continue
            elif self._links[name].stale():
                # attached but deaf: the heartbeat that flows from attach on
                # every server has stopped, so nothing would ever be heard
                self._recycle(name, "stale_recycle")
            elif self.supervisor.idle_recycle_due(name):
                self._recycle(name)
            else:
                self._sweep_idle(name, now)
        self._reap()
        self._note_drops()
        self._safe(None, "status", self._write_status)
        if idle:
            self._maintain(now)

    def _sweep_idle(self, name: str, now: float) -> None:
        """Between sessions the rx buffer must stay empty: a late echo left
        sitting there would be handed to the next session's deframer. The
        epoch bump also fences any straggler read still holding the old one.
        Deliberately not done at session start — the peer's HELLO can beat the
        CONNECTED notification through the loop, and eating it would wedge the
        session until the watchdog. The same HELLO is lost a tick later if the
        sweep is left unguarded: a drain long enough to outlast a tick leaves
        stale ticks queued ahead of the CONNECTED that opens the next session,
        and they are dispatched — as a burst, in an already-idle state — while
        the peer's HELLO sits in the rx buffer.

        So bytes are given tick_s of grace from the moment a sweep first sees
        them: any session that owns them opens within microseconds of that
        (its CONNECTED is already queued), while a genuine late echo just gets
        swept one tick later. Grace, not `client.connected`: a modem that never
        sends DISCONNECTED would keep that flag forever and lose the fence
        altogether."""
        client = self._links[name]
        # a one-byte peek: presence, without consuming what may be a live HELLO
        if client.peek(1, timeout=0.0):
            if now - self._sweep_seen.setdefault(name, now) < self.tick_s:
                return
        self._sweep_seen.pop(name, None)
        leftover = client.bump_epoch()
        if leftover:
            self.transcript.conf(name, "data_outside_session", "EXTRA",
                                 bytes=len(leftover), when="between_sessions")

    # -- session -----------------------------------------------------------

    def _begin_session(self, modem: str, connected_raw: str) -> None:
        self._winner = modem
        self._pending.clear()
        client = self._links[modem]
        for name, other in self._links.items():
            if name != modem:
                self._safe(name, "listen off", lambda o=other: o.set_listen(False))

        sid = f"{time.strftime('%Y%m%dT%H%M%S')}-{modem}-{secrets.token_hex(2)}"
        sid_dir = (self.results_root / self.config.site.name
                   / time.strftime("%Y-%m-%d") / sid)
        sid_dir.mkdir(parents=True, exist_ok=True)
        t = Transcript(str(sid_dir / "transcript.jsonl"), sid)
        # the CONNECTED that won arbitration landed in the daemon transcript;
        # copy it in so handshake latency has its zero point
        t.cmd_rx(modem, connected_raw)
        client.set_transcript(t)

        mcfg = self.config.modem(modem)
        self._sid, self._session_dir, self._session_t = sid, sid_dir, t
        self._meta = {
            "sid": sid, "site": self.config.site.name, "modem": modem,
            "version": client.version or "", "rev": mcfg.rev, "label": "",
            "scenario": "", "params": {},
            # the peer's own sid/call: the only key that joins this record to
            # the initiator's, two sites apart with no shared clock
            "peer_sid": "", "peer_call": "",
            "wall_start": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_end": "",
            "sim_time": mcfg.sim_time, "outcome": None,
        }
        self._cancel = threading.Event()
        self._cancel_reason = "complete"
        self._result = None
        self._deadline = time.monotonic() + self.config.responder.max_session_s
        fan = self._fanin.get(modem)
        self._hangups_at = fan.hangups if fan is not None else 0
        self.state = SESSION
        self._worker = threading.Thread(
            target=self._session_worker,
            args=(client, t, sid, self._cancel, client.epoch),
            name=f"responder-session-{modem}", daemon=True)
        self._worker.start()

    def _session_worker(self, client: Link, t: Transcript, sid: str,
                        cancel: threading.Event, epoch: int) -> None:
        mcfg = self.config.modem(client.name)
        hello_timeout = (mcfg.quirks.hello_timeout_s
                         or self.config.responder.hello_timeout_s)
        session = CwpSession(client, t, sid=sid, mycall=self.config.site.mycall,
                             cancel=cancel, epoch=epoch,
                             recv_timeout_s=self.config.responder.max_session_s)
        result = ScenarioResult("failed:no_result", {})
        try:
            try:
                hello = session.await_hello(timeout=hello_timeout)
            except NotCwp as exc:
                result = self._sink(session, exc.buffered)
            else:
                if hello is None:
                    result = self._sink(session, b"")
                else:
                    self._meta["scenario"] = hello.scenario
                    self._meta["params"] = dict(hello.params)
                    self._meta["peer_sid"] = hello.sid
                    self._meta["peer_call"] = hello.call
                    result = scenarios.respond(session, hello)
        except Desync as exc:
            session.record_desync(exc)
            result = ScenarioResult("failed:cwp_desync",
                                    {"sink_bytes": session.drain_sink()})
        except (SessionCancelled, EpochFenced):
            result = ScenarioResult("aborted:watchdog", {})
        except (ModemError, OSError) as exc:
            result = ScenarioResult(f"failed:{type(exc).__name__.lower()}", {})
        except Exception as exc:
            result = ScenarioResult("failed:internal", {})
            self._log(t, client.name, f"session worker error: {exc!r}")
        finally:
            self._result = result
            self._events.put((EV_DONE, client.name, sid))

    def _sink(self, session: CwpSession, buffered: bytes) -> ScenarioResult:
        if self.config.responder.on_plain_peer == "disconnect":
            session.note("sink", reason="not_cwp", len=len(buffered))
            self._safe(session.client.name, "disconnect",
                       lambda: session.client.disconnect())
            return ScenarioResult("plain_peer", {"policy": "disconnect",
                                                 "bytes": len(buffered)})
        return scenarios.fallback_sink(session, buffered, idle_s=self.sink_idle_s)

    def _end_session(self, reason: str) -> None:
        self._cancel_reason = reason
        self._cancel.set()
        self._drain()

    def _drain(self) -> None:
        self.state = DRAINING
        modem = self._winner
        client = self._links.get(modem)
        t = self._session_t

        self._sweep_seen.pop(modem, None)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(self.join_timeout_s)
        leftover = client.bump_epoch() if client is not None else b""
        if worker is not None and worker.is_alive():
            # the epoch bump fences its next read; give it a beat to die
            worker.join(min(self.join_timeout_s, 1.0))
        stuck = worker is not None and worker.is_alive()
        if t is not None:
            if stuck:
                t.conf(modem, "worker_stuck", "WARN",
                       evidence=f"scenario thread outlived join in {self._cancel_reason}")
            if leftover:
                t.conf(modem, "data_outside_session", "EXTRA",
                       bytes=len(leftover), when="after_session")

        if client is not None:
            self._hangup(modem, client, t)

        metrics = self._finalize(modem, t, stuck)
        if stuck and t is not None:
            # the worker still holds it; a later tick closes it once dead
            self._closing.append((lambda w=worker: not w.is_alive(), t))
        if client is not None:
            client.set_transcript(None)

        if modem is not None and (client is None or not client.attached):
            # a held attachment IS the liveness signal: probing a modem we
            # already own would queue a bogus zero-length session in the
            # server's own accept backlog, once per session, forever
            self._safe(modem, "ensure", lambda: self.supervisor.ensure(modem))
        self._pending.clear()
        for name, other in self._links.items():
            self._safe(name, "listen on", lambda o=other: o.set_listen(True))
        self._winner = self._sid = self._session_dir = self._session_t = None
        self.state = LISTEN_ALL
        # published last: sessions_completed means drained and listening again
        if metrics is not None:
            self.last_metrics = metrics
            self.sessions_completed += 1

    def _hangup(self, modem: str, client: Link,
                t: Transcript | None) -> None:
        """The initiator owns the hangup for every scenario (scenarios.py), so
        a completed session waits for its DISCONNECTED instead of racing it.
        The REPORT left on the data socket and DISCONNECT would leave on the
        cmd socket; nothing at the server orders the two, and a server that
        checks only its own tx queue would hang up before the REPORT landed.
        So: flush the modem's queue, let the peer hang up, and do it ourselves
        only if it never does."""
        if self._hung_up(modem) or not client.connected:
            return
        self._flush(modem, client, t)
        if self._cancel_reason == "complete" and self._await_hangup(modem):
            return
        self._safe(modem, "disconnect", lambda: client.disconnect())
        self._await_hangup(modem)

    def _hung_up(self, modem: str) -> bool:
        """Has the peer dropped the link since this session began? A count of
        DISCONNECTED events, never client.connected: a peer with no cool-down
        between sessions dials straight back in, and the flag is true again
        while the session it belongs to is over."""
        fan = self._fanin.get(modem)
        return fan is not None and fan.hangups > self._hangups_at

    def _await_hangup(self, modem: str) -> bool:
        """Wait out settle_s for that hangup. Polled, because the alternative —
        waiting on the next DISCONNECTED event — hangs up on the peer's *next*
        session when the one we were waiting for arrived a moment early."""
        deadline = time.monotonic() + self.settle_s
        while not self._hung_up(modem):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True

    def _flush(self, modem: str, client: Link,
               t: Transcript | None) -> None:
        deadline = time.monotonic() + self.settle_s
        while client.queue_bytes and client.connected:
            if time.monotonic() >= deadline:
                (t or self.transcript).conf(
                    modem, "flush_before_disconnect", "WARN",
                    evidence=f"{client.queue_bytes} B still queued after "
                             f"{self.settle_s:g}s")
                return
            time.sleep(0.02)

    def _finalize(self, modem: str, t: Transcript | None, stuck: bool):
        if t is None or self._session_dir is None:
            return None
        result = self._result or ScenarioResult("failed:no_result", {})
        outcome = self._final_outcome(result)
        self._meta["outcome"] = outcome
        self._meta["wall_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        t.note(modem, "session_end", outcome=outcome, reason=self._cancel_reason,
               stats=result.stats)
        path = self._session_dir / "transcript.jsonl"
        records, _ = read_transcript(str(path))
        m = compute(records, self._meta)
        write_session(self._session_dir, m)
        # a stuck worker still holds this transcript; closing it under the
        # thread would turn its next write into a crash
        if not stuck:
            t.close()
        return m

    def _final_outcome(self, result: ScenarioResult) -> str:
        if self._cancel_reason == "detach":
            return "failed:modem_lost"
        if self._cancel_reason == "watchdog":
            return "aborted:watchdog"
        if (self._cancel_reason == "disconnected"
                and result.outcome == "aborted:watchdog"):
            return "failed:link_lost"
        return result.outcome

    # -- maintenance -------------------------------------------------------

    def _maintain(self, now: float) -> None:
        """Retention and log rotation, between sessions only. rglob+stat over
        a month of results is tens of thousands of syscalls, so it is
        throttled and runs off the loop — arbitration must not wait on it."""
        if now - self._last_sweep < self.config.responder.retention_interval_s:
            return
        if self._sweeper is not None and self._sweeper.is_alive():
            return
        self._last_sweep = now
        self._safe(None, "rotate", self._rotate)
        sweeper = threading.Thread(target=self._sweep, daemon=True,
                                   name="responder-retention")
        sweeper.start()             # published only once joinable
        self._sweeper = sweeper

    def _sweep(self) -> None:
        self._safe(None, "retention", lambda: sweep_results(
            self.results_root,
            max_age_s=self.config.responder.retention_max_age_s,
            max_bytes=self.config.responder.retention_max_bytes))

    def _rotate(self) -> None:
        """responder.jsonl is append-only for the daemon's whole life; rotate
        it like a modem log so a burst of loop errors cannot fill the disk.
        The old writer is retired, not closed — a client's cmd reader may be
        one instruction from writing to it."""
        path = Path(self.transcript.path)
        try:
            if path.stat().st_size <= TRANSCRIPT_CAP:
                return
        except OSError:
            return
        old = self.transcript
        os.replace(path, path.with_name(path.name + ".1"))
        self.transcript = Transcript(str(path), "daemon")
        for client in self._links.values():
            client.set_transcript(self.transcript)
        retire_at = time.monotonic() + 2 * self.tick_s
        self._closing.append((lambda: time.monotonic() >= retire_at, old))

    def _reap(self) -> None:
        done = [(ready, t) for ready, t in self._closing if ready()]
        for _, t in done:
            t.close()
        self._closing = [e for e in self._closing if e not in done]

    def _note_drops(self) -> None:
        for name, fan in self._fanin.items():
            if fan.dropped > self._dropped.get(name, 0):
                self._dropped[name] = fan.dropped
                self.transcript.note(name, "fanin_overflow", dropped=fan.dropped)

    def _write_status(self) -> None:
        """The unattended end's health, refreshed every tick: without it the
        only way to ask after a remote responder is to rsync its results."""
        m = self.last_metrics
        status = {
            "site": self.config.site.name,
            "mycall": self.config.site.mycall,
            "wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "uptime_s": round(time.monotonic() - self.started_at, 1),
            "state": self.state,
            "winner": self.winner,
            "modems": self.modem_states,
            "pending": sorted(self._pending),
            "sessions_completed": self.sessions_completed,
            "last_session": None if m is None else {
                "sid": m.sid, "modem": m.modem, "scenario": m.scenario,
                "outcome": m.outcome, "bytes_tx": m.bytes_tx,
                "bytes_rx": m.bytes_rx, "wall_end": m.wall_end},
        }
        tmp = self.status_path.with_name(self.status_path.name + ".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.status_path)

    # -- helpers -----------------------------------------------------------

    def _safe(self, modem: str | None, what: str, fn):
        try:
            return fn()
        except Exception as exc:
            self._log(self.transcript, modem or "-", f"{what} failed: {exc}")
            return None

    @staticmethod
    def _log(t: Transcript, modem: str, text: str) -> None:
        try:
            t.error(modem, text)
        except ValueError:
            pass          # transcript closed under a straggler thread


def _describe(ev: LinkEvent) -> str:
    """How the session record names the connect that opened it. The VARA
    dialect has a raw line worth keeping verbatim; the structured one has
    fields, so they are rendered rather than invented."""
    raw = getattr(ev.raw, "raw", None)
    if isinstance(raw, str) and raw:
        return raw
    peer = ev.peer or "?"
    return f"CONNECTED {peer}"
