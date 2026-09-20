# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One session-level API over both host dialects.

A modem is reached either through the VARA-style two-socket ASCII dialect or
through the structured single-socket one. Those differ enormously in shape and
not at all in what a *scenario* needs: bring the modem up, listen or connect,
push bytes, pull bytes, notice the link came up or went away, tear it down.

So this is the seam. Scenarios, the initiator and the responder are written
against `Link` and never learn which dialect they are on; conformance is
deliberately *not* written against it, because grading a modem's fidelity to
its dialect is exactly the job that must see the dialect.

The event vocabulary is normalized rather than unioned: both dialects produce
`LinkEvent`s from the same small set of kinds. Where a dialect has no way to
express a kind, it simply never emits it — the VARA modems have no telemetry to
put in STATS, and the structured dialect has no PTT unless a client subscribes
to it. Absence is honest; a synthesized event would not be.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import hostapi
from .client import ModemClient
from .hostapi import HostApiClient
from .transcript import Transcript
from .wire import Line

CONNECTED = "connected"
DISCONNECTED = "disconnected"
PENDING = "pending"
CANCELPENDING = "cancelpending"
BUFFER = "buffer"
PTT = "ptt"
BUSY = "busy"
STATS = "stats"
CAPS = "caps"
ERROR = "error"
ATTACHED = "attached"
DETACHED = "detached"
OTHER = "other"


@dataclass(frozen=True, slots=True)
class LinkEvent:
    kind: str
    modem: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    @property
    def peer(self) -> str | None:
        return self.fields.get("peer")


class Link(Protocol):
    """What a scenario may assume of a modem, whatever it speaks."""

    name: str

    def configure(self, mycall: str) -> None:
        """Record desired state. No I/O: a modem may be configured before its
        process is even up, and the state is replayed on every attach."""
        ...
    def attach(self) -> None: ...
    def close(self) -> None: ...
    def set_listen(self, on: bool) -> None: ...
    def connect(self, dst: str) -> None: ...
    def disconnect(self) -> None: ...
    def abort(self) -> None: ...
    def send(self, data: bytes, label: str = "") -> None: ...
    def recv(self, n: int | None = None, timeout: float = 30.0,
             epoch: int | None = None, cancel: Any = None) -> bytes:
        """Exactly n bytes, or b"" on timeout without consuming; n=None takes
        whatever first arrives."""
        ...
    def peek(self, n: int, timeout: float, epoch: int | None = None,
             cancel: Any = None) -> bytes: ...
    def wait_for(self, kind: str, timeout: float, cancel: Any = None
                 ) -> LinkEvent | None:
        """Block for the next event of this kind. None on timeout or cancel."""
        ...
    def bump_epoch(self) -> bytes: ...
    def subscribe(self, q: queue.Queue) -> None: ...
    def set_transcript(self, transcript: Transcript | None) -> None: ...
    def stale(self, now: float | None = None) -> bool: ...

    @property
    def transcript(self) -> Transcript: ...
    def request_version(self) -> str: ...

    @property
    def attach_failures(self) -> int: ...
    @property
    def attached(self) -> bool: ...
    @property
    def connected(self) -> bool: ...
    @property
    def epoch(self) -> int: ...
    @property
    def queue_bytes(self) -> int: ...
    @property
    def version(self) -> str: ...


class VaraLink:
    """The two-socket ASCII dialect."""

    dialect = "vara"

    def __init__(self, client: ModemClient, cfg) -> None:
        self.client = client
        self.cfg = cfg
        self.name = client.name
        self._mycall = ""
        self._subs: list[queue.Queue] = []
        client.subscribe(_Adapter(self._emit, self._from_line))
        prior_attach, prior_detach = client.on_attach, client.on_detach
        client.on_attach = lambda c: (self._emit(LinkEvent(ATTACHED, c.name)),
                                      prior_attach and prior_attach(c))
        client.on_detach = lambda c: (self._emit(LinkEvent(DETACHED, c.name)),
                                      prior_detach and prior_detach(c))

    # -- lifecycle ---------------------------------------------------------

    def configure(self, mycall: str) -> None:
        self._mycall = mycall
        self.client.set_mycall(mycall)
        self.client.set_bandwidth(self.cfg.bandwidth)
        self.client.set_compression(self.cfg.compression)

    def attach(self) -> None:
        self.client.attach()

    def close(self) -> None:
        self.client.close()

    def set_listen(self, on: bool) -> None:
        self.client.set_listen(on)

    def connect(self, dst: str) -> None:
        self.client.command(f"CONNECT {self._mycall} {dst}")

    def disconnect(self) -> None:
        self.client.command("DISCONNECT")

    def abort(self) -> None:
        self.client.command("ABORT")

    # -- data --------------------------------------------------------------

    def send(self, data: bytes, label: str = "") -> None:
        self.client.send_data(data, label=label)

    def recv(self, n: int | None = None, timeout: float = 30.0,
             epoch: int | None = None, cancel=None) -> bytes:
        return self.client.read_data(n, timeout=timeout, epoch=epoch,
                                     cancel=cancel)

    def peek(self, n: int, timeout: float, epoch: int | None = None,
             cancel=None) -> bytes:
        return self.client.peek_accumulate(n, timeout=timeout, epoch=epoch,
                                           cancel=cancel)

    def bump_epoch(self) -> bytes:
        return self.client.bump_epoch()

    # -- state -------------------------------------------------------------

    def stale(self, now: float | None = None) -> bool:
        return self.client.stale(now)

    def request_version(self) -> str:
        return self.version

    def telemetry(self) -> dict:
        """The VARA dialect reports a queue depth and nothing else."""
        return {"queue_bytes": self.client.buffer_bytes}

    @property
    def attach_failures(self) -> int:
        return self.client.attach_failures

    @property
    def attached(self) -> bool:
        return self.client.attached

    @property
    def connected(self) -> bool:
        return self.client.connected

    @property
    def epoch(self) -> int:
        return self.client.epoch

    @property
    def queue_bytes(self) -> int:
        return self.client.buffer_bytes

    @property
    def version(self) -> str:
        if not self.cfg.quirks.version_reply:
            return ""
        return self.client.request_version() or ""

    # -- events ------------------------------------------------------------

    def subscribe(self, q: queue.Queue) -> None:
        self._subs.append(q)

    def wait_for(self, kind: str, timeout: float, cancel=None) -> LinkEvent | None:
        q: queue.Queue = queue.Queue()
        self.subscribe(q)
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (cancel is not None and cancel.is_set()):
                    return None
                try:
                    ev = q.get(timeout=min(remaining, 0.1)
                               if cancel is not None else remaining)
                except queue.Empty:
                    continue
                if ev.kind == kind:
                    return ev
        finally:
            self.unsubscribe(q)

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def set_transcript(self, transcript: Transcript | None) -> None:
        self.client.set_transcript(transcript)

    @property
    def transcript(self) -> Transcript:
        return self.client.transcript

    def _emit(self, ev: LinkEvent) -> None:
        for q in list(self._subs):
            q.put(ev)

    def _from_line(self, line: Line) -> LinkEvent:
        name, f = line.name, line.fields
        if name == "CONNECTED" and "src" in f and "dst" in f:
            # The peer is whichever end of the pair is not us. A CONNECTED that
            # did not parse never becomes a connect event: guessing a peer from
            # a malformed line is how a session gets attributed to the wrong
            # station, and Pat panics outright on the same input.
            src, dst = f["src"], f["dst"]
            peer = src if dst == self._mycall else dst
            return LinkEvent(CONNECTED, self.name,
                             {"peer": peer, "src": src, "dst": dst,
                              "bw": f.get("bw")}, line)
        if name == "DISCONNECTED":
            return LinkEvent(DISCONNECTED, self.name, {}, line)
        if name == "PENDING":
            return LinkEvent(PENDING, self.name, {}, line)
        if name == "CANCELPENDING":
            return LinkEvent(CANCELPENDING, self.name, {}, line)
        if name == "BUFFER":
            return LinkEvent(BUFFER, self.name,
                             {"queue_bytes": f.get("n", 0)}, line)
        if name == "PTT":
            return LinkEvent(PTT, self.name, {"on": f.get("on", False)}, line)
        if name == "BUSY":
            return LinkEvent(BUSY, self.name, {"on": f.get("on", False)}, line)
        return LinkEvent(OTHER, self.name, {"line": line.raw}, line)


class _Adapter:
    """A ModemClient subscriber that maps lines to LinkEvents on the reader
    thread — no forwarder thread, matching the responder's existing fan-in."""

    def __init__(self, emit, convert) -> None:
        self._emit, self._convert = emit, convert

    def put(self, line, block=True, timeout=None) -> None:
        try:
            self._emit(self._convert(line))
        except Exception:
            pass                     # a mapping bug must never kill a reader


class HostApiLink:
    """The structured single-socket dialect."""

    dialect = "hostapi"

    def __init__(self, client: HostApiClient, cfg) -> None:
        self.client = client
        self.cfg = cfg
        self.name = client.name
        self._attached = False
        self._closed = False
        self._subs: list[queue.Queue] = []
        self._sub = client.subscribe()
        self._pump = threading.Thread(target=self._run, name=f"link-{self.name}",
                                      daemon=True)
        self._pump.start()

    # -- lifecycle ---------------------------------------------------------

    def configure(self, mycall: str) -> None:
        self.client.set_identity(mycall)
        self.client.set_profile(_profile(self.cfg))
        self.client.subscribe_events(
            [hostapi.LINK_STATS, hostapi.CAPABILITIES, hostapi.PHYSICAL_STATE],
            stats_period=1.0)

    def attach(self) -> None:
        self.client.attach()
        self._attached = True
        self._emit(LinkEvent(ATTACHED, self.name))

    def close(self) -> None:
        self._closed = True
        self.client.close()

    def set_listen(self, on: bool) -> None:
        self.client.set_listen(on)

    def connect(self, dst: str) -> None:
        self.client.connect(dst)

    def disconnect(self) -> None:
        self.client.disconnect()

    def abort(self) -> None:
        self.client.abort()

    # -- data --------------------------------------------------------------

    def send(self, data: bytes, label: str = "") -> None:
        self.client.send_data(data, label=label)

    def recv(self, n: int | None = None, timeout: float = 30.0,
             epoch: int | None = None, cancel=None) -> bytes:
        return self.client.read_data(n, timeout=timeout, epoch=epoch,
                                     cancel=cancel)

    def peek(self, n: int, timeout: float, epoch: int | None = None,
             cancel=None) -> bytes:
        return self.client.peek_accumulate(n, timeout=timeout, epoch=epoch,
                                           cancel=cancel)

    def bump_epoch(self) -> bytes:
        return self.client.bump_epoch()

    # -- state -------------------------------------------------------------

    def stale(self, now: float | None = None) -> bool:
        """Always False: the structured dialect defines no heartbeat, so there
        is no silence to measure. A dead socket detaches the client outright,
        which is a stronger signal than a missed keepalive anyway."""
        return False

    def request_version(self) -> str:
        """The modem named itself in its Hello; there is nothing to ask for."""
        return self.version

    def telemetry(self) -> dict:
        return {k: v for k, v in (self.client.stats or {}).items() if k != "m"}

    @property
    def attach_failures(self) -> int:
        return self.client.attach_failures

    @property
    def attached(self) -> bool:
        return self.client.attached

    @property
    def connected(self) -> bool:
        return self.client.connected

    @property
    def epoch(self) -> int:
        return self.client.epoch

    @property
    def queue_bytes(self) -> int:
        return self.client.queue_bytes

    @property
    def version(self) -> str:
        return self.client.version or ""

    # -- events ------------------------------------------------------------

    def subscribe(self, q: queue.Queue) -> None:
        self._subs.append(q)

    def wait_for(self, kind: str, timeout: float, cancel=None) -> LinkEvent | None:
        q: queue.Queue = queue.Queue()
        self.subscribe(q)
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (cancel is not None and cancel.is_set()):
                    return None
                try:
                    ev = q.get(timeout=min(remaining, 0.1)
                               if cancel is not None else remaining)
                except queue.Empty:
                    continue
                if ev.kind == kind:
                    return ev
        finally:
            self.unsubscribe(q)

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def set_transcript(self, transcript: Transcript | None) -> None:
        self.client.set_transcript(transcript)

    @property
    def transcript(self) -> Transcript:
        return self.client.transcript

    def _emit(self, ev: LinkEvent) -> None:
        for q in list(self._subs):
            q.put(ev)

    def _run(self) -> None:
        while not self._closed:
            try:
                msg = self._sub.get(timeout=0.25)
            except queue.Empty:
                self._check_detach()
                continue
            ev = self._from_message(msg)
            if ev is not None:
                self._emit(ev)

    def _check_detach(self) -> None:
        """A reader that dies without saying so leaves a station reporting
        healthy forever, so the detach edge is published, not inferred."""
        if self._attached and not self.client.attached:
            self._attached = False
            self._emit(LinkEvent(DETACHED, self.name))

    def _from_message(self, msg: dict) -> LinkEvent | None:
        m = msg.get("m")
        if m == hostapi.STATE_CHANGED:
            state = msg.get("state")
            if state == hostapi.ST_CONNECTED:
                return LinkEvent(CONNECTED, self.name,
                                 {"peer": msg.get("peer_id")}, msg)
            if state == hostapi.ST_DISCONNECTED:
                return LinkEvent(DISCONNECTED, self.name,
                                 {"reason": hostapi.REASON_NAME.get(
                                     msg.get("reason"))}, msg)
            return LinkEvent(OTHER, self.name,
                             {"state": hostapi.STATE_NAME.get(state, state)}, msg)
        if m == hostapi.LINK_STATS:
            return LinkEvent(STATS, self.name,
                             {k: v for k, v in msg.items() if k != "m"}, msg)
        if m == hostapi.CAPABILITIES:
            return LinkEvent(CAPS, self.name,
                             {k: v for k, v in msg.items() if k != "m"}, msg)
        if m == hostapi.PHYSICAL_STATE:
            if "ptt" in msg:
                return LinkEvent(PTT, self.name, {"on": bool(msg["ptt"])}, msg)
            if "busy" in msg:
                return LinkEvent(BUSY, self.name, {"on": bool(msg["busy"])}, msg)
            return None
        if m == hostapi.ERROR:
            return LinkEvent(ERROR, self.name,
                             {"code": msg.get("code"),
                              "detail": msg.get("detail")}, msg)
        if m == hostapi.DATA_RECEIVED:
            return None              # the client buffered it; not a link event
        return LinkEvent(OTHER, self.name, {"m": m}, msg)


def _profile(cfg) -> int:
    """Until a site declares one, an amateur station is the only honest default;
    the structured dialect has no bandwidth knob to map the VARA enum onto."""
    return getattr(cfg, "profile", None) or hostapi.PF_AMATEUR


def open_link(cfg, host: str, transcript: Transcript, **kw) -> Link:
    """Build the Link for a modem, chosen by its configured dialect.

    The transcript is required rather than optional. Every client writes to it
    from `attach` onwards, so the `None` this used to accept could only ever
    surface as an attribute error on the first state change — a signature that
    offers something none of the three dialects can do.
    """
    if cfg.dialect == "hostapi":
        return HostApiLink(HostApiClient(cfg, transcript=transcript), cfg)
    if cfg.dialect == "ardop":
        # Imported here rather than at module scope: the ARDOP client is written
        # against this module's event vocabulary, so the dependency runs that
        # way round and only the dispatch needs to look back.
        from .ardop import ArdopClient, ArdopLink
        return ArdopLink(ArdopClient(cfg, host, transcript, **kw), cfg)
    return VaraLink(ModemClient(cfg, host, transcript, **kw), cfg)
