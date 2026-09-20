# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Client for the structured host interface (HOST-API v1.0).

One connection, one ordered stream: a 4-byte big-endian length prefix followed
by a CBOR map whose key 0 is the message-type code. Commands are asynchronous
and results arrive as events, so this client is an event pump with a small
amount of derived state layered on top — link state, queue depth, the negotiated
capability image — plus the same epoch-fenced receive buffer the VARA-dialect
client uses, so a scenario can be written once and run over either.

The registries below mirror the specification, which is the wire contract; they
are append-only and positional, so entries are never reordered or removed.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any, Callable

from . import cbor
from .client import AttachError, EpochFenced, ModemError, NotAttached
from .transcript import Kind, Transcript

PROTO = "1.0"
MAX_FRAME = 16 * 1024 * 1024

# message-type codes: 0-31 host->modem, 32-63 modem->host
HELLO, SET_IDENTITY, SET_PROFILE, LISTEN, CONNECT = 0, 1, 2, 3, 4
SEND, DISCONNECT, ABORT, SUBSCRIBE, CONFIGURE = 5, 6, 7, 8, 9
STATE_CHANGED, CAPABILITIES, LINK_STATS, DATA_RECEIVED = 32, 33, 34, 35
SEND_PROGRESS, ID_SENT, PEER_OBSERVED, PHYSICAL_STATE, ERROR = 36, 37, 38, 39, 40

ST_DISCONNECTED, ST_LISTENING, ST_CONNECTING, ST_CONNECTED, ST_DISCONNECTING = range(5)
STATE_NAME = {ST_DISCONNECTED: "DISCONNECTED", ST_LISTENING: "LISTENING",
              ST_CONNECTING: "CONNECTING", ST_CONNECTED: "CONNECTED",
              ST_DISCONNECTING: "DISCONNECTING"}

RS_REMOTE, RS_LOCAL, RS_LINK_FAILED, RS_REFUSED = 1, 2, 3, 4
REASON_NAME = {RS_REMOTE: "remote", RS_LOCAL: "local",
               RS_LINK_FAILED: "link_failed", RS_REFUSED: "refused"}

PF_AMATEUR, PF_UNRESTRICTED = 1, 2

ERR_INCOMPATIBLE, ERR_BAD_STATE, ERR_NO_IDENTITY, ERR_MALFORMED = 1, 2, 3, 4

MSG_NAME = {HELLO: "Hello", SET_IDENTITY: "SetIdentity", SET_PROFILE: "SetProfile",
            LISTEN: "Listen", CONNECT: "Connect", SEND: "Send",
            DISCONNECT: "Disconnect", ABORT: "Abort", SUBSCRIBE: "Subscribe",
            CONFIGURE: "Configure", STATE_CHANGED: "StateChanged",
            CAPABILITIES: "CapabilitiesNegotiated", LINK_STATS: "LinkStats",
            DATA_RECEIVED: "DataReceived", SEND_PROGRESS: "SendProgress",
            ID_SENT: "IdSent", PEER_OBSERVED: "PeerObserved",
            PHYSICAL_STATE: "PhysicalState", ERROR: "Error"}

_FIELDS = (
    "m", "proto", "features", "client", "modem", "profiles",
    "identity_required", "station_id", "aliases", "profile", "on", "peer_id",
    "deadline", "ref", "data", "stream", "priority", "deflate", "id",
    "events", "stats_period", "radio", "ptt", "state", "reason", "peer_capabilities",
    "usable", None, "peer_profiles", "gear", "rung", "snr3k_db",
    "group_snr_db", "control_tier", "throughput_bps", "queue_bytes", "eta_s",
    "harq_rounds", "rebuilds", "compression_ratio", "sent", "total",
    "delivered", "deflated", "t", "capabilities", "busy", "code", "detail",
    "addressee", "data_profile", "receive_profiles", "feedback_iters",
    "impulse_blank", "bandwidth_hz", "destination", "service", "message_id",
    "parity", "repeats", "inactivity_timeout_s",
)
KEY = {name: i for i, name in enumerate(_FIELDS) if name is not None}
NAME = {i: name for i, name in enumerate(_FIELDS) if name is not None}


class Incompatible(ModemError):
    """The modem refused our protocol major version."""


def pack(msg: dict[str, Any]) -> bytes:
    """A message dict -> a length-prefixed frame. An unregistered field is a
    bug here, not a must-ignore case: we only ever send what we know."""
    body = cbor.encode({KEY[name]: val for name, val in msg.items()})
    if len(body) > MAX_FRAME:
        raise ModemError(f"frame of {len(body)} bytes exceeds the 16 MiB limit")
    return len(body).to_bytes(4, "big") + body


def unpack(body: bytes) -> dict[str, Any]:
    """A frame body -> a message dict, dropping unregistered keys (must-ignore)."""
    raw = cbor.decode(body)
    if not isinstance(raw, dict):
        raise ModemError("host message is not a CBOR map")
    return {NAME[k]: v for k, v in raw.items() if k in NAME}


class HostApiClient:
    """One attachment to a modem speaking HOST-API v1.0.

    Mirrors ModemClient's surface where the two dialects genuinely mean the same
    thing — attach/close, send_data/read_data, connected, the rx epoch — so
    scenarios and metrics are written against one shape. Where the dialects
    diverge the difference is honest: there is no command/reply pump because
    every command here is asynchronous, and there is no BUFFER poll because
    queue depth arrives as telemetry.
    """

    def __init__(self, cfg, *, transcript: Transcript | None = None,
                 client_name: str = "creance"):
        self.cfg = cfg
        self.name = cfg.name
        self.transcript = transcript
        self.client_name = client_name

        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.RLock()
        self._closing = False
        self.attach_failures = 0

        self._rx = bytearray()
        self._rx_cv = threading.Condition(self._lock)
        self.epoch = 0

        self._subs: list[queue.Queue] = []
        self._ref = 0

        self.attached = False
        self.state = ST_DISCONNECTED
        self.peer: str | None = None
        self.last_reason: int | None = None
        self.queue_bytes = 0
        self.hello: dict[str, Any] = {}
        self.caps: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}
        self.errors: list[dict[str, Any]] = []

        # desired state, replayed on reattach
        self._identity: str | None = None
        self._profile: int | None = None
        self._listen = False
        self._subscribed: list[int] | None = None
        self._stats_period: float | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.state == ST_CONNECTED

    @property
    def version(self) -> str | None:
        return self.hello.get("modem")

    def set_transcript(self, transcript: Transcript | None) -> None:
        self.transcript = transcript

    def attach(self, timeout: float = 10.0) -> None:
        """Open the connection and complete the Hello exchange. On any failure
        the socket is closed — a half-open attach is what wedges these servers."""
        if self.attached:
            return
        deadline = time.monotonic() + timeout
        sock = None
        try:
            sock = socket.create_connection((self.host, self.port), timeout)
            # The dial deadline must not survive as a read deadline: a modem
            # with nothing to report is idle, not gone, and a recv timeout here
            # would tear the reader down mid-session without a word.
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._sock, self._closing, self.attached = sock, False, True
            self._reader = threading.Thread(target=self._read_loop,
                                            name=f"hostapi-{self.name}", daemon=True)
            self._reader.start()
            self._handshake(deadline)
            self._replay()
        except Incompatible:
            self.attach_failures += 1
            self._teardown()
            raise
        except Exception as exc:
            self.attach_failures += 1
            self._teardown()
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            raise AttachError(f"{self.name}: attach failed: {exc}") from exc
        self.attach_failures = 0

    @property
    def host(self) -> str:
        return getattr(self.cfg, "host", "127.0.0.1")

    @property
    def port(self) -> int:
        return self.cfg.cmd_port

    def _handshake(self, deadline: float) -> None:
        sub = self.subscribe(HELLO, ERROR)
        try:
            self.send(HELLO, proto=PROTO, features=[], client=self.client_name)
            while True:
                msg = self._await(sub, deadline)
                if msg["m"] == ERROR and msg.get("code") == ERR_INCOMPATIBLE:
                    raise Incompatible(f"{self.name}: modem refused proto {PROTO}: "
                                       f"{msg.get('detail')}")
                if msg["m"] == HELLO:
                    self.hello = msg
                    theirs = str(msg.get("proto", "")).split(".")[0]
                    if theirs and theirs != PROTO.split(".")[0]:
                        raise Incompatible(f"{self.name}: modem speaks proto "
                                           f"{msg.get('proto')}, we speak {PROTO}")
                    return
        finally:
            self.unsubscribe(sub)

    @staticmethod
    def _await(sub: queue.Queue, deadline: float) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for the modem")
        try:
            return sub.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError("timed out waiting for the modem") from None

    def _replay(self) -> None:
        if self._identity is not None:
            self.send(SET_IDENTITY, station_id=self._identity)
        if self._profile is not None:
            self.send(SET_PROFILE, profile=self._profile)
        if self._subscribed is not None:
            msg: dict[str, Any] = {"events": list(self._subscribed)}
            if self._stats_period is not None:
                msg["stats_period"] = self._stats_period
            self.send(SUBSCRIBE, **msg)
        if self._listen:
            self.send(LISTEN, on=True)

    def close(self) -> None:
        self._teardown()

    def _teardown(self) -> None:
        with self._lock:
            self._closing, self.attached = True, False
            sock, self._sock = self._sock, None
            self._rx.clear()
            self._rx_cv.notify_all()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

    # -- pub/sub -----------------------------------------------------------

    def subscribe(self, *types: int) -> queue.Queue:
        """Subscribe before sending, so a fast reply cannot be missed. An empty
        type list receives everything."""
        q: queue.Queue = queue.Queue()
        q._types = set(types)                                    # type: ignore[attr-defined]
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def wait_for(self, *types: int, timeout: float = 30.0) -> dict[str, Any]:
        sub = self.subscribe(*types)
        try:
            return self._await(sub, time.monotonic() + timeout)
        finally:
            self.unsubscribe(sub)

    # -- commands ----------------------------------------------------------

    def send(self, m: int, **fields: Any) -> None:
        frame = pack({"m": m, **fields})
        self._record(Kind.CMD_TX, m, fields)
        self._write(frame)

    def send_raw_frame(self, body: bytes) -> None:
        """Send a pre-encoded CBOR body under a correct length prefix. For
        conformance probing only: it is the only way to put a message type or
        field on the wire that this client's registry does not know."""
        self._write(len(body).to_bytes(4, "big") + body)

    def send_raw_prefix(self, prefix: bytes) -> None:
        """Send a bare length prefix with no body — for probing what a modem
        does with a frame length it must refuse."""
        self._write(prefix)

    def _write(self, frame: bytes) -> None:
        with self._lock:
            sock = self._sock
            if sock is None or self._closing:
                raise NotAttached(f"{self.name}: not attached")
        try:
            sock.sendall(frame)
        except OSError as exc:
            raise ModemError(f"{self.name}: send failed: {exc}") from exc

    def next_ref(self) -> int:
        with self._lock:
            self._ref += 1
            return self._ref

    def _desire(self, m: int, **fields: Any) -> None:
        """Record the intent, and send it only if we are attached. Desired
        state is replayed on every (re)attach, so configuring a modem whose
        process is not up yet is legal and lands when it comes up."""
        if self.attached:
            self.send(m, **fields)

    def set_identity(self, station_id: str, aliases: list[str] | None = None) -> None:
        self._identity = station_id
        msg: dict[str, Any] = {"station_id": station_id}
        if aliases:
            msg["aliases"] = list(aliases)
        self._desire(SET_IDENTITY, **msg)

    def set_profile(self, profile: int) -> None:
        self._profile = profile
        self._desire(SET_PROFILE, profile=profile)

    def set_listen(self, on: bool) -> None:
        self._listen = on
        self._desire(LISTEN, on=on)

    def subscribe_events(self, events: list[int], stats_period: float | None = None) -> None:
        self._subscribed, self._stats_period = list(events), stats_period
        msg: dict[str, Any] = {"events": list(events)}
        if stats_period is not None:
            msg["stats_period"] = stats_period
        self._desire(SUBSCRIBE, **msg)

    def connect(self, peer_id: str, *, profile: int | None = None,
                deadline: float | None = None) -> int:
        ref = self.next_ref()
        msg: dict[str, Any] = {"peer_id": peer_id, "ref": ref}
        if profile is not None:
            msg["profile"] = profile
        if deadline is not None:
            msg["deadline"] = deadline
        self.send(CONNECT, **msg)
        return ref

    def disconnect(self) -> int:
        ref = self.next_ref()
        self.send(DISCONNECT, ref=ref)
        return ref

    def abort(self) -> int:
        ref = self.next_ref()
        self.send(ABORT, ref=ref)
        return ref

    def send_data(self, data: bytes, label: str = "", *,
                  stream: int | None = None,
                  msg_id: int | None = None, deflate: bool | None = None,
                  priority: int | None = None) -> None:
        msg: dict[str, Any] = {"data": bytes(data)}
        if stream is not None:
            msg["stream"] = stream
        if msg_id is not None:
            msg["id"] = msg_id
        if deflate is not None:
            msg["deflate"] = deflate
        if priority is not None:
            msg["priority"] = priority
        self.send(SEND, **msg)
        if self.transcript is not None:
            try:
                self.transcript.data(self.name, "tx", bytes(data), label=label)
            except Exception:
                pass

    # -- receive buffer ----------------------------------------------------

    def bump_epoch(self) -> bytes:
        """Fence the buffer against a previous session's late arrivals, handing
        back whatever was stranded so the caller can record it."""
        with self._lock:
            self.epoch += 1
            left, self._rx = bytes(self._rx), bytearray()
            self._rx_cv.notify_all()
        return left

    def read_data(self, n: int | None = None, *, timeout: float = 30.0,
                  epoch: int | None = None,
                  cancel: threading.Event | None = None) -> bytes:
        """Consume n bytes (or, when n is None, whatever first arrives).
        Timeout returns b"" *without consuming* — a caller that asked for a
        count is framing something, and handing back a short read would lose
        that framing silently. Same contract as the VARA-dialect client, so a
        scenario reads identically over either.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            want = self.epoch if epoch is None else epoch
            while True:
                if want != self.epoch:
                    raise EpochFenced(f"{self.name}: epoch {want} fenced")
                have = len(self._rx)
                if have and (n is None or have >= n):
                    take = have if n is None else n
                    out, self._rx = bytes(self._rx[:take]), self._rx[take:]
                    return out
                if self._closing or (cancel is not None and cancel.is_set()):
                    return b""
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return b""
                self._rx_cv.wait(min(remaining, 0.1) if cancel is not None
                                 else remaining)

    def peek_accumulate(self, n: int, timeout: float = 30.0,
                        epoch: int | None = None,
                        cancel: threading.Event | None = None) -> bytes:
        """Accumulate until n bytes are buffered, then return them without
        consuming — the magic-peek the responder opens a session with."""
        deadline = time.monotonic() + timeout
        with self._lock:
            want = self.epoch if epoch is None else epoch
            while True:
                if want != self.epoch:
                    raise EpochFenced(f"{self.name}: epoch {want} fenced")
                if len(self._rx) >= n:
                    return bytes(self._rx[:n])
                if self._closing or (cancel is not None and cancel.is_set()):
                    return bytes(self._rx[:n])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return bytes(self._rx[:n])
                self._rx_cv.wait(min(remaining, 0.1) if cancel is not None
                                 else remaining)

    def consume(self, n: int) -> bytes:
        with self._lock:
            out, self._rx = bytes(self._rx[:n]), self._rx[n:]
            return out

    # -- reader ------------------------------------------------------------

    def _read_loop(self) -> None:
        buf = bytearray()
        try:
            while True:
                with self._lock:
                    sock = self._sock
                    if sock is None or self._closing:
                        return
                try:
                    chunk = sock.recv(65536)
                except (OSError, ValueError):
                    return
                if not chunk:
                    return
                buf += chunk
                while len(buf) >= 4:
                    size = int.from_bytes(buf[:4], "big")
                    if size > MAX_FRAME:
                        self._note_error(f"frame length {size} exceeds the limit")
                        return
                    if len(buf) < 4 + size:
                        break
                    body, buf = bytes(buf[4:4 + size]), buf[4 + size:]
                    try:
                        self._dispatch(unpack(body))
                    except Exception as exc:                     # a fault, not a crash
                        self._note_error(f"undecodable frame: {exc!r}")
        finally:
            # A reader that dies without detaching leaves a modem that looks
            # attached and is deaf; the station would report healthy forever.
            self._teardown()

    def _dispatch(self, msg: dict[str, Any]) -> None:
        m = msg.get("m")
        if not isinstance(m, int):
            self._note_error(f"message without a type code: {msg!r}")
            return
        self._record(Kind.CMD_RX, m, {k: v for k, v in msg.items() if k != "m"})

        if m == DATA_RECEIVED:
            data = msg.get("data") or b""
            if isinstance(data, (bytes, bytearray)) and data:
                with self._lock:
                    self._rx += data
                    self._rx_cv.notify_all()
                if self.transcript is not None:
                    try:
                        self.transcript.data(self.name, "rx", bytes(data))
                    except Exception:
                        pass
        elif m == STATE_CHANGED:
            state = msg.get("state")
            if isinstance(state, int):
                self.state = state
                self.peer = msg.get("peer_id") or (None if state == ST_DISCONNECTED
                                                   else self.peer)
                self.last_reason = msg.get("reason")
        elif m == CAPABILITIES:
            self.caps = msg
        elif m == LINK_STATS:
            self.stats = msg
            qb = msg.get("queue_bytes")
            if isinstance(qb, int):
                self.queue_bytes = qb
        elif m == PHYSICAL_STATE and "ptt" in msg:
            # Written as a PTT record, the same shape the VARA client uses, so
            # duty-cycle accounting has one source rather than two.
            if self.transcript is not None:
                try:
                    self.transcript.ptt(self.name, bool(msg["ptt"]))
                except Exception:
                    pass
        elif m == ERROR:
            self.errors.append(msg)

        with self._lock:
            subs = list(self._subs)
        for q in subs:
            types = getattr(q, "_types", set())
            if not types or m in types:
                q.put(msg)

    def _note_error(self, detail: str) -> None:
        self.errors.append({"m": ERROR, "code": None, "detail": detail})
        if self.transcript is not None:
            try:
                self.transcript.error(self.name, detail)
            except Exception:
                pass

    def _record(self, kind: str, m: int, fields: dict[str, Any]) -> None:
        t = self.transcript
        if t is None:
            return
        try:
            t.note(self.name, MSG_NAME.get(m, f"m{m}"), kind=kind,
                   fields=_loggable(fields))
        except Exception:
            pass                     # a closed transcript must never kill a reader


def _loggable(fields: dict[str, Any]) -> dict[str, Any]:
    """Payload bytes are summarized, not transcribed — a transcript is a record
    of the conversation, not a second copy of the traffic."""
    out = {}
    for k, v in fields.items():
        if isinstance(v, (bytes, bytearray)):
            out[k] = f"<{len(v)} bytes>"
        else:
            out[k] = v
    return out


def observer(client: HostApiClient, on_event: Callable[[dict], None],
             *types: int) -> threading.Thread:
    """Pump events to a callback on a daemon thread until the client detaches."""
    sub = client.subscribe(*types)

    def run() -> None:
        while client.attached:
            try:
                on_event(sub.get(timeout=0.5))
            except queue.Empty:
                continue
            except Exception:
                return

    t = threading.Thread(target=run, name=f"hostapi-obs-{client.name}", daemon=True)
    t.start()
    return t
