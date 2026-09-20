# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

# Adapted from an earlier VARA host client by the same author.
"""ModemClient: one attachment to a VARA-dialect modem (cmd + data sockets).

Attach is atomic — both sockets connect under one deadline or neither stays
open, because both target servers accept the cmd socket and then block
accepting the data socket: a half-attach wedges them until the process dies.

Every cmd/data byte lands in whichever Transcript is current: the responder
swaps in a per-session transcript via set_transcript() and reverts to the
daemon-level one between sessions.

The rx data buffer carries a session epoch: bump_epoch() fences off late
deliveries (the kestrel LoopbackModem has a real late-echo window) so they
cannot bleed into the next session's reads.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Callable, Optional

from .config import ModemConfig
from .transcript import Transcript
from .wire import (BANDWIDTHS, COMPRESSION_MODES, CR, Line, NOTIFICATION,
                   REPLY_OK, REPLY_WRONG, UNKNOWN, classify)


class ModemError(RuntimeError):
    pass


class AttachError(ModemError):
    pass


class NotAttached(ModemError):
    pass


class EpochFenced(ModemError):
    pass


_DETACHED = object()      # subscriber-queue sentinel: wakes waiters on detach
_CANCEL_POLL_S = 0.05     # a cancel Event cannot notify a Condition; bounded waits


def _hard_close(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class ModemClient:
    """Client for one modem's cmd+data TCP pair, with desired-state replay,
    pub/sub of classified cmd lines and epoch-guarded rx buffering.

    Reattachment is the caller's policy, not the client's: the responder's
    tick brings a detached modem back up, escalating to a respawn when the
    server is wedged, which no in-client retry loop could do."""

    _REPLAY_ORDER = ("mycall", "bandwidth", "compression", "listen")

    def __init__(self, cfg: ModemConfig, host: str, transcript: Transcript, *,
                 on_attach: Optional[Callable[["ModemClient"], None]] = None,
                 on_detach: Optional[Callable[["ModemClient"], None]] = None,
                 attach_timeout_s: float = 10.0) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.host = host
        self.on_attach = on_attach
        self.on_detach = on_detach
        self.attach_timeout_s = attach_timeout_s

        self._base_t = transcript
        self._session_t: Transcript | None = None

        self._cmd_sock: socket.socket | None = None
        self._data_sock: socket.socket | None = None
        self._cmd_tx_lock = threading.Lock()
        self._data_tx_lock = threading.Lock()

        self._attaching = threading.Lock()  # serializes whole attaches
        self._life = threading.Lock()      # attach/detach/close transitions
        self._gen = 0                      # fences stale reader threads
        self._closed = False

        self.attached = False
        self.connected = False
        self.buffer_bytes = 0
        self.last_connected: dict | None = None
        self.last_iamalive: float | None = None
        self.version: str | None = None
        self.attach_failures = 0
        self._attached_at = 0.0

        self._desired: dict[str, str] = {}

        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()

        self._rx = bytearray()
        self._rx_cv = threading.Condition()
        self._epoch = 0

    # -- transcript --------------------------------------------------------

    @property
    def transcript(self) -> Transcript:
        return self._session_t or self._base_t

    def set_transcript(self, t: Transcript | None) -> None:
        """Swap the current transcript; None reverts to the daemon-level one."""
        self._session_t = t

    # -- attach / detach ---------------------------------------------------

    def attach(self, timeout: float | None = None) -> None:
        """Atomically connect both sockets, start readers, replay desired
        state. On any failure both sockets are closed — never a half-attach.

        Held across replay by _attaching (never by _life, which the reader
        threads need to detach) so that a concurrent caller seeing attached=True
        also sees the desired state landed, not MYCALL/BW still in flight."""
        deadline = time.monotonic() + (self.attach_timeout_s if timeout is None
                                       else timeout)
        with self._attaching:
            with self._life:
                if self._closed:
                    raise ModemError(f"{self.name}: client closed")
                if self.attached:
                    return
                cmd = data = None
                try:
                    cmd = self._dial(self.cfg.cmd_port, deadline)
                    data = self._dial(self.cfg.data_port, deadline)
                except OSError as exc:
                    for s in (cmd, data):
                        if s is not None:
                            _hard_close(s)
                    self.attach_failures += 1
                    self.transcript.error(self.name, f"attach failed: {exc}")
                    raise AttachError(f"{self.name}: attach failed: {exc}") from exc
                self._cmd_sock, self._data_sock = cmd, data
                self._gen += 1
                gen = self._gen
                self.attached = True
                self.connected = False
                self.buffer_bytes = 0
                self.last_iamalive = None
                self._attached_at = time.monotonic()
                for target, sock, tag in ((self._cmd_reader, cmd, "cmd"),
                                          (self._data_reader, data, "data")):
                    threading.Thread(target=target, args=(gen, sock),
                                     name=f"{self.name}-{tag}-rx",
                                     daemon=True).start()
            self.transcript.state(self.name, "attached")
            try:
                self.replay(timeout=max(deadline - time.monotonic(), 1.0))
            except (ModemError, TimeoutError) as exc:
                self._detach(gen, f"replay failed: {exc}", deliberate=True)
                self.attach_failures += 1
                raise AttachError(f"{self.name}: {exc}") from exc
            self.attach_failures = 0
        if self.on_attach:
            self.on_attach(self)

    def _dial(self, port: int, deadline: float) -> socket.socket:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("attach deadline exhausted")
        s = socket.create_connection((self.host, port), timeout=remaining)
        s.settimeout(None)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def _detach(self, gen: int | None, reason: str, *, deliberate: bool) -> None:
        with self._life:
            if not self.attached or (gen is not None and gen != self._gen):
                return
            self._gen += 1             # fences the sibling reader thread
            self.attached = False
            self.connected = False
            for s in (self._data_sock, self._cmd_sock):
                if s is not None:
                    _hard_close(s)
            self._cmd_sock = self._data_sock = None
        self.transcript.state(self.name, "detached", detail=reason)
        self._publish(_DETACHED)
        with self._rx_cv:
            self._rx_cv.notify_all()
        if not deliberate and self.on_detach:
            self.on_detach(self)

    def close(self) -> None:
        with self._life:
            if self._closed:
                return
            self._closed = True
        self._detach(None, "closed", deliberate=True)

    # -- liveness ----------------------------------------------------------

    def stale(self, now: float | None = None) -> bool:
        """True when IAMALIVE (which flows from attach on both servers) has
        gone silent for 3x the configured interval."""
        interval = self.cfg.quirks.iamalive_s
        if not interval or not self.attached:
            return False
        base = self.last_iamalive if self.last_iamalive is not None else self._attached_at
        return ((time.monotonic() if now is None else now) - base) > 3 * interval

    # -- pub/sub of cmd lines ----------------------------------------------

    def subscribe(self, q: queue.Queue | None = None) -> queue.Queue:
        """Queue of every subsequently-received wire.Line. Subscribe before
        sending the triggering command; unsubscribe when done. Pass your own
        queue to receive lines somewhere other than a plain Queue — the
        responder registers one that forwards straight into its event loop."""
        if q is None:
            q = queue.Queue()
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self, item) -> None:
        with self._subs_lock:
            subs = list(self._subs)
        for q in subs:
            q.put(item)

    def command(self, text: str, timeout: float = 5.0) -> Line:
        """Send one command and return its single OK/WRONG reply, letting
        interleaved notifications pass by. TimeoutError if no reply."""
        q = self.subscribe()
        try:
            self._send_cmd(text)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{self.name}: no reply to {text!r}")
                try:
                    item = q.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError(f"{self.name}: no reply to {text!r}") from None
                if item is _DETACHED:
                    raise NotAttached(f"{self.name}: detached awaiting reply to {text!r}")
                if item.kind in (REPLY_OK, REPLY_WRONG):
                    return item
        finally:
            self.unsubscribe(q)

    def wait_for(self, predicate: Callable[[Line], bool], timeout: float,
                 cancel: threading.Event | None = None) -> Line | None:
        """Block until a received Line satisfies predicate. None on timeout,
        cancel, or detach."""
        q = self.subscribe()
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or (cancel is not None and cancel.is_set()):
                    return None
                wait = min(remaining, _CANCEL_POLL_S) if cancel is not None else remaining
                try:
                    item = q.get(timeout=wait)
                except queue.Empty:
                    continue
                if item is _DETACHED:
                    return None
                if predicate(item):
                    return item
        finally:
            self.unsubscribe(q)

    def request_version(self, timeout: float = 5.0) -> str | None:
        """VERSION replies with its version string, not OK — correlate here."""
        q = self.subscribe()
        try:
            self._send_cmd("VERSION")
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                try:
                    item = q.get(timeout=remaining)
                except queue.Empty:
                    return None
                if item is _DETACHED:
                    return None
                if item.kind == NOTIFICATION and item.name == "VERSION":
                    return item.fields["version"]
                if item.kind == REPLY_WRONG:
                    return None
        finally:
            self.unsubscribe(q)

    # -- desired-state replay ----------------------------------------------

    def set_mycall(self, *calls: str) -> Line | None:
        if not calls:
            raise ValueError("at least one callsign required")
        return self._desire("mycall", "MYCALL " + " ".join(calls))

    def set_listen(self, on: bool) -> Line | None:
        return self._desire("listen", "LISTEN ON" if on else "LISTEN OFF")

    def set_bandwidth(self, bw: str) -> Line | None:
        if bw not in BANDWIDTHS:
            raise ValueError(f"unsupported bandwidth {bw!r}")
        return self._desire("bandwidth", "BW" + bw)

    def set_compression(self, mode: str) -> Line | None:
        mode = mode.upper()
        if mode not in COMPRESSION_MODES:
            raise ValueError(f"unsupported compression mode {mode!r}")
        return self._desire("compression", "COMPRESSION " + mode)

    def _desire(self, slot: str, text: str) -> Line | None:
        self._desired[slot] = text
        return self.command(text) if self.attached else None

    def replay(self, timeout: float = 5.0) -> None:
        """Reissue all recorded desired state; attach() ends with this. A
        WRONG is transcribed as an error, not fatal — the responder's retry
        loop must not spin forever on a modem that rejects one setting."""
        for slot in self._REPLAY_ORDER:
            text = self._desired.get(slot)
            if text is None:
                continue
            line = self.command(text, timeout)
            if line.kind == REPLY_WRONG:
                self.transcript.error(self.name, f"replay rejected: {text}")

    # -- outbound ----------------------------------------------------------

    def _send_cmd(self, text: str) -> None:
        sock = self._cmd_sock
        if not self.attached or sock is None:
            raise NotAttached(f"{self.name}: not attached")
        self.transcript.cmd_tx(self.name, text)
        with self._cmd_tx_lock:
            try:
                sock.sendall(text.encode("ascii") + CR)
            except OSError as exc:
                raise NotAttached(f"{self.name}: cmd send failed: {exc}") from exc

    def send_data(self, blob: bytes, label: str = "") -> None:
        sock = self._data_sock
        if not self.attached or sock is None:
            raise NotAttached(f"{self.name}: not attached")
        self.transcript.data(self.name, "tx", blob, label=label)
        self.buffer_bytes += len(blob)   # authoritative value follows via BUFFER n
        with self._data_tx_lock:
            try:
                sock.sendall(blob)
            except OSError as exc:
                raise NotAttached(f"{self.name}: data send failed: {exc}") from exc

    # -- epoch-guarded rx data buffer --------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def bump_epoch(self) -> bytes:
        """Advance the session epoch, fencing all in-flight reads. Returns
        undrained leftovers — caller transcribes them as out-of-session."""
        with self._rx_cv:
            leftover = bytes(self._rx)
            self._rx.clear()
            self._epoch += 1
            self._rx_cv.notify_all()
            return leftover

    def read_data(self, n: int | None = None, *, timeout: float = 30.0,
                  epoch: int | None = None,
                  cancel: threading.Event | None = None) -> bytes:
        """Consume n bytes (or, when n is None, whatever first arrives).
        Timeout and cancel both return b"" without consuming — a short read
        would silently lose framing for a caller that asked for a count;
        EpochFenced if the buffer epoch moved past `epoch`."""
        deadline = time.monotonic() + timeout
        with self._rx_cv:
            want = self._epoch if epoch is None else epoch
            while True:
                if want != self._epoch:
                    raise EpochFenced(f"{self.name}: epoch {want} superseded by {self._epoch}")
                if cancel is not None and cancel.is_set():
                    return b""
                have = len(self._rx)
                if have and (n is None or have >= n):
                    take = have if n is None else n
                    out = bytes(self._rx[:take])
                    del self._rx[:take]
                    return out
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return b""
                self._rx_cv.wait(min(remaining, _CANCEL_POLL_S)
                                 if cancel is not None else remaining)

    def peek_accumulate(self, count: int, *, timeout: float = 30.0,
                        epoch: int | None = None,
                        cancel: threading.Event | None = None) -> bytes:
        """Wait until count bytes have accumulated and return a copy of them
        without consuming (the responder's magic peek). Short result on
        timeout or cancel; EpochFenced if the epoch moved on."""
        deadline = time.monotonic() + timeout
        with self._rx_cv:
            want = self._epoch if epoch is None else epoch
            while True:
                if want != self._epoch:
                    raise EpochFenced(f"{self.name}: epoch {want} superseded by {self._epoch}")
                if len(self._rx) >= count:
                    return bytes(self._rx[:count])
                if cancel is not None and cancel.is_set():
                    return bytes(self._rx[:count])
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return bytes(self._rx[:count])
                self._rx_cv.wait(min(remaining, _CANCEL_POLL_S)
                                 if cancel is not None else remaining)

    # -- reader threads ----------------------------------------------------

    def _cmd_reader(self, gen: int, sock: socket.socket) -> None:
        buf = b""
        reason = "cmd channel lost"
        # a reader thread that dies without detaching leaves the client
        # attached-but-deaf forever, so nothing here may escape the finally
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while CR in buf:
                    raw, buf = buf.split(CR, 1)
                    if raw:
                        self._on_cmd_line(raw.decode("ascii", "replace"))
        except Exception as exc:
            reason = f"cmd reader failed: {exc!r}"
        finally:
            self._detach(gen, reason, deliberate=False)

    def _on_cmd_line(self, text: str) -> None:
        t = self.transcript
        t.cmd_rx(self.name, text)
        line = classify(text)
        if line.kind == UNKNOWN:
            # unknown lines are conformance findings on every session
            t.conf(self.name, "vocabulary", "EXTRA", line=text)
        elif line.kind == NOTIFICATION:
            name = line.name
            if name == "CONNECTED":
                self.connected = True
                self.last_connected = line.fields
                t.state(self.name, "connected", detail=text)
            elif name == "DISCONNECTED":
                self.connected = False
                self.buffer_bytes = 0
                t.state(self.name, "disconnected")
            elif name == "PTT":
                t.ptt(self.name, line.fields["on"])
            elif name == "BUFFER":
                self.buffer_bytes = line.fields["n"]
            elif name == "IAMALIVE":
                self.last_iamalive = time.monotonic()
            elif name == "VERSION":
                self.version = line.fields["version"]
        self._publish(line)

    def _data_reader(self, gen: int, sock: socket.socket) -> None:
        reason = "data channel lost"
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                with self._rx_cv:
                    self._rx += chunk
                    self._rx_cv.notify_all()
                self.transcript.data(self.name, "rx", chunk)
        except Exception as exc:
            reason = f"data reader failed: {exc!r}"
        finally:
            self._detach(gen, reason, deliberate=False)
