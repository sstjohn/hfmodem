# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Client for the ARDOP host interface: a command socket and a counted data socket.

Written from ``docs/protocols/ardop/10-HOST-API.md``. This is the client half,
written from the specification rather than from any server — which is what makes
creance's grading of an ARDOP modem mean anything.

From a distance the dialect looks like the VARA one: two TCP sockets, one host
at a time, CR-terminated ASCII on the first. It differs in the two places that
matter. There is no OK/WRONG, so a command is answered by an echo of its own
verb and correlation is by verb rather than by position. And the data socket is
framed rather than raw — a two-byte big-endian count, with a three-character tag
inside modem-to-host blocks saying whether the bytes are session data, an FEC
broadcast, a block that could not be corrected, or a decoded ID.

Asynchronous status shares the command socket with those echoes, so nothing here
may assume the next line is the reply to the last command. That assumption is
the characteristic failure of this dialect: one NEWSTATE arriving between a
command and its echo desynchronizes a positional reader for the rest of the
session, and every later reply is attributed to the wrong command. The reader
below classifies each line on its own and hands the pump only the lines that
carry the verb it is waiting for.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Callable, Optional

from .client import AttachError, EpochFenced, ModemError, NotAttached
from .config import ModemConfig
from .link import (ATTACHED, BUFFER, BUSY, CANCELPENDING, CONNECTED, DETACHED,
                   DISCONNECTED, ERROR, OTHER, PENDING, PTT, LinkEvent,
                   _Adapter)
from .transcript import Transcript
from .wire import Line, NOTIFICATION, UNKNOWN

CR = b"\r"
DEFAULT_CMD_PORT = 8515          # the data port is always this one plus 1 (§1)

# Line kinds. The dialect has no OK/WRONG: a command's answer is its own verb
# echoed back, or a FAULT.
ECHO = "echo"
FAULT = "fault"

#: Protocol states, upper-cased for comparison. ARDOP folds every command it
#: receives but emits its state tokens verbatim, so IRStoISS really does arrive
#: mixed-case; a client that compared literally would fail to recognize the one
#: state that says a turnover is under way. The token itself is kept as sent.
STATES = ("OFFLINE", "DISC", "ISS", "IRS", "IDLE", "IRSTOISS", "FECSEND", "FECRCV")

#: ARQ session bandwidths in Hz. MAX negotiates down to what the path will
#: carry; FORCED refuses anything narrower.
ARQ_BANDWIDTHS = (200, 500, 1000, 2000)
ARQ_BW_TOKENS = tuple(f"{hz}{suffix}" for hz in ARQ_BANDWIDTHS
                      for suffix in ("MAX", "FORCED"))

#: CONREQ frames per ARQCALL. One is enough for the parser and far too few for
#: a real path; ten is what Pat dials with.
CONNECT_REQUESTS = 10

TAG_LEN = 3
TAG_ARQ, TAG_FEC, TAG_ERR, TAG_IDF = b"ARQ", b"FEC", b"ERR", b"IDF"
MAX_BLOCK = 0xFFFF               # the largest count a two-byte prefix can carry

#: Every verb the deployed modem answers to (§3.1). Its purpose is to tell a
#: command echo from a line nobody recognizes: without it an unknown verb and a
#: reply to something we sent are indistinguishable, and neither could be
#: reported honestly.
COMMANDS = frozenset("""
    ABORT ARQBW ARQCALL ARQTIMEOUT AUTOBREAK BREAK BUFFER BUSYBLOCK BUSYDET
    CALLBW CAPTURE CAPTUREDEVICES CL CLOSE CMDTRACE CONSOLELOG CWID DATATOSEND
    DD DEBUGLOG DISCONNECT DRIVELEVEL ENABLEPINGACK EXTRADELAY FASTSTART FECID
    FECMODE FECREPEATS FECSEND FSKONLY GRIDSQUARE INITIALIZE INPUTNOISE LEADER
    LISTEN LOGLEVEL MONITOR MYAUX MYCALL PING PLAYBACK PLAYBACKDEVICES
    PROTOCOLMODE PURGEBUFFER RADIOFREQ RADIOHEX RADIOPTTOFF RADIOPTTON RXLEVEL
    SENDID SQUELCH STATE TRAILER TUNINGRANGE TWOTONETEST TXLEVEL USE600MODES
    VERSION
""".split())


class ArdopFault(ModemError):
    """The modem answered a command with FAULT."""


def _bare(args: list[str]) -> dict | None:
    return {} if not args else None


def _flag(args: list[str]) -> dict | None:
    if len(args) == 1 and args[0].upper() in ("TRUE", "FALSE"):
        return {"on": args[0].upper() == "TRUE"}
    return None


def _connected(args: list[str]) -> dict | None:
    # "CONNECTED <call> <session bandwidth in Hz>".
    if len(args) == 2 and args[1].isdigit():
        return {"peer": args[0], "bw": int(args[1])}
    return None


def _newstate(args: list[str]) -> dict | None:
    if len(args) == 1 and args[0].upper() in STATES:
        return {"state": args[0]}
    return None


def _count(args: list[str]) -> dict | None:
    if len(args) == 1 and args[0].isdigit():
        return {"n": int(args[0])}
    return None


def _call(args: list[str]) -> dict | None:
    return {"call": args[0]} if len(args) == 1 else None


def _text(args: list[str]) -> dict | None:
    return {"text": " ".join(args)} if args else None


def _ping(args: list[str]) -> dict | None:
    # "PING <caller>><target> <snr> <quality>" -- the callsign pair is one token
    # split by '>', which is why this cannot be a positional parse.
    if len(args) == 3 and ">" in args[0] and args[1].lstrip("-").isdigit():
        caller, _, target = args[0].partition(">")
        return {"caller": caller, "target": target,
                "snr": int(args[1]), "quality": _int_or_none(args[2])}
    return None


def _pingack(args: list[str]) -> dict | None:
    if len(args) == 2:
        return {"snr": _int_or_none(args[0]), "quality": _int_or_none(args[1])}
    return None


def _peaks(args: list[str]) -> dict | None:
    if len(args) == 2:
        return {"min": _int_or_none(args[0]), "max": _int_or_none(args[1])}
    return None


def _int_or_none(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        return None


_NOTIFICATIONS = {
    "NEWSTATE": _newstate,
    "CONNECTED": _connected,
    "DISCONNECTED": _bare,
    "PENDING": _bare,
    "CANCELPENDING": _bare,
    "PTT": _flag,
    "BUSY": _flag,
    "BUFFER": _count,
    "TARGET": _call,
    "STATUS": _text,
    "FREQUENCY": _count,
    "PING": _ping,
    "PINGACK": _pingack,
    "PINGREPLY": _bare,
    "REJECTEDBW": _call,
    "REJECTEDBUSY": _call,
    "INPUTPEAKS": _peaks,
}


def _strip_now(rest: str) -> str:
    """A setter echoes ``<CMD> now <value>``; DISCONNECT answers ``NOW TRUE`` in
    upper case. The token is matched without regard to case, as Pat matches it,
    so both shapes yield the value alone."""
    return rest[4:].lstrip() if rest[:4].upper() == "NOW " else rest


def classify(line: str) -> Line:
    """Classify one line from the command socket.

    Stateless, and decided by the verb alone. The asynchronous vocabulary is
    disjoint from the command vocabulary except for BUFFER and PING, which the
    modem uses both as a query reply and as an unsolicited report; those
    classify as notifications, and the command pump correlates on the verb
    rather than on the kind so that a query still finds its answer.

    Leading and trailing whitespace goes first: every NEWSTATE carries a
    trailing space (§7), and a classifier that let it through would make every
    state token unrecognizable.

    Malformed arguments to a known verb classify as UNKNOWN — a grammar
    violation is a conformance finding, not something to guess at.
    """
    text = line.strip()
    parts = text.split()
    if not parts:
        return Line(UNKNOWN, "", raw=line)
    verb, args = parts[0].upper(), parts[1:]
    if verb == "FAULT":
        return Line(FAULT, verb, {"text": " ".join(args)}, line)
    parser = _NOTIFICATIONS.get(verb)
    if parser is not None:
        fields = parser(args)
        if fields is not None:
            return Line(NOTIFICATION, verb, fields, line)
        return Line(UNKNOWN, verb, raw=line)
    if verb in COMMANDS:
        return Line(ECHO, verb, {"value": _strip_now(" ".join(args))}, line)
    return Line(UNKNOWN, verb, raw=line)


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


class ArdopClient:
    """One attachment to an ARDOP TNC's command and data sockets.

    Mirrors ModemClient's surface — attach/close, command, send_data/read_data,
    the epoch-fenced receive buffer, desired-state replay — so a scenario reads
    identically over either dialect. What differs is the dialect's own doing:
    the reply pump matches verbs instead of a status token, and the data socket
    is deframed rather than passed through.
    """

    _REPLAY_ORDER = ("protocolmode", "bandwidth", "mycall", "listen")

    def __init__(self, cfg: ModemConfig, host: str, transcript: Transcript, *,
                 on_attach: Optional[Callable[["ArdopClient"], None]] = None,
                 on_detach: Optional[Callable[["ArdopClient"], None]] = None,
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
        self.state = ""
        self.peer: str | None = None
        self.version: str | None = None
        self.attach_failures = 0

        self._desired: dict[str, str] = {}

        self._subs: list[queue.Queue] = []
        self._subs_lock = threading.Lock()

        self._rx = bytearray()
        self._rx_cv = threading.Condition()
        self._epoch = 0

    @property
    def data_port(self) -> int:
        """The dialect fixes the data port at the command port plus one, so a
        configuration that leaves it unset is complete rather than wrong."""
        return self.cfg.data_port or self.cfg.cmd_port + 1

    # -- transcript --------------------------------------------------------

    @property
    def transcript(self) -> Transcript:
        return self._session_t or self._base_t

    def set_transcript(self, t: Transcript | None) -> None:
        self._session_t = t

    # -- attach / detach ---------------------------------------------------

    def attach(self, timeout: float | None = None) -> None:
        """Atomically connect both sockets, start readers, replay desired state.

        Both or neither: a TNC that has accepted the command socket and is
        waiting on the data socket has already displaced whatever host was there
        before, so a half-attach leaves the modem host-less and looking busy.
        """
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
                    data = self._dial(self.data_port, deadline)
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

    # -- pub/sub of cmd lines ----------------------------------------------

    def subscribe(self, q: queue.Queue | None = None) -> queue.Queue:
        """Queue of every subsequently-received Line. Subscribe before sending
        the triggering command; unsubscribe when done."""
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
        """Send one command and return the line that answers it.

        The answer is the first line carrying our own verb, whatever arrives
        before it; the modem interleaves status freely and a reader that took
        the next line would attribute a NEWSTATE to the command that provoked
        it. Correlation is by verb and not by kind, so a BUFFER query is
        answered by the BUFFER report even though that verb is also asynchronous
        — the two carry the same number.

        FAULT raises. The dialect gives a fault no correlation of its own, so
        one arriving while a command is outstanding is attributed to it; that is
        the modem's own ordering and the only reading available.
        """
        head = text.split(maxsplit=1)
        verb = head[0].upper() if head else ""
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
                if item.kind == FAULT:
                    raise ArdopFault(f"{self.name}: {text} -> {item.fields['text']}")
                if item.name == verb:
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
        try:
            return self.command("VERSION", timeout).fields["value"]
        except (ModemError, TimeoutError):
            return None

    # -- desired-state replay ----------------------------------------------

    def set_mycall(self, call: str) -> Line | None:
        return self._desire("mycall", f"MYCALL {call}")

    def set_listen(self, on: bool) -> Line | None:
        return self._desire("listen", "LISTEN " + ("TRUE" if on else "FALSE"))

    def set_bandwidth(self, token: str) -> Line | None:
        if token not in ARQ_BW_TOKENS:
            raise ValueError(f"unsupported ARQ bandwidth {token!r}")
        return self._desire("bandwidth", f"ARQBW {token}")

    def set_protocol_mode(self, mode: str) -> Line | None:
        return self._desire("protocolmode", f"PROTOCOLMODE {mode}")

    def _desire(self, slot: str, text: str) -> Line | None:
        self._desired[slot] = text
        return self.command(text) if self.attached else None

    def replay(self, timeout: float = 5.0) -> None:
        """Clear the modem's queued state, then reissue everything desired.

        INITIALIZE first because it is what a host sends first; LISTEN last
        because a modem that starts answering calls before MYCALL has landed
        answers them as somebody else. A FAULT on one setting is transcribed
        rather than raised: a modem that rejects one knob is still worth
        attaching to, and the alternative is a retry loop that never converges.
        """
        self.command("INITIALIZE", timeout)
        for slot in self._REPLAY_ORDER:
            text = self._desired.get(slot)
            if text is None:
                continue
            try:
                self.command(text, timeout)
            except ArdopFault as exc:
                self.transcript.error(self.name, f"replay rejected: {text}: {exc}")

    # -- session -----------------------------------------------------------

    def connect(self, dst: str, requests: int = CONNECT_REQUESTS) -> Line:
        """Start an ARQ call. The echo says only that the call was accepted for
        transmission; the outcome arrives later as CONNECTED, as DISCONNECTED,
        or as a FAULT with no command outstanding to attach it to."""
        return self.command(f"ARQCALL {dst} {requests}")

    def disconnect(self) -> Line:
        return self.command("DISCONNECT")

    def abort(self) -> Line:
        return self.command("ABORT")

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
        """Enqueue payload for transmission, in counted blocks.

        Host-to-modem blocks carry no tag — the tag exists to tell the host what
        a received block is, and there is only one thing to send. Anything
        longer than a two-byte count can address goes as several blocks, which
        the modem concatenates into its transmit buffer; the alternative is a
        silent truncation at 64 KiB.
        """
        sock = self._data_sock
        if not self.attached or sock is None:
            raise NotAttached(f"{self.name}: not attached")
        self.transcript.data(self.name, "tx", blob, label=label)
        self.buffer_bytes += len(blob)   # authoritative value follows via BUFFER
        blocks = [blob[i:i + MAX_BLOCK] for i in range(0, len(blob), MAX_BLOCK)]
        frames = b"".join(len(b).to_bytes(2, "big") + b for b in blocks)
        with self._data_tx_lock:
            try:
                sock.sendall(frames)
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
        without consuming. Short result on timeout or cancel; EpochFenced if
        the epoch moved on."""
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
                # Lines are CR-terminated (§1). Splitting on CR alone and
                # stripping what remains accepts a modem that sends CRLF too,
                # which costs nothing and is not worth a second code path.
                while CR in buf:
                    raw, buf = buf.split(CR, 1)
                    text = raw.decode("ascii", "replace").strip()
                    if text:
                        self._on_cmd_line(text)
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
        elif line.kind == FAULT:
            t.error(self.name, f"FAULT {line.fields['text']}")
        elif line.kind == ECHO and line.name == "VERSION":
            self.version = line.fields["value"]
        elif line.kind == NOTIFICATION:
            name = line.name
            if name == "CONNECTED":
                self.connected = True
                self.peer = line.fields["peer"]
                t.state(self.name, "connected", detail=text)
            elif name == "DISCONNECTED":
                self.connected = False
                self.peer = None
                self.buffer_bytes = 0
                t.state(self.name, "disconnected")
            elif name == "NEWSTATE":
                # Recorded, never turned into a disconnect event: the modem
                # sends DISCONNECTED for every session end anyway, and a second
                # one synthesized here could land after the next CONNECTED and
                # end a session that had only just begun.
                self.state = line.fields["state"]
            elif name == "PTT":
                t.ptt(self.name, line.fields["on"])
            elif name == "BUFFER":
                self.buffer_bytes = line.fields["n"]
        self._publish(line)

    def _data_reader(self, gen: int, sock: socket.socket) -> None:
        buf = bytearray()
        reason = "data channel lost"
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                # Two-byte big-endian count covering everything after it in the
                # block and never itself (§6), so a reader takes 2 + length. The
                # 3-char type tag is INSIDE the count modem->host: five payload
                # bytes arrive as 00 08, "ARQ", then the five. Over TCP there is
                # no multiplex prefix and no CRC; both belong to the serial
                # transport.
                while len(buf) >= 2:
                    size = int.from_bytes(buf[:2], "big")
                    if len(buf) < 2 + size:
                        break
                    block = bytes(buf[2:2 + size])
                    del buf[:2 + size]
                    self._on_block(block)
        except Exception as exc:
            reason = f"data reader failed: {exc!r}"
        finally:
            self._detach(gen, reason, deliberate=False)

    def _on_block(self, block: bytes) -> None:
        """Route one deframed block by its three-character tag."""
        if len(block) < TAG_LEN:
            self.transcript.conf(self.name, "data_framing", "SHORT",
                                 length=len(block))
            return
        tag, payload = block[:TAG_LEN], block[TAG_LEN:]
        label = tag.decode("ascii", "replace")
        if tag in (TAG_ARQ, TAG_FEC):
            with self._rx_cv:
                self._rx += payload
                self._rx_cv.notify_all()
            self.transcript.data(self.name, "rx", payload, label=label)
        elif tag == TAG_ERR:
            # The modem's own verdict on a block it could not repair. Recorded
            # and not delivered: a scenario that read these would count corrupt
            # bytes as received, which is exactly the measurement creance exists
            # to get right.
            self.transcript.note(self.name, "uncorrected data",
                                 length=len(payload))
        elif tag == TAG_IDF:
            self.transcript.note(self.name, "id frame",
                                 text=payload.decode("ascii", "replace"))
        else:
            self.transcript.conf(self.name, "data_tag", "EXTRA", tag=label)


class ArdopLink:
    """The ARDOP dialect behind the Link seam."""

    dialect = "ardop"

    def __init__(self, client: ArdopClient, cfg) -> None:
        self.client = client
        self.cfg = cfg
        self.name = client.name
        self._subs: list[queue.Queue] = []
        client.subscribe(_Adapter(self._emit, self._from_line))
        prior_attach, prior_detach = client.on_attach, client.on_detach
        client.on_attach = lambda c: (self._emit(LinkEvent(ATTACHED, c.name)),
                                      prior_attach and prior_attach(c))
        client.on_detach = lambda c: (self._emit(LinkEvent(DETACHED, c.name)),
                                      prior_detach and prior_detach(c))

    # -- lifecycle ---------------------------------------------------------

    def configure(self, mycall: str) -> None:
        self.client.set_protocol_mode("ARQ")
        self.client.set_bandwidth(_arqbw(self.cfg))
        self.client.set_mycall(mycall)

    def attach(self) -> None:
        self.client.attach()

    def close(self) -> None:
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
        """Always False: ARDOP defines no heartbeat, so there is no silence to
        measure. A dead socket detaches the client outright, which says more
        than a missed keepalive would."""
        return False

    def request_version(self) -> str:
        return self.version

    def telemetry(self) -> dict:
        """ARDOP reports a transmit-queue depth and nothing else."""
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
        if line.kind == FAULT:
            return LinkEvent(ERROR, self.name, {"detail": f["text"]}, line)
        if line.kind == NOTIFICATION:
            if name == "CONNECTED":
                # ARDOP names the remote station outright, so unlike the VARA
                # dialect there is no pair to disambiguate and no way to
                # attribute a session to the wrong callsign.
                return LinkEvent(CONNECTED, self.name,
                                 {"peer": f["peer"], "bw": f["bw"]}, line)
            if name == "DISCONNECTED":
                return LinkEvent(DISCONNECTED, self.name, {}, line)
            if name == "PENDING":
                return LinkEvent(PENDING, self.name, {}, line)
            if name == "CANCELPENDING":
                return LinkEvent(CANCELPENDING, self.name, {}, line)
            if name == "BUFFER":
                return LinkEvent(BUFFER, self.name, {"queue_bytes": f["n"]}, line)
            if name == "PTT":
                return LinkEvent(PTT, self.name, {"on": f["on"]}, line)
            if name == "BUSY":
                return LinkEvent(BUSY, self.name, {"on": f["on"]}, line)
        return LinkEvent(OTHER, self.name, {"line": line.raw}, line)


def _arqbw(cfg) -> str:
    """The widest ARQ bandwidth that fits the one the site configured.

    The configured enum is VARA's (500, 2300, 2750 Hz) and ARDOP's is
    200/500/1000/2000, so they do not line up and ARDOP has nothing as wide as
    2300. Rounding down keeps a run inside the channel the site was told to use;
    rounding up would put a station outside it. MAX rather than FORCED, so a
    session still negotiates down to whatever the path will carry.
    """
    want = int(getattr(cfg, "bandwidth", None) or ARQ_BANDWIDTHS[0])
    fits = [hz for hz in ARQ_BANDWIDTHS if hz <= want]
    return f"{fits[-1] if fits else ARQ_BANDWIDTHS[0]}MAX"
