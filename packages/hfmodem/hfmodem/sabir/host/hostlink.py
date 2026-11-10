# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The structured host front-end (HOST-API.md): one framed CBOR connection
carrying typed commands and events.

`HostLink` is one attached application. It runs the HELLO handshake, dispatches
command frames to a :class:`ModemCore`, and — as that modem's
:class:`ModemObserver` — encodes link events back as frames. A reader loop runs
on its own thread, with observer callbacks arriving
from the air thread, one send lock. `HostClient` is the peer side, used by the
native Pat backend and the tests.
"""

from __future__ import annotations

import socket
import threading
from typing import Optional

from . import messages as M
from .messages import PROTO
from .modem_core import ModemCore, ModemObserver

MODEM_NAME = "sabir"
FEATURES = ("data-profiles", "objects")
PROFILES = (M.PF_AMATEUR, M.PF_UNRESTRICTED)

# events a subscription gates; the rest are always delivered
_OPT_IN = frozenset({M.LINK_STATS, M.ID_SENT, M.PHYSICAL_STATE, M.PEER_OBSERVED})


def _recvn(conn: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = conn.recv(n - len(buf))
        except TimeoutError:                   # alive, no data yet -> propagate
            raise
        except OSError:                        # socket closed under us -> EOF
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


MAX_FRAME = 16 * 1024 * 1024               # HOST-API §2: a larger prefix closes


class FrameTooLarge(ValueError):
    """A length prefix past MAX_FRAME -- a protocol error, not a frame."""


def read_frame(conn: socket.socket) -> Optional[bytes]:
    """One length-prefixed frame body, or None at EOF. Raises FrameTooLarge on
    an oversize prefix -- before reading or allocating the body (§2)."""
    hdr = _recvn(conn, 4)
    if hdr is None:
        return None
    n = int.from_bytes(hdr, "big")
    if n > MAX_FRAME:
        raise FrameTooLarge(n)
    return _recvn(conn, n)


class HostLink(ModemObserver):
    def __init__(self, conn: socket.socket, modem: ModemCore, *, log=None):
        self._conn = conn
        self._modem = modem
        self._log = log or (lambda *a: None)
        self._lock = threading.Lock()
        self._closed = False
        self._subs: frozenset = frozenset()
        self._identity: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:
        self._modem.start(self)
        try:
            if self._handshake():
                self._reader()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for verb in (lambda: self._modem.submit(self._modem.abort), self._modem.stop):
            try:
                verb()
            except Exception:
                pass
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()

    def _send(self, msg: dict) -> None:
        if self._closed:
            return
        with self._lock:
            try:
                self._conn.sendall(M.encode(msg))
                self._log("modem->app", msg)
            except OSError:
                pass

    def _emit(self, msg: dict) -> None:
        if msg["m"] in _OPT_IN and msg["m"] not in self._subs:
            return
        self._send(msg)

    def _error(self, code: int, ref=None) -> None:
        msg = {"m": M.ERROR, "code": code}
        if ref is not None:                 # ref:u? -- absent, never null (§5)
            msg["ref"] = ref
        self._send(msg)

    # -- handshake ---------------------------------------------------------
    def _handshake(self) -> bool:
        try:
            frame = read_frame(self._conn)
            if frame is None:
                return False
            hello = M.decode(frame)
            if hello.get("m") != M.HELLO or not isinstance(hello.get("proto"), str):
                raise ValueError("expected Hello")
        except Exception:
            self._error(M.ERR_MALFORMED)
            return False
        self._send({"m": M.HELLO, "proto": PROTO, "features": list(FEATURES),
                    "modem": MODEM_NAME, "profiles": list(PROFILES),
                    "identity_required": True})
        peer_major = str(hello.get("proto", "0")).split(".")[0]
        if peer_major != PROTO.split(".")[0]:
            self._error(M.ERR_INCOMPATIBLE)
            return False
        return True

    # -- inbound: command frames -------------------------------------------
    def _reader(self) -> None:
        while not self._closed:
            try:
                frame = read_frame(self._conn)
            except FrameTooLarge:               # §2: report, then close
                self._error(M.ERR_MALFORMED)
                break
            if frame is None:
                break
            try:
                msg = M.decode(frame)
            except Exception:
                self._error(M.ERR_MALFORMED)
                continue
            self._modem.submit(lambda msg=msg: self._execute(msg))

    def _execute(self, msg: dict) -> None:
        if self._closed:
            return
        try:
            self._dispatch(msg)
        except Exception:
            ref = msg.get("ref")
            self._error(M.ERR_MALFORMED, ref if type(ref) is int and ref >= 0 else None)

    def _need_identity(self, ref) -> bool:
        if self._identity:
            return False
        self._error(M.ERR_NO_IDENTITY, ref)
        return True

    def _dispatch(self, msg: dict) -> None:
        self._log("app->modem", msg)
        m = msg.get("m")
        ref = msg.get("ref")
        for field in ("ref", "id", "addressee", "service", "repeats", "feedback_iters", "bandwidth_hz"):
            if field in msg and (type(msg[field]) is not int or msg[field] < 0):
                raise ValueError(f"{field} must be an unsigned integer")
        if m == M.SET_IDENTITY:
            if self._modem.connected:
                raise ValueError("identity is fixed during a session")
            identity = msg["station_id"].upper()
            from hfmodem.sabir.arq.wire import station_bytes
            station_bytes(identity)
            if not identity:
                raise ValueError("identity must not be empty")
            if msg.get("aliases"):
                raise ValueError("aliases are unsupported")
            self._modem.set_identity(identity)
            self._identity = identity
        elif m == M.SET_PROFILE:
            if type(msg["profile"]) is not int or msg["profile"] not in PROFILES:
                raise ValueError("unsupported profile")
            self._modem.set_profile(msg["profile"])
        elif m == M.LISTEN:
            if self._need_identity(ref):
                return
            if type(msg.get("on")) is not bool:
                raise ValueError("Listen.on must be boolean")
            if "station_id" in msg:
                if self._modem.connected:
                    raise ValueError("identity is fixed during a session")
                from hfmodem.sabir.arq.wire import station_bytes
                identity = msg["station_id"].upper()
                station_bytes(identity)
                self._modem.set_identity(identity)
            self._modem.set_listen(msg["on"])
        elif m == M.CONNECT:
            if self._need_identity(ref):
                return
            if "profile" in msg or "deadline" in msg:
                raise ValueError("unsupported Connect attributes")
            from hfmodem.sabir.arq.wire import station_bytes
            src = msg.get("station_id", self._identity).upper()
            dst = msg["peer_id"].upper()
            station_bytes(src)
            station_bytes(dst)
            self._modem.connect(src, dst)
        elif m == M.SEND:
            data = msg["data"]
            if not isinstance(data, bytes) or type(msg.get("deflate", False)) is not bool:
                raise ValueError("invalid Send fields")
            if msg.get("stream", 0) != 0 or msg.get("priority", 0) != 0 or "deadline" in msg:
                raise ValueError("unsupported Send attributes")
            deflated = self._modem.transmit(data, msg_id=msg.get("id"),
                                             deflate=msg.get("deflate", False))
            if "id" in msg:
                self._send({"m": M.SEND_PROGRESS, "id": msg["id"],
                            "sent": len(data), "total": len(data),
                            "delivered": False,
                            "deflated": deflated})
        elif m == M.DISCONNECT:
            self._modem.disconnect()
        elif m == M.ABORT:
            self._modem.abort()
        elif m == M.SUBSCRIBE:
            events = msg["events"]
            if not isinstance(events, list) or any(type(e) is not int or e < 0 for e in events):
                raise ValueError("invalid event subscription")
            self._subs = frozenset(events)
        elif m == M.BEACON:
            if self._need_identity(ref):
                return
            self._modem.beacon(msg.get("addressee", 0))
        elif m == M.CONFIGURE:
            if "radio" in msg or "ptt" in msg:
                raise ValueError("radio binding is configured by the station")
            keys = ("data_profile", "receive_profiles", "feedback_iters", "impulse_blank", "bandwidth_hz", "inactivity_timeout_s")
            self._modem.configure_data(**{k: msg[k] for k in keys if k in msg})
        elif m == M.SEND_OBJECT:
            if self._need_identity(ref):
                return
            if not isinstance(msg["data"], bytes) or type(msg.get("parity", True)) is not bool:
                raise ValueError("invalid SendObject fields")
            keys = ("destination", "service", "data_profile", "parity", "repeats", "message_id")
            mid = self._modem.send_object(msg["data"], **{k: msg[k] for k in keys if k in msg})
            self._send({"m": M.SEND_PROGRESS, "message_id": mid,
                        "sent": 0, "total": len(msg["data"]), "delivered": False})
        # unknown m: must-ignore (§10.1)

    # -- ModemObserver: link events -> frames ------------------------------
    def modem_connected(self, src, dst, bw) -> None:
        self._send({"m": M.STATE_CHANGED, "state": M.ST_CONNECTED,
                    "peer_id": dst})

    def modem_disconnected(self, reason: int = 0) -> None:
        msg = {"m": M.STATE_CHANGED, "state": M.ST_DISCONNECTED}
        if reason:                          # DISC_* == wire RS_* (§5); omit if 0
            msg["reason"] = reason
        self._send(msg)

    # LISTENING/CONNECTING/DISCONNECTING; CONNECTED/DISCONNECTED ride the
    # peer_id-carrying callbacks above, so skip them here to avoid a double.
    _INTERMEDIATE = {"LISTENING": M.ST_LISTENING, "CONNECTING": M.ST_CONNECTING,
                     "DISCONNECTING": M.ST_DISCONNECTING}

    def modem_state_changed(self, state: str) -> None:
        code = self._INTERMEDIATE.get(state)
        if code is not None:
            self._send({"m": M.STATE_CHANGED, "state": code})

    def modem_capabilities(self, image: dict) -> None:
        self._send({"m": M.CAPABILITIES, **image})

    def modem_link_stats(self, snapshot: dict) -> None:
        self._emit({"m": M.LINK_STATS, **snapshot})

    def modem_id_sent(self, station_id: str, t: float) -> None:
        self._emit({"m": M.ID_SENT, "station_id": station_id, "t": float(t)})

    def modem_send_progress(self, msg_id: int, delivered: bool) -> None:
        self._send({"m": M.SEND_PROGRESS, "id": msg_id, "delivered": delivered})

    def modem_peer_observed(self, image: dict) -> None:
        self._emit({"m": M.PEER_OBSERVED, **image})

    def modem_data_received(self, blob: bytes) -> None:
        self._send({"m": M.DATA_RECEIVED, "data": bytes(blob), "stream": 0})

    def modem_object_received(self, image: dict) -> None:
        self._send({"m": M.OBJECT_RECEIVED, **image})

    def modem_ptt(self, on: bool) -> None:
        self._emit({"m": M.PHYSICAL_STATE, "ptt": bool(on)})

    def modem_busy(self, on: bool) -> None:
        self._emit({"m": M.PHYSICAL_STATE, "busy": bool(on)})



class HostLinkServer:
    """Accepts one application at a time on a single framed-CBOR port."""

    def __init__(self, modem_factory, host: str = "127.0.0.1", port: int = 8400,
                 log=None):
        self._factory = modem_factory
        self._log = log or (lambda *a: None)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self._running = False
        self._link: Optional[HostLink] = None

    def serve_forever(self) -> None:
        self._running = True
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._log("server", "application attached")
            self._link = HostLink(conn, self._factory(), log=self._log)
            try:
                self._link.run()
            finally:
                self._link = None
                self._log("server", "application detached")

    def start_background(self) -> None:
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self._link:
            self._link.close()
        try:
            self._srv.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._srv.close()


class HostClient:
    """The application side of a HostLink: frame a command, read events."""

    def __init__(self, conn: socket.socket):
        self._conn = conn

    def send(self, msg: dict) -> None:
        self._conn.sendall(M.encode(msg))

    def recv(self) -> Optional[dict]:
        frame = read_frame(self._conn)
        return None if frame is None else M.decode(frame)

    def hello(self, features=()) -> dict:
        self.send({"m": M.HELLO, "proto": PROTO, "features": list(features),
                   "client": "sabir-test"})
        return self.recv()

    def close(self) -> None:
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()
