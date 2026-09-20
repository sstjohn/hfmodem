# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A scriptable fake modem speaking the structured host interface (HOST-API v1.0).

One connection, one ordered stream, length-prefixed CBOR — so unlike the
VARA-dialect fake there is no half-attach hazard to reproduce. What this does
reproduce is the parts a client can get wrong: the symmetric Hello (and its
major-version refusal), commands that answer only with events, echo of Send as
DataReceived, and the ability to emit an unknown message type and an unknown
field so the client's must-ignore path is exercised rather than assumed.
"""

from __future__ import annotations

import socket
import threading
from typing import Any

from .. import cbor
from ..hostapi import (CONNECT, DATA_RECEIVED, DISCONNECT, ERR_BAD_STATE,
                       ERR_INCOMPATIBLE, MAX_FRAME,
                       ERR_NO_IDENTITY, ERROR, HELLO, LISTEN, PROTO,
                       SEND, SET_IDENTITY, SET_PROFILE, ST_CONNECTED,
                       ST_DISCONNECTED, ST_LISTENING, STATE_CHANGED, SUBSCRIBE,
                       ABORT, KEY, NAME, pack, unpack)


class FakeHostApiModem:
    """A HOST-API modem for tests.

    on_command(msg, modem) -> None runs for every decoded command; return value
    is ignored, it is a hook for injecting events. echo=True turns Send into
    DataReceived, which is what makes a round-trip scenario testable in one
    process. require_identity mirrors a profile that mandates a station id.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0,
                 modem_version: str = "fake/1.0", proto: str = PROTO,
                 features: list[int] | None = None,
                 profiles: list[int] | None = None,
                 require_identity: bool = False, echo: bool = False,
                 on_command=None):
        self.host = host
        self.modem_version = modem_version
        self.proto = proto
        self.features = features if features is not None else []
        self.profiles = profiles if profiles is not None else [1, 2]
        self.require_identity = require_identity
        self.echo = echo
        self.on_command = on_command

        self.log: list[dict[str, Any]] = []
        self.sent: bytearray = bytearray()   # payload the host has submitted
        self.identity: str | None = None
        self.profile: int | None = None
        self.listening = False
        self.connected = False
        self.accepted = 0            # connections accepted, for reattach tests
        self.subscribed: list[int] = []

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]

        self._conn: socket.socket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # -- test controls -----------------------------------------------------

    def emit(self, m: int, **fields: Any) -> None:
        self._write(pack({"m": m, **fields}))

    def emit_raw(self, body: bytes) -> None:
        """Send an arbitrary CBOR body — for unknown types and unknown keys."""
        self._write(len(body).to_bytes(4, "big") + body)

    def emit_unknown_type(self) -> None:
        self.emit_raw(cbor.encode({0: 200, 1: "from the future"}))

    def emit_unknown_field(self) -> None:
        body = cbor.encode({0: STATE_CHANGED, KEY["state"]: ST_DISCONNECTED,
                            len(NAME) + 7: "a field you do not know"})
        self.emit_raw(body)

    def send_data(self, data: bytes, stream: int | None = None) -> None:
        msg: dict[str, Any] = {"data": data}
        if stream is not None:
            msg["stream"] = stream
        self.emit(DATA_RECEIVED, **msg)

    def state(self, state: int, peer_id: str | None = None,
              reason: int | None = None) -> None:
        msg: dict[str, Any] = {"state": state}
        if peer_id is not None:
            msg["peer_id"] = peer_id
        if reason is not None:
            msg["reason"] = reason
        self.emit(STATE_CHANGED, **msg)

    def drop(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        self.drop()
        try:
            self._sock.close()
        except OSError:
            pass

    def commands(self, m: int | None = None) -> list[dict[str, Any]]:
        return [c for c in self.log if m is None or c.get("m") == m]

    # -- server ------------------------------------------------------------

    def _write(self, frame: bytes) -> None:
        with self._lock:
            conn = self._conn
        if conn is None:
            return
        try:
            conn.sendall(frame)
        except OSError:
            pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._conn = conn
                self.accepted += 1
            self._session(conn)

    def _session(self, conn: socket.socket) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = conn.recv(65536)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while len(buf) >= 4:
                size = int.from_bytes(buf[:4], "big")
                if size > MAX_FRAME:
                    # spec §2: a larger prefix is a protocol error and the
                    # modem closes rather than trying to allocate it
                    self.drop()
                    return
                if len(buf) < 4 + size:
                    break
                body, buf = bytes(buf[4:4 + size]), buf[4 + size:]
                try:
                    self._handle(unpack(body))
                except Exception:
                    self.emit(ERROR, code=4, detail="undecodable frame")

    def _handle(self, msg: dict[str, Any]) -> None:
        self.log.append(msg)
        m = msg.get("m")

        if m == HELLO:
            theirs = str(msg.get("proto", "")).split(".")[0]
            self.emit(HELLO, proto=self.proto, features=self.features,
                      modem=self.modem_version, profiles=self.profiles,
                      identity_required=self.require_identity)
            if theirs != self.proto.split(".")[0]:
                self.emit(ERROR, code=ERR_INCOMPATIBLE,
                          detail=f"modem speaks {self.proto}")
                self.drop()
                return
        elif m == SET_IDENTITY:
            self.identity = msg.get("station_id")
        elif m == SET_PROFILE:
            self.profile = msg.get("profile")
        elif m == SUBSCRIBE:
            self.subscribed = list(msg.get("events") or [])
        elif m == LISTEN:
            if self.require_identity and self.identity is None:
                self.emit(ERROR, code=ERR_NO_IDENTITY, detail="SetIdentity first")
                return
            self.listening = bool(msg.get("on"))
            # A modem already in session reports the session, not the listen
            # flag. Announcing LISTENING here would let a late-arriving Listen
            # overwrite a Connected this modem's peer has already reported, and
            # leave a linked station reading as unlinked for good.
            if not self.connected:
                self.state(ST_LISTENING if self.listening else ST_DISCONNECTED)
        elif m == CONNECT:
            if self.require_identity and self.identity is None:
                self.emit(ERROR, code=ERR_NO_IDENTITY, detail="SetIdentity first",
                          ref=msg.get("ref"))
                return
            self.connected = True
            self.state(ST_CONNECTED, peer_id=msg.get("peer_id"))
        elif m == SEND:
            blob = msg.get("data") or b""
            with self._lock:
                self.sent += blob
            if self.echo:
                self.send_data(blob, msg.get("stream"))
        elif m in (DISCONNECT, ABORT):
            if self.connected:
                self.connected = False
                self.state(ST_DISCONNECTED, reason=2)
            else:
                # §4: a command that cannot be honored yields an Error
                # correlated by the command's ref. Silence would leave a host
                # waiting for a state change that is never coming.
                self.emit(ERROR, code=ERR_BAD_STATE,
                          detail="not connected", ref=msg.get("ref"))

        if self.on_command is not None:
            self.on_command(msg, self)
