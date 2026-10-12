# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A KiwiSDR's sound stream, over the standard library.

Written here rather than vendored. kiwiclient, the client everyone else uses,
carries no licence at all (github.com/jks-prv/kiwiclient issue 157, open since
2025-12), so it cannot ship inside an AGPL tree however it is copied. What a
receive-only stream needs of it is small: one WebSocket, a dozen ``SET``
commands, and the layout of a ``SND`` frame.

Two settings are chosen, not defaulted, because every consumer here measures
the signal rather than listening to it. Compression is off: the Kiwi's default
audio is IMA-ADPCM, lossy on exactly the amplitude detail a varied-modulus
constellation carries. AGC is off with a fixed gain: a receiver that re-levels
across a burst edge puts an amplitude error on the thing being measured.

Samples leave as the Kiwi sends them, big-endian int16, interleaved I and Q in
IQ mode. Nothing here converts them; the caller hands the bytes to ffmpeg.
"""

from __future__ import annotations

import base64
import os
import socket
import struct
import time
import urllib.parse
from typing import Iterator

CLOSE, PING, PONG = 0x8, 0x9, 0xA
IQ_HEADER = 10                      # GPS solution flag, pad, seconds, nanoseconds
KEEPALIVE_S = 1.0
MAX_REDIRECTS = 2


class KiwiError(RuntimeError):
    pass


class Redirect(KiwiError):
    def __init__(self, host: str) -> None:
        super().__init__(f"redirected to {host}")
        self.host = host


class WebSocket:
    """The client half of RFC 6455, as much of it as a Kiwi speaks."""

    def __init__(self, sock, host: str = "") -> None:
        self.sock = sock
        self.host = host
        self._buf = b""

    @classmethod
    def open(cls, host: str, port: int, path: str,
             timeout: float = 20.0) -> "WebSocket":
        for hop in range(MAX_REDIRECTS + 1):
            sock = socket.create_connection((host, port), timeout)
            try:
                return cls.handshake(sock, host, port, path)
            except Redirect as moved:
                sock.close()
                if hop == MAX_REDIRECTS:
                    raise KiwiError(f"{host}:{port}{path}: too many redirects") from None
                host = moved.host
                print(f"kiwi redirected to {host}", flush=True)
        raise AssertionError("unreachable")

    @classmethod
    def handshake(cls, sock, host: str, port: int, path: str) -> "WebSocket":
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                      "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Key: {key}\r\n"
                      "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        ws = cls(sock, host)
        head = ws._until(b"\r\n\r\n").decode("latin-1")
        status = head.split(" ", 2)
        if len(status) > 1 and status[1] in ("301", "302", "307", "308"):
            for line in head.splitlines()[1:]:
                name, _, value = line.partition(":")
                if name.strip().lower() == "location" and value.strip():
                    target = urllib.parse.urlsplit(value.strip())
                    if target.hostname:
                        raise Redirect(target.hostname)
        if len(status) < 2 or status[1] != "101":
            raise KiwiError(f"{host}:{port}{path} refused the WebSocket: "
                            f"{head.splitlines()[0] if head else 'no reply'}")
        return ws

    def _fill(self) -> None:
        chunk = self.sock.recv(65536)
        if not chunk:
            raise EOFError
        self._buf += chunk

    def _until(self, mark: bytes) -> bytes:
        while mark not in self._buf:
            self._fill()
        head, self._buf = self._buf.split(mark, 1)
        return head

    def _take(self, n: int) -> bytes:
        while len(self._buf) < n:
            self._fill()
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def send(self, text: str, opcode: int = 0x1) -> None:
        payload = text.encode() if isinstance(text, str) else text
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(head + mask + body)

    def messages(self) -> Iterator[bytes]:
        """Whole messages, continuations joined, until the server closes."""
        parts: list[bytes] = []
        while True:
            try:
                b0, b1 = self._take(2)
            except EOFError:
                return
            n = b1 & 0x7F
            if n == 126:
                n, = struct.unpack(">H", self._take(2))
            elif n == 127:
                n, = struct.unpack(">Q", self._take(8))
            mask = self._take(4) if b1 & 0x80 else None
            payload = self._take(n)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            opcode = b0 & 0x0F
            if opcode == CLOSE:
                return
            if opcode == PING:
                self.send(payload, PONG)
                continue
            if opcode == PONG:
                continue
            parts.append(payload)
            if b0 & 0x80:
                yield b"".join(parts)
                parts = []

    def close(self) -> None:
        try:
            self.send(struct.pack(">H", 1001), CLOSE)
        except OSError:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class KiwiStream:
    """One Kiwi receiver channel, tuned once, yielding raw sample payloads.

    ``mode`` is ``"iq"`` or a sideband; ``lo_hz``/``hi_hz`` are the passband the
    Kiwi filters to, signed about the dial in IQ mode. ``sample_rate`` is known
    once :meth:`start` returns, which it does when the first samples arrive.
    """

    def __init__(self, ws: WebSocket, dial_khz: float, *, mode: str = "iq",
                 lo_hz: int = -6000, hi_hz: int = 6000, gain: int = 50,
                 ident: str = "hfmodem", name: str = "") -> None:
        self.ws = ws
        self.dial_khz = dial_khz
        self.mode = mode
        self.lo_hz, self.hi_hz = lo_hz, hi_hz
        self.gain = gain
        self.ident = ident
        self.name = name
        self.sample_rate: float | None = None
        self._foff_khz = 0.0
        self._messages = ws.messages()
        self._first: bytes | None = None
        self._sent = 0.0

    @classmethod
    def open(cls, host: str, port: int, dial_khz: float, *,
             wideband: bool = False, **kw) -> "KiwiStream":
        path = f"{'/wb' if wideband else ''}/{int(time.time())}/SND"
        ws = WebSocket.open(host, port, path)
        return cls(ws, dial_khz, name=f"{ws.host}:{port}", **kw).start()

    def _set(self, command: str) -> None:
        self.ws.send(f"SET {command}")

    def _tune(self) -> None:
        self._set(f"ident_user={self.ident}")
        self._set(f"mod={self.mode} low_cut={self.lo_hz} high_cut={self.hi_hz} "
                  f"freq={self.dial_khz - self._foff_khz:.3f}")
        self._set(f"agc=0 hang=0 thresh=-100 slope=6 decay=1000 manGain={self.gain}")
        self._set("compression=0")
        self._set("keepalive")

    def _msg(self, body: str) -> None:
        for pair in body.split(" "):
            name, _, value = pair.partition("=")
            value = urllib.parse.unquote(value)
            if name == "too_busy":
                raise KiwiError(f"{self.name}: all {value} channels are taken")
            if name == "redirect":
                raise KiwiError(f"{self.name}: redirected to {value}")
            if name == "badp" and value != "0":
                raise KiwiError(f"{self.name}: refused the connection (badp={value}), "
                                "a password or an address it will not serve")
            if name == "down":
                raise KiwiError(f"{self.name}: the receiver is down")
            if name == "freq_offset":
                self._foff_khz = float(value)
            elif name == "audio_rate":
                self._set(f"AR OK in={int(float(value))} out=44100")
            elif name == "sample_rate":
                self.sample_rate = float(value)
                self._set("squelch=0 max=0")
                self._set("genattn=0")
                self._set("gen=0 mix=-1")
                self._tune()

    def _samples(self) -> Iterator[bytes]:
        for message in self._messages:
            tag, body = message[:3], message[3:]
            if tag == b"MSG":
                self._msg(body[1:].decode("latin-1"))
            elif tag == b"SND":
                now = time.monotonic()
                if now - self._sent >= KEEPALIVE_S:
                    self._set("keepalive")
                    self._sent = now
                flags, = struct.unpack("<B", body[:1])
                if flags & 0x10:
                    raise KiwiError(f"{self.name}: sent compressed audio after "
                                    "compression=0 was asked for")
                data = body[7:]
                yield data[IQ_HEADER:] if self.mode == "iq" else data

    def start(self) -> "KiwiStream":
        self._set("auth t=kiwi p=")
        self._it = self._samples()
        self._first = next(self._it, None)
        if self._first is None or self.sample_rate is None:
            raise KiwiError(f"{self.name}: closed before sending any audio")
        return self

    def blocks(self) -> Iterator[bytes]:
        if self._first is not None:
            first, self._first = self._first, None
            yield first
        yield from self._it

    def close(self) -> None:
        self.ws.close()
