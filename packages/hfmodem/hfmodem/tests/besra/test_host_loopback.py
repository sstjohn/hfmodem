# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""End-to-end host-server test: drive besra's ARDOP dialect as a client would.

Exercises the two-socket transport, the reply/echo grammar, the reproduced
byte-exact quirks, and the full connect → data round-trip → disconnect
choreography against the LoopbackModem — no radio.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from hfmodem.besra.host.modem_core import LoopbackModem
from hfmodem.besra.host.server import HostServer


class Client:
    """A minimal ARDOP host client: CR-terminated command socket + length-
    prefixed data socket, with a background reader that collects command lines."""

    def __init__(self, host: str, cport: int, dport: int) -> None:
        self.cmd = socket.create_connection((host, cport), timeout=5)
        self.data = socket.create_connection((host, dport), timeout=5)
        self.lines: list[str] = []
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._seen = 0                       # persistent read cursor
        self._run = True
        self.eof = threading.Event()         # set when the modem closes on us
        threading.Thread(target=self._read_cmd, daemon=True).start()

    def _read_cmd(self) -> None:
        try:
            while self._run:
                try:
                    chunk = self.cmd.recv(4096)
                except OSError:
                    return
                if not chunk:
                    return
                self._buf.extend(chunk)
                while b"\r" in self._buf:
                    line, _, rest = self._buf.partition(b"\r")
                    self._buf = bytearray(rest)
                    with self._lock:
                        self.lines.append(line.decode("latin-1"))
        finally:
            self.eof.set()

    def send(self, line: str) -> None:
        self.cmd.sendall(line.encode("latin-1") + b"\r")

    def send_data(self, blob: bytes) -> None:
        self.data.sendall(struct.pack(">H", len(blob)) + blob)

    def recv_data_block(self, timeout: float = 5.0) -> tuple[str, bytes]:
        self.data.settimeout(timeout)
        hdr = self._recv_exactly(self.data, 2)
        (length,) = struct.unpack(">H", hdr)
        body = self._recv_exactly(self.data, length)
        return body[:3].decode("ascii"), bytes(body[3:])

    @staticmethod
    def _recv_exactly(sock: socket.socket, n: int) -> bytearray:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed mid-block")
            buf.extend(chunk)
        return buf

    def wait_present(self, prefix: str, timeout: float = 5.0) -> str:
        """Scan the whole history (not the cursor) for a line with this prefix.
        For notifications whose order relative to a command reply is not fixed."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for ln in self.lines:
                    if ln.startswith(prefix):
                        return ln
            time.sleep(0.01)
        raise AssertionError(f"no line starting {prefix!r}; saw {self.lines}")

    def wait_line(self, prefix: str, timeout: float = 5.0) -> str:
        """Read forward from a persistent cursor for the next line with this
        prefix — so a later query reply is never satisfied by an earlier echo."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                while self._seen < len(self.lines):
                    ln = self.lines[self._seen]
                    self._seen += 1
                    if ln.startswith(prefix):
                        return ln
            time.sleep(0.01)
        raise AssertionError(f"no line starting {prefix!r}; saw {self.lines}")

    def close(self) -> None:
        self._run = False
        self.cmd.close()
        self.data.close()


@pytest.fixture
def server():
    srv = HostServer(LoopbackModem(bandwidth=500), host="127.0.0.1",
                     control_port=0, quiet=True)
    cport, dport = srv.start()          # ephemeral ports; never collides
    yield srv, cport, dport
    srv.stop()


def test_init_and_reply_grammar(server):
    srv, cport, dport = server
    c = Client("127.0.0.1", cport, dport)
    try:
        c.send("INITIALIZE")
        assert c.wait_line("INITIALIZE") == "INITIALIZE"

        c.send("VERSION")
        assert c.wait_line("VERSION").startswith("VERSION besra-")

        # Set-echo grammar: "<CMD> now <value>".
        c.send("ARQTIMEOUT 120")
        assert c.wait_line("ARQTIMEOUT") == "ARQTIMEOUT now 120"

        # Query grammar: "<CMD> <value>".
        c.send("ARQTIMEOUT")
        assert c.wait_line("ARQTIMEOUT ") == "ARQTIMEOUT 120"

        # Booleans upper-cased.
        c.send("listen false")
        assert c.wait_line("LISTEN") == "LISTEN now FALSE"
    finally:
        c.close()


def test_byte_exact_quirks(server):
    srv, cport, dport = server
    c = Client("127.0.0.1", cport, dport)
    try:
        # The misspelled unknown-command fault, verbatim.
        c.send("NOSUCHVERB")
        assert c.wait_line("FAULT") == "FAULT CMD NOSUCHVERB not recoginized"

        # CONSOLELOG echoes WITHOUT "now".
        c.send("CONSOLELOG 3")
        assert c.wait_line("CONSOLELOG") == "CONSOLELOG 3"

        # ARQBW suffix is FORCED, not FORCE.
        c.send("ARQBW 500FORCED")
        assert c.wait_line("ARQBW") == "ARQBW now 500FORCED"

        # DISCONNECT while idle is IGNORED (not a "now" echo).
        c.send("DISCONNECT")
        assert c.wait_line("DISCONNECT") == "DISCONNECT IGNORED"

        # PROTOCOLMODE accepts anything and echoes the raw parameter.
        c.send("PROTOCOLMODE GIBBERISH")
        assert c.wait_line("PROTOCOLMODE") == "PROTOCOLMODE now GIBBERISH"
    finally:
        c.close()


def test_connect_data_roundtrip_disconnect(server):
    srv, cport, dport = server
    c = Client("127.0.0.1", cport, dport)
    try:
        c.send("MYCALL W9SSJ")
        assert c.wait_line("MYCALL") == "MYCALL now W9SSJ"
        c.send("PROTOCOLMODE ARQ")
        c.wait_line("PROTOCOLMODE")

        # Dial: ARQCALL echoes the original line, then NEWSTATE ISS + CONNECTED.
        c.send("ARQCALL K7ABC 5")
        assert c.wait_line("ARQCALL") == "ARQCALL K7ABC 5"
        state = c.wait_line("NEWSTATE ISS")
        assert state == "NEWSTATE ISS "            # reproduced trailing space
        assert c.wait_line("CONNECTED") == "CONNECTED K7ABC 500"

        # Data round-trip: bytes written to the data port loop back tagged ARQ.
        c.send_data(b"hello ardop")
        tag, payload = c.recv_data_block()
        assert tag == "ARQ" and payload == b"hello ardop"

        # Graceful disconnect flushes then drops. The reply and the async
        # teardown notifications may interleave in any order — assert presence.
        c.send("DISCONNECT")
        assert c.wait_present("DISCONNECT NOW TRUE") == "DISCONNECT NOW TRUE"
        c.wait_present("DISCONNECTED")
        assert c.wait_present("NEWSTATE DISC ") == "NEWSTATE DISC "
    finally:
        c.close()


def test_mycall_validation(server):
    srv, cport, dport = server
    c = Client("127.0.0.1", cport, dport)
    try:
        c.send("MYCALL 12")                        # too short / not a call
        assert c.wait_line("FAULT").startswith("FAULT Syntax Err: MYCALL")
        c.send("ARQCALL K7ABC 5")                  # no MYCALL set yet
        assert c.wait_line("FAULT") == "FAULT MYCALL not set"
    finally:
        c.close()


def _connected_client(cport: int, dport: int) -> Client:
    """A client with an ARQ session up, for the lifecycle tests below."""
    c = Client("127.0.0.1", cport, dport)
    c.send("MYCALL W9SSJ")
    c.wait_line("MYCALL")
    c.send("ARQCALL K7ABC 5")
    c.wait_present("CONNECTED K7ABC 500")
    return c


def test_purgebuffer_empties_the_buffer_without_ending_the_session(server):
    """PURGEBUFFER is a buffer verb, not a session verb (spec §3.1): a host
    clearing a stalled queue mid-session keeps its link."""
    srv, cport, dport = server
    c = _connected_client(cport, dport)
    try:
        c.send("PURGEBUFFER")
        assert c.wait_present("BUFFER 0") == "BUFFER 0"
        assert c.wait_present("PURGEBUFFER") == "PURGEBUFFER"

        c.send("STATE")
        assert c.wait_line("STATE") == "STATE ISS"
        assert not any(ln.startswith("DISCONNECTED") for ln in c.lines)
    finally:
        c.close()


def test_a_new_host_displaces_the_previous_one_and_keeps_its_session(server):
    """One host per socket (spec §1). The incumbent's socket is closed on arrival
    of a replacement, and when that abandoned socket finally unwinds it must not
    run the host-link failsafe over the newcomer's live session."""
    srv, cport, dport = server
    first = _connected_client(cport, dport)
    try:
        second = Client("127.0.0.1", cport, dport)
        try:
            assert first.eof.wait(2.0), "the incumbent connection stayed open"
            time.sleep(0.2)                      # let the displaced handler unwind

            assert srv.modem.connected
            second.send("STATE")
            assert second.wait_line("STATE") == "STATE ISS"
            assert not any(ln.startswith("DISCONNECTED") for ln in second.lines)
        finally:
            second.close()
    finally:
        first.close()


def test_a_dropped_data_socket_ends_the_session(server):
    """The host-link failsafe covers *either* socket (spec §1): a host whose data
    socket dies would otherwise leave besra in a session nothing can feed."""
    srv, cport, dport = server
    c = _connected_client(cport, dport)
    try:
        c.data.close()
        assert c.wait_present("DISCONNECTED") == "DISCONNECTED"
        assert c.wait_present("NEWSTATE DISC ") == "NEWSTATE DISC "
    finally:
        c.close()
