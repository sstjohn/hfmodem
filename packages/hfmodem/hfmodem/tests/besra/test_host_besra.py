# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The whole system over TCP: a client → host dialect → real BesraModem →
virtual air → echoing peer → back. No radio, but every ARDOP frame is really
modulated and demodulated. This is the top-to-bottom integration gate.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import pytest

from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.host.server import HostServer
from hfmodem.besra.sim.echo import besra_with_echo_peer


@pytest.fixture
def besra_server():
    modem = besra_with_echo_peer("BESRA-1", bandwidth=500)
    srv = HostServer(modem, host="127.0.0.1", control_port=0, quiet=True)
    yield srv.start()                   # ephemeral ports; never collides
    srv.stop()


class _Cmd:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=5)
        self.lines: list[str] = []
        self._buf = bytearray()
        self._run = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while self._run:
            try:
                chunk = self.sock.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            self._buf.extend(chunk)
            while b"\r" in self._buf:
                line, _, rest = self._buf.partition(b"\r")
                self._buf = bytearray(rest)
                self.lines.append(line.decode("latin-1"))

    def send(self, line):
        self.sock.sendall(line.encode("latin-1") + b"\r")

    def wait(self, prefix, timeout=40.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for ln in list(self.lines):
                if ln.startswith(prefix):
                    return ln
            time.sleep(0.05)
        raise AssertionError(f"no {prefix!r}; saw {self.lines}")


@pytest.mark.slow
def test_client_connects_through_real_besra_and_echoes(besra_server):
    cport, dport = besra_server
    cmd = _Cmd("127.0.0.1", cport)
    data = socket.create_connection(("127.0.0.1", dport), timeout=5)

    cmd.send("INITIALIZE")
    cmd.send("MYCALL W9SSJ")
    cmd.wait("MYCALL now W9SSJ")
    cmd.send("PROTOCOLMODE ARQ")
    cmd.wait("PROTOCOLMODE now ARQ")

    # Dial the built-in echo peer; wait for the ARQ link to come up over real audio.
    cmd.send("ARQCALL BESRA-1 5")
    cmd.wait("CONNECTED BESRA-1")

    # Send bytes on the data port; the peer echoes them back through the waveform.
    payload = b"besra end to end"
    data.sendall(struct.pack(">H", len(payload)) + payload)

    data.settimeout(60)
    got = bytearray()
    while len(got) < len(payload):
        hdr = data.recv(2)
        (length,) = struct.unpack(">H", hdr)
        body = data.recv(length)
        got += body[3:]                      # strip the 3-char tag
    assert bytes(got) == payload

    cmd.send("DISCONNECT")
    cmd.wait("DISCONNECTED")


def test_buffer_and_datatosend_report_the_real_modem_queue():
    """Flow control reads queue depth through the `ModemCore` seam, so a host
    pacing its writes gets the truth from the real modem and not only from the
    loopback stand-in."""
    modem = BesraModem(bandwidth=500)
    srv = HostServer(modem, control_port=0, quiet=True)
    modem.set_mycall("W9SSJ")
    modem.start(srv, threaded=False)          # srv is the observer; no sockets bound
    modem.transmit(b"x" * 40)

    assert modem.queued == 40
    assert srv._table["BUFFER"]([], "BUFFER") == "BUFFER 40"
    assert srv._table["DATATOSEND"]([], "DATATOSEND") == "DATATOSEND 40"

    srv._table["PURGEBUFFER"]([], "PURGEBUFFER")
    assert modem.queued == 0
