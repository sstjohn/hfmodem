# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""rig.py against a fake rigctld."""

import socket
import threading

import pytest

from hfhost.config import RigConfig
from hfhost.rig import Rig, RigError


class FakeRigctld:
    """Accepts one client at a time, records commands, replies per script."""

    def __init__(self, replies=None):
        self.replies = dict(replies or {})
        self.commands: list[str] = []
        self._srv = socket.socket()
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(2)
        self.port = self._srv.getsockname()[1]
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while self._running:
                    try:
                        chunk = conn.recv(4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        cmd = raw.decode().strip()
                        self.commands.append(cmd)
                        reply = self.replies.get(cmd.split()[0], "RPRT 0")
                        conn.sendall(reply.encode() + b"\n")

    def stop(self):
        self._running = False
        self._srv.close()


@pytest.fixture
def rigctld():
    server = FakeRigctld()
    yield server
    server.stop()


def test_cat_and_ptt(rigctld):
    with Rig(host="127.0.0.1", port=rigctld.port) as rig:
        rig.set_freq(14_105_000)
        rig.set_mode("usb", 2700)
        rig.ptt(True)
        rig.ptt(False)
    assert rigctld.commands == ["F 14105000", "M USB 2700", "T 1", "T 0"]


def test_config_supplies_host_and_port(rigctld):
    rig = Rig(RigConfig(host="127.0.0.1", port=rigctld.port))
    try:
        rig.ptt(True)
    finally:
        rig.close()
    assert rigctld.commands == ["T 1"]


def test_error_reply_raises_and_drops_the_socket():
    server = FakeRigctld({"T": "RPRT -1"})
    try:
        rig = Rig(host="127.0.0.1", port=server.port)
        with pytest.raises(RigError, match="RPRT -1"):
            rig.ptt(True)
        rig.set_freq(7_100_000)          # reconnects transparently
        rig.close()
    finally:
        server.stop()
    assert server.commands == ["T 1", "F 7100000"]


def test_unreachable_rigctld():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    with pytest.raises(RigError):
        Rig(host="127.0.0.1", port=port).ptt(False)
