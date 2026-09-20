# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""End-to-end tests for the host-side VARA-protocol server.

The headline test wires the project's OWN oracle client
(``oracle/vara_client.py``) to THIS server and drives a full session:
handshake -> CONNECT -> data transfer -> DISCONNECT. Client and server are two
independent implementations of the same documented spec, so a green run is a
mutual correctness check on both. A second test pokes the raw command socket to
confirm OK/WRONG syntax handling and BUFFER flow-control cadence.

Run:  python3 test_loopback.py          (or) python3 -m pytest test_loopback.py
"""

from __future__ import annotations

import os
import socket
import tempfile
import time
import unittest

from hfmodem.kestrel.host.modem_core import LoopbackModem
from hfmodem.kestrel.host.protocol import CR
from hfmodem.kestrel.host.server import HostSession, VaraServer
from hfmodem.tests.kestrel.corpora import harness

payloads = harness("payloads")
Transcript = harness("transcript").Transcript
VaraClient = harness("vara_client").VaraClient


def _make_server():
    srv = VaraServer(host="127.0.0.1", cmd_port=0, data_port=0,
                     iamalive_interval=3600.0)  # long: no heartbeat noise in tests
    srv.start_background()
    return srv


def _make_transcript():
    tmp = tempfile.mkdtemp(prefix="kestrel-hostapi-")
    return Transcript(path=os.path.join(tmp, "t.jsonl"),
                      epoch=time.monotonic(), echo=False)


class TestOracleClientLoopback(unittest.TestCase):
    """The oracle client completes a real session against this server."""

    def test_connect_transfer_disconnect(self):
        srv = _make_server()
        t = _make_transcript()
        c = VaraClient("APP", mycall="N0CALL", transcript=t,
                       host="127.0.0.1", cmd_port=srv.cmd_port,
                       data_port=srv.data_port, scheme="varahf")
        try:
            c.connect_tcp()  # opens both sockets, runs the documented handshake

            # VERSION round-trips (newer/Pat-Vara command; server answers it).
            ver = c.version(timeout=5.0)
            self.assertIsNotNone(ver, "server did not answer VERSION")

            c.set_bandwidth("500")

            # CONNECT -> the loopback modem fabricates a link and reports CONNECTED.
            ok = c.connect("N0DX", timeout=10.0, p2p=True)
            self.assertTrue(ok, "did not reach CONNECTED")
            self.assertEqual(c.state, "connected")

            # Data transfer: payload written to the data port is echoed back.
            payload = payloads.build("prbs9", 4096)
            c.send_data(payload, label="prbs9")
            got = c.recv_data(len(payload), timeout=10.0)
            self.assertEqual(got, payload, "echoed payload is not byte-exact")

            # Flow control: TX buffer drains to 0 (BUFFER 0 seen by the client).
            self.assertTrue(c.flush(timeout=10.0), "TX buffer did not drain")
            self.assertEqual(c.buffer_bytes, 0)

            # Graceful disconnect.
            self.assertTrue(c.disconnect(timeout=10.0), "no clean DISCONNECTED")
            self.assertEqual(c.state, "disconnected")
        finally:
            c.close()
            srv.stop()
            t.close()

    def test_multiple_transfers(self):
        """Several sends in one session all echo back in order."""
        srv = _make_server()
        t = _make_transcript()
        c = VaraClient("APP", mycall="N0CALL", transcript=t,
                       host="127.0.0.1", cmd_port=srv.cmd_port,
                       data_port=srv.data_port, scheme="varahf")
        try:
            c.connect_tcp()
            self.assertTrue(c.connect("N0DX", timeout=10.0))
            chunks = [payloads.build("counter", 256),
                      payloads.build("prbs9", 1000),
                      payloads.build("ones", 512)]
            for ch in chunks:
                c.send_data(ch)
            self.assertTrue(c.flush(timeout=10.0))
            got = c.recv_data(sum(len(x) for x in chunks), timeout=10.0)
            self.assertEqual(got, b"".join(chunks))
            self.assertTrue(c.disconnect(timeout=10.0))
        finally:
            c.close()
            srv.stop()
            t.close()


class TestRawProtocol(unittest.TestCase):
    """Talk to the command socket directly to check syntax + BUFFER cadence."""

    def _readlines(self, sock, want, timeout=5.0):
        """Read until `want` complete CR-terminated messages are seen."""
        sock.settimeout(timeout)
        buf = b""
        out = []
        deadline = time.monotonic() + timeout
        while len(out) < want and time.monotonic() < deadline:
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
            while CR in buf:
                line, buf = buf.split(CR, 1)
                if line:
                    out.append(line.decode("ascii", errors="replace"))
        return out

    def test_ok_wrong_and_version(self):
        srv = _make_server()
        cmd = socket.create_connection(("127.0.0.1", srv.cmd_port), timeout=5)
        data = socket.create_connection(("127.0.0.1", srv.data_port), timeout=5)
        try:
            # First async message the modem sends after attach is registration.
            first = self._readlines(cmd, 1)
            self.assertIn("LINK REGISTERED", first)

            cmd.sendall(b"MYCALL N0CALL" + CR)
            self.assertIn("OK", self._readlines(cmd, 1))

            cmd.sendall(b"TOTALLY UNKNOWN CMD" + CR)
            self.assertIn("WRONG", self._readlines(cmd, 1))

            cmd.sendall(b"BW9999" + CR)  # invalid bandwidth
            self.assertIn("WRONG", self._readlines(cmd, 1))

            cmd.sendall(b"VERSION" + CR)
            got = self._readlines(cmd, 1)
            self.assertTrue(any(g.startswith("VERSION") for g in got), got)
        finally:
            cmd.close()
            data.close()
            srv.stop()

    def test_buffer_flow_control(self):
        """After CONNECT, writing N bytes yields BUFFER N then BUFFER 0."""
        srv = _make_server()
        cmd = socket.create_connection(("127.0.0.1", srv.cmd_port), timeout=5)
        data = socket.create_connection(("127.0.0.1", srv.data_port), timeout=5)
        try:
            self._readlines(cmd, 1)  # drain LINK REGISTERED
            cmd.sendall(b"CONNECT N0CALL N0DX" + CR)
            # expect OK, PTT ON, PTT OFF, BUSY ON, CONNECTED. No PENDING: the
            # modem is the initiator here, and PENDING is a responder-role event
            # [test_loopback_honesty.py].
            msgs = self._readlines(cmd, 5, timeout=5.0)
            self.assertTrue(any(m.startswith("CONNECTED") for m in msgs), msgs)
            self.assertNotIn("PENDING", msgs, msgs)
            self.assertNotIn("BUSY OFF", msgs, "channel freed while the link is up")

            payload = b"\x5a" * 2048
            data.sendall(payload)
            # expect BUFFER 2048, PTT ON, BUFFER 0, PTT OFF
            msgs = self._readlines(cmd, 4, timeout=5.0)
            self.assertFalse(any(m.startswith("SN") for m in msgs),
                             f"SN emitted without CHAT ON: {msgs}")
            self.assertFalse(any(m.startswith("BITRATE") for m in msgs),
                             f"loopback has no gearshift to report: {msgs}")
            buffers = [m for m in msgs if m.startswith("BUFFER")]
            self.assertTrue(buffers, f"no BUFFER messages: {msgs}")
            self.assertIn("BUFFER 2048", buffers)      # enqueue
            self.assertEqual(buffers[-1], "BUFFER 0")   # drained

            # the payload is echoed back on the data port
            data.settimeout(5.0)
            echoed = b""
            while len(echoed) < len(payload):
                chunk = data.recv(4096)
                if not chunk:
                    break
                echoed += chunk
            self.assertEqual(echoed, payload)
        finally:
            cmd.close()
            data.close()
            srv.stop()


class TestNotifications(unittest.TestCase):
    """BITRATE per over, SN gated on CHAT ON, pre-connect buffering (spec §7.4)."""

    def _readlines(self, sock, want, timeout=5.0):
        sock.settimeout(timeout)
        buf, out = b"", []
        deadline = time.monotonic() + timeout
        while len(out) < want and time.monotonic() < deadline:
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
            while CR in buf:
                line, buf = buf.split(CR, 1)
                if line:
                    out.append(line.decode("ascii", errors="replace"))
        return out

    def _drain_until(self, sock, pred, timeout=5.0):
        """Collect cmd lines until one satisfies pred (inclusive) or timeout."""
        sock.settimeout(timeout)
        buf, out = b"", []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                break
            if not chunk:
                break
            buf += chunk
            while CR in buf:
                line, buf = buf.split(CR, 1)
                if line:
                    msg = line.decode("ascii", errors="replace")
                    out.append(msg)
                    if pred(msg):
                        return out
        return out

    # BITRATE and SN are host-API messages that no kestrel modem currently
    # drives: the real core has neither a gearshift hook nor an SNR estimate,
    # and the loopback is forbidden from inventing them. The wire formatting is
    # still a contract worth holding, so it is tested where it lives — at the
    # session, called directly, rather than through a modem that would have to
    # fabricate the numbers to reach it.

    def _session(self):
        """A HostSession wired to socketpairs, without starting its readers."""
        cmd_a, cmd_b = socket.socketpair()
        data_a, data_b = socket.socketpair()
        sess = HostSession(cmd_a, data_a, LoopbackModem(), iamalive_interval=3600.0)
        self.addCleanup(sess.close)
        for s in (cmd_a, cmd_b, data_a, data_b):
            self.addCleanup(s.close)
        return sess, cmd_b

    def test_bitrate_line_shape(self):
        sess, peer = self._session()
        sess.modem_bitrate(4, 3000, tx=True)
        sess.modem_bitrate(2, 512, tx=False)
        msgs = self._readlines(peer, 2)
        self.assertEqual(msgs, ["BITRATE (4) 3000 bps TX", "BITRATE (2) 512 bps RX"])

    def test_sn_gated_on_chat(self):
        sess, peer = self._session()

        sess.chat = False
        sess.modem_snr(15)
        sess.chat = True
        sess.modem_snr(-7)          # negative SN is ordinary on a real link
        msgs = self._readlines(peer, 1)
        self.assertEqual(msgs, ["SN -7"], "SN must be silent until CHAT ON")

        sess.chat = False
        sess.modem_snr(3)
        self.assertEqual(self._readlines(peer, 1, timeout=0.5), [],
                         "SN resumed after CHAT OFF")

    def test_preconnect_data_buffered_and_flushed(self):
        srv = _make_server()
        cmd = socket.create_connection(("127.0.0.1", srv.cmd_port), timeout=5)
        data = socket.create_connection(("127.0.0.1", srv.data_port), timeout=5)
        try:
            self._readlines(cmd, 1)  # LINK REGISTERED
            payload = bytes((i * 37) & 0xFF for i in range(1024))
            # Write to the data port BEFORE connecting: must be buffered, not dropped.
            data.sendall(payload)
            cmd.sendall(b"CONNECT N0CALL N0DX" + CR)
            self._drain_until(cmd, lambda m: m.startswith("CONNECTED"))
            # The buffered bytes are flushed on CONNECT and echoed back.
            data.settimeout(5.0)
            echoed = b""
            while len(echoed) < len(payload):
                chunk = data.recv(4096)
                if not chunk:
                    break
                echoed += chunk
            self.assertEqual(echoed, payload,
                             "pre-connect data was not buffered+flushed (spec §7.4)")
        finally:
            cmd.close()
            data.close()
            srv.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
