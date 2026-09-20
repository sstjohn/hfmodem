# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""When rigctld does not answer, the log must say which way it failed.

The four unkey timeouts of 2026-08-09 went into the log as one sentence --
"rigctld took the command and did not answer" -- and a two-day theory grew where
evidence should have been. That sentence covers failures with different causes: a
daemon not listening, a daemon alive on the socket and dead to the radio, a reply
cut off mid-line, a verdict the parser could not read. These tests stand up one
fake for each shape and hold the log to naming it, with the raw bytes and timing
alongside, so the next occurrence on the air produces a record rather than
another inference.

Everything runs against fakes on loopback ephemeral ports. Nothing here may touch
4533: there is a real transmitter on the end of it.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest

from hfmodem.tests.kestrel import corpora, fake_rigctld

bridge = corpora.harness("vara_rig_bridge")


@pytest.fixture
def rigctld():
    yield from fake_rigctld.serving()


@pytest.fixture
def rig():
    """An armed `Rig` keeping its log, stood down after the test."""
    made = []

    def make(port: int, **kw):
        logged: list[str] = []
        r = bridge.Rig(f"127.0.0.1:{port}", armed=True, log=logged.append)
        r.logged = logged
        for k, v in kw.items():
            setattr(r, k, v)
        made.append(r)
        return r

    yield make
    for r in made:
        r._stop.set()


class _Answers(threading.Thread):
    """A daemon that answers every connection with exactly ``payload``, then hangs
    up (``then_close``) or holds the connection open. `FakeRigctld` models a
    rigctld behaving; this models the wire failing in the specific shapes the
    bridge's `_Wire` record has to tell apart."""

    def __init__(self, payload: bytes, then_close: bool = True):
        super().__init__(daemon=True)
        self.payload, self.then_close = payload, then_close
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._open: list[socket.socket] = []
        self._closed = False

    def run(self):
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                conn.recv(64)
                conn.sendall(self.payload)
            except OSError:
                pass
            if self.then_close:
                conn.close()
            else:
                self._open.append(conn)

    def close(self):
        self._closed = True
        self._sock.close()
        for c in self._open:
            c.close()


@pytest.fixture
def answers():
    made = []

    def start(payload: bytes, **kw) -> _Answers:
        s = _Answers(payload, **kw)
        s.start()
        made.append(s)
        return s

    yield start
    for s in made:
        s.close()


def _attempts(r) -> list[str]:
    return [m for m in r.logged if "PTT OFF attempt" in m]


# -- one fake per failure shape, one name per fake ---------------------------------


def test_a_refused_connection_is_named_with_bytes_and_held_time(rig):
    r = rig(1, keyed=True, key_since=time.time() - 5)   # nothing listens on port 1
    r._unkey(0.3, 0.1)
    line = _attempts(r)[0]
    assert "refused the connection" in line, line
    assert r"sent b'T 0\n'" in line, line
    assert "PTT held" in line, line
    assert "total" in line and "ms" in line, line


def test_a_daemon_that_accepts_and_goes_silent_is_not_called_refused(rigctld, rig):
    server = rigctld(answer=False)
    r = rig(server.port, keyed=True, key_since=time.time())
    r._unkey(0.3, 0.1)
    line = _attempts(r)[0]
    assert "accepted the command and went silent" in line, line
    assert "got b''" in line, line
    assert "timeout at" in line, line
    assert "t -> no answer" in line, line


def test_a_reply_cut_off_mid_verdict_is_not_called_a_refusal(rig, answers):
    """The old record for this shape was "rigctld refused: 'RPRT'" -- a daemon
    mid-collapse dressed as a rig saying no."""
    server = answers(b"RPRT", then_close=False)     # the verdict never finishes
    r = rig(server.port, keyed=True, key_since=time.time())
    r._unkey(0.3, 0.1)
    line = _attempts(r)[0]
    assert "truncated verdict" in line, line
    assert "got b'RPRT'" in line, line
    assert "refused" not in line, line


def test_a_peer_that_closes_mid_reply_is_named(rig, answers):
    server = answers(b"RPRT -1 " + b"x" * 56)       # one full 64B read, then gone
    r = rig(server.port, keyed=True, key_since=time.time())
    r._unkey(0.3, 0.1)
    line = _attempts(r)[0]
    assert "closed the connection mid-reply" in line, line
    assert "EOF at" in line, line


def test_a_connection_closed_without_a_byte_is_its_own_mode(rig, answers):
    server = answers(b"")
    r = rig(server.port, keyed=True, key_since=time.time())
    r._unkey(0.3, 0.1)
    line = _attempts(r)[0]
    assert "closed it without a byte" in line, line


def test_an_in_band_refusal_is_reported_as_the_rigs_answer(rigctld, rig):
    server = rigctld(refuse=("T",))
    r = rig(server.port)
    assert r.key(True, "burst") is False
    line = next(m for m in r.logged if "PTT ON attempt" in m)
    assert "refused in-band" in line, line
    assert r"sent b'T 1\n'" in line and r"got b'RPRT -1\n'" in line, line


# -- the parser takes every framing the daemon has actually used -------------------


def test_rprt_framing_variants_all_parse():
    """The daemon has answered the extended form with ``RPRT 0 RPRT 0`` on one
    line as well as one per line. Neither is a failure -- and a negative anywhere
    in either framing is, where the line-based reading stopped at the first code
    it saw."""
    ok = bridge.Rig._rprt_ok
    assert ok("RPRT 0\n")
    assert ok("RPRT 0\nRPRT 0\n")
    assert ok("RPRT 0 RPRT 0\n")
    assert not ok("RPRT 0 RPRT -9\n")
    assert not ok("RPRT -1\n")
    assert not ok("RPRT")               # cut off mid-verdict is not a pass
    assert ok("2\n")                    # no verdict at all: nothing to contradict


# -- and evidence is for failures, not ceremony ------------------------------------


def test_a_clean_unkey_writes_no_forensics(rigctld, rig):
    server = rigctld()
    server.ptt = 1
    r = rig(server.port, keyed=True, key_since=time.time())
    assert r.key(False) is True
    assert not _attempts(r), r.logged
