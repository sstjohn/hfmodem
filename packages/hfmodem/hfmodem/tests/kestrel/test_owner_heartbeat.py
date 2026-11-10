# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A responsive TCP reader cannot impersonate a responsive live VARA owner."""
import socket

import pytest

from hfmodem.kestrel.host.modem_core import LoopbackModem
from hfmodem.kestrel.host.server import HostSession
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.tests.kestrel.test_compression import IO


class OneInterval:
    def __init__(self):
        self.elapsed = False

    def wait(self, timeout):
        previous, self.elapsed = self.elapsed, True
        return previous

    def set(self):
        self.elapsed = True


@pytest.fixture
def attachment():
    pairs = [socket.socketpair(), socket.socketpair()]
    modem = VaraHostModem(IO())
    session = HostSession(pairs[0][0], pairs[1][0], modem)
    lines = []
    session._send_cmd = lines.append
    modem.start(session)
    modem.poll()
    lines.clear()
    yield session, modem, lines
    session.close()
    modem.poll()
    session.ensure_retired()
    for _, peer in pairs:
        peer.close()


def interval(session):
    session._done = OneInterval()
    session._heartbeat()


def test_heartbeat_waits_for_actual_owner_service(attachment):
    session, modem, lines = attachment
    interval(session)
    assert 'IAMALIVE' not in lines
    # Ordinary TCP command handling remains responsive while the owner stalls.
    session._dispatch('VERSION')
    assert any(line.startswith('VERSION') for line in lines)
    assert 'IAMALIVE' not in lines
    modem.poll()
    assert lines.count('IAMALIVE') == 1
    modem.poll()
    assert lines.count('IAMALIVE') == 1


def test_stalled_intervals_coalesce_without_filling_the_command_queue(attachment):
    session, modem, lines = attachment
    for _ in range(200):
        interval(session)
    assert 'IAMALIVE' not in lines
    assert modem._events.empty()
    modem.poll()
    assert lines.count('IAMALIVE') == 1
    interval(session)
    modem.poll()
    assert lines.count('IAMALIVE') == 2


def test_attachment_retirement_cancels_unserviced_heartbeat(attachment):
    session, modem, lines = attachment
    interval(session)
    session.close()
    modem.poll()
    assert 'IAMALIVE' not in lines
    assert modem._retired_attachment


def test_failed_owner_poll_cannot_emit_alive(attachment, monkeypatch):
    session, modem, lines = attachment
    interval(session)
    poll = modem._poll
    def fail():
        raise RuntimeError('synthetic owner fault')
    monkeypatch.setattr(modem, '_poll', fail)
    with pytest.raises(RuntimeError, match='owner fault'):
        modem.poll()
    assert 'IAMALIVE' not in lines
    monkeypatch.setattr(modem, '_poll', poll)
    modem.poll()
    assert lines.count('IAMALIVE') == 1


def test_loopback_heartbeat_retains_its_immediate_test_backend_behavior():
    pairs = [socket.socketpair(), socket.socketpair()]
    modem = LoopbackModem()
    session = HostSession(pairs[0][0], pairs[1][0], modem)
    lines = []
    session._send_cmd = lines.append
    modem.start(session)
    try:
        interval(session)
        assert lines.count('IAMALIVE') == 1
    finally:
        session.close()
        modem._worker.join(1)
        assert not modem._worker.is_alive()
        session.ensure_retired()
        for _, peer in pairs:
            peer.close()
