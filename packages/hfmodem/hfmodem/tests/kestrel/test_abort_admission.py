# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owner-completed ABORT retires host admission, without inventing wire events."""
from types import SimpleNamespace

import pytest

from hfmodem.kestrel.host.server import HostSession, SessionState
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.vara import vara_arq as A
from .test_compression import IO


@pytest.fixture
def endpoint(monkeypatch):
    # No synthesis, audio, sockets, or physical keying. Only command ownership.
    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', lambda *a, **kw: True)
    modem = VaraHostModem(IO())
    host = HostSession(None, None, modem)
    lines = []
    host._send_cmd = lines.append
    modem.start(host)
    return host, modem, lines


def test_failed_call_abort_listen_then_next_call_is_admitted(endpoint):
    host, modem, lines = endpoint
    host._dispatch('CONNECT W9SSJ K0EPI')
    modem.poll()
    assert host.state == SessionState.CONNECTING
    host._dispatch('ABORT')
    assert lines[-1] == 'OK'
    assert host.state == SessionState.CONNECTING
    host._dispatch('CONNECT W9SSJ KL7QR')
    assert lines[-1] == 'WRONG', 'owner has not consumed abort yet'
    modem.poll()
    assert host.state == SessionState.DISCONNECTED
    assert 'DISCONNECTED' not in lines, 'an unestablished attempt is not a disconnected link'
    host._dispatch('LISTEN CQ')
    modem.poll()
    host._dispatch('LISTEN OFF')
    host._dispatch('CONNECT W9SSJ KL7QR')
    assert lines[-1] == 'OK'
    modem.poll()
    assert host.state == SessionState.CONNECTING
    assert modem.station.called == 'KL7QR'


def test_connected_abort_keeps_existing_disconnect_notification(endpoint):
    host, modem, lines = endpoint
    modem._connected = True
    modem.station.state = A.VaraState.CONNECTED
    host.modem_connected('W9SSJ', 'K0EPI', '500')
    host._dispatch('ABORT')
    assert host.state == SessionState.CONNECTED
    modem.poll()
    assert host.state == SessionState.DISCONNECTED
    assert lines.count('DISCONNECTED') == 1


def test_old_abort_completion_cannot_overwrite_new_connect_epoch():
    callbacks = []
    modem = SimpleNamespace(abort_with_completion=callbacks.append, connect=lambda *a: None)
    host = HostSession(None, None, modem)
    lines = []
    host._send_cmd = lines.append
    host._dispatch('CONNECT W9SSJ K0EPI')
    host._dispatch('ABORT')
    host._dispatch('ABORT')
    callbacks[0]()
    host._dispatch('CONNECT W9SSJ KL7QR')
    assert lines[-1] == 'OK' and host.state == SessionState.CONNECTING
    callbacks[1]()
    assert host.state == SessionState.CONNECTING


def test_abort_completion_uses_current_listen_policy(endpoint):
    host, modem, _ = endpoint
    host._dispatch('CONNECT W9SSJ K0EPI')
    modem.poll()
    host._dispatch('ABORT')
    host._dispatch('LISTEN CQ')
    assert host.state == SessionState.CONNECTING
    modem.poll()
    assert host.state == SessionState.LISTENING


def test_abort_does_not_erase_data_queued_after_it(endpoint):
    host, modem, lines = endpoint
    host._dispatch('CONNECT W9SSJ K0EPI')
    modem.poll()
    modem.transmit(b'old payload')
    host._dispatch('ABORT')
    modem.transmit(b'new payload')
    modem.poll()
    assert host.state == SessionState.DISCONNECTED
    assert modem.station.payload_ledger.pending == len(b'new payload')
    assert modem.retired[-1]['unconfirmed'] == b'old payload'
    assert 'DISCONNECTED' not in lines


def test_ordinary_cq_ignored_abort_does_not_report_owner_completion(endpoint):
    host, modem, _ = endpoint
    modem._cq_pending = True
    completed = []
    modem.abort_with_completion(lambda: completed.append(True))
    modem.poll()
    assert completed == []


def test_generic_modem_abort_contract_is_unchanged():
    calls = []
    host = HostSession(None, None, SimpleNamespace(abort=lambda: calls.append('abort')))
    host._send_cmd = calls.append
    host._dispatch('ABORT')
    assert calls == ['abort', 'OK']


def test_abort_queued_during_owner_frame_completes_after_unkey(endpoint, monkeypatch):
    host, modem, lines = endpoint
    order = []
    original = modem.abort_with_completion
    def traced(completed=None):
        def after():
            assert lines[-1] == 'PTT OFF'
            order.append('abort consumed after unkey')
            completed()
        original(after)
    monkeypatch.setattr(modem, 'abort_with_completion', traced)
    def frame(station, kind):
        station._key(True)
        try:
            host._dispatch('ABORT')
            assert host.state == SessionState.CONNECTING
            order.append('frame still keyed')
        finally:
            station._key(False)
        return True
    monkeypatch.setattr(A.VaraStationHandshake, '_send_burst', frame)
    host._dispatch('CONNECT W9SSJ K0EPI')
    modem.poll()
    assert order == ['frame still keyed', 'abort consumed after unkey']
    assert host.state == SessionState.DISCONNECTED


def test_owner_completion_does_not_wait_for_command_operation_lock(endpoint):
    import threading
    host, modem, _ = endpoint
    host._dispatch('CONNECT W9SSJ K0EPI')
    modem.poll()
    host._dispatch('ABORT')
    done = threading.Event()
    def consume():
        modem.poll()
        done.set()
    # A command may hold this lock while waiting for the modem owner lock.
    with host._operation_lock:
        owner = threading.Thread(target=consume)
        owner.start()
        completed = done.wait(1)
    owner.join(1)
    assert completed, 'abort callback introduced an operation-lock/modem-lock inversion'
    assert not owner.is_alive()
