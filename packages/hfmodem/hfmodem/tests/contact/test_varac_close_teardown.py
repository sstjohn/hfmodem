# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A failed stepped /bye exits the real terminal and stops the local runtime."""
from contextlib import ExitStack, contextmanager
import io
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.tests.contact.test_varac import Host as FakeHost
from hfmodem.tests.contact.test_varac_launcher import arguments, launch as launch  # noqa: F401


def test_stalled_owner_cannot_block_server_cleanup_before_keyer_shutdown(launch):
    order = []
    def server_stop():
        order.append('server stop would wait for the stalled modem owner lock')
    host = launch.Host(SimpleNamespace(stop=lambda: order.append('audio stopped')),
                       runtime_factory=lambda transport: SimpleNamespace(modem=None))
    host.server = SimpleNamespace(stop=server_stop)
    host.thread = SimpleNamespace(ident=1, is_alive=lambda: True,
                                  join=lambda timeout: order.append(('join', timeout)))
    with pytest.raises(launch.HostError, match='runtime did not stop'):
        with ExitStack() as held:
            held.callback(lambda: order.append('keyer retired'))
            held.callback(host.close)
    assert order == [('join', 10), 'audio stopped', 'keyer retired']


@pytest.mark.parametrize('stage', ['terminal', 'audio', 'host'])
def test_startup_failure_always_retires_keyer(launch, monkeypatch, tmp_path, stage):
    import kestrel_connect
    from hfmodem.contact import slots
    from hfmodem.core import config

    order = []
    def fail(*args, **kwargs):
        raise RuntimeError('startup failed')
    monkeypatch.setattr(launch, 'Terminal', fail if stage == 'terminal' else lambda *a: io.StringIO())
    monkeypatch.setattr(config, 'tx_drive', lambda: .4)
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', fail if stage == 'audio' else
                        lambda *a, **k: SimpleNamespace(taps=[]))
    monkeypatch.setattr(slots, 'SlotWatch', lambda *a, **k: SimpleNamespace(
        close=lambda: order.append('slots stopped')))
    monkeypatch.setattr(launch, 'Host', fail)
    rig = SimpleNamespace(shutdown=lambda: order.append('keyer retired'))
    with pytest.raises(RuntimeError, match='startup failed'):
        launch.run(arguments(launch, tmp_path), rig, SimpleNamespace(opened=lambda: None))
    assert order[-1:] == ['keyer retired']


@pytest.mark.parametrize('operation', ['bye', 'quit-connect', 'quit-retune'])
def test_failed_close_runs_terminal_finally_and_runtime_teardown_before_keyer(
        launch, monkeypatch, tmp_path, operation):
    import kestrel_connect
    from hfmodem.contact import slots
    from hfmodem.core import config
    from hfmodem.contact.varac import frame_message
    from hfmodem.kestrel.host.audio_runtime import RetuneRequest

    order = []
    fake = FakeHost(replies={'VERSION': ['VERSION VARA HF v4.9.0'],
                             'CONNECT W9SSJ KE8PLM': ['OK', 'PENDING', 'PTT ON'],
                             'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    fake.data.respond = lambda data: fake.lines('BUFFER 10', 'PTT ON')
    class Tty(io.StringIO):
        reads = 0
        def readline(self):
            self.reads += 1
            if self.reads == 1:
                if operation == 'quit-connect':
                    return '/call KE8PLM'
                fake.lines('CONNECTED W9SSJ KE8PLM 500')
            elif self.reads == 2:
                if operation == 'quit-connect':
                    self.quit_at = fake.now
                    return '/quit'
                if operation == 'quit-retune':
                    fake.data.incoming.append(frame_message('<Q:75>'))
                    return None
                return '/bye'
            elif self.reads == 3 and operation == 'quit-retune':
                self.quit_at = fake.now
                self.quit_wall = time.monotonic()
                return '/quit'
            return None
    tty = Tty()
    monkeypatch.setattr(launch, 'Terminal', lambda held, history: tty)
    monkeypatch.setattr(config, 'tx_drive', lambda: .4)
    monkeypatch.setattr(slots, 'SlotWatch', lambda *a, **k: SimpleNamespace(
        put=lambda frame: None, poll=lambda: [], table=lambda: ([], 0), close=lambda: order.append('slots stopped')))
    def audio_stop():
        order.append('audio stopped')
        fake.client.ptt = False
    audio = SimpleNamespace(taps=[], start=lambda: None, stop=audio_stop)
    monkeypatch.setattr(kestrel_connect, 'AudioVaraIO', lambda *a, **k: audio)
    modem = SimpleNamespace(station=object(), received_cursor=10,
                            stop=lambda: order.append('modem stopped'),
                            poll=lambda: order.append('modem polled'))
    releases = []
    def request_retune(callback, **kwargs):
        request = RetuneRequest(callback, **kwargs)
        request._lock.acquire()
        release = threading.Timer(1, request._lock.release)
        releases.append(release)
        release.start()
        return request
    runtime = SimpleNamespace(modem=modem, modem_factory=lambda: modem,
                              step=lambda: time.sleep(.001), request_retune=request_retune)
    server = SimpleNamespace(cmd_port=100, data_port=101, start_background=lambda: None,
                             stop=lambda: order.append('server stopped'))
    real_host = launch.Host
    host = real_host(audio, runtime_factory=lambda transport: runtime,
                     server_factory=lambda **kwargs: server)
    monkeypatch.setattr(launch, 'Host', lambda transport: host)
    @contextmanager
    def open_client(**kwargs):
        fake.client.event = kwargs['event']
        try:
            yield fake.client
        finally:
            order.append('client closed')
            fake.client.close()
    monkeypatch.setattr(launch.Client, 'open', open_client)
    def shutdown():
        assert not host.thread.is_alive()
        assert not fake.client.ptt
        order.append('keyer retired')
    rig = SimpleNamespace(retired=False, keyed=False, shutdown=shutdown, armed=False, power_w=lambda w=None: None,
                          line_state=lambda: fake.client.ptt, meter_reports=queue.Queue())
    if operation == 'quit-retune':
        fake.replies['ABORT'] = ['OK', 'PTT OFF']
    try:
        with pytest.raises(TimeoutError, match='retune' if operation == 'quit-retune' else 'confirm ABORT'):
            launch.run(arguments(launch, tmp_path), rig, SimpleNamespace(opened=lambda: None))
        if operation == 'quit-retune':
            assert time.monotonic() - tty.quit_wall < .5
    finally:
        for release in releases:
            release.join()
    assert fake.sent.count('ABORT') == 1
    assert fake.sent.count('DISCONNECT') == 1
    if operation == 'quit-retune':
        assert fake.now - tty.quit_at < 1
    elif operation == 'quit-connect':
        assert 5 <= fake.now - tty.quit_at < 5.2
    else:
        assert 35 <= fake.now < 37, 'terminal finally must not run a second close deadline'
    assert order == ['client closed', 'server stopped', 'modem stopped', 'modem polled',
                     'audio stopped', 'slots stopped', 'keyer retired']


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_terminal_preserves_initial_fault_and_still_attempts_close(launch, cleanup_fails):
    original = ValueError('invalid incoming application length')
    secondary = RuntimeError('host connection closed during cleanup')
    calls = []
    def tick():
        raise original
    def quit():
        calls.append('close attempted')
        if cleanup_fails:
            raise secondary
    session = SimpleNamespace(tick=tick, quit=quit)
    with pytest.raises(ValueError) as caught:
        launch.terminal(session, io.StringIO())
    assert caught.value is original
    assert calls == ['close attempted']
    if cleanup_fails:
        assert caught.value.__cause__ is secondary


def test_terminal_without_initial_fault_propagates_close_failure(launch):
    failure = TimeoutError('cannot confirm unkey')
    def quit():
        raise failure
    session = SimpleNamespace(tick=lambda: None, quit=quit)
    with pytest.raises(TimeoutError) as caught:
        launch.terminal(session, io.StringIO())
    assert caught.value is failure
