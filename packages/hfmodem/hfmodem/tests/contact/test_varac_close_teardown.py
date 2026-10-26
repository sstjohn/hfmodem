# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A failed stepped /bye exits the real terminal and stops the local runtime."""
from contextlib import contextmanager
import io
import queue
import time
from types import SimpleNamespace

import pytest

from hfmodem.tests.contact.test_varac import Host as FakeHost
from hfmodem.tests.contact.test_varac_launcher import arguments, launch as launch  # noqa: F401


def test_failed_bye_runs_terminal_finally_and_runtime_teardown_before_keyer(
        launch, monkeypatch, tmp_path):
    import kestrel_connect
    from hfmodem.contact import slots
    from hfmodem.core import config

    order = []
    fake = FakeHost(replies={'VERSION': ['VERSION VARA HF v4.9.0'],
                             'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    fake.data.respond = lambda data: fake.lines('BUFFER 10', 'PTT ON')
    class Tty(io.StringIO):
        reads = 0
        def readline(self):
            self.reads += 1
            if self.reads == 1:
                fake.lines('CONNECTED W9SSJ KE8PLM 500')
            elif self.reads == 2:
                return '/bye'
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
    modem = SimpleNamespace(stop=lambda: order.append('modem stopped'),
                            poll=lambda: order.append('modem polled'))
    runtime = SimpleNamespace(modem=modem, modem_factory=lambda: modem,
                              step=lambda: time.sleep(.001))
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
    rig = SimpleNamespace(retired=False, keyed=False, shutdown=shutdown, armed=False,
                          line_state=lambda: fake.client.ptt, meter_reports=queue.Queue())
    with pytest.raises(TimeoutError, match='confirm ABORT'):
        launch.run(arguments(launch, tmp_path), rig, SimpleNamespace(opened=lambda: None))
    assert fake.sent.count('ABORT') == 1
    assert fake.sent.count('DISCONNECT') == 1
    assert fake.now < 37, 'terminal finally must not run a second close deadline'
    assert order == ['client closed', 'server stopped', 'modem stopped', 'modem polled',
                     'audio stopped', 'keyer retired', 'slots stopped']


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
