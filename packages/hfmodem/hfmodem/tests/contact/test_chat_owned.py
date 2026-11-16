# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Local runtime injection preserves line-launcher cleanup with fake owners."""
from contextlib import ExitStack
from threading import get_ident

import pytest

from hfmodem.contact.chat_session import OwnedSessionAdapter
from hfmodem.contact.chat_controller import Controller
from hfmodem.tests.contact.test_varac import Desk


def adapter(events, *, line=False, quit_error=None):
    desk = Desk()
    desk.line = line
    desk.session.quit = lambda **kwargs: events.append('quit')
    if quit_error is not None:
        def quit(*, unwinding=None):
            events.append('quit')
            raise quit_error
        desk.session.quit = quit
    cleanup = ExitStack()
    return OwnedSessionAdapter(desk.session, cleanup=cleanup), cleanup


def test_local_injected_exitstack_preserves_dependency_order_once():
    events = []
    port, cleanup = adapter(events)
    for name in ('bbs-service', 'rig-finalizer', 'catalog', 'terminal', 'streams',
                 'recording-saves', 'reporter', 'slots', 'host', 'closing-notice', 'client'):
        cleanup.callback(events.append, name)
    port.quit()
    port.quit()
    assert events == ['quit', 'client', 'closing-notice', 'host', 'slots', 'reporter',
                      'recording-saves', 'streams', 'terminal', 'catalog', 'rig-finalizer', 'bbs-service']


def test_local_factory_and_finalizers_run_on_the_controller_owner():
    calls = []

    def factory():
        desk = Desk()
        cleanup = ExitStack()
        cleanup.callback(lambda: calls.append(('client', get_ident())))
        desk.session.quit = lambda **kwargs: calls.append(('quit', get_ident()))
        calls.append(('factory', get_ident()))
        return OwnedSessionAdapter(desk.session, cleanup=cleanup)

    controller = Controller(factory)
    try:
        controller.ready.result(timeout=2)
    finally:
        controller.shutdown().result(timeout=2)
        assert controller.join(2)
    assert [kind for kind, _ in calls] == ['factory', 'quit', 'client']
    assert len({ident for _, ident in calls}) == 1
    assert calls[0][1] != get_ident()


@pytest.mark.parametrize('message', ['owner pipe EOF', 'Host failure'])
def test_local_runtime_failure_preserves_initiating_error_and_runs_finalizers(message):
    events = []
    original = EOFError(message)
    port, cleanup = adapter(events, quit_error=original)
    cleanup.callback(events.append, 'terminal')

    def host_close():
        events.append('host')
        raise RuntimeError('Host close failed')

    cleanup.callback(host_close)
    with pytest.raises(EOFError, match=message) as caught:
        port.quit()
    assert caught.value is original
    assert str(caught.value.__cause__) == 'Host close failed'
    assert events == ['quit', 'host', 'terminal']
    with pytest.raises(EOFError, match=message):
        port.quit()
    assert events == ['quit', 'host', 'terminal']


@pytest.mark.parametrize('line', [True, None])
def test_local_high_or_unknown_line_fails_after_resource_cleanup(line):
    events = []
    port, cleanup = adapter(events, line=line)
    cleanup.callback(events.append, 'terminal')
    with pytest.raises(RuntimeError, match='low-line proof'):
        port.quit()
    assert events == ['quit', 'terminal']


def test_local_failed_save_continues_terminal_and_outer_finalizers():
    events = []
    port, cleanup = adapter(events)
    cleanup.callback(events.append, 'bbs-service')
    cleanup.callback(events.append, 'terminal')

    def save():
        events.append('save')
        raise RuntimeError('recording save failed')

    cleanup.callback(save)
    with pytest.raises(RuntimeError, match='recording save failed'):
        port.quit()
    assert events == ['quit', 'save', 'terminal', 'bbs-service']


def test_final_projection_never_reads_a_closed_watch():
    class Watch:
        closed = False
        def table(self):
            if self.closed:
                raise ValueError('queue closed')
            return (), None
        def close(self):
            self.closed = True
    watch = Watch()
    def factory():
        desk = Desk()
        desk.session.slots = watch
        desk.session.tick = lambda *args: None
        cleanup = ExitStack()
        cleanup.callback(watch.close)
        return OwnedSessionAdapter(desk.session, cleanup=cleanup)
    controller = Controller(factory)
    controller.ready.result(2)
    controller.shutdown().result(2)
    assert controller.join(2) and watch.closed
    assert controller.snapshot().tx.physical_line is False
