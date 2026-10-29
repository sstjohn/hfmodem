# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.contact import varac as v
from hfmodem.tests.contact.test_bbs_capture import offered
from hfmodem.tests.contact.test_varac import Desk, FRAME


def test_bye_disables_admission_before_graceful_disconnect():
    desk = Desk()
    desk.hear('CONNECTED W9SSJ K7QB 500')
    before = len(desk.sent)
    desk.session.command('/bye')
    desk.drive(until=lambda: 'DISCONNECT' in desk.sent[before:])
    commands = desk.sent[before:]
    assert commands.index('LISTEN OFF') < commands.index('DISCONNECT')
    desk.drive()


@pytest.mark.parametrize('operation', ['bye', 'stop'])
@pytest.mark.parametrize('arrival', ['PENDING', 'CONNECTED W9SSJ N0XYZ 500'])
@pytest.mark.parametrize('unkey', [False, True])
def test_closing_refuses_new_admission_without_waiting_for_farewell_timeout(operation, arrival, unkey):
    desk = Desk(**{'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    desk.hear('CONNECTED W9SSJ K7QB 500')
    desk.session.command('/' + operation)
    desk.drive(until=lambda: 'DISCONNECT' in desk.sent)
    before, start, texts = len(desk.sent), desk.host.now, desk.texts[:]
    desk.host.client._line('DISCONNECTED')
    desk.host.client._line(arrival)
    assert desk.sent[before:] == ['LISTEN OFF', 'ABORT']
    desk.host.client._line('PTT ON')
    desk.host.replies['ABORT'] = ['OK', 'PTT OFF'] if unkey else ['OK']
    if unkey:
        desk.host.lines('PTT OFF')
        desk.drive()
    else:
        with pytest.raises(TimeoutError, match='confirm ABORT'):
            desk.drive()
    assert desk.host.now - start < 10.5
    assert desk.texts == texts
    assert desk.session.peer == 'K7QB'


@pytest.mark.parametrize('operation', ['bye', 'stop'])
def test_operator_close_returns_to_answering_and_the_next_caller_is_answered(operation):
    desk = Desk()
    desk.hear('CONNECTED W9SSJ K7QB 500')
    before = len(desk.sent)
    desk.do('/' + operation)
    assert desk.session._auto_answer and desk.armed[-1]
    assert desk.sent[-1] == 'LISTEN CQ' and 'LISTEN CQ' in desk.sent[before:]
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer on'
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    assert desk.session.contact['peer'] == 'N0XYZ' and desk.host.client.connected
    assert desk.texts[-1] == '3 <Q>'


def test_auto_off_stays_off_across_a_close():
    desk = Desk(**{'CONNECT W9SSJ K7QB': ['OK', 'CONNECTED W9SSJ K7QB 500']})
    desk.do('/auto off')
    desk.do('/call K7QB 2')
    assert desk.host.client.connected
    desk.do('/bye')
    assert not desk.session._auto_answer and desk.sent[-1] == 'LISTEN OFF'
    assert desk.legacy_said[-1] == 'monitoring 14105.000 kHz at 500 Hz, auto-answer off (/auto on to answer)'


def test_quit_stops_before_bounded_retune_cancellation(tools):
    from hfmodem.kestrel.host.audio_runtime import RetuneRequest
    from varac_chat import received_retune_adapter

    desk = Desk()
    desk.hear('CONNECTED W9SSJ K7QB 500')
    requests = []
    def request(callback, **kwargs):
        item = RetuneRequest(callback, **kwargs)
        requests.append(item)
        return item
    runtime = SimpleNamespace(modem=SimpleNamespace(station=object(), received_cursor=10),
                              request_retune=request)
    desk.flow.received_retune_steps = received_retune_adapter(
        runtime, desk.host.client, desk.tunes.append, lambda: True,
        quitting=lambda: desk.session._quitting)
    desk.receive('<Q:75>')
    item, = requests
    item._lock.acquire()
    observed = []
    cancel = item.cancel
    def cancellation(*args, **kwargs):
        observed.extend(desk.sent[before:])
        return cancel(*args, **kwargs)
    item.cancel = cancellation
    release = threading.Timer(1, item._lock.release)
    release.start()
    before, start = len(desk.sent), time.monotonic()
    try:
        with pytest.raises(TimeoutError, match='retune'):
            desk.session.quit()
        assert time.monotonic() - start < .5
        assert observed == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
        assert desk.session.steps is None
        assert not desk.host.client.connected
        assert not item.valid()
    finally:
        release.join()


@pytest.mark.parametrize('unkey', [False, True])
@pytest.mark.parametrize('operation', ['cq', 'beacon', 'connect', 'connect-aborted', 'file', 'file-close'])
def test_quit_cancels_nested_wait_before_any_more_work(tmp_path, operation, unkey):
    desk = offered(tmp_path) if operation.startswith('file') else Desk()
    desk.host.replies.update({'DISCONNECT': ['OK'], 'ABORT': ['OK']})
    if operation == 'cq':
        desk.host.replies[FRAME] = ['OK', 'BUSY ON', 'BUSY OFF']
        desk.session.command('/cq 1')
        desk.drive(until=lambda: FRAME in desk.sent)
    elif operation == 'beacon':
        desk.session._run(desk.session._beaconing(once=True))
    elif operation.startswith('connect'):
        desk.host.replies['CONNECT W9SSJ K7QB'] = ['OK', 'PENDING', 'PTT ON']
        desk.session.command('/call K7QB')
        if operation == 'connect-aborted':
            desk.session.command('/stop')
            desk.drive(until=lambda: 'ABORT' in desk.sent)
    else:
        desk.session.command('/bbs capture')
        if operation == 'file-close':
            desk.session.files.stop()
    desk.host.client.ptt = True
    start, before, texts = desk.host.now, len(desk.sent), desk.texts[:]
    def abort():
        assert desk.host.now == start, 'ABORT must precede every wait'
        return ['OK', 'PTT OFF'] if unkey else ['OK']
    desk.host.replies['ABORT'] = abort
    ready = desk.host.ready
    def bounded(sockets, timeout):
        assert desk.host.now - start < 6, 'quit failed to bound its wait'
        return ready(sockets, timeout)
    desk.host.client.ready = bounded
    if unkey:
        desk.session.quit()
        assert desk.host.now - start < 1
    else:
        with pytest.raises(TimeoutError, match='confirm ABORT'):
            desk.session.quit()
        assert 5 <= desk.host.now - start < 5.2
    assert desk.sent[before:] == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    assert desk.texts == texts
    assert desk.session.steps is None
    assert not desk.session.files.active


@pytest.mark.parametrize('auto', [False, True])
@pytest.mark.parametrize('operation', ['file', 'chat'])
def test_stop_contact_rearms_answering_only_when_auto_is_on(tmp_path, auto, operation):
    desk = offered(tmp_path)
    desk.session.set_auto(auto)
    if operation == 'file':
        desk.session.command('/bbs capture')
    before, start = len(desk.sent), desk.host.now
    desk.session.command('/stop')
    assert desk.host.now == start
    assert desk.sent[before:] == ['LISTEN OFF', 'CLEANTXBUFFER', 'DISCONNECT', 'VERSION', 'ABORT']
    desk.drive()
    assert desk.flow.state is v.State.MONITORING
    assert desk.session.auto is auto
    assert desk.armed[-1] is auto
    assert ('LISTEN CQ' in desk.sent[before:]) is auto
    if not auto:
        texts = desk.texts[:]
        desk.hear('CONNECTED W9SSJ K7QB 500')
        desk.receive('<Q:375>')
        desk.drive()
        assert desk.texts == texts
        desk.session.set_auto(True)
    assert desk.sent[-1] == 'LISTEN CQ'
    assert desk.armed[-1] is True
