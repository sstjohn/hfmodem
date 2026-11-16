# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
import time

import pytest

from hfmodem.contact import chat_state as s
from hfmodem.contact.chat_fake import FakeClock
from hfmodem.contact.chat_controller import Disposition
from hfmodem.station.chat import Chat, ControlFence
from hfmodem.station.service import Service
from hfmodem.station.sessions import Registry, Refused
from hfmodem.tests.contact.test_chat_controller import Port


def until(predicate):
    end = time.monotonic() + 3
    while time.monotonic() < end:
        if predicate():
            return
        Event().wait(.005)
    assert predicate()


class FakeAttachment:
    def __init__(self, chat):
        self.chat = chat
        self._closed = False
        self.port = Port()
        client, lease, generation = chat.fence.identity
        self.port.state = replace(self.port.state, ownership=s.Ownership(
            'station', client, lease, generation, None, 'ready', True))
        self.port.quit = self.close
        chat.created.append(self)

    def open(self):
        self.chat.entered.set()
        assert self.chat.start_barrier.wait(3)
        try:
            self.chat.fence.check()
            if self.chat.start_error:
                raise self.chat.start_error
        except BaseException:
            self.close()
            raise
        return self.port

    def close(self, *, unwinding=None):
        self.port.quit_count += 1
        self._closed = True

    def release(self):
        assert self._closed


@pytest.fixture
def service(tmp_path):
    clock = FakeClock(10)
    registry = Registry(clock=clock)
    registry.admit()
    station = SimpleNamespace(sessions=registry, _stop=Event(), transitioning=False,
                              arbiter=SimpleNamespace(stop=lambda *args: None))
    station.stop_transmit = lambda: result.chat.revoke()
    result = Service(tmp_path / 'station.toml', station=station, port=8737)
    result.chat = Chat(result, station, attachment_factory=FakeAttachment)
    result.chat.created = []
    result.chat.entered, result.chat.start_barrier = Event(), Event()
    result.chat.start_barrier.set()
    result.chat.start_error = None
    result.clock = clock
    result.control = registry.control('operator')
    worker = Thread(target=result._dispatch)
    worker.start()
    yield result
    result.chat.start_barrier.set()
    result.close()
    worker.join(3)
    assert not worker.is_alive()


def start(service, id='start', **args):
    control = service.control
    return dict(id=id, kind='chat.start', client=control['client'],
                control_id=control['control_id'], generation=control['generation'], args=dict(
                    expected_incarnation=service.incarnation,
                    expected_lifecycle_generation=service.chat.generation,
                    expected_controller_id=None, **args))


def terminal(service, id):
    def done():
        result = service.results.lookup(service.incarnation, 'operator', id)
        return result if result['status'] == 'found' else None
    until(done)
    return done()['result']


def operation(service, action, id='op', actor='first', *, observer=False):
    controller = service.chat.controller
    request = s.Request(id, controller.controller_id, controller.snapshot().link.generation,
                        actor, None if observer else service.control['control_id'],
                        None if observer else service.control['generation'], action)
    return dict(id=id, kind=action.kind, client='observer' if observer else 'operator',
                control_id=request.control_id, generation=request.control_generation,
                args=s.to_wire(request))


def test_start_dispatch_is_nonblocking_and_stop_has_priority(service):
    service.chat.start_barrier.clear()
    service.submit(start(service))
    assert service.chat.entered.wait(2)
    assert service.results.lookup(service.incarnation, 'operator', 'start')['status'] == 'expired'
    service.submit(dict(id='stop', client='any-view', kind='stop'))
    assert not service.chat.fence()
    assert service.chat.state == 'starting'
    service.chat.start_barrier.set()
    result = terminal(service, 'start')
    assert result['state'] == 'failed' and result['reason'] == 'chat-start-cancelled'
    assert service.chat.created[0].port.quit_count == 1
    assert service.chat.state == 'idle'


@pytest.mark.parametrize('failure', [None, RuntimeError('startup failed'), Refused('bbs-config')])
def test_start_replay_after_exit_or_failure_and_both_cache_evictions(service, failure):
    service.chat.start_error = failure
    original = start(service)
    service.submit(original)
    result = terminal(service, 'start')
    if failure is None:
        assert result['state'] == 'done'
        service.chat.close()
    else:
        assert result['state'] == ('refused' if isinstance(failure, Refused) else 'failed')
    assert service.chat.state == 'idle'
    registry = service.station.sessions
    registry.heartbeat(dict(kind='heartbeat', **service.control))
    for n in range(1001):
        service._remember(f'evict-{n}', {'id': str(n)})
        service.results.remember(service.incarnation, 'other', str(n), {'state': 'done'})
    assert service.results.lookup(service.incarnation, 'operator', 'start')['status'] == 'expired'
    service.submit(original)
    assert len(service.chat.created) == 1
    assert service.chat.state == 'idle' and service.chat.generation == 1
    events = service.events.after(0)
    assert any(e['data'].get('reason') == 'chat-lifecycle-stale' for e in events)
    assert service.results.lookup(service.incarnation, 'operator', 'start')['status'] == 'expired'
    service.chat.start_error = None
    service.submit(start(service, 'fresh'))
    assert terminal(service, 'fresh')['state'] == 'done'
    assert len(service.chat.created) == 2


@pytest.mark.parametrize('factory', ['attachment_factory', 'controller_factory'])
def test_failed_start_before_controller_creation_stays_consumed_after_eviction(service, factory):
    original_factory = getattr(service.chat, factory)
    calls = []
    def fail(*args):
        calls.append(factory)
        raise RuntimeError('before controller creation')
    setattr(service.chat, factory, fail)
    original = start(service)
    service.submit(original)
    assert terminal(service, 'start')['state'] == 'failed'
    assert service.chat.state == 'idle'
    assert all(item._closed for item in service.chat.created)
    for n in range(1001):
        service._remember(str(n), {})
        service.results.remember(service.incarnation, 'other', str(n), {'state': 'done'})
    service.station.sessions.heartbeat(dict(kind='heartbeat', **service.control))
    service.submit(original)
    assert calls == [factory]
    assert service.chat.generation == 1 and service.chat.state == 'idle'
    assert service.results.lookup(service.incarnation, 'operator', 'start')['status'] == 'expired'
    setattr(service.chat, factory, original_factory)
    service.submit(start(service, 'fresh'))
    assert terminal(service, 'fresh')['state'] == 'done'


def test_concurrent_starts_consume_one_generation_and_stale_incarnation_refuses(service):
    first, second = start(service), start(service, 'second')
    service.submit(first)
    ack = service.submit(second)
    assert service.events.after(ack['seq'] - 1)[0]['data']['reason'] == 'chat-lifecycle-stale'
    assert terminal(service, 'start')['state'] == 'done'
    wrong = start(service, 'wrong')
    wrong['args']['expected_incarnation'] = 'previous-service'
    ack = service.submit(wrong)
    assert service.events.after(ack['seq'] - 1)[0]['data']['reason'] == 'chat-lifecycle-stale'
    assert len(service.chat.created) == 1


def test_two_views_one_owner_observer_projection_check_and_no_recovery_effect(service):
    service.submit(start(service))
    assert terminal(service, 'start')['state'] == 'done'
    controller = service.chat.controller
    attach = start(service, 'attach')
    attach['args']['expected_controller_id'] = controller.controller_id
    service.submit(attach)
    assert terminal(service, 'attach')['result']['controller_id'] == controller.controller_id
    assert len(service.chat.created) == 1 and service.chat.generation == 2
    port = service.chat.created[0].port
    before = tuple(port.calls)
    snapshot = service.chat.read('/chat/snapshot', {})
    projection = service.chat.read('/chat/projection', {'controller_id': [controller.controller_id]})
    assert snapshot['through_seq'] <= projection['through_seq']
    assert 'records' not in str(projection)
    service.submit(operation(service, s.ChatCheck(s.VerboseSet(True)), observer=True))
    until(lambda: isinstance(controller.outcome('op'), s.RequestResult))
    assert controller.outcome('op').state.state == 'done'
    assert not any(call == 'verbose.set' for call in port.calls[len(before):])
    with pytest.raises(Refused, match='controller-stale'):
        service.chat.read('/chat/projection', {'controller_id': ['wrong']})


@pytest.mark.parametrize('active', ['chat.qsy', 'broadcast.send', 'vmail.compose', 'beacon.once'])
@pytest.mark.parametrize('takeover', [False, True])
def test_expiry_and_heartbeating_takeover_retire_active_chat(service, active, takeover):
    service.submit(start(service))
    terminal(service, 'start')
    port = service.chat.created[0].port
    actions = {'chat.qsy': s.ChatQSY(2), 'broadcast.send': s.BroadcastSend('hello'),
               'vmail.compose': s.VMailCompose('subject'), 'beacon.once': s.BeaconOnce(False)}
    def execute(request):
        port.note(request.action.kind)
        return Disposition('running', stage='waiting')
    port.execute = execute
    controller = service.chat.controller
    service.submit(operation(service, actions[active]))
    until(lambda: isinstance(controller.outcome('op'), s.RequestResult)
          and controller.outcome('op').state.state == 'running')
    old_fence = service.chat.fence
    if takeover:
        replacement = service.station.sessions.control('replacement')
        for _ in range(3):
            service.clock.advance(5)
            service.station.sessions.heartbeat(dict(kind='heartbeat', **replacement))
    else:
        service.clock.advance(31)
        service.station.sessions.expire()
    assert not old_fence()
    until(lambda: service.chat.state == 'idle')
    assert port.quit_count == 1
    assert port.calls.count(active) == 1
    assert controller.outcome('op').state.state in ('cancelled', 'unknown')


def test_fence_never_revives_after_replacement_even_with_same_identity():
    clock = FakeClock(10)
    command = dict(client='a', control_id='one', generation=1)
    fence = ControlFence(command, clock=clock)
    fence.update(command, 40)
    assert fence()
    fence.update(dict(command, control_id='two'), 60)
    fence.update(command, 90)
    assert not fence()


def test_close_before_dispatch_consumes_start_and_retires_watch(tmp_path):
    registry = Registry(clock=FakeClock(10))
    registry.admit()
    station = SimpleNamespace(sessions=registry, _stop=Event(), transitioning=False)
    service = Service(tmp_path / 'station.toml', station=station, port=8737)
    service.control = registry.control('operator')
    created = []
    service.chat.attachment_factory = lambda chat: created.append(chat)
    original = start(service)
    service.submit(original)
    assert service.chat.state == 'starting'
    service.chat.close()
    assert service.chat.state == 'idle'
    assert service.chat.generation == 1 and not created
    result = service.results.lookup(service.incarnation, 'operator', 'start')['result']
    assert result['state'] == 'failed' and result['reason'] == 'chat-start-cancelled'
    with pytest.raises(Refused, match='chat-lifecycle-stale'):
        service.chat.dispatch(original, lambda *args, **kwargs: None)
    for n in range(1001):
        service._remember(str(n), {})
        service.results.remember(service.incarnation, 'other', str(n), {'state': 'done'})
    registry.heartbeat(dict(kind='heartbeat', **service.control))
    service.submit(original)
    assert service.chat.state == 'idle' and not created
    assert service.results.lookup(service.incarnation, 'operator', 'start')['status'] == 'expired'
    service.close()


def test_cleanup_does_not_hold_lifecycle_lock_while_waiting_for_registry(service, monkeypatch):
    service.submit(start(service))
    assert terminal(service, 'start')['state'] == 'done'
    entered = Event()
    unwatch = service.station.sessions.unwatch_control

    def observe(callback):
        entered.set()
        unwatch(callback)

    monkeypatch.setattr(service.station.sessions, 'unwatch_control', observe)
    done, snapshots = Event(), []
    closer = Thread(target=lambda: (service.chat.close(), done.set()))
    reader = Thread(target=lambda: snapshots.append(service.chat.lifecycle()))
    with service.station.sessions.lock:
        closer.start()
        assert entered.wait(2)
        reader.start()
        reader.join(2)
        assert not reader.is_alive()
        assert snapshots[0]['state'] == 'closing'
        assert not done.is_set()
    closer.join(2)
    assert done.is_set() and not closer.is_alive()


@pytest.mark.parametrize('lease_loss', (False, True))
def test_operator_exit_retains_fence_until_owner_quit_but_lease_loss_revokes_first(service, lease_loss):
    service.submit(start(service))
    assert terminal(service, 'start')['state'] == 'done'
    item = service.chat.created[0]
    entered, release = Event(), Event()
    live_at_quit = []
    def close(**kwargs):
        live_at_quit.append(service.chat.fence())
        entered.set()
        assert release.wait(3)
        item._closed = True
    item.port.quit = close
    try:
        if lease_loss:
            service.station.sessions.control('replacement')
            service.chat.pump()
        else:
            service.submit(operation(service, s.ChatExit(), 'exit'))
        assert entered.wait(2)
        until(lambda: service.chat.state == 'closing')
        assert live_at_quit == [not lease_loss]
        assert service.chat.fence() is (not lease_loss)
    finally:
        release.set()
        until(lambda: service.chat.state == 'idle')
    assert not service.chat.fence()
