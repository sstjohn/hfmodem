# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from datetime import datetime, timedelta, timezone
import errno
import json
import threading

import pytest

from hfmodem.station import client, hosts, wire
from hfmodem.station.qsolog import SessionRecords
from hfmodem.station.service import Service
from hfmodem.station.sessions import Refused, Registry
from .test_process import fake_claim as fake_claim, station as station
from .test_service import socketpair_http as socketpair_http


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def wall(self):
        return datetime(2026, 10, 18, tzinfo=timezone.utc) + timedelta(seconds=self.now)


@pytest.fixture
def registry(tmp_path):
    clock = Clock()
    events = []
    registry = Registry(records=SessionRecords(tmp_path / 'sessions'), clock=clock,
                        wall=clock.wall, publish=lambda type, data: events.append((type, data)))
    registry.admit()
    registry.events = events
    return registry


def start(registry, **kw):
    return registry.start(**(dict(kind='monitor', lane='fake', attended=False,
                                  directory='capture', client='test') | kw))


def envelope(kind, id='one', client='test', control=None, **kw):
    command = dict(kind=kind, id=id, client=client, args={})
    if control:
        command.update({k: control[k] for k in ('control_id', 'generation')})
    return command | kw


def result(service, ack):
    return next(e['data'] for e in service.events.after(ack['seq'] - 1)
                if e['type'] == 'command' and e['data']['id'] == ack['id'])


def test_one_holder_and_nonholding_witness(registry):
    holder = start(registry, kind='flight')
    witness = start(registry, holds_rig=False)
    with pytest.raises(Refused) as error:
        start(registry)
    assert error.value.reason == wire.REFUSAL_CLAIMED
    assert holder.id in str(error.value) and 'test' in str(error.value)
    assert error.value.holder['id'] == holder.id
    state = registry.snapshot()
    assert state['owner']['id'] == holder.id
    assert state['session'] == state['owner']
    assert {s['id'] for s in state['sessions']} == {holder.id, witness.id}
    registry.end(holder.id, outcome='completed')
    assert registry.snapshot()['owner'] is None
    assert start(registry).id != holder.id


def test_atomic_holder_admission(registry):
    ready = threading.Barrier(3)
    sessions, refusals = [], []

    def enter():
        ready.wait(2)
        try:
            sessions.append(start(registry))
        except Refused as exc:
            refusals.append(exc)

    threads = [threading.Thread(target=enter) for _ in range(2)]
    for thread in threads:
        thread.start()
    ready.wait(2)
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert len(sessions) == len(refusals) == 1
    assert refusals[0].holder['id'] == sessions[0].id


def test_lifecycle_authority_records_and_copies(registry):
    registry.authority = lambda id: (id, 17)
    session = start(registry, attended=True, recording_origin_index=24000)
    assert session.authority == (session.id, 17)
    assert registry.attended(session.id)
    assert registry.running(session.id).state == 'running'
    with pytest.raises(ValueError, match='running'):
        registry.running(session.id)
    registry.suspend()
    assert not registry.attended(session.id)
    assert registry.snapshot()['owner']['state'] == 'stopping'
    with pytest.raises(Refused, match='not accepting'):
        start(registry, holds_rig=False)
    verdict = {'line': 'down'}
    ended = registry.end(session.id, outcome='stopped', verdict=verdict, hand_stopped=True)
    verdict['line'] = 'changed'
    ended.verdict['line'] = 'changed too'
    snapshot = registry.snapshot()
    assert snapshot['owner'] is None
    assert snapshot['sessions'][0]['verdict'] == {'line': 'down'}
    assert snapshot['sessions'][0]['recording_origin_index'] == 24000
    assert snapshot['sessions'][0]['returncode'] == 0
    assert snapshot['sessions'][0]['ended_utc'] == '2026-10-18T00:00:00Z'
    count = len(registry.events)
    registry.end(session.id, outcome='different')
    assert len(registry.events) == count
    records = [data for type, data in registry.events if type == 'session']
    assert [r['state'] for r in records] == ['starting', 'running', 'stopping', 'ended']
    assert registry.records.read() == [records[-1]]


@pytest.mark.parametrize('state', ['starting', 'running', 'stopping'])
def test_reconstruction_preserves_provenance_and_closes_only_unended(registry, state):
    registry.provenance = dict(holder_pid=123, incarnation='old', station_url='http://fake')
    old = start(registry, recording_origin_index=42)
    if state != 'starting':
        registry.running(old.id)
    if state == 'stopping':
        registry.suspend()
    registry.end(old.id, outcome='completed', verdict={'line': 'down'})
    complete = registry.records.read()[0]
    registry.admit()
    pending = start(registry)
    if state != 'starting':
        registry.running(pending.id)
    if state == 'stopping':
        registry.suspend()
    (registry.records.directory / 'partial.tmp').write_text('{')
    fresh = Registry(records=registry.records, wall=registry.wall)
    fresh.recover()
    records = {r['id']: r for r in fresh.snapshot()['sessions']}
    assert records[old.id] == complete
    recovered = records[pending.id]
    assert recovered['state'] == 'ended' and recovered['outcome'] == 'interrupted'
    assert recovered['returncode'] == 1 and recovered['hand_stopped'] is False
    assert recovered['verdict'] == {'line': 'unconfirmed', 'reason': 'station gone'}
    assert recovered['incarnation'] == 'old' and recovered['holder_pid'] == 123
    assert fresh.snapshot()['owner'] is None
    fresh.recover()
    again = Registry(records=registry.records)
    again.recover()
    assert again.snapshot() == fresh.snapshot()


def test_record_failure_blocks_admission_but_corruption_does_not_block_recovery(registry, monkeypatch):
    def fail(record):
        raise OSError('disk failed')
    monkeypatch.setattr(registry.records, 'write', fail)
    with pytest.raises(OSError, match='disk failed'):
        start(registry)
    assert registry.snapshot()['sessions'] == [] and not registry.events
    registry.records.directory.mkdir()
    (registry.records.directory / 'corrupt.json').write_text('{')
    registry.recover()
    assert (registry.records.directory / 'corrupt.json.corrupt').read_text() == '{'
    assert registry.events[-1][0] == 'warning'


def test_atomic_record_replacement_survives_failed_replace(registry, monkeypatch):
    from hfmodem.station import qsolog
    session = start(registry)
    before = registry.records.read()
    def fail(*args):
        raise OSError('replace failed')
    monkeypatch.setattr(qsolog.os, 'replace', fail)
    with pytest.raises(OSError, match='replace failed'):
        registry.running(session.id)
    assert registry.records.read() == before
    assert registry.snapshot()['owner']['state'] == 'starting'
    assert list(registry.records.directory.glob('*.tmp')) == []


def test_control_expiry_requires_a_fresh_acknowledged_heartbeat(registry):
    control = registry.control('test')
    assert control == dict(client='test', control_id=control['control_id'], generation=1,
                           since_utc='2026-10-18T00:00:00Z', expires_utc='2026-10-18T00:00:30Z')
    registry.clock.now = 29
    registry.authorize(envelope('own', control=control))
    assert registry.snapshot()['control']['expires_utc'] == control['expires_utc']
    renewed = registry.heartbeat(envelope('heartbeat', control=control))
    assert renewed['expires_utc'] == '2026-10-18T00:00:59Z'
    registry.clock.now = 58.999
    registry.authorize(envelope('ack', control=control))
    registry.clock.now = 59
    with pytest.raises(Refused) as error:
        registry.heartbeat(envelope('heartbeat', control=control))
    assert error.value.reason == wire.REFUSAL_NOT_CONTROLLER
    assert registry.snapshot()['control'] is None
    assert registry.events[-1] == ('owner', {'owner': None, 'control': None})
    assert registry.control('next')['generation'] == 2


@pytest.mark.parametrize('field,value', [('client', 'other'), ('control_id', 'wrong'),
                                        ('generation', 2), ('generation', True),
                                        ('generation', '1'), ('generation', None)])
def test_control_identity_is_checked(registry, field, value):
    control = registry.control('test')
    with pytest.raises(Refused) as error:
        registry.authorize(envelope('own', control=control) | {field: value})
    assert error.value.reason == wire.REFUSAL_NOT_CONTROLLER


@pytest.mark.parametrize('kind', ['own', 'release', 'fly', 'heartbeat', 'ack'])
def test_service_refuses_unauthorized_actions(station, monkeypatch, kind):
    service = Service(station.station_file, station=station, port=0)
    station.sessions.control('other')
    ack = service.submit(envelope(kind))
    assert result(service, ack)['reason'] == wire.REFUSAL_NOT_CONTROLLER
    assert service.commands.empty()


def test_control_reply_takeover_and_old_cached_acknowledgements(station, monkeypatch):
    service = Service(station.station_file, station=station, port=0)
    first = service.submit(envelope('control', 'control-one'))
    assert first == service.submit(envelope('control', 'control-one'))
    assert service.snapshot()['control']['control_id'] == first['control_id']
    old_ack = envelope('ack', 'old-ack', control=first)
    service.submit(old_ack)
    next_control = service.submit(envelope('control', 'control-two', client='next'))
    assert next_control['generation'] == first['generation'] + 1
    for command in (envelope('control', 'control-one'), old_ack):
        ack = service.submit(command)
        assert result(service, ack)['reason'] == wire.REFUSAL_SUPERSEDED
    stopped = threading.Event()
    monkeypatch.setattr(station, 'stop_transmit', stopped.set)
    service.submit(envelope('stop', 'old-stop', control=first))
    assert stopped.is_set()
    assert service.snapshot()['control']['client'] == 'next'
    assert service.commands.empty()


def test_duplicate_heartbeat_does_not_extend_expiry(station):
    service = Service(station.station_file, station=station, port=0)
    clock = Clock()
    station.sessions.clock, station.sessions.wall = clock, clock.wall
    control = service.submit(envelope('control'))
    heartbeat = envelope('heartbeat', 'heartbeat', control=control)
    clock.now = 10
    ack = service.submit(heartbeat)
    clock.now = 39
    assert service.submit(heartbeat) == ack
    assert service.snapshot()['control']['expires_utc'] == '2026-10-18T00:00:40Z'
    clock.now = 40
    assert result(service, service.submit(heartbeat))['reason'] == wire.REFUSAL_NOT_CONTROLLER


def test_control_and_heartbeat_bypass_queue_and_transition(station):
    service = Service(station.station_file, station=station, port=0)
    first = service.submit(envelope('control', 'control'))
    for n in range(64):
        service.submit(envelope('release', str(n), control=first))
    with station._lifecycle:
        control = service.submit(envelope('control', 'takeover'))
        ack = service.submit(envelope('heartbeat', 'heartbeat', control=control))
    assert control['generation'] == first['generation'] + 1
    assert result(service, ack)['state'] == 'accepted'
    assert service.commands.qsize() == 64


def test_worker_expires_control_without_a_reader(station, monkeypatch):
    import queue
    service = Service(station.station_file, station=station, port=0)
    clock = Clock()
    station.sessions.clock = clock
    service.submit(envelope('control'))
    clock.now = 30
    def finish(**kwargs):
        service._closed.set()
        raise queue.Empty
    monkeypatch.setattr(service.commands, 'get', finish)
    service._dispatch()
    owners = [e['data'] for _, e in service.events.ring if e['type'] == 'owner']
    assert owners[-1]['control'] is None


@pytest.mark.parametrize('change', ['takeover', 'expiry'])
def test_queued_commands_recheck_authority(station, change, monkeypatch):
    service = Service(station.station_file, station=station, port=0)
    clock = Clock()
    station.sessions.clock = clock
    control = service.submit(envelope('control', 'control'))
    service.submit(envelope('own', control=control))
    acted = threading.Event()
    monkeypatch.setattr(station, 'own', lambda **kw: acted.set() or {})
    if change == 'takeover':
        service.submit(envelope('control', 'new-control', client='next'))
    else:
        clock.now = 30
    worker = threading.Thread(target=service._dispatch)
    worker.start()
    def terminal():
        return next((e['data'] for _, e in service.events.ring
                     if e['type'] == 'command' and e['data']['id'] == 'one'
                     and e['data']['state'] in ('refused', 'done')), None)
    try:
        with service.events.condition:
            assert service.events.condition.wait_for(terminal, timeout=2)
            if change == 'takeover':
                assert terminal()['reason'] == wire.REFUSAL_SUPERSEDED
                assert not acted.is_set()
            else:
                assert terminal()['state'] == 'done'
                assert acted.is_set()
    finally:
        service._closed.set()
        worker.join(2)
    assert not worker.is_alive() and station.calls == []


def test_fly_names_holder_and_state_events_agree(station):
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    station.own()
    control = service.submit(envelope('control', 'control'))
    holder = start(station.sessions, kind='flight')
    station.sessions.running(holder.id)
    ack = service.submit(envelope('fly', control=control))
    refusal = result(service, ack)
    state = service.snapshot()
    assert refusal['reason'] == wire.REFUSAL_CLAIMED
    assert refusal['claim_holder'] == refusal['detail']
    assert holder.id in refusal['claim_holder']
    assert holder.id in refusal['detail']
    records = [e['data'] for e in service.events.after(0) if e['type'] == 'session']
    assert records[-1] == state['session']
    assert state['owner']['incarnation'] == service.incarnation
    assert state['owner']['holder_pid'] == state['service']['pid']
    station.release()


def test_release_stops_sessions_before_hosts_and_reown_admits(station, monkeypatch):
    station.own()
    holder = start(station.sessions)
    station.sessions.running(holder.id)
    witness = start(station.sessions, holds_rig=False)
    stopped = threading.Event()
    original = station.stop_transmit
    def stop():
        original()
        stopped.set()
    def stop_hosts(*args):
        assert stopped.is_set()
        assert all(s['state'] == 'stopping' for s in station.snapshot()['sessions']
                   if s['state'] != 'ended')
        with pytest.raises(Refused):
            start(station.sessions)
        return True
    monkeypatch.setattr(station, 'stop_transmit', stop)
    monkeypatch.setattr(hosts, 'stop_all', stop_hosts)
    station.release()
    state = station.snapshot()
    assert state['owner'] is None
    assert {s['id'] for s in state['sessions']} == {holder.id, witness.id}
    assert all(s['state'] == 'ended' and s['outcome'] == 'stopped' for s in state['sessions'])
    assert all(s['verdict']['line'] == 'down' for s in state['sessions'])
    with pytest.raises(Refused):
        start(station.sessions)
    station.own()
    assert start(station.sessions).state == 'starting'
    station.release()


def test_transition_refuses_sessions_and_arbiter_supplies_authority(station):
    from types import SimpleNamespace
    station.own()
    with station._lifecycle, pytest.raises(Refused):
        start(station.sessions)
    station.arbiter = SimpleNamespace(authority=lambda id: (id, 9))
    try:
        holder = start(station.sessions)
        assert holder.authority == (holder.id, 9)
    finally:
        station.arbiter = None
        station.release()


def test_unconfirmed_release_ends_session_without_claiming_success(station, monkeypatch):
    station.own()
    holder = start(station.sessions)
    line = station.rig.ptt
    monkeypatch.setattr(line, 'release', lambda: None)
    from hfmodem.core.rig import RigError
    with pytest.raises(RigError, match='unconfirmed'):
        station.release()
    record = next(s for s in station.snapshot()['sessions'] if s['id'] == holder.id)
    assert record['state'] == 'ended' and record['outcome'] == 'unconfirmed'
    assert record['returncode'] == 1 and record['verdict']['line'] == 'unconfirmed'
    assert station.claim is not None and not station.sessions.accepting
    monkeypatch.setattr(line, 'release', lambda: setattr(line, 'released_low', True))
    station.release()


def test_failed_teardown_leaves_record_for_recovery(station, monkeypatch):
    from hfmodem.core.operations import LeaseTimeout
    station.own()
    holder = start(station.sessions)
    monkeypatch.setattr(hosts, 'stop_all', lambda *args: False)
    with pytest.raises(LeaseTimeout):
        station.release()
    assert station.snapshot()['owner']['id'] == holder.id
    assert station.snapshot()['owner']['state'] == 'stopping'
    assert not station.sessions.accepting
    fresh = Registry(records=station.sessions.records)
    fresh.recover()
    assert fresh.snapshot()['sessions'][0]['outcome'] == 'interrupted'
    monkeypatch.setattr(hosts, 'stop_all', lambda *args: True)
    station.release()


def test_automatic_session_survives_control_loss(registry):
    automatic = start(registry)
    control = registry.control('test')
    registry.clock.now = 30
    registry.expire()
    assert registry.snapshot()['owner']['id'] == automatic.id
    assert registry.snapshot()['owner']['state'] == 'starting'
    registry.control('next')
    with pytest.raises(Refused):
        registry.heartbeat(envelope('ack', control=control))
    assert registry.snapshot()['owner']['id'] == automatic.id


@pytest.mark.parametrize('late_station', [False, True])
def test_http_records_and_reconstruction_after_bind(station, socketpair_http, late_station):
    station.own()
    old = start(station.sessions)
    records = station.sessions.records
    station.sessions = Registry(records=records)
    service = Service(station.station_file, station=None if late_station else station)
    station.service = service
    service.bind()
    if late_station:
        service.station = station
    try:
        attached = client.connect(station.station_file)
        state = attached.state()
        assert state['owner'] is None
        record = next(s for s in state['sessions'] if s['id'] == old.id)
        assert record['state'] == 'ended' and record['outcome'] == 'interrupted'
        event = next(e for e in service.events.after(0) if e['type'] == 'session')
        stream = attached.events(event['seq'] - 1, duration=2)
        try:
            assert next(stream)['data'] == record
        finally:
            stream.close()
        control = attached.command('control', wait=True)['result']
        assert attached.state()['control'] == control
        with pytest.raises(client.CommandFailed) as error:
            attached.release()
        assert error.value.result['reason'] == wire.REFUSAL_NOT_CONTROLLER
    finally:
        service.close()
        station.release()


def test_failed_bind_does_not_reconstruct_another_stations_records(station, monkeypatch):
    from hfmodem.station import service as module
    def refuse(*args):
        raise OSError(errno.EPERM, 'fake bind failure')
    monkeypatch.setattr(module, 'ThreadingHTTPServer', refuse)
    monkeypatch.setattr(station.sessions, 'recover', lambda: pytest.fail('recovered before bind'))
    with pytest.raises(OSError):
        Service(station.station_file, station=station).bind()


def test_session_records_reject_path_traversal(tmp_path):
    records = SessionRecords(tmp_path / 'sessions')
    with pytest.raises(ValueError, match='session id'):
        records.write({'id': '../escape'})
    assert not list(tmp_path.iterdir())


def test_record_round_trip_is_json(registry):
    session = start(registry)
    path = registry.records.directory / f'{session.id}.json'
    assert json.loads(path.read_text()) == registry.snapshot()['owner']


@pytest.mark.parametrize('control_state', ['none', 'current', 'expired', 'superseded'])
@pytest.mark.parametrize('kind', ['stop', 'shutdown'])
def test_stop_and_shutdown_always_act(station, monkeypatch, kind, control_state):
    service = Service(station.station_file, station=station, port=0)
    clock = Clock()
    station.sessions.clock = clock
    control = None
    if control_state != 'none':
        control = station.sessions.control('test')
    if control_state == 'expired':
        clock.now = 30
    elif control_state in ('current', 'superseded'):
        station.sessions.control('other')
    acted = threading.Event()
    monkeypatch.setattr(station, 'stop_transmit', acted.set)
    monkeypatch.setattr(station, 'release', lambda: acted.set() or {'state': 'observing'})
    command = envelope(kind, control=control if control_state != 'current' else None)
    ack = service.submit(command)
    assert result(service, ack)['state'] == 'accepted'
    if kind == 'shutdown':
        # Take over again between submit and dispatch: shutdown still acts.
        station.sessions.control('newer')
        dispatch = threading.Thread(target=service._dispatch, daemon=True)
        dispatch.start()
        dispatch.join(5)
        if dispatch.is_alive():
            service._closed.set()
        assert station._stop.is_set()
    assert acted.is_set()


@pytest.mark.parametrize('lapsed', [False, True])
def test_console_own_and_legacy_handoff_over_http(station, socketpair_http, tools, lapsed):
    from consolelib import claim as console_claim
    service = Service(station.station_file, station=station)
    station.service = service
    clock = Clock()
    station.sessions.clock = clock
    service.bind()
    try:
        if lapsed:
            station.sessions.control('old-desk')
            clock.now = 30
        assert console_claim.own(station.station_file)['result']['state'] == 'owning'
        start(station.sessions)
        confirmations = []
        console_claim.release(station.station_file,
                              confirm=lambda state: confirmations.append(state) or True)
        assert len(confirmations) == 1
        assert station.state == 'observing' and station.claim is None
        assert client.connect(station.station_file).state()['control'] is None
    finally:
        service.close()
        station.release()


@pytest.mark.parametrize('failing_state', ['stopping', 'ended', 'both'])
def test_release_record_failure_warns_and_completes(station, monkeypatch, failing_state):
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    station.own()
    start(station.sessions)
    original = station.sessions.records.write
    def write(record):
        assert 'rig-close' in station.calls
        if failing_state in (record['state'], 'both'):
            raise OSError('disk full')
        original(record)
    monkeypatch.setattr(station.sessions.records, 'write', write)
    reply = station.release()
    assert reply['state'] == 'observing'
    assert reply['warnings'] and all('disk full' in w for w in reply['warnings'])
    warnings = [e['data']['detail'] for e in service.events.after(0) if e['type'] == 'warning']
    assert warnings == reply['warnings']
    assert station.claim is station.rig is station.audio is None
    assert station.lanes == station.hosts == {}
    assert 'lane-stop' in station.calls and 'rig-close' in station.calls
    assert not any(isinstance(c, tuple) and c[0] == 'fatal' for c in station.calls)
    assert station.snapshot()['sessions'][0]['state'] == 'ended'
    assert station.release()['state'] == 'observing'


def test_suspend_write_failure_is_a_warning(registry, monkeypatch):
    session = start(registry)
    def fail(record):
        raise OSError('disk full')
    monkeypatch.setattr(registry.records, 'write', fail)
    registry.suspend()
    assert registry.snapshot()['owner']['state'] == 'stopping'
    assert not registry.accepting
    assert registry.warnings == [f'{session.id}: disk full']
    assert any(kind == 'warning' for kind, _ in registry.events)


def test_blocked_record_write_does_not_delay_stop_or_rig_close(station, monkeypatch):
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    station.own()
    session = start(station.sessions)
    entered, unblock, stopped, closed = (threading.Event() for _ in range(4))
    original_write = station.sessions.records.write
    original_close = station.rig.close
    def write(record):
        entered.set()
        assert unblock.wait(5)
        original_write(record)
    def close():
        original_close()
        closed.set()
    monkeypatch.setattr(station.sessions.records, 'write', write)
    monkeypatch.setattr(station, 'stop_transmit', stopped.set)
    monkeypatch.setattr(station.rig, 'close', close)
    writer = threading.Thread(target=lambda: station.sessions.running(session.id))
    stopper = threading.Thread(target=lambda: service.submit(envelope('stop')))
    releaser = threading.Thread(target=station.release)
    writer.start()
    try:
        assert entered.wait(2)
        # Even non-STOP registry users must not be serialized behind fsync.
        assert station.sessions.lock.acquire(timeout=.2)
        station.sessions.lock.release()
        stopper.start()
        assert stopped.wait(1)
        releaser.start()
        assert closed.wait(1)
        assert not unblock.is_set()
    finally:
        unblock.set()
        for thread in (writer, stopper, releaser):
            if thread.ident is not None:
                thread.join(3)
                assert not thread.is_alive()
    assert station.claim is None


def test_attended_none_means_attended_rig_holder(registry):
    witness = start(registry, attended=True, holds_rig=False)
    assert registry.attended(witness.id)
    assert not registry.attended(None)
    holder = start(registry, attended=True)
    assert registry.attended(None)
    registry.running(holder.id)
    assert registry.attended(None)
    registry.suspend()
    assert not registry.attended(None)
    registry.finish_all(outcome='stopped')
    assert not registry.attended(None)


def test_stage_zero_keeps_attended_safety_limits_and_auto_id(station):
    from hfmodem.station import process
    from hfmodem.station.arbiter import Identity, Refused as ArbiterRefused
    from hfmodem.tests.core.test_arbiter import Card, LineRig, lease_of, req, centred
    station.state = 'owning'
    station.sessions.admit()
    holder = start(station.sessions, attended=True)
    station.audio = Card()
    station.rig = LineRig(station.audio)
    process.Station.build_arbiter(station)
    arb = station.arbiter
    try:
        assert station.sessions.attended(None)
        arb.identity = Identity('N0CALL', interval_s=0)
        with pytest.raises(ArbiterRefused, match='identification is due'):
            arb._policy(req(authority=arb.authority(holder.id)))
        assert arb.identify_if_free(centred(7_100_000, 500))
        arb._errand.join(2)
        assert not arb._errand.is_alive()
        arb.identity = None
        with lease_of(arb, authority=arb.authority(holder.id)) as lease:
            assert lease.ceiling_s == station.rig.max_key_s
            lease.end()
    finally:
        arb.close()
        station.arbiter = station.audio = station.rig = None


def test_sessions_keep_active_and_latest_100_finished(registry):
    holder = start(registry)
    recent = []
    for n in range(105):
        registry.clock.now = n
        session = start(registry, holds_rig=False)
        registry.end(session.id, outcome='completed')
        recent.append(session.id)
    assert {s['id'] for s in registry.snapshot()['sessions']} == {holder.id, *recent[-100:]}
    assert len(registry._sessions) == 101
    # Recovery ends the old holder, making it the newest finished session.
    registry.clock.now = 200
    recovered = Registry(records=registry.records, wall=registry.wall)
    recovered.recover()
    assert {s['id'] for s in recovered.snapshot()['sessions']} == {holder.id, *recent[-99:]}
    assert len(recovered._sessions) == 100


def test_owner_events_have_one_shape(station):
    service = Service(station.station_file, station=station, port=0)
    station.service = service
    station.own()
    start(station.sessions)
    station.sessions.control('desk')
    station.release()
    events = [e['data'] for e in service.events.after(0) if e['type'] == 'owner']
    assert len(events) >= 4
    assert all(set(e) == {'owner', 'control'} for e in events)


@pytest.mark.parametrize('contents', ['{', '{}', '[]', '{"id": "bad", "authority": null}'])
def test_corrupt_record_startup_warns_and_recovers_good_records(station, socketpair_http, contents):
    records = station.sessions.records
    records.directory.mkdir(parents=True)
    corrupt = records.directory / 'bad.json'
    corrupt.write_text(contents)
    service = Service(station.station_file, station=station)
    station.service = service
    service.bind()
    try:
        assert client.connect(station.station_file).state()['sessions'] == []
        assert not corrupt.exists()
        assert corrupt.with_name('bad.json.corrupt').read_text() == contents
        assert any(e['type'] == 'warning' and 'bad.json' in e['data']['detail']
                   for e in service.events.after(0))
    finally:
        service.close()


def test_record_fsyncs_file_before_replace_and_directory_after(registry, monkeypatch):
    import os
    import stat
    from hfmodem.station import qsolog
    calls = []
    fsync, replace = qsolog.os.fsync, qsolog.os.replace
    def sync(fd):
        calls.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        fsync(fd)
    def rename(src, dst):
        calls.append('replace')
        replace(src, dst)
    monkeypatch.setattr(qsolog.os, 'fsync', sync)
    monkeypatch.setattr(qsolog.os, 'replace', rename)
    start(registry)
    assert calls == ['file', 'replace', 'directory']


def test_only_post_authorization_refusals_are_cached(station):
    service = Service(station.station_file, station=station, port=0)
    control = station.sessions.control('desk')
    command = envelope('own', client='desk')
    refused = service.submit(command)
    assert result(service, refused)['reason'] == 'not-controller'
    authorized = command | {k: control[k] for k in ('control_id', 'generation')}
    accepted = service.submit(authorized)
    assert result(service, accepted)['state'] == 'accepted'
    assert accepted != refused
    with station._lifecycle:
        blocked = service.submit(authorized | {'id': 'transition'})
    assert result(service, blocked)['reason'] == 'transitioning'
    assert service.submit(authorized | {'id': 'transition'}) == blocked
    assert service.commands.qsize() == 1


def test_starting_refusal_does_not_cache_the_command_id(station):
    service = Service(station.station_file, port=0)
    command = envelope('own')
    refused = service.submit(command)
    assert result(service, refused)['reason'] == 'starting'
    service.station = station
    accepted = service.submit(command)
    assert result(service, accepted)['state'] == 'accepted'
    assert accepted != refused
    assert service.commands.qsize() == 1


def test_external_claim_refusal_uses_same_string_shape(station, monkeypatch):
    from hfmodem.core import claim
    service = Service(station.station_file, station=station, port=0)
    def own(**kwargs):
        service._closed.set()
        raise claim.ClaimError('legacy holder pid=123')
    monkeypatch.setattr(station, 'own', own)
    service.submit(envelope('own'))
    dispatch = threading.Thread(target=service._dispatch, daemon=True)
    dispatch.start()
    dispatch.join(5)
    if dispatch.is_alive():
        service._closed.set()
        pytest.fail('dispatch never reached own')
    refusal = [e['data'] for e in service.events.after(0)
               if e['type'] == 'command' and e['data']['state'] == 'refused'][-1]
    assert refusal['reason'] == 'claimed'
    assert refusal['claim_holder'] == refusal['detail'] == 'legacy holder pid=123'


def test_invalid_record_field_is_moved_aside(registry):
    session = start(registry)
    path = registry.records.directory / f'{session.id}.json'
    record = json.loads(path.read_text())
    record['ended_utc'] = {'invalid': 'timestamp'}
    path.write_text(json.dumps(record))
    fresh = Registry(records=registry.records)
    fresh.recover()
    assert fresh.snapshot()['sessions'] == []
    assert path.with_name(path.name + '.corrupt').exists()
    assert fresh.warnings


def bound(station):
    service = Service(station.station_file, station=station)
    station.service = service
    service.bind()
    return service


def test_unwritable_records_at_startup_warn_and_still_recover(station, socketpair_http):
    records = station.sessions.records
    earlier = Registry(records=records)
    earlier.admit()
    pending = start(earlier)
    records.directory.chmod(0o555)
    try:
        service = bound(station)
        try:
            [recovered] = client.connect(station.station_file).state()['sessions']
            assert recovered['id'] == pending.id
            assert recovered['state'] == 'ended' and recovered['outcome'] == 'interrupted'
            assert any(e['type'] == 'warning' and pending.id in e['data']['detail']
                       for e in service.events.after(0))
        finally:
            service.close()
    finally:
        records.directory.chmod(0o755)
    assert json.loads((records.directory / f'{pending.id}.json').read_text())['state'] == 'starting'


@pytest.mark.parametrize('case', ['mode-000', 'directory', 'rename-fails'])
def test_unreadable_or_stuck_records_are_skipped_at_startup(station, socketpair_http, case):
    records = station.sessions.records
    earlier = Registry(records=records)
    earlier.admit()
    good = earlier.end(start(earlier).id, outcome='completed')
    bad = records.directory / 'bad.json'
    if case == 'directory':
        bad.mkdir()
    else:
        bad.write_text('{')
    if case == 'mode-000':
        bad.chmod(0)
    if case == 'rename-fails':
        records.directory.chmod(0o555)
    try:
        service = bound(station)
        try:
            [kept] = client.connect(station.station_file).state()['sessions']
            assert kept == good.record()
            assert any(e['type'] == 'warning' and 'bad.json' in e['data']['detail']
                       for e in service.events.after(0))
        finally:
            service.close()
    finally:
        records.directory.chmod(0o755)
        if case == 'mode-000':
            bad.chmod(0o644)
    assert bad.exists() and not bad.with_name('bad.json.corrupt').exists()


def test_repeat_end_after_pruning_is_a_no_op(registry):
    first = start(registry, holds_rig=False)
    registry.end(first.id, outcome='completed')
    for n in range(1, 102):
        registry.clock.now = n
        registry.end(start(registry, holds_rig=False).id, outcome='completed')
    assert first.id not in registry._sessions
    count = len(registry.events)
    assert registry.end(first.id, outcome='different') is None
    assert len(registry.events) == count
    assert json.loads((registry.records.directory / f'{first.id}.json').read_text())['outcome'] == 'completed'


def test_admission_write_in_flight_across_suspend_lands_stopping(registry, monkeypatch):
    entered, unblock = threading.Event(), threading.Event()
    original = registry.records.write
    def write(record):
        if not entered.is_set():
            entered.set()
            assert unblock.wait(5)
        original(record)
    monkeypatch.setattr(registry.records, 'write', write)
    admission = threading.Thread(target=lambda: start(registry), daemon=True)
    admission.start()
    assert entered.wait(2)
    teardown = threading.Thread(target=registry.suspend, daemon=True)
    teardown.start()
    teardown.join(.2)
    unblock.set()
    for thread in (admission, teardown):
        thread.join(5)
        assert not thread.is_alive()
    assert registry.snapshot()['owner']['state'] == 'stopping'
    assert [r['state'] for r in registry.records.read()] == ['stopping']


def test_valid_record_under_another_name_is_moved_aside(registry):
    session = start(registry)
    path = registry.records.directory / f'{session.id}.json'
    stray = path.with_name('elsewhere.json')
    stray.write_text(path.read_text())
    fresh = Registry(records=registry.records)
    fresh.recover()
    assert [s['id'] for s in fresh.snapshot()['sessions']] == [session.id]
    assert not stray.exists() and stray.with_name('elsewhere.json.corrupt').exists()
    assert any('elsewhere.json' in w for w in fresh.warnings)
