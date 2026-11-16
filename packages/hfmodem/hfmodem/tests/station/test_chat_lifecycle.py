# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Attachment teardown proofs with injected owners, never sockets or hardware."""
from types import SimpleNamespace
import threading
import time

import pytest

from hfmodem.bbs.store import Store
from hfmodem.station.chat import Attachment, AttachmentGate
from hfmodem.station.sessions import Refused
from hfmodem.kestrel.host.server import HostSession


class Fence:
    identity = ('operator', 'lease', 1)

    def check(self):
        pass

    def __call__(self):
        return True


def attachment(tmp_path, monkeypatch, *, bbs_path=None, line='low'):
    events = []

    class Gate(AttachmentGate):
        def release(self, owner):
            events.append('gate-release')
            super().release(owner)

    gate = Gate()
    runtime = SimpleNamespace(attachments=gate,
        modem_factory=lambda **kwargs: pytest.fail('modem factory must not run'))

    def start(**kwargs):
        events.append('register')
        return SimpleNamespace(id='chat-session', authority=object())

    sessions = SimpleNamespace(start=start,
        end=lambda *args, **kwargs: events.append('session-end'))
    station = SimpleNamespace(
        lanes={'vara': SimpleNamespace(runtime=runtime)},
        arbiter=SimpleNamespace(stop=lambda sid: events.append('authority-revoke')),
        rig=SimpleNamespace(line_state=line, dial=lambda: 14_105_000),
        cfg=SimpleNamespace(bbs=SimpleNamespace(config=bbs_path),
            station=SimpleNamespace(mycall='W9SSJ'), rig=SimpleNamespace(filter_hz=3000)),
        station_file=None, sessions=sessions, audio=SimpleNamespace(samples=0))
    chat = SimpleNamespace(station=station, fence=Fence(), session=None,
                           service=SimpleNamespace(incarnation='test-station'), generation=1)
    monkeypatch.setenv('HFMODEM_DATA_DIR', str(tmp_path))
    return Attachment(chat), events


def reserved(a):
    a.runtime = a.station.lanes['vara'].runtime
    a.runtime.attachments.reserve(a)
    a._reserved = True


def assert_reserved(a):
    with pytest.raises(Refused):
        a.runtime.attachments.reserve(object())


def resource(events, name, **attributes):
    return SimpleNamespace(close=lambda **kwargs: events.append(name), **attributes)


def test_no_rig_startup_refusal_requires_no_unowned_low_line_proof(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    a.station.rig = None
    with pytest.raises(Refused, match='owned station VARA lane required'):
        a.open()
    assert a._closed
    a.release()
    assert events == []


def test_bbs_owner_conflict_refuses_before_modem_factory(tmp_path, monkeypatch):
    config = tmp_path / 'bbs.toml'
    config.write_text('[bbs]\nstate_root = "state"\n')
    a, events = attachment(tmp_path, monkeypatch, bbs_path=str(config))
    with Store(tmp_path / 'state'):
        with pytest.raises(Refused, match='BBS|bbs|owner|owned'):
            a.open()
        assert a._closed
        assert_reserved(a)
        a.release()
    assert events == ['gate-release']


def test_startup_failure_after_bbs_acquisition_closes_store_before_release(tmp_path, monkeypatch):
    config = tmp_path / 'bbs.toml'
    config.write_text('[bbs]\nstate_root = "state"\n')
    a, events = attachment(tmp_path, monkeypatch, bbs_path=str(config))

    def fail(**kwargs):
        raise ValueError('registration failed')

    a.station.sessions.start = fail
    with pytest.raises(ValueError, match='registration failed'):
        a.open()
    assert a.bbs.store is None
    with Store(tmp_path / 'state'):
        pass
    assert_reserved(a)
    a.release()
    assert events == ['gate-release']


def test_teardown_order_and_exactly_once_session_quit(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)
    a.chat.session = SimpleNamespace(id='registered')
    a.session = SimpleNamespace(quit=lambda **kwargs: events.append('quit'))
    a.client = resource(events, 'client')
    a.host = resource(events, 'host', submitter_live=True,
        ensure_retired=lambda **kwargs: events.append('host-retired'),
        _modem=SimpleNamespace(_retired_attachment=True))
    a.worker = SimpleNamespace(join=lambda timeout: events.append('host-join'), is_alive=lambda: False)
    a.capture = SimpleNamespace(close=lambda **kwargs: events.append('capture'))
    a.bbs = SimpleNamespace(close=lambda **kwargs: events.append('bbs'))
    a._held.callback(events.append, 'stream-first')
    a._held.callback(events.append, 'stream-last')
    a.close()
    a.close()
    a.release()
    a.release()
    assert not a.host.submitter_live
    assert events == ['quit', 'client', 'host', 'host-join', 'host-retired',
                      'capture', 'bbs', 'stream-last', 'stream-first',
                      'authority-revoke', 'session-end', 'gate-release']


def test_pending_bbs_retains_gate_and_close_retry_does_not_repeat_quit(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)
    a.session = SimpleNamespace(quit=lambda **kwargs: events.append('quit'))
    pending = [True]

    def close_bbs(**kwargs):
        events.append('bbs')
        if pending[0]:
            raise RuntimeError('BBS worker pending; owner lock retained')

    a.bbs = SimpleNamespace(close=close_bbs)
    a._held.callback(events.append, 'streams')
    with pytest.raises(RuntimeError, match='BBS worker pending'):
        a.close()
    assert_reserved(a)
    with pytest.raises(RuntimeError, match='have not retired'):
        a.release()
    pending[0] = False
    a.close()
    a.release()
    assert events == ['quit', 'bbs', 'streams', 'bbs', 'gate-release']


@pytest.mark.parametrize('line', ['high', 'unknown'])
def test_high_or_unknown_unkey_fails_and_retains_reservation(tmp_path, monkeypatch, line):
    a, events = attachment(tmp_path, monkeypatch, line=line)
    reserved(a)
    a.capture = SimpleNamespace(close=lambda **kwargs: events.append('capture'))
    with pytest.raises(RuntimeError, match='low-line proof'):
        a.close()
    assert events == ['capture']
    assert_reserved(a)
    with pytest.raises(RuntimeError, match='have not retired'):
        a.release()


def test_failed_save_does_not_skip_bbs_or_stream_cleanup(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)

    def fail(**kwargs):
        events.append('capture')
        raise RuntimeError('recording save failed')

    a.capture = SimpleNamespace(close=fail)
    a.bbs = SimpleNamespace(close=lambda **kwargs: events.append('bbs'))
    a._held.callback(events.append, 'streams')
    with pytest.raises(RuntimeError, match='recording save failed'):
        a.close()
    assert events == ['capture', 'bbs', 'streams']
    assert_reserved(a)


def test_quit_failure_is_not_hidden_by_retry_and_other_cleanup_continues(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)

    def quit(*, unwinding=None):
        events.append('quit')
        raise ValueError('quit failed')

    a.session = SimpleNamespace(quit=quit)
    a.client = resource(events, 'client')
    for _ in range(2):
        with pytest.raises(RuntimeError, match='quit failed'):
            a.close()
    assert events == ['quit', 'client', 'client']
    assert_reserved(a)


def test_session_end_must_succeed_before_gate_release(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)
    a.chat.session = SimpleNamespace(id='registered')
    pending = [True]

    def end(*args, **kwargs):
        events.append('session-end')
        if pending[0]:
            raise RuntimeError('registry unavailable')

    a.station.sessions.end = end
    a.close()
    with pytest.raises(RuntimeError, match='registry unavailable'):
        a.release()
    assert_reserved(a)
    pending[0] = False
    a.release()
    a.release()
    assert events == ['authority-revoke', 'session-end', 'session-end', 'gate-release']


def test_live_host_worker_blocks_gate_release_even_with_low_line(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)
    a.host = resource(events, 'host', submitter_live=True,
        ensure_retired=lambda **kwargs: pytest.fail('reader verification before host worker exit'),
        _modem=SimpleNamespace(_retired_attachment=True))
    a.worker = SimpleNamespace(join=lambda timeout: events.append('host-join'), is_alive=lambda: True)
    with pytest.raises(RuntimeError, match='host worker did not retire'):
        a.close()
    assert_reserved(a)
    with pytest.raises(RuntimeError, match='have not retired'):
        a.release()


def test_host_close_lock_acquisition_honors_deadline_and_preserves_default_api():
    host = HostSession.__new__(HostSession)
    host._operation_lock = threading.Lock()
    events = []
    host._close = lambda: events.append('close')
    host._operation_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='close operation still pending'):
            host.close(deadline=time.monotonic() - 1)
        assert events == []
    finally:
        host._operation_lock.release()
    # An already-free lock still allows immediate cleanup after the deadline.
    host.close(deadline=time.monotonic() - 1)
    host.close()
    assert events == ['close', 'close']


def test_data_writer_exit_is_required_and_retry_uses_remaining_deadline(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)
    alive, waits = [True], []
    writer = SimpleNamespace(_thread=SimpleNamespace(
        join=lambda timeout: waits.append(timeout), is_alive=lambda: alive[0]))
    a.client = resource(events, 'client', data_writer=writer)
    a.capture = SimpleNamespace(close=lambda **kwargs: events.append('capture'))
    a.deadline = time.monotonic() - 1
    with pytest.raises(RuntimeError, match='data writer did not retire'):
        a.close()
    assert waits == [0]
    assert events == ['client', 'capture']
    assert_reserved(a)
    with pytest.raises(RuntimeError, match='have not retired'):
        a.release()
    alive[0] = False
    a.close()
    a.release()
    assert events == ['client', 'capture', 'client', 'capture', 'gate-release']


def test_failed_authority_revocation_prevents_session_end_and_gate_release(tmp_path, monkeypatch):
    a, events = attachment(tmp_path, monkeypatch)
    reserved(a)
    a.chat.session = SimpleNamespace(id='registered')
    pending = [True]

    def stop(sid):
        assert sid == 'registered'
        events.append('authority-revoke')
        if pending[0]:
            raise RuntimeError('authority revocation pending')

    a.station.arbiter.stop = stop
    a.close()
    with pytest.raises(RuntimeError, match='authority revocation pending'):
        a.release()
    assert events == ['authority-revoke']
    assert_reserved(a)
    pending[0] = False
    a.release()
    a.release()
    assert events == ['authority-revoke', 'authority-revoke', 'session-end', 'gate-release']


def test_capture_constructor_failure_preserves_failed_resource_cleanup(tmp_path, monkeypatch):
    from hfmodem.contact import slots
    from hfmodem.core import wav

    a, events = attachment(tmp_path, monkeypatch)

    def close_slots():
        events.append('slots-close')
        raise RuntimeError('decoder did not retire')

    def recorder(*args, **kwargs):
        raise ValueError('recorder construction failed')

    monkeypatch.setattr(slots, 'SlotWatch', lambda *args, **kwargs: SimpleNamespace(close=close_slots))
    monkeypatch.setattr(wav, 'CaptureRecorder', recorder)
    with pytest.raises(ValueError, match='recorder construction failed'):
        a.open()
    assert 'slots-close' in events
    assert_reserved(a)
    with pytest.raises(RuntimeError, match='decoder did not retire'):
        a.close()



def test_operator_close_keeps_live_armed_authority_for_disconnect_and_owed_id(tmp_path, monkeypatch):
    from hfmodem.tests.contact.test_varac import Desk
    a, events = attachment(tmp_path, monkeypatch)
    desk = Desk()
    desk.hear('CONNECTED W9SSJ N0XYZ 500')
    desk.session.owed = True
    desk.session._tx_dial = desk.flow.dial
    a.session = desk.session
    a._armed = True
    def disconnect():
        assert a.allowed(), 'DISCONNECT lost operator authority'
        events.append('disconnect')
        return ['OK', 'DISCONNECTED']
    identify = desk.session.identify
    def closing_identify():
        assert a.allowed(), 'closing ID lost operator authority'
        events.append('ident')
        return identify()
    desk.host.replies['DISCONNECT'] = disconnect
    desk.session.identify = closing_identify
    a.close()
    assert events == ['disconnect', 'ident']
    assert a._armed is False and a._closed


def test_attached_operator_exit_closes_link_and_id_before_monitor_cleanup(tmp_path, monkeypatch):
    from hfmodem.contact import chat_state as s
    from hfmodem.contact.chat_controller import Controller
    from hfmodem.station.chat import AttachedSession
    from hfmodem.tests.contact.test_varac import Desk
    a, events = attachment(tmp_path, monkeypatch)
    a.fence.expires_utc = None
    class Watch:
        closed = False
        def table(self):
            if self.closed:
                raise ValueError('monitor closed')
            return (), None
        def close(self):
            self.closed = True
    watch = Watch()
    def factory():
        desk = Desk()
        desk.hear('CONNECTED W9SSJ N0XYZ 500')
        desk.session.owed = True
        desk.session._tx_dial = desk.flow.dial
        desk.session.slots = watch
        desk.session.tick = lambda *args: None
        a.session = desk.session
        a.capture = SimpleNamespace(close=lambda **kwargs: watch.close())
        a._armed = True
        identify = desk.session.identify
        def closing_identify():
            a.fence.check()
            assert a.allowed()
            events.append('ident')
            return identify()
        def disconnect():
            assert a.allowed()
            events.append('disconnect')
            return ['OK', 'DISCONNECTED']
        desk.session.identify = closing_identify
        desk.host.replies['DISCONNECT'] = disconnect
        return AttachedSession(a)
    controller = Controller(factory, controller_id='exit')
    snapshot = controller.ready.result(2)
    req = s.Request('quit', 'exit', snapshot.link.generation, 'view', 'lease', 1, s.ChatExit())
    assert controller.submit(req).queued
    assert controller.join(2)
    controller.shutdown().result(0)
    assert controller.outcome('quit').state.state == 'done'
    assert events == ['disconnect', 'ident']
    assert watch.closed and not a._armed
    assert controller.snapshot().tx.physical_line is False
