# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import random
import threading
import time
from types import SimpleNamespace

import pytest

from hfmodem.ax25.connected import Connection, State
from hfmodem.ax25.frame import parse
from hfmodem.brubru.host.mail import run_exchange
from hfmodem.brubru.host.runtime import PacketRuntime
from hfmodem.station.arbiter import RequestState
from hfmodem.winlink import B2FSession, MailClient
from hfmodem.winlink.message import compose


class InstantAir:
    def __init__(self, *, mode='mail', outbox=()):
        self.host = self.packet_engine = None
        self.mode = mode
        self.peer = Connection('N0CALL', self.receive_frame)
        self.peer.t2 = .001
        self.session = B2FSession('N0CALL', role='answering', target='W9SSJ', outbox=list(outbox))
        client = MailClient(self.session, self.peer.write)
        self.peer.receive = client.on_link_data if mode == 'mail' else lambda _: None
        self.peer.state_changed = lambda state, *_: client.link_up() if (
            state is State.CONNECTED and mode == 'mail') else None

    def receive_frame(self, frame):
        self.host.send_frame(frame)
        return True

    def enqueue(self, frame):
        if self.mode != 'no-link':
            self.peer.on_frame(frame)
        return SimpleNamespace(completion=('sent', ''), state=RequestState())

    def drop_pending(self):
        pass

    def start(self):
        runtime = PacketRuntime(self, mycall='W9SSJ')
        original_tick = runtime.engine.tick
        def tick():
            if self.mode == 'loss' and runtime.engine.state is State.CONNECTED:
                self.peer.disconnect()
            self.peer.tick()
            original_tick()
        runtime.engine.tick = tick
        runtime.engine.t2 = .001
        runtime.start()
        return runtime


def test_mail_exchange_both_directions_large_write_and_owner_thread(tmp_path, capsys):
    body = random.Random(1).randbytes(2048)
    outbound = compose('W9SSJ', 'N0CALL', 'outbound', body, mid='OWN000000001')
    inbound = compose('N0CALL', 'W9SSJ', 'inbound', b'return\r\n', mid='OWN000000002')
    air = InstantAir(outbox=[inbound])
    runtime = air.start()
    called = set()
    def instrument(name):
        original = getattr(runtime.engine, name)
        def check(*args, **kw):
            assert threading.current_thread() is runtime._thread
            called.add(name)
            return original(*args, **kw)
        setattr(runtime.engine, name, check)
    def configure():
        runtime.engine.buffer_limit = 64
        for name in ('connect', 'write', 'disconnect', 'on_frame', 'tick', 'abort'):
            instrument(name)
    runtime._call(configure)
    session = B2FSession('W9SSJ', target='N0CALL', outbox=[outbound])
    try:
        assert run_exchange(runtime, session, 'N0CALL', timeout=8, outdir=tmp_path) == 0
        assert session.done and not session.failure
        assert session.sent_mids == [outbound.mid]
        assert [m.render() for m in air.session.inbox] == [outbound.render()]
        assert (tmp_path/f'{inbound.mid}.b2f').read_bytes() == inbound.render()
        assert called == {'connect', 'write', 'disconnect', 'on_frame', 'tick', 'abort'}
        assert runtime._attachment is None and not runtime.active
        assert runtime.engine.accepted == runtime.engine.delivered > 2048
        assert runtime.engine.discarded == 0
        transcript = capsys.readouterr().out
        assert 'CONNECTED N0CALL' in transcript and 'link down' in transcript
        assert 'mail: exchange complete — closing' in transcript
    finally:
        runtime.stop()


@pytest.mark.parametrize('mode, expected', [
    ('no-link', 'mail: NO LINK'), ('idle', 'mail: out of time — closing the link'),
    ('loss', 'the link went down mid-exchange'), ('stop', 'the run was stopped')])
def test_mail_failure_timeout_link_loss_and_stop(mode, expected, capsys):
    air = InstantAir(mode=mode)
    runtime = air.start()
    try:
        session = B2FSession('W9SSJ', target='N0CALL')
        stop = lambda: mode == 'stop' and runtime._call(lambda: runtime.engine.state is State.CONNECTED)
        assert run_exchange(runtime, session, 'N0CALL', timeout=.15, stopped=stop, close_timeout=1) == 1
        assert expected in capsys.readouterr().out
        assert runtime._attachment is None and not runtime.active
    finally:
        runtime.stop()


def test_mail_protocol_failure_is_never_exchange_complete(capsys):
    air = InstantAir()
    air.session.client_sid = 'BadPeer-1'
    runtime = air.start()
    try:
        session = B2FSession('W9SSJ', target='N0CALL')
        # A gateway's explicit rejection ends a session before any proposal.
        air.peer.receive = lambda _: air.peer.write(b'*** rejected\r')
        assert run_exchange(runtime, session, 'N0CALL', timeout=2) == 1
        transcript = capsys.readouterr().out
        assert 'mail: exchange failed' in transcript
        assert 'mail: exchange complete' not in transcript
    finally:
        runtime.stop()


def test_clean_session_with_declined_outbound_is_nonzero():
    message = compose('W9SSJ', 'N0CALL', 'duplicate', b'already held', mid='OWN000000001')
    air = InstantAir()
    air.session.inbox.append(message)
    runtime = air.start()
    session = B2FSession('W9SSJ', target='N0CALL', outbox=[message])
    try:
        assert run_exchange(runtime, session, 'N0CALL', timeout=2) == 1
        assert session.done and not session.failure and not session.sent_mids
        assert session.proposals[0].status == 'rejected'
    finally:
        runtime.stop()


def test_retirement_waits_for_committed_rf_completion():
    air = InstantAir(mode='no-link')
    finished = threading.Event()
    timers = []
    def enqueue(frame):
        request = SimpleNamespace(completion=None, state=RequestState())
        request.state.commit()
        def complete():
            request.completion = ('sent', '')
            finished.set()
        timer = threading.Timer(.15, complete)
        timers.append(timer)
        timer.start()
        return request
    air.enqueue = enqueue
    runtime = air.start()
    try:
        assert run_exchange(runtime, B2FSession('W9SSJ'), 'N0CALL', timeout=.02) == 1
        assert finished.is_set() and runtime._transmission is None
    finally:
        runtime.stop()
        for timer in timers:
            timer.join(1)


def test_stop_during_committed_iframe_retires_before_return(monkeypatch, capsys):
    from hfmodem.brubru.host import mail
    air = InstantAir()
    enqueue = air.enqueue
    committed, finished = threading.Event(), threading.Event()
    requests, timers = [], []
    def transmit(frame):
        if parse(frame).control & 1:
            return enqueue(frame)
        request = SimpleNamespace(completion=None, state=RequestState())
        assert request.state.commit()
        requests.append(request)
        def complete():
            request.completion = ('sent', '')
            finished.set()
        timer = threading.Timer(4.3, complete)
        timers.append(timer)
        timer.start()
        committed.set()
        return request
    air.enqueue = transmit
    runtime = air.start()
    monkeypatch.setattr(mail, 'STOP_CLOSE_S', .05)
    detach = runtime.detach_and_wait
    def retire(attachment, *, timeout):
        assert timeout == 5. + mail.RETIRE_MARGIN_S
        assert not finished.is_set()
        detach(attachment, timeout=timeout)
        assert finished.is_set()
    monkeypatch.setattr(runtime, 'detach_and_wait', retire)
    try:
        result = run_exchange(runtime, B2FSession('W9SSJ', target='N0CALL'),
                              'N0CALL', timeout=2, stopped=committed.is_set, burst_limit=5.)
        assert result == 1 and committed.is_set() and finished.is_set()
        assert len(requests) == 1 and requests[0].state.value == 'committed'
        assert runtime._attachment is None and not runtime.active
        assert runtime._transmission is None and runtime.engine.state is State.DISCONNECTED
        transcript = capsys.readouterr().out
        assert 'mail: exchange complete' not in transcript
        assert 'packet RF work did not retire' not in transcript
    finally:
        runtime.stop()
        for timer in timers:
            timer.join(6)


class TerminalAir(InstantAir):
    def __init__(self, outcome):
        super().__init__()
        self.outcome = outcome
        self.terminal = None
        self.ticks = self.deferrals = 0
        receive = self.peer.receive
        def finish(data):
            receive(data)
            if self.session.done:
                self.peer.disconnect()
        self.peer.receive = finish

    def enqueue(self, frame):
        if parse(frame).control & 0xEF == 0x63 and self.packet_engine.engine.state is State.DISCONNECTED:
            self.terminal = SimpleNamespace(completion=None, state=RequestState(), frame=frame)
            return self.terminal
        return super().enqueue(frame)

    def drop_pending(self):
        if self.terminal and self.terminal.state.value == 'cancelled':
            self.terminal.completion = ('cancelled', '')

    def start(self):
        runtime = super().start()
        def configure():
            tick, transmit = runtime.engine.tick, runtime.engine.transmit
            def defer(frame):
                if (parse(frame).control & 0xEF == 0x63 and
                        runtime.engine.state is State.DISCONNECTED and self.deferrals < 3):
                    self.deferrals += 1
                    return False
                return transmit(frame)
            def complete():
                if self.terminal and self.terminal.completion is None:
                    self.ticks += 1
                    if self.ticks >= 8 and self.outcome != 'blocked':
                        if self.outcome == 'sent':
                            self.peer.on_frame(self.terminal.frame)
                        self.terminal.completion = (self.outcome, 'deferred UA')
                tick()
            runtime.engine.tick, runtime.engine.transmit = complete, defer
        runtime._call(configure)
        return runtime


@pytest.mark.parametrize('outcome', ['sent', 'error', 'blocked'])
def test_final_ua_drains_through_rf_completion_before_retirement(outcome, capsys):
    air = TerminalAir(outcome)
    runtime = air.start()
    try:
        result = run_exchange(runtime, B2FSession('W9SSJ', target='N0CALL'),
                              'N0CALL', timeout=2, close_timeout=.3)
        assert result == (0 if outcome == 'sent' else 1)
        assert air.terminal is not None and air.deferrals == 3
        assert runtime._attachment is None and not runtime.active
        transcript = capsys.readouterr().out
        if outcome == 'sent':
            assert air.terminal.completion[0] == 'sent'
            assert air.terminal.state.value != 'cancelled'
            assert air.peer.state is State.DISCONNECTED
            assert not air.peer.active and not air.peer.controls
            assert transcript.index('link down') < transcript.index('mail: exchange complete')
        else:
            assert 'mail: exchange complete' not in transcript
            assert 'mail: exchange failed' in transcript
            assert 'session ended: the exchange finished' not in transcript
    finally:
        runtime.stop()


class LostFinalAir(InstantAir):
    def enqueue(self, frame):
        if parse(frame).info == b'FQ\r':
            self.packet_engine.engine.rtt = .01
            return SimpleNamespace(completion=('sent', ''), state=RequestState())
        return super().enqueue(frame)


def test_final_fq_loss_is_not_published_as_complete(capsys):
    air = LostFinalAir()
    runtime = air.start()
    runtime._call(lambda: setattr(runtime.engine, 'retry_limit', 1))
    runtime._call(lambda: setattr(runtime.engine, 'rtt', .02))
    message = compose('W9SSJ', 'N0CALL', 'outbound', b'final FQ loss', mid='OWN000000001')
    session = B2FSession('W9SSJ', target='N0CALL', outbox=[message])
    try:
        assert run_exchange(runtime, session, 'N0CALL', timeout=2, close_timeout=1) == 1
        assert session.done and not air.session.done
        assert runtime.engine.discarded == 3
        transcript = capsys.readouterr().out
        assert 'mail: exchange complete' not in transcript
        assert 'mail: exchange failed — link close did not finish before the deadline' in transcript
        assert 'session ended: link close did not finish before the deadline' in transcript
    finally:
        runtime.stop()


def test_stop_during_close_shortens_deadline_and_aborts(monkeypatch, capsys):
    from hfmodem.brubru.host import mail
    air = InstantAir()
    enqueue = air.enqueue
    def drop_disc(frame):
        if parse(frame).control & 0xEF == 0x43:
            return SimpleNamespace(completion=('sent', ''), state=RequestState())
        return enqueue(frame)
    air.enqueue = drop_disc
    runtime = air.start()
    monkeypatch.setattr(mail, 'STOP_CLOSE_S', .1)
    stopped = lambda: runtime._call(lambda: runtime.engine.state is State.DISCONNECTING)
    start = time.monotonic()
    try:
        assert run_exchange(runtime, B2FSession('W9SSJ', target='N0CALL'),
                            'N0CALL', timeout=2, stopped=stopped) == 1
        assert time.monotonic() - start < 2
        assert not runtime.active and runtime._attachment is None
        transcript = capsys.readouterr().out
        assert 'mail: exchange complete' not in transcript
        assert 'session ended: link close did not finish before the deadline' in transcript
    finally:
        runtime.stop()


def test_failed_rf_retirement_cannot_publish_completion(monkeypatch, capsys):
    air = InstantAir()
    runtime = air.start()
    detach = runtime.detach_and_wait
    def fail(attachment, **kwargs):
        detach(attachment, **kwargs)
        raise RuntimeError('RF retirement failure')
    monkeypatch.setattr(runtime, 'detach_and_wait', fail)
    try:
        with pytest.raises(RuntimeError, match='RF retirement failure'):
            run_exchange(runtime, B2FSession('W9SSJ', target='N0CALL'), 'N0CALL', timeout=2)
        transcript = capsys.readouterr().out
        assert 'mail: exchange complete' not in transcript
        assert 'session ended: packet RF work did not retire successfully' in transcript
    finally:
        runtime.stop()


def test_station_mail_host_has_contact_owner_and_no_endpoint():
    from hfmodem.core.config import ProtocolConfig
    from hfmodem.station import hosts
    from hfmodem.tests.brubru.test_station import station
    st = station()
    st.cfg.station = SimpleNamespace(mycall='W9SSJ')
    st.cfg.protocols = {'rpr': ProtocolConfig(enabled=True, host='mail')}
    st.enabled, st.lanes = {'rpr'}, {'rpr': st.link}
    lines = []
    served = hosts.build(st.cfg, station=st, log=lines.append)
    try:
        assert served['rpr'].ports == served['rpr'].endpoints == ()
        assert isinstance(served['rpr'].server, PacketRuntime)
        assert served['rpr'].server.contact is not None
        assert lines == ['rpr: mail in process']
    finally:
        hosts.stop_all(served)
        st.arbiter.close()
