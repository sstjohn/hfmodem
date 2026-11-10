# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Connected facades and raw KISS qualify local AX.25 generations separately."""
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from hfmodem.ax25 import connected as AX
from hfmodem.ax25.frame import Frame, address, ui_frame
from hfmodem.ax25.kiss import encode
from hfmodem.brubru.host.runtime import PacketRuntime
from hfmodem.brubru.host.server import KissServer
from hfmodem.station.arbiter import TxResult
from hfmodem.station.link import RprLink
from hfmodem.tests.station.recording_live_support import LOCAL, PEER, OTHER, Clock


def frame(src, dst, control, *, command=True, via=(), completed=True, pf=True):
    calls = (dst, src, *via)
    addresses = tuple(address(call, last=i == len(calls) - 1,
                              flag=command if i == 0 else not command if i == 1 else completed)
                      for i, call in enumerate(calls))
    return Frame(addresses, control | (0x10 if pf else 0), None, b'').encode()


class PacketLink:
    def __init__(self):
        self.host = self.packet_engine = None
        self.txdelay = 100
        self._condition = threading.Condition()
        self.access = NS(persistence=63, slot_ms=100, _lock=threading.Lock())
        self.requests = []

    def enqueue(self, raw):
        request = NS(frame=raw, completion=None, state=NS(cancel=lambda: None))
        self.requests.append(request)
        return request

    def drop_pending(self):
        pass


def runtime(owner, monkeypatch, personality='tracker'):
    from hfmodem.brubru.host import mail

    monkeypatch.setattr(mail, 'MailClient', lambda session, send: NS(
        link_up=lambda: None, on_link_data=lambda data: None))
    result = owner.bind(PacketRuntime(PacketLink(), mycall=LOCAL, clock=Clock()))
    owner.bind(result.engine)
    monkeypatch.setattr(result, '_call', lambda fn: fn())
    attachment = result.attach(personality, send=lambda packet: None, session=NS())
    if personality == 'agwpe':
        attachment.host.registered = LOCAL
    result.engine.listen = True
    return result, attachment


def connected(runtime, *, incoming=False):
    if incoming:
        runtime.send_frame(frame(PEER, LOCAL, AX.SABM))
    else:
        runtime.engine.connect(PEER)
        runtime.send_frame(frame(PEER, LOCAL, AX.UA, command=False))
    runtime._step()


@pytest.mark.parametrize('personality', ['tracker', 'agwpe', 'mail'])
@pytest.mark.parametrize('incoming', [False, True])
def test_packet_facades_keep_boundary_before_replaceable_state_changed(live, monkeypatch,
                                                                     personality, incoming):
    owner = live('off')
    rt, attachment = runtime(owner, monkeypatch, personality)
    delivered = []
    callback = rt.engine.state_changed

    def observe(*args):
        delivered.append((args[0], len(owner.starts), len(owner.ends)))
        callback(*args)

    rt.engine.state_changed = observe
    connected(rt, incoming=incoming)
    row = owner.one()
    assert row['key'][1] == 'rpr' and row['peer'] == PEER
    assert row['key'][-1] == rt.engine.generation
    assert [e for e in delivered if e[0] is AX.State.CONNECTED][0][1] == 1
    rt._retire(attachment)
    row = owner.one()
    assert row['state'] == 'partial'
    assert len(owner.starts) == len(owner.ends) == 1
    assert [e for e in delivered if e[0] is AX.State.DISCONNECTED][0][2] == 1


@pytest.mark.parametrize('end', ['peer-disc', 'peer-dm', 'peer-frmr', 'retry-exhausted',
                                 'peer-reset', 'host-abort', 'owner-stop'])
def test_connected_packet_end_seams_retire_one_generation(live, monkeypatch, end):
    owner = live('off')
    rt, _ = runtime(owner, monkeypatch)
    connected(rt)
    if rt._transmission is not None:
        rt._transmission.completion = ('sent', '')
        rt._step()
    if end == 'retry-exhausted':
        rt.engine.retries = rt.engine.retry_limit
        rt.engine._deadline = rt.clock.now
        rt._step()
    elif end == 'host-abort':
        rt.engine.abort('host abort')
    elif end == 'owner-stop':
        rt._stop.set()
        rt._run()
    else:
        if end == 'peer-reset':
            rt.engine.accepted = 1
        control = {'peer-disc': AX.DISC, 'peer-dm': AX.DM,
                   'peer-frmr': AX.FRMR, 'peer-reset': AX.SABM}[end]
        rt.send_frame(frame(PEER, LOCAL, control, command=end in ('peer-disc', 'peer-reset')))
        rt._step()
    row = owner.one()
    assert row.get('end_positions') is not None
    assert len(owner.starts) == len(owner.ends) == 1
    if end != 'peer-disc':
        assert row['state'] == 'partial'


def test_packet_source_stamp_precedes_facade_queue_and_stale_generation(live, monkeypatch):
    owner = live()
    owner.advance(65)
    rt, _ = runtime(owner, monkeypatch)
    connected(rt)
    first = owner.one()['id']
    rt.engine.abort('replacement')
    rt._transmission = None
    rt.engine.transmission_finished(True)
    rt.engine.connect(OTHER)
    rt.send_frame(frame(OTHER, LOCAL, AX.UA, command=False))
    rt._step()
    rows = owner.rows()
    assert len(rows) == 2
    assert rows[first]['positions'] == {'card': 520, 'dac': 325}
    owner.edges.ended(tuple(rows[first]['key']), {'card': 999, 'dac': 999}, 'stale packet callback')
    assert next(row for cid, row in owner.rows().items() if cid != first).get('end_positions') is None


class Client:
    def setblocking(self, value):
        pass

    def close(self):
        pass


@pytest.fixture
def raw(live, monkeypatch):
    from hfmodem.brubru import rx, tx, waveform
    from hfmodem.station import link as link_module

    monkeypatch.setattr(link_module, 'time', NS(monotonic=Clock()))
    monkeypatch.setattr(rx, 'Receiver', lambda: NS())
    monkeypatch.setattr(waveform, 'Decimator', lambda: NS())
    monkeypatch.setattr(tx, 'onset_s', lambda *a: 0)
    monkeypatch.setattr(tx, 'lead_s', lambda *a: 0)
    monkeypatch.setattr(tx, 'transmit', lambda *a, **kw: [NS(
        audio=np.zeros(1, np.float32), lead_s=lambda *a: 0)])

    def make():
        reports = []
        owner = live('off', report=reports.append)
        arbiter = NS(receive_boundary=(0, 0), scan_held=False, disable=lambda *a: None,
                     submit_result=lambda req: TxResult('sent', 'committed', end=0,
                                                        unkey_confirmed=True))
        station = NS(capture=owner.capture, arbiter=arbiter,
                     audio=NS(sample_now=lambda: 0),
                     rig=NS(dial=lambda: 7100000, max_key_s=30),
                     cfg=NS(protocols={'rpr': NS(options={})}, declared_power_w=1,
                            station=NS(mycall=LOCAL), rig=NS(ptt=NS(settle_s=0))))
        link = owner.bind(RprLink(station))
        monkeypatch.setattr(link, '_fresh', lambda: None)
        monkeypatch.setattr(link.access, 'contend', lambda *a: True)
        server = KissServer(link, log=reports.append)
        assert server._accept(Client())
        link.host = server
        return owner, link, server, reports
    return make


def submit(link, server, raw, *, outcome='sent'):
    server._receive(encode(raw))
    request = link._queue.popleft()
    if outcome == 'sent':
        link._serve(request)
    else:
        link._record(request, outcome, 'synthetic refusal')
    return request


@pytest.mark.parametrize('setup', [AX.SABM, 0x6F])
@pytest.mark.parametrize('incoming', [False, True])
def test_raw_kiss_setup_ua_both_ways_only_after_successful_tx(raw, setup, incoming):
    owner, link, server, _ = raw()
    if incoming:
        link.on_frame(frame(PEER, LOCAL, setup))
        assert not owner.rows()
        submit(link, server, frame(LOCAL, PEER, AX.UA, command=False))
    else:
        submit(link, server, frame(LOCAL, PEER, setup))
        assert not owner.rows()
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    row = owner.one()
    assert row['peer'] == PEER and row['key'][1] == 'rpr'
    assert len(owner.starts) == 1
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert len(owner.rows()) == 1
    assert server._out, 'raw KISS receive delivery must survive boundary observation'


def test_raw_tx_boundary_precedes_journal_and_failed_ua_cannot_qualify(raw, monkeypatch):
    owner, link, server, _ = raw()
    link.on_frame(frame(PEER, LOCAL, AX.SABM))
    submit(link, server, frame(LOCAL, PEER, AX.UA, command=False), outcome='refused')
    assert not owner.rows()
    before_journal = []
    record = link._record

    def journal(*args, **kwargs):
        before_journal.append(len(owner.starts))
        return record(*args, **kwargs)

    monkeypatch.setattr(link, '_record', journal)
    link.on_frame(frame(PEER, LOCAL, AX.SABM))
    submit(link, server, frame(LOCAL, PEER, AX.UA, command=False))
    assert before_journal == [1], 'successful UA boundary must precede _record journal work'
    assert owner.one()['peer'] == PEER


@pytest.mark.parametrize('invalid', ['ui', 'unrelated', 'unfinished-route', 'bad-cr', 'bad-pf'])
def test_raw_unqualified_traffic_is_excluded_then_valid_handshake_works(raw, invalid):
    owner, link, server, _ = raw()
    if invalid == 'ui':
        submit(link, server, ui_frame(LOCAL, PEER, b'CQ'))
    elif invalid == 'unrelated':
        link.on_frame(frame(OTHER, PEER, AX.SABM))
        link.on_frame(frame(PEER, OTHER, AX.UA, command=False))
    elif invalid == 'unfinished-route':
        link.on_frame(frame(PEER, LOCAL, AX.SABM, via=(OTHER,), completed=False))
        submit(link, server, frame(LOCAL, PEER, AX.UA, command=False, via=(OTHER,)))
    else:
        submit(link, server, frame(LOCAL, PEER, AX.SABM))
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=invalid == 'bad-cr',
                            pf=invalid != 'bad-pf'))
    assert not owner.rows()
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert owner.one()['peer'] == PEER


@pytest.mark.parametrize('end', ['disc-ua', 'dm', 'frmr', 'detach', 'retune', 'lane-stop'])
def test_raw_observed_close_or_owner_retirement_ends_contact(raw, end):
    owner, link, server, _ = raw()
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    if end == 'disc-ua':
        submit(link, server, frame(LOCAL, PEER, AX.DISC))
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    elif end in ('dm', 'frmr'):
        link.on_frame(frame(PEER, LOCAL, AX.DM if end == 'dm' else AX.FRMR, command=False))
    elif end == 'detach':
        server._disconnect()
    elif end == 'retune':
        link.retuned(1000)
    else:
        link.stop()
    row = owner.one()
    assert len(owner.starts) == len(owner.ends) == 1
    assert row.get('end_positions') is not None
    if end != 'disc-ua':
        assert row['state'] == 'partial'


def test_raw_silent_external_timeout_keeps_uncertainty_until_observed_close(raw):
    owner, link, server, _ = raw()
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    row = owner.one()
    assert row.get('end_positions') is None
    assert row.get('uncertainty'), 'external client timeout is not observable on raw KISS'
    server._disconnect()
    assert owner.one()['state'] == 'partial'


def test_raw_candidate_and_active_overflow_report_without_breaking_kiss(raw):
    owner, link, server, reports = raw()
    tracker = getattr(link, 'recording_tracker', None)
    assert tracker is not None, 'RprLink needs its bounded raw AX.25 boundary observer'
    for n in range(tracker.candidate_limit + 1):
        submit(link, server, frame(LOCAL, f'N{n:05d}', AX.SABM))
    assert len(tracker.candidates) <= tracker.candidate_limit
    owner.rows()
    assert any('overflow' in str(report).lower() for report in reports)
    reports.clear()
    for n in range(tracker.active_limit + 1):
        peer = f'W{n:05d}'
        submit(link, server, frame(LOCAL, peer, AX.SABM))
        link.on_frame(frame(peer, LOCAL, AX.UA, command=False))
    assert len(tracker.active) <= tracker.active_limit
    owner.rows()
    assert any('overflow' in str(report).lower() for report in reports)
    link.on_frame(ui_frame(PEER, LOCAL, b'text remains available'))
    assert server._out


def test_raw_reset_after_traffic_retires_old_generation_but_repeated_setup_does_not(raw):
    owner, link, server, _ = raw()
    for _ in range(2):
        submit(link, server, frame(LOCAL, PEER, AX.SABM))
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    first = owner.one()['id']
    data = Frame((address(LOCAL, flag=True), address(PEER, last=True)), 0, 0xF0, b'payload')
    link.on_frame(data.encode())
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    rows = owner.rows()
    assert len(rows) == 2
    assert rows[first]['state'] == 'partial'
    assert rows[first].get('end_positions') is not None
    assert next(row for cid, row in rows.items() if cid != first).get('end_positions') is None


def test_refused_raw_setup_and_unsolicited_ua_do_not_establish(raw):
    owner, link, server, _ = raw()
    submit(link, server, frame(LOCAL, PEER, AX.SABM), outcome='refused')
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert not owner.rows()
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert owner.one()['peer'] == PEER


def test_tracker_output_overflow_cannot_erase_established_radio_contact(live, monkeypatch):
    owner = live('off')
    rt, attachment = runtime(owner, monkeypatch)
    attachment.host.events.extend((False, 'queued text', '') for _ in range(16))
    connected(rt)
    row = owner.one()
    assert row['state'] == 'partial' and row.get('end_positions') is not None
    assert attachment.failure == 'Tracker link output overflow'
    assert len(owner.starts) == len(owner.ends) == 1


def test_raw_completed_digipeater_route_qualifies_its_local_endpoint(raw):
    owner, link, server, _ = raw()
    submit(link, server, frame(LOCAL, PEER, AX.SABM, via=(OTHER,), completed=False))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False, via=(OTHER,), completed=True))
    row = owner.one()
    assert row['peer'] == PEER and row['key'][3] == LOCAL


def test_raw_attachment_replacement_ignores_old_ua_and_gets_fresh_generation(raw):
    owner, link, server, _ = raw()
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    first = owner.one()['id']
    server._disconnect()
    assert server._accept(Client())
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert len(owner.rows()) == 1
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    rows = owner.rows()
    assert len(rows) == 2
    assert rows[first]['state'] == 'partial'
    successor = next(row for cid, row in rows.items() if cid != first)
    assert successor['key'] != rows[first]['key']


def test_raw_concurrent_peers_end_independently(raw):
    owner, link, server, _ = raw()
    for peer in (PEER, OTHER):
        submit(link, server, frame(LOCAL, peer, AX.SABM))
        link.on_frame(frame(peer, LOCAL, AX.UA, command=False))
    link.on_frame(frame(PEER, LOCAL, AX.DM, command=False))
    rows = owner.rows()
    assert len(rows) == 2 and len(owner.ends) == 1
    assert next(row for row in rows.values() if row['peer'] == OTHER).get('end_positions') is None


def test_raw_setup_without_attached_client_cannot_seed_next_attachment(raw):
    owner, link, server, _ = raw()
    server._disconnect()
    link.on_frame(frame(PEER, LOCAL, AX.SABM))
    assert server._accept(Client())
    submit(link, server, frame(LOCAL, PEER, AX.UA, command=False))
    assert not owner.rows()
    link.on_frame(frame(PEER, LOCAL, AX.SABM))
    submit(link, server, frame(LOCAL, PEER, AX.UA, command=False))
    assert owner.one()['peer'] == PEER


def test_raw_old_transmission_completion_cannot_seed_new_attachment(raw):
    owner, link, server, _ = raw()
    server._receive(encode(frame(LOCAL, PEER, AX.SABM)))
    request = link._queue.popleft()
    server._disconnect()
    assert server._accept(Client())
    link._serve(request)
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert not owner.rows()
    submit(link, server, frame(LOCAL, PEER, AX.SABM))
    link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert owner.one()['peer'] == PEER


def test_connected_packet_lane_stop_retires_its_shared_boundary(raw, monkeypatch):
    owner, link, server, _ = raw()
    server._disconnect()
    rt = PacketRuntime(link, mycall=LOCAL, clock=Clock())
    monkeypatch.setattr(rt, '_call', lambda fn: fn())
    rt.attach('tracker')
    rt.engine.listen = True
    link.host = link.packet_engine = rt
    connected(rt)
    assert len(owner.starts) == 1
    link.stop()
    assert owner.one()['state'] == 'partial'
    assert len(owner.ends) == 1
    rt.engine.abort('packet owner stopped')
    assert len(owner.ends) == 1


def test_same_peer_setup_while_disconnecting_retires_recording_generation(live, monkeypatch):
    owner = live('off')
    rt, attachment = runtime(owner, monkeypatch)
    connected(rt)
    attachment.host.events.clear()
    first = owner.one()['id']
    rt.engine.disconnect()
    assert rt.engine.state is AX.State.DISCONNECTING
    owner.feed(owner.rx, 0, 80)
    rt.engine.on_frame(frame(PEER, LOCAL, AX.SABM))
    rows = owner.rows()
    assert len(rows) == 2
    assert rows[first]['state'] == 'partial'
    assert rows[first]['end_positions']['card'] == 80
    successor = next(row for cid, row in rows.items() if cid != first)
    assert successor['key'][-1] == rt.engine.generation
    assert successor['positions'] == {'card': 80, 'dac': 0}


@pytest.mark.parametrize('pf', [False, True])
@pytest.mark.parametrize('incoming', [False, True])
def test_raw_setup_and_disconnect_match_both_poll_values(raw, pf, incoming):
    owner, link, server, _ = raw()
    if incoming:
        link.on_frame(frame(PEER, LOCAL, AX.SABM, pf=pf))
        submit(link, server, frame(LOCAL, PEER, AX.UA, command=False, pf=not pf))
        assert not owner.rows()
        submit(link, server, frame(LOCAL, PEER, AX.UA, command=False, pf=pf))
    else:
        submit(link, server, frame(LOCAL, PEER, AX.SABM, pf=pf))
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False, pf=not pf))
        assert not owner.rows()
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False, pf=pf))
    assert len(owner.rows()) == 1
    if incoming:
        link.on_frame(frame(PEER, LOCAL, AX.DISC, pf=pf))
        submit(link, server, frame(LOCAL, PEER, AX.UA, command=False, pf=pf))
    else:
        submit(link, server, frame(LOCAL, PEER, AX.DISC, pf=pf))
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False, pf=pf))
    assert owner.one().get('end_positions') is not None


@pytest.mark.parametrize('refusal', [AX.DM, AX.FRMR])
@pytest.mark.parametrize('incoming', [False, True])
def test_raw_refusal_retires_pending_attempt_before_delayed_ua(raw, refusal, incoming):
    owner, link, server, _ = raw()
    if incoming:
        link.on_frame(frame(PEER, LOCAL, AX.SABM))
        submit(link, server, frame(LOCAL, PEER, refusal, command=False))
        submit(link, server, frame(LOCAL, PEER, AX.UA, command=False))
    else:
        submit(link, server, frame(LOCAL, PEER, AX.SABM))
        link.on_frame(frame(PEER, LOCAL, refusal, command=False))
        link.on_frame(frame(PEER, LOCAL, AX.UA, command=False))
    assert not owner.rows()


def test_raw_overflow_with_failed_reporter_preserves_frame_delivery(raw):
    owner, link, server, _ = raw()
    reported = threading.Event()
    threads = []
    def report(message):
        threads.append(threading.get_ident())
        reported.set()
        raise OSError('synthetic reporter failure')
    owner.session.report = report
    link.recording_tracker.candidate_limit = 1
    link.on_frame(frame(PEER, LOCAL, AX.SABM))
    queued = len(server._out)
    link.on_frame(frame(OTHER, LOCAL, AX.SABM))
    assert len(server._out) > queued
    assert reported.wait(2)
    assert threading.get_ident() not in threads
    owner.session.report = lambda message: None
