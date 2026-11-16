# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The ID deadline on a simulated capture clock; no radio or device I/O."""
import json
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.core import cwid
from hfmodem.shrike import onair, p4breakintx, p4probe, p4robusttx
from hfmodem.shrike.p4receiver import Clock as ReceiveClock, Emission, Expect, Kind, Role
from hfmodem.shrike.p4observation import Registered, RegisteredObservation
from hfmodem.shrike.p4turn import CLOSE_TIMEOUT_S, P4Turn, PeerWatch, State
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx
from hfmodem.tests.shrike.test_p4turn import ack_frame, body, machine, word
from hfmodem.tests.shrike.test_identify import Clock, _flown
from hfmodem.tests.shrike.test_grid import _Rig


def closing_cw(tx, stream, tmp_path):
    clock = Clock()
    clock.audio, clock.limit = stream.audio, len(stream.audio)
    clock._advance(stream.pos)
    clock.pos = stream.pos
    cw = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=tx.settle)
    cw.live, cw.p4_probe = clock, tx.p4_probe
    cw.burst_level = (.1, .3)
    cw.aim(tx.raster, 1)
    cw.identify('W9SSJ')
    assert tx.p4_probe.id[-1]['kind'] == 'cw'
    assert not tx.p4_probe.result()['missing_closing_id']
    return tx.p4_probe.id[-1]['end_sample']


@pytest.mark.parametrize('refresh', [False, True])
def test_long_arm_closes_and_finishes_real_cw_before_id_deadline(tmp_path, monkeypatch, refresh):
    stream = Stream(np.zeros(1200*FS, np.float32))
    origin = 10*FS
    stream.pos = origin
    tx = Tx(stream)
    probe = tx.p4_probe
    probe.mycall = probe.cw_call = 'W9SSJ'
    probe.note_id(b'W9SSJ', 'p1', 0, FS)
    tx.p4_link_options.p4_link_timeout = 900
    tx.p4_link_options.p4_data = 'body'
    probe.emissions = [dict(audio_start=origin, audio_end=origin+round(.758*FS))]
    tx.sessrx.p4.note_emission(Emission(Kind.NORMAL, Role.ISS, origin, .758))
    keyed = P4Turn.keyed
    observe = P4Turn.observe
    refreshed = False

    def peer(m, at=None):
        sent = keyed(m, at)
        name = ('V27' if sent.status == 0x20 or sent.value == 37 else
                'V27' if sent.kind == Kind.NORMAL and sent.status & 1 else 'X27')
        wave = p4robusttx.robust_word(name)
        start = sent.stream_sample+round(.920*FS)
        stream.audio[start:start+len(wave)] = wave
        return sent

    def seed(m, *args, **kwargs):
        m._opening = m._opening_wait = False
        m.managed_application = True
        m._idle(origin)
        m.keyed(origin)

    def receive(m, reg, now=None):
        nonlocal refreshed
        result = observe(m, reg, now=now)
        if refresh and not refreshed and now >= origin+300*FS and m.state == State.ISS_HOLD:
            refreshed = True
            tx.p4_link_host.arq.on_host_data(b';FW: W9SSJ\r')
            m._begin_data_turn(m.next_raster(now))
            from hfmodem.shrike.p4waveforms import prepare
            prepare(probe.link_waveforms, m)
        return result

    monkeypatch.setattr(P4Turn, 'seed_opening', seed)
    monkeypatch.setattr(P4Turn, 'keyed', peer)
    monkeypatch.setattr(P4Turn, 'observe', receive)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        origin, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    clocks = [e for e in probe.link_events if e['event'] == 'IdClock']
    assert len(clocks) == (2 if refresh else 1)
    if refresh:
        assert clocks[-1]['deadline_sample'] > clocks[0]['deadline_sample']+300*FS
        assert stream.pos > clocks[0]['deadline_sample']
    close = [e for e in probe.emissions if e.get('status') == 0x20]
    assert close and any(e.get('value') == 37 for e in probe.emissions)
    assert probe.reason == 'close acknowledged'
    assert not any(e['event'] == 'Refused' for e in probe.link_events)

    assert closing_cw(tx, stream, tmp_path) < clocks[-1]['deadline_sample']
    budget = onair._p4_id_budget(tx)
    assert budget['cw_seconds'] == pytest.approx(len(cwid.audio('W9SSJ'))/FS)
    assert budget['margin_seconds'] == 3.75


def test_id_close_keeps_the_accepted_reply_on_its_grid_slot():
    m, a = machine(role='irs', data='body')
    m.observe(body(0), now=round(.8*FS))
    ack = m.decide(round(.846*FS))
    assert ack.kind == Kind.WORD and ack.word == 'X27'
    assert ack.stream_sample == round(.920*FS)
    m.close_for_identification(round(.846*FS))
    sent = m.keyed()
    assert (sent.kind, sent.word, sent.stream_sample) == (ack.kind, ack.word, ack.stream_sample)


@pytest.mark.parametrize('ending', [None, KeyboardInterrupt(), SystemExit('SIGTERM'), RuntimeError('reader failed'),
                                  onair.PttError('key failed')])
def test_arm_probe_and_verdict_account_for_closing_id(tmp_path, monkeypatch, capsys, ending):
    monkeypatch.setattr(p4probe, 'code_revision', lambda *a: None)
    if ending is not None:
        def fail(*args):
            raise ending
        def replay(host):
            stream = Stream(np.zeros(5*FS, np.float32))
            packet = p4breakintx.changeover_packet(b'peer greeting', 0)
            at = round(.923*FS)
            stream.audio[at:at+len(packet)] = packet
            tx = Tx(stream)
            probe = host.peer.p4_probe
            probe.emissions = tx.p4_probe.emissions
            probe.request(0, b'', 0)
            probe.id.append(dict(kind='cw', call='W9SSJ', start_sample=FS,
                                 end_sample=FS+round(.244*FS)))
            tx.p4_probe = probe
            tx._tx = fail
            onair._run_p4_link(tx, stream, onair._MasterGrid(
                0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
        monkeypatch.setattr(onair.PtcHost, 'tick', replay)
    if ending is not None and not isinstance(ending, onair.PttError):
        with pytest.raises(type(ending)):
            _flown(tmp_path, monkeypatch, '--after-connect', 'p1-robust', '--p4-link',
                   '--max-cycles', '1', max_key=40)
    else:
        _flown(tmp_path, monkeypatch, '--after-connect', 'p1-robust', '--p4-link',
               '--max-cycles', '1', max_key=40)
    result = json.loads((tmp_path/'capture/p4-probe.json').read_text())
    assert result['missing_closing_id'] is (ending is not None)
    output = capsys.readouterr().out
    assert ('P4 identification: closing CW ID missing -- send W9SSJ by hand'
            if ending is not None else 'P4 identification: closing CW ID present') in output


def test_interrupted_arm_writes_missing_id_even_if_teardown_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p4probe, 'code_revision', lambda *a: None)
    def interrupt(_host):
        raise KeyboardInterrupt
    def broken_close():
        raise RuntimeError('capture close failed')
    monkeypatch.setattr(onair.PtcHost, 'tick', interrupt)
    clock = Clock()
    monkeypatch.setattr(clock, 'close', broken_close)
    with pytest.raises(KeyboardInterrupt):
        _flown(tmp_path, monkeypatch, '--after-connect', 'p1-robust', '--p4-link',
               '--max-cycles', '1', max_key=40, live=clock)
    result = json.loads((tmp_path/'capture/p4-probe.json').read_text())
    assert result['missing_closing_id']
    assert 'closing CW ID missing -- send W9SSJ by hand' in capsys.readouterr().out


def test_an_earlier_cw_id_does_not_identify_a_later_transmission():
    probe = p4probe.EntryProbe()
    probe.id.append(dict(kind='cw', start_sample=FS, end_sample=2*FS))
    assert not probe.result()['missing_closing_id']
    probe.emissions.append(dict(audio_start=3*FS, audio_end=4*FS))
    assert probe.result()['missing_closing_id']


def test_failed_closing_cw_is_not_reported_as_identified(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p4probe, 'code_revision', lambda *a: None)
    _flown(tmp_path, monkeypatch, '--after-connect', 'p1-robust', '--p4-link',
           '--max-cycles', '1')
    result = json.loads((tmp_path/'capture/p4-probe.json').read_text())
    assert result['missing_closing_id']
    assert 'closing CW ID missing -- send W9SSJ by hand' in capsys.readouterr().out


@pytest.mark.parametrize('data', ['none', 'body', 'long'])
def test_id_close_acquires_the_turn_without_queued_application_bytes(data):
    m, a = machine(role='irs', data=data)
    m.observe(body(0), now=round(.8*FS))
    m.close_for_identification(round(.8*FS))
    request = m.decide(round(.846*FS))
    assert request.kind == Kind.BODY and request.status == 0x40
    assert request.stream_sample == round(.920*FS)
    m.keyed()
    grant_at = request.stream_sample+round(.930*FS)
    m.observe(body(grant_at, b'', 0x40), now=grant_at+round(.8*FS))
    m.decide(grant_at+round(.846*FS))
    m.close_for_identification(grant_at+round(.846*FS))
    close = m.keyed()
    assert close.kind == Kind.BODY and close.status == 0x20
    m.silence(close.stream_sample+FS)
    end = m.keyed()
    assert end.kind == Kind.SHORT and end.value == 37
    m.observe(word('V27', end.stream_sample+round(.930*FS),
                   responds_to=end.stream_sample))
    assert m.state == State.DOWN and m.reason == 'close acknowledged'


def test_id_close_retires_the_peer_watch_instead_of_waiting_for_reoffer():
    m, a = machine(role='irs', data='long')
    m.peer_watch = PeerWatch(0, round(3.75*FS), 0., 1, attempts=4, last_attempt=FS)
    m.next_offer_at = 21*FS
    m.close_for_identification(FS)
    assert m.peer_watch is None and m.next_offer_at is None
    assert m.app_done and m._wants_turn()


@pytest.mark.parametrize('refused', [False, True])
def test_unlimited_receive_hold_finishes_its_reply_then_releases_for_cw(tmp_path, monkeypatch, refused):
    origin = 480*FS
    stream = Stream(np.zeros(650*FS, np.float32))
    stream.pos = origin
    tx = Tx(stream)
    transmit = tx._tx
    refusals = []
    def admit(audio, what):
        if refused and stream.pos >= 540*FS:
            tx.refused = True
            refusals.append(stream.pos)
        else:
            transmit(audio, what)
    monkeypatch.setattr(tx, '_tx', admit)
    probe = tx.p4_probe
    probe.mycall = probe.cw_call = 'W9SSJ'
    probe.note_id(b'W9SSJ', 'p1', 0, FS)
    probe.emissions = [dict(audio_start=origin, audio_end=origin+round(.758*FS))]
    tx.p4_link_options.p4_link_timeout = 900
    tx.p4_link_options.p4_v53_hold = 0
    rx = tx.sessrx.p4
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, origin, .758))
    first = origin+round(.923*FS)
    rx.clock = ReceiveClock(Expect.PACKET, first, .935, 1.25, 0.)
    for k, at in enumerate(range(first, 645*FS, 60000)):
        wave = p4robusttx.robust_short(53, 'B' if k else 'A')
        stream.audio[at:at+len(wave)] = wave
    seed_opening = P4Turn.seed_opening
    def seed(m, *args, **kwargs):
        seed_opening(m, *args, **kwargs)
        m._opening = m._opening_wait = False
        m.state = State.IRS_ANSWER
    monkeypatch.setattr(P4Turn, 'seed_opening', seed)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        origin, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert probe.reason == 'identification close timeout'
    clock = next(e for e in probe.link_events if e['event'] == 'IdClock')
    assert clock['release_at_sample'] <= stream.pos < clock['release_at_sample']+round(clock['packet_seconds']*FS)
    assert clock['close_seconds'] >= CLOSE_TIMEOUT_S+clock['packet_seconds']
    if refused:
        assert len(refusals) > 2 and all(e['audio_start'] < 540*FS for e in probe.emissions)
    replies = probe.emissions[1:]
    assert replies and all(e['word'] == 'U27' for e in replies)
    assert len(replies) == round((replies[-1]['audio_start']-replies[0]['audio_start'])/60000)+1
    assert all(abs(e['audio_start']-(first+k*60000+round(.920*FS))) < round(.035*FS)
               for k, e in enumerate(replies))
    assert closing_cw(tx, stream, tmp_path) < 600*FS


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
@pytest.mark.parametrize('crossing', ['receive', 'selected'])
@pytest.mark.parametrize('expiry', ['sample', 'wall'])
def test_close_expiry_finishes_current_packet_and_owed_reply(kind, crossing, expiry):
    clock = [0.]
    m, _ = machine(role='irs', wall_clock=lambda: clock[0])
    m._body(round(11.250*FS), 0x20, State.ISS_CLOSE)
    m.keyed()
    m.state = State.IRS_ANSWER
    end = m.close_context.end
    due = end+round((.585 if crossing == 'receive' else .020)*FS)
    at = due-round(m.t[kind]*FS)
    packet = ack_frame(at, b'peer data', 1, kind)
    if expiry == 'wall':
        m.close_context.end += 100*FS
    if crossing == 'receive':
        partial = replace(packet.observation, frame=None, provisional=True)
        m.observe(RegisteredObservation(partial, Registered.NEW), now=at+FS//4)
    else:
        m.observe(packet, now=due-m.lead)
        selected = m.decide(due-m.lead)
        assert selected is not None and selected.stream_sample == due
    if expiry == 'wall':
        clock[0] = CLOSE_TIMEOUT_S
    assert m.decide(end) is (None if crossing == 'receive' else selected)
    assert m.state != State.DOWN
    if crossing == 'receive':
        m.silence(end)
        m.observe(replace(packet, outcome=Registered.ENRICHED), now=due-m.lead)
    reply = m.decide(max(end, due-m.lead))
    assert reply is not None and (reply.kind, reply.word, reply.stream_sample) == (Kind.WORD, 'X27', due)
    assert m.can_key(reply)
    assert m.keyed().stream_sample == due
    assert m.decide(due+round(.275*FS)) is None
    assert m.state == State.DOWN and m.reason == 'close deadline'


def test_interrupted_arm_prints_id_verdict_before_failed_probe_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(p4probe, 'code_revision', lambda *a: None)
    def interrupt(_host):
        raise KeyboardInterrupt
    write = onair._capture_write_text
    def broken_write(path, text):
        if path.name == 'p4-probe.json':
            assert 'P4 identification: closing CW ID missing -- send W9SSJ by hand' in capsys.readouterr().out
            raise OSError('probe write failed')
        return write(path, text)
    monkeypatch.setattr(onair.PtcHost, 'tick', interrupt)
    monkeypatch.setattr(onair, '_capture_write_text', broken_write)
    with pytest.raises(OSError, match='probe write failed'):
        _flown(tmp_path, monkeypatch, '--after-connect', 'p1-robust', '--p4-link',
               '--max-cycles', '1', max_key=40)


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
def test_packet_first_registered_after_close_deadline_still_gets_its_reply(kind):
    m, _ = machine(role='irs')
    m._body(round(11.250*FS), 0x20, State.ISS_CLOSE)
    m.keyed()
    m.state = State.IRS_ANSWER
    end = m.close_context.end
    due = end+round(.585*FS)
    packet = ack_frame(due-round(m.t[kind]*FS), b'peer data', 1, kind)
    m.observe(packet, now=due-m.lead)
    reply = m.decide(due-m.lead)
    assert reply is not None and (reply.word, reply.stream_sample) == ('X27', due)
    assert m.keyed().stream_sample == due
    m.silence(due+round(.275*FS))
    assert m.state == State.DOWN and m.reason == 'close deadline'


def test_later_packets_cannot_extend_pending_close_expiry():
    m, _ = machine(role='irs')
    m._body(round(11.250*FS), 0x20, State.ISS_CLOSE)
    m.keyed()
    m.state = State.IRS_ANSWER
    end = m.close_context.end
    first = replace(ack_frame(end-FS, b'peer data', 1, Kind.LONG).observation,
                    frame=None, provisional=True)
    m.observe(RegisteredObservation(first, Registered.NEW), now=end-FS//2)
    assert m.decide(end) is None and m.state != State.DOWN
    later = replace(first, stream_sample=end+FS)
    m.observe(RegisteredObservation(later, Registered.NEW), now=end+FS)
    assert m.decide(end+round(3.75*FS)) is None
    assert m.state == State.DOWN and m.reason == 'close deadline'


def test_close_expiry_preserves_a_selected_close_grant():
    m, _ = machine()
    m._body(0, 0x20, State.ISS_CLOSE)
    close = m.keyed()
    at = round(.935*FS)
    due = at+round(.920*FS)
    m.close_context.end = due-round(.020*FS)
    m.observe(word('Y27', at, extent=.244, responds_to=close.stream_sample),
              now=at+round(.300*FS))
    grant = m.decide(due-m.lead)
    assert grant is not None and grant.purpose == 'close-grant'
    assert m.decide(m.close_context.end) is grant
    assert m.keyed().stream_sample == due
    m.silence(due+round(1.25*FS))
    assert m.state == State.DOWN and m.reason == 'close deadline'


@pytest.mark.parametrize('shape', ['empty', 'bare', 'peer-close'])
@pytest.mark.parametrize('expiry', ['sample', 'wall'])
def test_close_expiry_drains_a_turn_back_reply(shape, expiry):
    wall = [0.]
    m, _ = machine(wall_clock=lambda: wall[0])
    m._body(0, 0x20, State.ISS_CLOSE)
    close = m.keyed()
    request = round(.935*FS)
    m.observe(word('Y27', request, extent=.244, responds_to=close.stream_sample))
    grant = m.decide(request+round(.920*FS)-m.lead)
    assert grant.purpose == 'close-grant'
    m.keyed()
    ctx = m.close_context
    end = ctx.end
    at = end-round(.500*FS)
    due = end+round(.420*FS)
    head = word('Y27', at, partial=True, responds_to=None)
    m.observe(head, now=at+round(.250*FS))
    packet = (body(at, b'', 0x20 if shape == 'peer-close' else 0x40) if shape != 'bare' else
              word('Y27', at, extent=.244, responds_to=None,
                   outcome=Registered.CONFIRMED, replaced=head.observation))
    if expiry == 'wall':
        ctx.end += 100*FS
        wall[0] = CLOSE_TIMEOUT_S
    frozen_end = ctx.end
    assert m.decide(end) is None and m.state != State.DOWN
    m.observe(packet, now=at+round(.800*FS))
    assert ctx.expired_at == end
    boundary = ctx.expired_at
    reply = m.decide(due-m.lead)
    assert reply is not None
    if shape == 'peer-close':
        assert (reply.kind, reply.word, reply.state) == (Kind.WORD, 'V27', State.IRS_CLOSE)
    else:
        assert (reply.kind, reply.status, reply.state) == (Kind.BODY, 0x20, State.ISS_CLOSE)
    assert reply.stream_sample == due
    assert reply.current_slot == at
    assert ctx.resumed is (shape != 'peer-close') and ctx.grants == 1
    m.silence(due-m.lead)
    assert m.state != State.DOWN and m.decide(due-m.lead) is reply
    assert m.can_key(reply)
    assert m.keyed().stream_sample == due
    assert ctx.end == frozen_end and ctx.expired_at == boundary
    span = round((.275 if shape == 'peer-close' else 1.25)*FS)
    assert m.decide(due+span-1) is None
    assert m.state != State.DOWN
    m.silence(due+span)
    assert m.state == State.DOWN and m.reason == 'close deadline'


def test_close_expiry_preserves_a_selected_turn_back_reply():
    m, _ = machine()
    m._body(0, 0x20, State.ISS_CLOSE)
    close = m.keyed()
    request = round(.935*FS)
    m.observe(word('Y27', request, extent=.244, responds_to=close.stream_sample))
    m.decide(request+round(.920*FS)-m.lead)
    grant = m.keyed()
    m.close_context.end = grant.stream_sample+2*FS
    end = m.close_context.end
    due = end+round(.020*FS)
    at = due-round(.920*FS)
    m.observe(body(at, b'', 0x40), now=due-m.lead)
    reply = m.decide(due-m.lead)
    assert reply is not None and reply.current_slot == at
    m.silence(end)
    assert m.state != State.DOWN and m.decide(end) is reply
    assert m.keyed().stream_sample == due
    m.silence(due+round(1.25*FS))
    assert m.state == State.DOWN and m.reason == 'close deadline'


@pytest.mark.parametrize('shape', ['empty', 'bare', 'peer-close'])
def test_id_close_keys_turn_back_reply_and_finishes_cw_before_deadline(tmp_path, monkeypatch, shape):
    stream = Stream(np.zeros(650*FS, np.float32))
    tx = Tx(stream)
    probe = tx.p4_probe
    probe.mycall = probe.cw_call = 'W9SSJ'
    probe.note_id(b'W9SSJ', 'p1', 0, FS)
    tx.p4_link_options.p4_link_timeout = 900
    budget = onair._p4_id_budget(tx)
    reserve = sum(v for k, v in budget.items() if k != 'nominal_close_seconds')
    end = round((600-reserve+budget['turn_seconds']+budget['close_seconds'])*FS)
    at = end-round(.500*FS)
    due = end+round(.420*FS)
    grant_at = at-round(.330*FS)
    stream.pos = grant_at
    probe.emissions = [dict(audio_start=grant_at, audio_end=grant_at+round(.798*FS))]
    wave = (p4breakintx.changeover_packet(b'', 0x20 if shape == 'peer-close' else 0x40)
            if shape != 'bare' else
            p4robusttx.robust_word('Y27'))
    stream.audio[at:at+len(wave)] = wave
    turns = []
    rx = tx.sessrx.p4

    def seed(m, *args, **kwargs):
        turns.append(m)
        m._opening = m._opening_wait = False
        m._body(end-round(CLOSE_TIMEOUT_S*FS), 0x20, State.ISS_CLOSE)
        m.keyed()
        request_at = grant_at-round(.920*FS)
        m._body(request_at-round(.935*FS), 0x20, State.ISS_CLOSE)
        close = m.keyed()
        m.observe(word('Y27', request_at, extent=.244, responds_to=close.stream_sample))
        grant = m.decide(grant_at-m.lead)
        assert grant.purpose == 'close-grant' and grant.stream_sample == grant_at
        m.keyed()
        rx.note_emission(Emission(Kind.BODY, Role.IRS, grant_at, .798,
                                  packet_reply=True), close_grant=True)

    monkeypatch.setattr(P4Turn, 'seed_opening', seed)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        grant_at, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    replies = [e for e in probe.emissions if e.get('kind')]
    assert len(replies) == 1
    assert replies[0]['audio_start'] == due
    if shape == 'peer-close':
        assert replies[0]['word'] == 'V27'
    else:
        assert replies[0]['status'] == 0x20
    ctx = turns[0].close_context
    assert ctx.end == ctx.expired_at == end
    assert ctx.resumed is (shape != 'peer-close') and ctx.grants == 1
    assert turns[0].last.current_slot == at
    assert probe.reason in ('close deadline', 'identification close timeout')
    assert stream.pos >= due+round((.275 if shape == 'peer-close' else 1.25)*FS)
    clock = next(e for e in probe.link_events if e['event'] == 'IdClock')
    assert closing_cw(tx, stream, tmp_path) < clock['deadline_sample'] == 600*FS


@pytest.mark.parametrize('kind', [Kind.NORMAL, Kind.LONG])
@pytest.mark.parametrize('crossing', ['deferred', 'receive'])
def test_close_expiry_drains_a_deferred_reply(kind, crossing):
    m, _ = machine(role='irs')
    m._body(0, 0x20, State.ISS_CLOSE)
    m.keyed()
    m.state = State.IRS_ANSWER
    end = m.close_context.end
    at = end-round((m.t[kind]+.010 if crossing == 'deferred' else .010)*FS)
    due = at+round(m.t[kind]*FS)
    packet = ack_frame(at, b'peer data', 1, kind)
    if crossing == 'receive':
        partial = replace(packet.observation, frame=None, provisional=True)
        m.observe(RegisteredObservation(partial, Registered.NEW), now=at)
        assert m.decide(end) is None and m.state != State.DOWN
    m.observe(replace(packet, outcome=Registered.ENRICHED), now=due+1)
    assert m.decide(due+1) is None
    assert m._deferred is not None
    later = m._deferred[1]['due']
    assert later == due+round(3.75*FS if kind == Kind.LONG else 1.25*FS)
    assert m.decide(max(end, due+1)) is None and m.state != State.DOWN
    if kind == Kind.LONG and crossing == 'receive':
        assert m.decide(end+round(3.75*FS)) is None and m.state != State.DOWN
    reply = m.decide(later-m.lead)
    assert reply is not None and (reply.word, reply.stream_sample) == ('X27', later)
    assert m.can_key(reply)
    m.keyed()
    assert m.decide(later+round(.100*FS)) is None and m.state != State.DOWN
    m.silence(later+round(.275*FS))
    assert m.state == State.DOWN and m.reason == 'close deadline'


@pytest.mark.parametrize('registered', [False, True])
@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('onset_lead', [.800, .010])
def test_id_release_drains_an_incrementally_acquired_long(tmp_path, monkeypatch, registered, late, onset_lead):
    stream = Stream(np.zeros(650*FS, np.float32))
    stream.wall = 0
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    tx = Tx(stream)
    probe = tx.p4_probe
    probe.mycall = probe.cw_call = 'W9SSJ'
    probe.note_id(b'W9SSJ', 'p1', 0, FS)
    tx.p4_link_options.p4_link_timeout = 900
    tx.p4_link_options.p4_rx_scan = 'incremental'
    budget = onair._p4_id_budget(tx)
    reserve = sum(v for k, v in budget.items() if k != 'nominal_close_seconds')
    release = round((600-reserve+budget['turn_seconds']+budget['close_seconds'])*FS)
    at = release-round(onset_lead*FS)
    due = at+round(3.400*FS)
    stream.pos = release-FS
    probe.emissions = [dict(audio_start=stream.pos, audio_end=stream.pos+round(.758*FS))]
    rx = tx.sessrx.p4
    packet = ack_frame(at, b'peer data', 1, Kind.LONG)
    acquired = completed = False

    def seed(m, *args, **kwargs):
        m._opening = m._opening_wait = False
        m.state = State.IRS_ANSWER
        rx.clock = None

    def acquire(*, now, **kwargs):
        nonlocal acquired
        if acquired or now < release:
            return []
        acquired = True
        rx.clock = ReceiveClock(Expect.PACKET, at, .330, 3.75, 0.)
        result = []
        if registered:
            rx._emit(replace(packet.observation, frame=None, provisional=True), result)
        return result

    def finish(*args, **kwargs):
        nonlocal completed
        if completed or stream.pos < at+round(3.300*FS):
            return []
        completed = True
        if late:
            stream.wall = due+1
        result = []
        rx._emit(packet.observation, result)
        return result

    monkeypatch.setattr(P4Turn, 'seed_opening', seed)
    monkeypatch.setattr(rx, 'scan_idle', acquire)
    monkeypatch.setattr(rx, 'read_partial', lambda *a, **kw: [])
    monkeypatch.setattr(rx, 'read_window', finish)
    monkeypatch.setattr(rx, 'refine', lambda **kw: [])
    monkeypatch.setattr(rx, 'prepare_sf8', lambda *a: None)
    monkeypatch.setattr(rx, 'prepare_long', lambda *a: None)
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        stream.pos, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert acquired and completed
    assert probe.reason == 'identification close timeout'
    replies = [e for e in probe.emissions if e.get('kind')]
    # Every LONG passes enqueue admission before source delivery, so a copy
    # finished past that limit earns no deferred ACK: its slot asks for the
    # packet again (V27 under the toggle). A timely copy is accepted and
    # answered with the release's break-in word.
    assert [(e['audio_start'], e['word']) for e in replies] == [
        (due, 'V27' if late else 'Y27')]
    assert not [e for e in probe.link_events if e['event'] == 'answer-deferred']
    accepted = [e for e in probe.link_events if e['event'] == 'source-accept']
    assert len(accepted) == (0 if late else 1)
    keyed, = [e for e in probe.link_events if e['event'] == 'Keyed']
    assert (keyed['purpose'], keyed['current_slot_accepted'], keyed['state']) == (
        ('repeat-request', False, State.IRS_ANSWER) if late
        else (None, True, State.IRS_BREAKIN))
    clock = next(e for e in probe.link_events if e['event'] == 'IdClock')
    assert clock['release_at_sample'] == release
    assert stream.pos < release+round(budget['packet_seconds']*FS)
    assert replies[0]['audio_end'] < release+round(budget['packet_seconds']*FS)
    assert budget['packet_seconds'] >= 3.400+3.750+len(p4robusttx.robust_word('X27'))/FS
    assert budget['close_seconds'] >= CLOSE_TIMEOUT_S+budget['packet_seconds']
    assert closing_cw(tx, stream, tmp_path) < clock['deadline_sample'] == 600*FS
