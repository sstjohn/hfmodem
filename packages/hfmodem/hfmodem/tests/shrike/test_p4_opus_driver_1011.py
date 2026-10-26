# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Charged real-reader and actual-key controls, without radio or decoder patches."""
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.shrike import onair, p4chirp, p4longtx, p4turn, p4receiver, p4robusttx, p4waveforms
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
from hfmodem.tests.shrike.test_p4_part_a_driver_1011 import AbsoluteStream
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, CYCLE, packet

CORPUS = CAPTURES
GRID = onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800)


class ChargedStream(AbsoluteStream):
    def __init__(self, audio, origin):
        super().__init__(audio, origin)
        self.work = origin
        self.charges = []
        self.key_notice = round(.020*FS)

    def sample_now(self):
        return max(self.pos, self.work)

    def clamp_late(self, at):
        return max(0, self.sample_now()+self.key_notice-at)

    def charge(self, name, seconds):
        before = self.sample_now()
        self.work = before+round(seconds*FS)
        self.charges.append((name, self.pos, before, self.work))


def input_for(branch):
    onset = 10*FS
    if branch == 'sf8':
        path = require_input(CORPUS/'onair-1002-2154/stream.flac')
        source = onair.session.load_wav(str(path), FS)
        at = 3528586
        return source[at-24000:at+168000], at, .69, str(path)
    if branch == 'other':
        path = require_input(CORPUS/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590/stream.flac')
        source = onair.session.load_wav(str(path), FS)
        at = 8748670
        return source[at-24000:at+168000], at, -.19, str(path)
    wave = (p4chirp.entry_packet(b'hello\r', 0x61) if branch == 'placed'
            else p4longtx.long_packet(b'hello\r', 0x61))
    audio = np.zeros(192000, np.float32)
    audio[24000:24000+len(wave)] = wave
    return audio, onset, 0., 'synthetic '+branch+' waveform, real decode'


@pytest.mark.parametrize('branch,cost', [('sf8', .040), ('head', .025), ('other', .030), ('placed', .020)])
@pytest.mark.parametrize('coincident', [False, True])
def test_f2_real_primary_retains_parent_budget(branch, cost, coincident, monkeypatch):
    audio, onset, cfo, source = input_for(branch)
    live = ChargedStream(audio, onset-24000)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions = []
    tx.p4_probe.peer_breakin = True
    tx.p4_link_options.p4_ladder, tx.p4_link_options.p4_data = 'none', 'body'
    tx.p4_link_options.p4_link_timeout = len(audio)/FS
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, onset, .330, 3.75, cfo)
    rx.cfo_hz, rx.expected = cfo, True
    reads, preparations, admissions, keys, machines = [], [], [], [], []
    for name in ('read_partial', 'read_window', 'read_chirp', 'prepare_sf8', 'prepare_long'):
        original = getattr(rx, name)
        def real(*a, _name=name, _original=original, **kw):
            clock = rx.clock
            end = a[1]+len(a[0])
            started = live.sample_now()
            result = _original(*a, **kw)
            seconds = cost if _name in ('read_window', 'read_chirp') and end >= onset+157000 else .001
            live.charge(_name, seconds)
            row = (_name, clock.expects, clock.next_due, a[1], end, started,
                   live.sample_now(), [(r.observation.kind, r.observation.stream_sample,
                                       r.observation.frame.status.byte if r.observation.frame else None)
                                      for r in result] if isinstance(result, list) else bool(result))
            (reads if _name.startswith('read') else preparations).append(row)
            return result
        setattr(rx, name, real)
    if not coincident:
        # Parent primary-formula comparator only: remove the optional side-watch
        # dispatcher from this fixture. No source/reader/result/CRC alteration.
        rx.sync_peer_watch = lambda *a, **kw: None
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        m.first_body = False
        seed_onset = onset+(1250 if branch == 'head' else 0)-CYCLE
        m.observe(packet(seed_onset, Kind.LONG, 0x62), now=seed_onset+158400)
        m.decide(seed_onset+163200-m.lead)
        m.keyed()
        rx.sync_peer_watch(m.peer_watch)
        observe, can_key, decide = m.observe, m.can_key, m.decide
        def feed(*a, **kw):
            result = observe(*a, **kw)
            live.charge('feed', .001)
            return result
        def admit(p):
            live.charge('admission', .001)
            ok = can_key(p)
            admissions.append((id(p), p.stream_sample, p.accepted_target, live.sample_now(), ok))
            return ok
        m.observe, m.can_key = feed, admit
        def decision(*a, **kw):
            result = decide(*a, **kw)
            live.charge('decide', .001)
            return result
        m.decide = decision
        machines.append(m)
        return m
    original_emit = tx._tx
    def key(audio, what):
        if tx.p4_tx_admit: tx.p4_tx_admit()
        p = machines[0]._next
        assert live.sample_now()+live.key_notice <= p.stream_sample
        original_emit(audio, what)
        keys.append((id(p), p.stream_sample, p.accepted_target, p.word, live.sample_now()))
    tx._tx = key
    build = onair._p4_link_waveforms
    def render(*a, **kw):
        result = build(*a, **kw)
        live.charge('render', .002)
        return result
    prepare = p4waveforms.prepare
    def cache_prepare(*a, **kw):
        result = prepare(*a, **kw)
        live.charge('waveform_prepare', .001)
        return result
    # The processing clock is monotonic and separate from capture position.
    base = onair.time.monotonic()
    mono = lambda: base+(live.sample_now()-live.origin)/FS
    monkeypatch.setattr(onair.time, 'monotonic', mono)
    monkeypatch.setattr(p4receiver, 'monotonic', mono)
    with patch.object(p4turn, 'P4Turn', factory), patch.object(onair, '_p4_link_waveforms', render), \
         patch.object(p4waveforms, 'prepare', cache_prepare):
        onair._run_p4_link(tx, live, GRID)
    full = [r for r in reads if r[0] in ('read_window', 'read_chirp') and r[4] >= onset+157000]
    print(dict(branch=branch, coincident=coincident, source=source, reads=reads,
               preparations=preparations, admissions=admissions, keys=keys, charges=live.charges))
    assert full
    expected = {'sf8': 157440, 'head': 160560, 'other': 159840,
                'placed': round(p4chirp.PACKET_S*FS)}[branch]
    # Refined heads can legitimately move the primary onset; report the actual endpoint.
    head = next((r.observation.stream_sample for r in rx.registrations
                 if r.observation.kind == Kind.CHIRP), onset)
    schedule_origin = head if branch in ('head', 'placed') else onset
    assert full[0][4] == schedule_origin+expected
    if branch == 'sf8': assert any(e['event'] == 'sf8-long' and e.get('crc_valid') for e in rx.live_events)
    assert keys and keys[0][2] is not None
    heard = next((r for r in tx.p4_probe.heard if r.get('status') is not None), None)
    deadline = (head if branch == 'head' else heard['onset_sample'])+163200
    if branch == 'head':
        # This genuine wideband candidate has no chirp CRC. Its parent ordinary
        # unread branch must retain the held policy answer, without source credit.
        assert keys[0][2] == onset+1250-CYCLE and machines[0].unread > 0
    assert keys[0][1] == deadline, 'first feasible primary answer must survive charged real work'
    assert sum(ok and ident == keys[0][0] for ident, due, target, now, ok in admissions) >= 2
    assert all(now+live.key_notice <= due for ident, due, target, now, ok in admissions if ok)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0


@pytest.mark.parametrize('boundary', ['initial', 'final'])
@pytest.mark.parametrize('managed', [False, True])
def test_f3_driver_never_keys_captured_canceled_request(boundary, managed):
    start, stop = 8016000, KEY+2*CYCLE+FS
    path = require_input(CORPUS/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590/stream.flac')
    source = onair.session.load_wav(str(path), FS)
    live = AbsoluteStream(source[start:stop], start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    args = tx.p4_link_options
    args.p4_ladder, args.p4_data, args.p4_link_timeout = 'none', 'body', (stop-start)/FS
    tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    rx.cfo_hz = -.19
    rx.expected = True
    cls, machines, captured, keyed = p4turn.P4Turn, [], [], []
    def cancel(m, p):
        captured.append(p)
        m.allow_breakin = False
        m._recover_peer_turn(p.stream_sample)
    def factory(*a, **kw):
        kw['role'], kw['managed_application'] = 'irs', managed
        m = cls(*a, **kw)
        machines.append(m)
        can_key = m.can_key
        def admit(p):
            if boundary == 'initial' and not captured and p.word == 'Y27': cancel(m, p)
            return can_key(p)
        m.can_key = admit
        return m
    emit = tx._tx
    def key(audio, what):
        m = machines[0]
        p = m._next
        if boundary == 'final' and not captured and p.word == 'Y27': cancel(m, p)
        if tx.p4_tx_admit: tx.p4_tx_admit()
        assert np.array_equal(audio, p4robusttx.robust_word(p.word)), 'waveform/current proposal mismatch'
        emit(audio, what)
        keyed.append(p)
    tx._tx = key
    with patch.object(p4turn, 'P4Turn', factory):
        onair._run_p4_link(tx, live, GRID)
    assert captured and not any(p is captured[0] for p in keyed)
    assert keyed and all(p.word == 'X27' and p.state == p4turn.State.IRS_ANSWER for p in keyed)
    events = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert len(events) == len(keyed)
    assert all((e['at'], e['word']) == (p.stream_sample, p.word) for e, p in zip(events, keyed))
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    print(dict(boundary=boundary, managed=managed, captured=[id(p) for p in captured],
               actual=[(id(p), p.stream_sample, p.word) for p in keyed]))


@pytest.mark.parametrize('dispatch,cost', [(True, .004), (True, .040), (False, .004)])
def test_f2_independent_c_watch_real_recovery(dispatch, cost, monkeypatch):
    path = require_input(CORPUS/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590/stream.flac')
    source = onair.session.load_wav(str(path), FS)
    live = ChargedStream(source[8016000:9504000], 8016000)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    args = tx.p4_link_options
    args.p4_ladder, args.p4_data, args.p4_link_timeout = 'none', 'body', 31
    tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    rx.cfo_hz, rx.expected = -.19, True
    supports, feeds, keys, cycles, machines = [], [], [], [], []
    for name in ('read_partial', 'read_window', 'read_chirp', 'prepare_sf8', 'prepare_long'):
        original = getattr(rx, name)
        def read(*a, _name=name, _original=original, **kw):
            at, owner = rx.clock.next_due, rx._watch_read_generation
            start = live.sample_now()
            result = _original(*a, **kw)
            live.charge(_name, cost if owner is not None and _name == 'read_window' else .001)
            supports.append((_name, at, owner, a[1], a[1]+len(a[0]), start,
                             live.sample_now(), [(r.observation.kind, r.observation.stream_sample,
                              r.observation.frame.status.byte if r.observation.frame else None)
                              for r in result] if isinstance(result, list) else bool(result)))
            return result
        setattr(rx, name, read)
    if not dispatch: rx.sync_peer_watch = lambda *a, **kw: None
    finish = rx.finish_cycle
    def counted(at):
        previous, before = rx._counted_cycle, rx.silent_cycles
        result = finish(at)
        cycles.append((at, live.sample_now(), previous, before, result))
        return result
    rx.finish_cycle = counted
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        machines.append(m)
        observe, can_key, decide = m.observe, m.can_key, m.decide
        def feed(reg, **kw):
            live.charge('feed', .001)
            prior = m.state
            result = observe(reg, **kw)
            feeds.append((reg.observation.stream_sample, reg.peer_watch_generation,
                          prior, m.state, live.sample_now(), m._next,
                          m._deferred[0] if m._deferred else None))
            return result
        def admit(p):
            live.charge('admission', .001)
            return can_key(p)
        m.observe, m.can_key = feed, admit
        def decision(*a, **kw):
            result = decide(*a, **kw)
            live.charge('decide', .001)
            return result
        m.decide = decision
        return m
    emit = tx._tx
    def key(audio, what):
        if tx.p4_tx_admit: tx.p4_tx_admit()
        p = machines[0]._next
        assert live.sample_now()+live.key_notice <= p.stream_sample
        emit(audio, what)
        keys.append((id(p), p.stream_sample, p.word, p.state, p.accepted_target, live.sample_now()))
    tx._tx = key
    build = onair._p4_link_waveforms
    def render(*a, **kw):
        result = build(*a, **kw)
        live.charge('render', .002)
        return result
    prepare = p4waveforms.prepare
    def cache_prepare(*a, **kw):
        result = prepare(*a, **kw)
        live.charge('waveform_prepare', .001)
        return result
    base = onair.time.monotonic()
    mono = lambda: base+(live.sample_now()-live.origin)/FS
    monkeypatch.setattr(onair.time, 'monotonic', mono)
    monkeypatch.setattr(p4receiver, 'monotonic', mono)
    with patch.object(p4turn, 'P4Turn', factory), patch.object(onair, '_p4_link_waveforms', render), \
         patch.object(p4waveforms, 'prepare', cache_prepare):
        onair._run_p4_link(tx, live, GRID)
    print(dict(source=str(path), interval=[8016000,9504000], dispatch=dispatch, cost=cost,
               supports=supports, feeds=feeds, keys=keys, cycles=cycles, charges=live.charges))
    longs = [r for r in tx.p4_probe.heard if r['kind'] == Kind.LONG and r.get('status') == 0x42]
    if not dispatch:
        assert not longs, 'dispatch-disabled fixture must lose the claimed independent recovery'
        return
    assert [r['onset_sample'] for r in longs] == pytest.approx([8748327,8928325,9108324], abs=48)
    first = longs[0]['onset_sample']
    recovered = [row for row in feeds if row[0] == first and row[1] is not None]
    assert recovered and recovered[0][2] == p4turn.State.IRS_BREAKIN and recovered[0][3] == p4turn.State.IRS_ANSWER
    assert [(r['onset_sample'], r['peer_watch_generation']) for r in tx.p4_probe.heard
            if r.get('peer_watch_generation') is not None] == [
                (r[0], r[1]) for r in feeds if r[1] is not None]
    assert any(owner is not None and abs(at-8748670) <= 1680 and end >= at+161040
               for name, at, owner, begin, end, start, now, results in supports)
    answers = [row for row in keys if row[4] == first]
    assert answers and answers[0][1] == first+(163200 if cost < .020 else 343200)
    assert all(row[2] == 'X27' and row[3] == p4turn.State.IRS_ANSWER for row in answers)
    assert not any(row[3] == p4turn.State.IRS_BREAKIN for row in keys if row[1] > recovered[0][4])
    assert tx.p4_link_host.arq.tx_acked_bytes == 0 and not machines[0].outbox.inflight
    assert bytes(tx.p4_link_host.arq._outbuf) == b'TEST\r'
    assert live.pos == 9504000
    for at, now, previous, before, after in cycles:
        if previous is not None and abs(at-previous) < FS//10: assert before == after


@pytest.mark.parametrize('advance', ['wall_only', 'capture_ahead'])
def test_f3_application_expiry_at_final_key_with_separate_clocks(advance):
    start, stop = 8016000, KEY+2*CYCLE+FS
    path = require_input(CORPUS/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590/stream.flac')
    source = onair.session.load_wav(str(path), FS)
    live = ChargedStream(source[start:stop], start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    tx.p4_link_options.p4_ladder, tx.p4_link_options.p4_data = 'none', 'body'
    tx.p4_link_options.p4_link_timeout = (stop-start)/FS
    tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    rx.cfo_hz = -.19
    rx.expected = True
    wall = [0.]
    for name in ('read_partial', 'read_window', 'read_chirp', 'prepare_sf8', 'prepare_long'):
        original = getattr(rx, name)
        def real(*a, _name=name, _original=original, **kw):
            result = _original(*a, **kw)
            live.charge(_name, .001)
            return result
        setattr(rx, name, real)
    cls, machines, captured, keys = p4turn.P4Turn, [], [], []
    def factory(*a, **kw):
        kw.update(role='irs', managed_application=True,
                  wall_clock=(lambda: wall[0]) if advance == 'wall_only' else
                             (lambda: live.sample_now()/FS))
        m = cls(*a, **kw)
        m.update_application_progress(20. if advance == 'wall_only' else KEY/FS)
        observe, can_key, decide = m.observe, m.can_key, m.decide
        def feed(*a, **kw):
            result = observe(*a, **kw)
            live.charge('feed', .001)
            return result
        def admit(*a, **kw):
            live.charge('admission', .001)
            return can_key(*a, **kw)
        def decision(*a, **kw):
            result = decide(*a, **kw)
            live.charge('decide', .001)
            return result
        m.observe, m.can_key, m.decide = feed, admit, decision
        machines.append(m)
        return m
    emit = tx._tx
    def key(audio, what):
        m = machines[0]
        p = m._next
        if not captured and p.peer_watch_generation is not None:
            captured.append(p)
            before = live.sample_now()
            if advance == 'wall_only':
                wall[0] = 20.
                assert live.sample_now() == before
            else:
                work = live.work
                live.take_until(p.stream_sample+1)
                assert live.pos > work and live.work == work
        tx.p4_tx_admit()
        emit(audio, what)
        keys.append(p)
    tx._tx = key
    build, prepare = onair._p4_link_waveforms, p4waveforms.prepare
    def render(*a, **kw):
        result = build(*a, **kw)
        live.charge('render', .002)
        return result
    def cache_prepare(*a, **kw):
        result = prepare(*a, **kw)
        live.charge('waveform_prepare', .001)
        return result
    with patch.object(p4turn, 'P4Turn', factory), patch.object(onair, '_p4_link_waveforms', render), \
         patch.object(p4waveforms, 'prepare', cache_prepare):
        onair._run_p4_link(tx, live, GRID)
    assert captured and not any(p is captured[0] for p in keys)
    assert machines[0].app_done and machines[0]._closing_failure and machines[0].peer_watch is None
    assert bytes(tx.p4_link_host.arq._outbuf) == b'TEST\r'
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    print(dict(advance=advance, captured=id(captured[0]),
               keys=[(id(p), p.stream_sample, p.word, p.state) for p in keys], charges=live.charges))
