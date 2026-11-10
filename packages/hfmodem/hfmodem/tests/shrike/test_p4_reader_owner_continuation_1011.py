# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Charged genuine-reader continuation after natural request render/selection."""
from dataclasses import asdict
import hashlib
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.shrike import onair, p4turn, p4receiver, p4chirp, p4robusttx, p4waveforms
from hfmodem.shrike.p4receiver import Clock, Expect
from hfmodem.tests.shrike.test_p4_link_0927 import Tx, FS
from hfmodem.tests.shrike.test_p4_opus_driver_1011 import ChargedStream, GRID
from hfmodem.tests.shrike.reader_owner_support_1011 import record, freeze_guard


def counters(m, a):
    w = m.peer_watch
    return (w.attempts if w else None, w.last_attempt if w else None,
            m.next_offer_at, m.breakins, m.grants, bytes(a._outbuf),
            m.outbox.inflight, a.tx_acked_bytes)


@pytest.mark.parametrize('managed', [False, True])
def test_c3_natural_zero_key_continuation_and_distinct_final_witness(managed, monkeypatch):
    onset, cycle = 480000, 180000
    origin, stop = onset-24000, onset+5*cycle
    audio = np.zeros(stop-origin, np.float32)
    initial = p4chirp.entry_packet(b'\x1e'*22, 0x41)
    continuation = p4chirp.entry_packet(b'new peer source\r', 0x62)
    for at, wave in [(onset, initial), (onset+2*cycle, continuation)]:
        audio[at-origin:at-origin+len(wave)] = wave
    live = ChargedStream(audio, origin)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    args = tx.p4_link_options
    args.p4_ladder, args.p4_data, args.p4_link_timeout = 'none', 'body', len(audio)/FS
    a = tx.p4_link_host.arq
    a.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, onset, .330, 3.75, 0.)
    rx.expected = True
    machines, selected, final_guards, keys, timeline, delivered = [], [], [], [], [], []

    def event(stage, **fields):
        row = dict(order=len(timeline), stage=stage, capture=live.pos,
                             now=live.sample_now(), **fields)
        timeline.append(row)
        record('c3_timeline', managed=managed, **row)

    def deliver(b):
        delivered.append(b)
        event('source_callback', data=b.hex(), counters=counters(machines[0], a))
    tx.p4_link_host.deliver = deliver
    for name in ('read_partial', 'read_window', 'read_chirp', 'prepare_sf8', 'prepare_long'):
        original = getattr(rx, name)
        def read(*args, _name=name, _original=original, **kw):
            at, owner, expects = rx.clock.next_due, rx._watch_read_generation, rx.clock.expects
            event('read_start', method=_name, at=at, owner=owner, expects=expects,
                  support=[args[1], args[1]+len(args[0])])
            result = _original(*args, **kw)
            cost = .030 if owner is not None and abs(at-(onset+cycle)) <= 48 and _name == 'read_window' else .001
            live.charge(_name, cost)
            event('read_end', method=_name, at=at, owner=owner, cost=cost,
                  results=[asdict(r) for r in result] if isinstance(result, list) else bool(result))
            return result
        setattr(rx, name, read)
    for name in ('set_receive_phase', 'sync_peer_watch'):
        original = getattr(rx, name)
        def delegate(*args, _name=name, _original=original, **kw):
            event(_name+'_before', values=str(args), keywords=kw)
            result = _original(*args, **kw)
            event(_name+'_after', clock=asdict(rx.clock) if rx.clock else None,
                  watch=asdict(rx.peer_watch) if rx.peer_watch else None)
            return result
        setattr(rx, name, delegate)
    cls = p4turn.P4Turn
    def factory(*args, **kw):
        kw.update(role='irs', managed_application=managed)
        m = cls(*args, **kw)
        machines.append(m)
        observe, decide, admit = m.observe, m.decide, m.can_key
        def feed(reg, **kw):
            prior = m._next
            event('observe_before', reg=asdict(reg), proposal=asdict(prior) if prior else None,
                  counters=counters(m, a))
            result = observe(reg, **kw)
            live.charge('feed', .001)
            event('observe_after', proposal=asdict(m._next) if m._next else None,
                  pending=str(m.pending), deferred=str(m._deferred), counters=counters(m, a))
            return result
        def decision(*args, **kw):
            event('decide_before', args=args, counters=counters(m, a))
            p = decide(*args, **kw)
            live.charge('decide', .001)
            event('decide_after', proposal=asdict(p) if p else None, counters=counters(m, a))
            return p
        def admission(p):
            before = counters(m, a)
            ok = admit(p)
            event('admission', proposal=id(p), due=p.stream_sample, ok=ok,
                  current=id(m._next), before=before, after=counters(m, a))
            return ok
        m.observe, m.decide, m.can_key = feed, decision, admission
        return m
    build = onair._p4_link_waveforms
    class Cache(dict):
        def get(self, key, default=None):
            wave = super().get(key, default)
            m = machines[0]
            p = m._next
            if wave is not None and p is not None:
                event('render_selected', proposal=id(p), fields=asdict(p), cache_key=str(key),
                      waveform=hashlib.sha256(wave.tobytes()).hexdigest(), authorized=m.can_key(p),
                      counters=counters(m, a))
                selected.append((p, wave, len(timeline)-1))
            return wave
    def render(*args, **kw):
        cache = Cache(build(*args, **kw))
        live.charge('render', .002)
        return cache
    original_prepare = p4waveforms.prepare
    def prepare(*args, **kw):
        result = original_prepare(*args, **kw)
        live.charge('waveform_prepare', .001)
        return result
    sink = tx._tx
    probes = []
    def key(wave, what):
        m = machines[0]
        fn = tx.p4_tx_admit
        assert fn is not None
        frozen, cells = freeze_guard(fn)
        assert cells['waveform'] is wave and cells['machine'] is m and cells['aimed'] is tx.raster
        fn()
        current = m._next
        assert current is cells['proposal'] if managed else current == cells['proposal']
        assert np.array_equal(wave, p4robusttx.robust_word(current.word))
        assert live.sample_now()+live.key_notice <= current.stream_sample
        if current.word == 'Y27':
            final_guards.append((current, frozen, cells, wave))
        elif final_guards and not probes:
            # Distinct witnesses: earlier genuinely installed final guard has
            # one actual key; naturally initial-canceled work has zero total.
            stale, retained, oldcells, oldwave = final_guards[0]
            unkeyed = next(p for p, w, order in selected if p.word == 'Y27' and all(p is not k for k in keys))
            before = counters(m, a), m._next, m._accepted_proposal, len(tx.slots_used), len(keys)
            assert sum(p is stale for p in keys) == 1
            assert all(p is not unkeyed for p in keys)
            for witness, boundary, captured, guard in [
                ('natural-unkeyed', 'initial', unkeyed, lambda: m.can_key(unkeyed)),
                ('prior-installed', 'initial', stale, lambda: m.can_key(stale)),
                ('prior-installed', 'final', stale, retained)]:
                entered = []
                if boundary == 'final':
                    assert retained.__code__ is fn.__code__
                    assert retained.__globals__ is fn.__globals__
                    assert retained.__defaults__ is fn.__defaults__
                    assert retained.__kwdefaults__ is fn.__kwdefaults__
                    assert all(c.cell_contents is oldcells[n] for n, c in zip(retained.__code__.co_freevars, retained.__closure__))
                    assert oldcells['proposal'] is stale and oldcells['waveform'] is oldwave
                    with pytest.raises(onair._MissedTxSlot, match='proposal no longer keyable'):
                        guard()  # Cancellation predicate precedes any expiry/sink check.
                        entered.append(oldwave)
                else:
                    assert guard() is False
                assert not entered
                assert before == (counters(m, a), m._next, m._accepted_proposal, len(tx.slots_used), len(keys))
                probes.append(dict(witness=witness, boundary=boundary, proposal=id(captured),
                    prior_keys=sum(p is captured for p in keys), additional_keys=0, sink_entries=0,
                    before=before[0], after=counters(m, a), current=id(m._next)))
        event('sink_after_final_admit', proposal=id(current), fields=asdict(current),
              waveform=hashlib.sha256(wave.tobytes()).hexdigest(), counters=counters(m, a))
        sink(wave, what)
        keys.append(current)
    tx._tx = key
    base = onair.time.monotonic()
    mono = lambda: base+(live.sample_now()-live.origin)/FS
    monkeypatch.setattr(onair.time, 'monotonic', mono)
    monkeypatch.setattr(p4receiver, 'monotonic', mono)
    with patch.object(p4turn, 'P4Turn', factory), patch.object(onair, '_p4_link_waveforms', render), patch.object(p4waveforms, 'prepare', prepare):
        onair._run_p4_link(tx, live, GRID)
    m = machines[0]
    record('c3_charged_continuation', managed=managed, timeline=timeline, probes=probes,
           charges=live.charges, lead=m.lead, driver_lead=onair._prekey_lead(live, round(tx.settle*FS)),
           key_notice=live.key_notice, input_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
           keys=[dict(identity=id(p), fields=asdict(p)) for p in keys],
           events=tx.p4_probe.link_events, emissions=tx.p4_probe.emissions)
    canceled = [(p, wave, order) for p, wave, order in selected if p.word == 'Y27' and all(p is not k for k in keys)]
    assert len(canceled) == 1, 'natural selected/rendered Y27 must cancel before its first actual key'
    p, wave, order = canceled[0]
    missed = next(r for r in timeline if r['stage'] == 'read_end' and r['owner'] is not None and abs(r['at']-(onset+cycle)) <= 48 and r['method'] == 'read_window')
    nominal = missed['at']+163200
    assert missed['now'] > nominal-m.lead
    assert p.stream_sample == nominal+cycle
    cancel_read = next(r for r in timeline if r['stage'] == 'read_start' and r['owner'] is not None and abs(r['at']-(onset+2*cycle)) <= 48)
    assert order < cancel_read['order']
    assert cancel_read['support'][1] == cancel_read['at']+161040
    refusals = [r for r in timeline if r['stage'] == 'admission' and r['proposal'] == id(p) and not r['ok']]
    assert refusals and all(r['before'] == r['after'] for r in refusals)
    assert sum(k.word == 'Y27' for k in keys) == 1
    answers = [k for k in keys if k.accepted_target is not None and abs(k.accepted_target-(onset+2*cycle)) <= 48]
    assert answers and all(k.state == p4turn.State.IRS_ANSWER for k in answers)
    assert answers[0].ack_identity[1].rstrip(b'\x1e') == b'new peer source\r'
    assert delivered == [b'new peer source\r']
    assert len(probes) == 3 and [r['prior_keys'] for r in probes] == [0, 1, 1]
    assert a.tx_acked_bytes == 0 and not m.outbox.inflight and bytes(a._outbuf) == b'TEST\r'
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert len(keyed) == len(keys) == len(tx.p4_probe.emissions)
    for k, e, emission in zip(keys, keyed, tx.p4_probe.emissions):
        assert k.kind == p4turn.Kind.WORD and k.state != p4turn.State.ISS_HOLD
        assert k.stream_sample == e['at'] == emission['audio_start']
        assert k.word == e['word'] == emission['word']
        assert k.accepted_target == e['accepted_target'] and k.current_slot == e['current_slot']
        assert hashlib.sha256(p4robusttx.robust_word(k.word).tobytes()).hexdigest() == emission['waveform_sha256']
