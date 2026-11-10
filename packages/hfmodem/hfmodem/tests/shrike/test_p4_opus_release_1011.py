# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Charged short grant enrichment wins the unrelated long-watch deadline."""
from unittest.mock import patch
import numpy as np
from hfmodem.shrike import onair, p4turn, p4breakintx, p4robusttx, p4waveforms
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Clock, Expect, Emission, Role
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
from hfmodem.tests.shrike.test_p4_opus_driver_1011 import ChargedStream, GRID
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, packet


def test_charged_checked_body_release_keeps_primary_920_opportunity():
    grant, due = 8236750, 8280910
    start, stop = KEY+12000, due+3*FS
    audio = np.zeros(stop-start, np.float32)
    wave = p4breakintx.changeover_packet(b'', 0x40)
    audio[grant-start:grant-start+len(wave)] = wave
    ack_at, ack = due+44880, p4robusttx.robust_word('X27')
    audio[ack_at-start:ack_at-start+len(ack)] = ack
    live = ChargedStream(audio, start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    tx.p4_link_options.p4_ladder, tx.p4_link_options.p4_data = 'none', 'body'
    tx.p4_link_options.p4_link_timeout = (stop-start)/FS
    tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx, supports, machines, keys = tx.sessrx.p4, [], [], []
    for name in ('read_partial', 'read_window'):
        original = getattr(rx, name)
        def real(*a, _name=name, _original=original, **kw):
            owner, onset = rx._watch_read_generation, rx.clock.next_due
            result = _original(*a, **kw)
            live.charge(_name, .015 if _name == 'read_window' else .001)
            supports.append((_name, onset, owner, a[1], a[1]+len(a[0]), live.sample_now()))
            return result
        setattr(rx, name, real)
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        m.observe(packet(), now=AT+158400)
        m.decide(KEY-m.lead)
        assert m.keyed().word == 'Y27'
        rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, 0.)
        rx.sync_peer_watch(m.peer_watch)
        rx.note_emission(Emission(Kind.WORD, Role.ISS, KEY, .244))
        observe, can_key, decide = m.observe, m.can_key, m.decide
        def feed(*a, **kw):
            live.charge('feed', .001)
            return observe(*a, **kw)
        def admit(p):
            live.charge('admission', .001)
            return can_key(p)
        m.observe, m.can_key = feed, admit
        def decision(*a, **kw):
            result = decide(*a, **kw)
            live.charge('decide', .001)
            return result
        m.decide = decision
        machines.append(m)
        return m
    emit = tx._tx
    def key(audio, what):
        if tx.p4_tx_admit: tx.p4_tx_admit()
        p = machines[0]._next
        assert live.sample_now()+live.key_notice <= p.stream_sample
        if p.payload: assert tx.p4_link_host.arq.tx_acked_bytes == 0
        emit(audio, what)
        keys.append((id(p), p.stream_sample, p.status, p.payload, live.sample_now()))
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
    body = next(r for r in tx.p4_probe.heard if r['kind'] == Kind.BODY and r.get('status') == 0x40)
    assert abs(body['onset_sample']-grant) <= 48 and body['responds_to'] == KEY
    payload = next(row for row in keys if row[3] == b'TEST\r')
    assert abs(payload[1]-due) <= 48 and payload[2] == 0
    assert not any(owner is not None and now < payload[4] for name, at, owner, begin, end, now in supports)
    assert machines[0].peer_watch is None and tx.p4_link_host.arq.tx_acked_bytes == 5
    print(dict(source='synthetic BODY40 and synthetic full ACK waveforms, real readers',
               supports=supports, keys=keys, charges=live.charges))
