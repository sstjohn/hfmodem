# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""C driver acceptance; all capture samples, including historical TX, retained."""
import hashlib
import json
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.tests.evidence import CAPTURES, require_input
from hfmodem.shrike import onair, p4turn, p4breakintx, p4robusttx, p4chirp
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Clock, Expect, Emission, Role
from hfmodem.tests.shrike.test_p4_link_0927 import Tx, Stream, FS
from hfmodem.tests.shrike.test_p4_part_a_1011 import AT, KEY, packet
from hfmodem.tests.shrike.p4_declared_clock_1111 import declared_time


def assert_long42_copies(heard, events):
    """Pin every recovered physical copy and its own accepted answer."""
    rows = [r for r in heard if r['kind'] == Kind.LONG and r.get('status') == 0x42
            and r.get('data') == '1e'*40 and not r['provisional']]
    # Complete-copy remembered agreement retains the same exact SL2 identity
    # for both edge copies.
    assert [r['onset_sample'] for r in rows] == pytest.approx(
        [8568325, 8748327, 8928324, 9108324, 9288324], abs=48)
    engines = [e for e in events if e['event'] == 'long-engine'
               and e['crc_valid'] and e['timely'] and not e['relaxed']]
    # Retain the original three-copy correction-count evidence as well as
    # the expanded physical-copy and answer checks below.
    for nominal in (AT+180000*k for k in (4, 5, 6)):
        assert any(abs(e['at']-nominal) <= 1680 and e['corrected_bits'] == 0
                   for e in engines)
    keyed = [e for e in events if e['event'] == 'Keyed']
    for row in rows:
        assert row['speedlevel'] == '2' and row['placed']
        data = bytes.fromhex(row['data'])
        assert len(data) == 40 and hashlib.sha256(data).hexdigest() == (
            '412153b93609307dee339415db04279f3579a6aeb1d5f42cb36eebd7da76bc50')
        assert any(abs(e['at']-row['onset_sample']) <= 48
                   and (e['state'] == 'unique' and e['tier'] >= 0
                        or e['state'] == 'repeat' and e['tier'] is None)
                   for e in engines)
        answers = [e for e in keyed if e.get('accepted_target') == row['onset_sample']]
        assert answers and all(e['state'] == p4turn.State.IRS_ANSWER
                               and e['word'] == 'V27' for e in answers)
        assert any(e['at'] == row['onset_sample']+163200
                   and e['current_slot_accepted']
                   and abs(e['current_slot']-row['onset_sample']) <= 48
                   for e in answers)
    return rows


def selected_watch_support(rx, at):
    """The physical form/head visible before this particular reader call."""
    heads = [row.stream_sample for row in rx.observations
             if row.kind == Kind.CHIRP and abs(row.stream_sample-at) < FS//10
             and rx.has_chirp_header(row.stream_sample)]
    if heads:
        return 'chirp', heads[-1]
    if rx.has_normal_long(at):
        obs = next(row['obs'] for key, row in rx._long_cache.items()
                   if row.get('kind') == 'normal' and row.get('obs') is not None
                   and abs(key[0]-at) < FS//20)
        return 'normal', obs.stream_sample
    return ('sf8' if rx.sf8_prepared() else 'sf16'), at


def assert_watch_support(selected, at, end):
    family, physical = selected
    # The selected form determines actual complete support. A placed
    # chirp must be measured from its header, rather than the predicted slot.
    seconds = dict(chirp=p4chirp.PACKET_S, normal=3.300, sf8=3.280, sf16=3.330)[family]
    assert end >= physical+round(seconds*FS), (selected, at, end)


class AbsoluteStream(Stream):
    def __init__(self, audio, origin):
        super().__init__(audio)
        self.pos = self.origin = origin

    def take_until(self, end):
        end = min(end, self.origin+len(self.audio))
        out = self.audio[self.pos-self.origin:end-self.origin]
        self.pos = end
        return out


@pytest.mark.parametrize('historical', [False, True])
def test_recording_c_real_dispatch(historical):
    path = require_input(CAPTURES/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590'/'stream.flac')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '880a6ab6541fd0c22e0f747fa5a3f3270b2c82301c59eabe98b789d93056f47e'
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert metadata['samplerate'] == FS and metadata['samples'] == 9524096
    # load_wav calls the existing artifact reader/identity validation and FLAC decoder.
    source = onair.session.load_wav(str(path), FS)
    assert len(source) == 9524096 and np.isfinite(source).all()
    stream = AbsoluteStream(source[8016000:9504000], 8016000)
    tx = Tx(stream)
    tx.settle = .025
    tx.p4_probe.emissions = []
    tx.p4_probe.peer_breakin = True

    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_link_timeout = 31
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, -.19)
    rx.cfo_hz = -.19
    rx.expected = True
    supports = []
    selections = {}
    cycles = []
    finish_cycle = rx.finish_cycle
    def counted(onset):
        previous = rx._counted_cycle
        before = rx.silent_cycles
        result = finish_cycle(onset)
        cycles.append((onset, stream.pos, previous, before, result))
        return result
    rx.finish_cycle = counted
    for name in ('read_partial', 'read_window'):
        method = getattr(rx, name)
        def read(audio, origin, _method=method, _name=name, **kw):
            owner = rx._watch_read_generation
            at, expect = rx.clock.next_due, rx.clock.expects
            selected = selected_watch_support(rx, at)
            result = _method(audio, origin, **kw)
            row = (_name, expect, at, origin, origin+len(audio), owner,
                             stream.sample_now(), [(r.observation.kind, r.observation.stream_sample,
                                 r.observation.frame.status.byte if r.observation.frame else None)
                                 for r in result])
            supports.append(row)
            selections[id(row)] = selected
            return result
        setattr(rx, name, read)
    machines = []
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        machines.append(m)
        if historical:
            keyed = m.keyed
            injected = False
            def external(at=None):
                nonlocal injected
                if not injected:
                    assert m._next.stream_sample == KEY
                    injected = True
                    # Historical external key, after the fixed planner's WORD admission.
                    m._next = p4turn.Proposal(Kind.BODY, KEY, status=0x40, state=p4turn.State.ISS_GRANT)
                return keyed(at)
            m.keyed = external
        return m
    if historical:
        emit = tx._tx
        def historical_key(audio, what):
            if not tx.slots_used:
                audio = p4breakintx.changeover_packet(b'', 0x40)
            return emit(audio, what)
        tx._tx = historical_key
        note = rx.note_emission
        def historical_note(em, **kw):
            from dataclasses import replace
            if em.stream_sample == KEY:
                em = replace(em, duration_s=(tx.tx_end-KEY)/FS)
            return note(em, **kw)
        rx.note_emission = historical_note
    # This instantaneous capture fixture has one declared
    # sample clock, including arithmetic and every publication/key deadline.
    with declared_time(stream.sample_now, fs=FS), patch.object(p4turn, 'P4Turn', factory):
        onair._run_p4_link(tx, stream, onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    heard = tx.p4_probe.heard
    longs = assert_long42_copies(heard, tx.p4_probe.link_events)
    assert not any(r.get('status') == 0x41 and 8200000 < r['onset_sample'] < 8250000 for r in heard)
    # The two earlier unread windows remain negative controls; copies3/7
    # become positive through the exact engine/agreement above.
    for nominal in (AT+180000*k for k in (1, 2)):
        assert not any(r.get('status') is not None and r['kind'] in (Kind.LONG, Kind.CHIRP)
                       and abs(r['onset_sample']-nominal) <= 1680 for r in heard)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    assert stream.pos == 9504000 and machines[0].reason in ('session ceiling', 'capture ended')
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert all(e['kind'] == Kind.WORD and e['state'] == p4turn.State.IRS_ANSWER
               for e in keyed[1 if historical else 0:])
    if not historical:
        assert (keyed[0]['kind'], keyed[0]['word'], keyed[0]['at']) == (Kind.WORD, 'X27', KEY)
    assert not any(e['event'] == 'Decision' and e.get('state') == p4turn.State.DOWN and e['at'] < 9504000 for e in tx.p4_probe.link_events)
    for nominal in (AT+180000*k for k in range(1, 8)):
        full = [r for r in supports if r[0] == 'read_window' and r[1] == Expect.PACKET
                and abs(r[2]-nominal) <= 1680 and r[4] >= r[2]+157440]
        assert full, (nominal, supports)
        for row in full:
            _, expect, at, origin, end, owner, feed_at, outcomes = row
            assert origin <= max(8016000, at-24000)
            if owner is not None:
                assert_watch_support(selections[id(row)], at, end)
            else:
                # Coincident PACKET uses the parent's first full read; a later
                # conditional retry/refinement reports its actual support too.
                assert end >= at+157440
    if historical:
        assert any(expect == Expect.WORD and at == 8236750
                   for _, expect, at, origin, end, owner, feed_at, outcomes in supports)
        assert any(owner is not None and expect == Expect.PACKET and end >= at+round(3.330*FS)
                   for _, expect, at, origin, end, owner, feed_at, outcomes in supports)
    assert cycles
    for onset, end, previous, before, after in cycles:
        if previous is None or onset > previous+FS//10:
            assert any(r[1] == Expect.PACKET and abs(r[2]-onset) <= 1680
                       and r[4] <= end and r[4] >= r[2]+157440 for r in supports)
        else:
            assert after == before
    print({'source': str(path), 'interval': [8016000, 9504000], 'onsets': [r['onset_sample'] for r in longs],
           'historical': historical, 'supports': supports, 'cycles': cycles, 'reason': machines[0].reason})


def test_primary_checked_body_enrichment_keeps_920_data_opportunity():
    grant, due = 8236750, 8280910
    start, stop = KEY+12000, due+3*FS
    audio = np.zeros(stop-start, np.float32)
    body = p4breakintx.changeover_packet(b'', 0x40)
    audio[grant-start:grant-start+len(body)] = body
    # Synthetic full payload ACK waveform, genuinely decoded by the receiver.
    ack = p4robusttx.robust_word('X27')
    ack_at = due+44880
    audio[ack_at-start:ack_at-start+len(ack)] = ack
    live = AbsoluteStream(audio, start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions = []
    tx.p4_probe.peer_breakin = True
    args = tx.p4_link_options
    args.p4_data, args.p4_link_timeout = 'body', (stop - start) / FS
    tx.p4_link_host.arq.on_host_data(b'TEST\r')
    rx = tx.sessrx.p4
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        seed = packet()
        m.observe(seed, now=AT+158400)
        m.decide(KEY-m.lead)
        assert m.keyed().word == 'Y27'
        rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, 0.)
        rx.sync_peer_watch(m.peer_watch)
        rx.note_emission(Emission(Kind.WORD, Role.ISS, KEY, .244))
        return m
    with patch.object(p4turn, 'P4Turn', factory):
        onair._run_p4_link(tx, live, onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    body = next(r for r in tx.p4_probe.heard if r['kind'] == Kind.BODY and r.get('status') == 0x40)
    assert abs(body['onset_sample']-grant) <= 48 and body['responds_to'] == KEY
    emitted = next(e for e in tx.p4_probe.link_events if e['event'] == 'Keyed' and e.get('payload') == b'TEST\r'.hex())
    assert emitted['status'] == 0 and abs(emitted['at']-due) <= 48
    assert tx.p4_link_host.arq.tx_acked_bytes == 5


def test_final_admission_rechecks_current_rendered_peer_span():
    """Synthetic final-admission phase displacement, after initial admission."""
    from dataclasses import replace
    start, stop = AT+158400, KEY+FS
    live = AbsoluteStream(np.zeros(stop-start, np.float32), start)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions = []
    tx.p4_probe.peer_breakin = True
    args = tx.p4_link_options
    args.p4_data, args.p4_link_timeout = 'body', (stop - start) / FS
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, AT, .330, 3.75, 0.)
    cls, machines = p4turn.P4Turn, []
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        m.observe(packet(), now=start)
        machines.append(m)
        return m
    attempted = []
    def emit(audio, what):
        m = machines[0]
        p = m._next
        assert m.peer_span_fits(p, len(audio), 4800)
        # A newer protected phase can arrive between the two admission checks.
        # Isolate span from identity/generation; never shorten the rendered word.
        m.peer_phase = replace(m.peer_phase, onset=m.peer_phase.onset-960)
        assert not m.peer_span_fits(p, len(audio), 4800)
        attempted.append((p.stream_sample, len(audio)))
        tx.p4_tx_admit()
        pytest.fail('final protected-span guard admitted an overlapping waveform')
    tx._tx = emit
    with patch.object(p4turn, 'P4Turn', factory):
        onair._run_p4_link(tx, live, onair._MasterGrid(0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    assert attempted and attempted[0][0] == KEY and not tx.slots_used
    assert any(e['event'] == 'Refused' and e['at'] == KEY for e in tx.p4_probe.link_events)
