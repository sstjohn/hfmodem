# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual dispatch branch/work recorder and genuine WORD-origin speculation."""
from dataclasses import asdict
from pathlib import Path
import sys
from functools import partial
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.shrike import onair, p4budget, p4turn, p4chirp, p4longtx
from hfmodem.shrike.p4receiver import Expect, Emission, Role
from hfmodem.tests.shrike.test_p4_part_a_1011 import packet
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike.test_p4_link_0927 import Tx, FS
from hfmodem.tests.shrike.test_p4_opus_driver_1011 import ChargedStream, GRID, test_f2_real_primary_retains_parent_budget as primary
from hfmodem.tests.shrike.reader_owner_support_1011 import record
from hfmodem.tests.timing import within


class DispatchTrace:
    def __init__(self):
        self.rows = []
        source = Path(onair.__file__).read_text().splitlines()
        self.lines = {i+1:s.strip() for i,s in enumerate(source) if s.strip().startswith((
            'if not borrowed_word_delay:', 'if not capture(max(live.pos, full_read)):',
            'read_cost = round(', 'due = _p4_link_due(', 'breakin_long_generation = machine.'))}
    def trace(self, frame, event, arg):
        if frame.f_code.co_filename != onair.__file__ or frame.f_code.co_name != '_run_p4_link':
            return None
        if event == 'line' and frame.f_lineno in self.lines:
            v = frame.f_locals
            live, rx = v.get('live'), v.get('receiver')
            m = v.get('machine')
            if live and rx and m:
                row = dict(line=frame.f_lineno, source=self.lines[frame.f_lineno],
                    capture=live.pos, now=live.sample_now(),
                    clock=asdict(rx.clock) if rx.clock else None,
                    local_clock=asdict(v['clock']) if v.get('clock') else None,
                    owner_same=rx.clock is v.get('clock'), watch=asdict(rx.peer_watch) if rx.peer_watch else None,
                    heads=v.get('heads'), pending=v.get('pending'), onset=v.get('onset'),
                    due=v.get('due'), decision=v.get('decision'), full_read=v.get('full_read'),
                    borrowed=v.get('borrowed_word_delay'), delay=v.get('delay'),
                    completed_generation=v.get('breakin_long_generation'))
                self.rows.append(row)
                record('dispatch_branch', **row)
        return self.trace
    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self.trace)
        return self
    def __exit__(self, *args):
        sys.settrace(self.previous)


@pytest.mark.parametrize('branch,cost', [('sf8', .040), ('head', .025), ('other', .030), ('placed', .020)])
def test_r1_primary_dispatch_entry_and_complete_charged_work(branch, cost, monkeypatch):
    with DispatchTrace() as trace:
        primary(branch, cost, True, monkeypatch)
    starts = [r for r in trace.rows if r['source'].startswith('if not capture(max(live.pos, full_read))')]
    assert starts and all(r['borrowed'] is False for r in starts)
    assert all(r['clock']['expects'] == Expect.PACKET or r['heads'] for r in starts)
    # Primary helper asserts the actual charged readers, complete endpoints,
    # preparations, can_key boundaries and first feasible genuine sink.
    record('primary_dispatch_work', branch=branch, cost=cost, rows=trace.rows)


# The long is read by the LONG engine, whose Python fallback answers on time only
# when the machine keeps up.
@pytest.mark.parametrize('kind', [pytest.param('long', marks=pytest.mark.realtime), 'chirp'])
def test_r1_word_origin_real_packet_evidence_still_completes(kind):
    onset, origin = 480000, 456000
    wave = (p4chirp.entry_packet(b'ordinary promoted\r', 0x62) if kind == 'chirp' else
            p4longtx.long_packet(b'ordinary promoted\r', 0x62))
    audio = np.zeros(5*FS, np.float32)
    packet_at = onset
    audio[onset-origin:onset-origin+len(wave)] = wave
    live = ChargedStream(audio, origin)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_link_timeout = len(audio)/FS
    rx = tx.sessrx.p4
    # Semantic prior history is a genuine selected/keyed ordinary BODY grant,
    # not a LONG invitation or a fabricated IRS WORD. Its real note_emission
    # installs WORD ownership; PCM readers alone promote that owner thereafter.
    rx.expected = True
    reads = []
    for name in ('read_partial', 'read_window', 'read_chirp'):
        original = getattr(rx, name)
        def read(audio, origin, _name=name, _original=original, **kw):
            before = asdict(rx.clock)
            result = _original(audio, origin, **kw)
            live.charge(_name, .001)
            reads.append(dict(method=_name, before=before, after=asdict(rx.clock),
                              end=origin+len(audio), now=live.sample_now(), records=[asdict(r) for r in result] if result else []))
            return result
        setattr(rx, name, read)
    jobs = []
    start_long = rx._start_long
    def started(key, *a, **kw):
        entry = start_long(key, *a, **kw)
        if all(entry is not job for job in jobs):
            jobs.append(entry)
        return entry
    rx._start_long = started
    engine = []
    advance_long = rx._advance_long
    def advanced(entry, **kw):
        seen = len(rx.live_events)
        out = advance_long(entry, **kw)
        event = rx.live_events[-1] if len(rx.live_events) > seen else {}
        if event.get('event') == 'long-engine' and not event['relaxed']:
            # Budgets only shrink, so what is left just after the call is a
            # floor on what the engine and the publication check saw.
            limits = entry['limits']
            engine.append(dict(event, search=entry['search'], spent=entry['search'].elapsed,
                               work_left=limits.work.remaining() if limits else None,
                               publication_left=limits.publication.remaining() if limits else None))
        return out
    rx._advance_long = advanced
    deadlines = []
    class Granted(p4budget.LongDeadline):
        def __init__(self, fs, onset, *a, **kw):
            super().__init__(fs, onset, *a, **kw)
            self.onset, self.granted = onset, self.work.wall_deadline-self.work.wall_now()
            deadlines.append(self)
    history = []
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        seed_at = onset-44880-44160
        m.observe(packet(seed_at, Kind.BODY, 0x40), now=seed_at+38640)
        previous = m.decide(onset-44880-m.lead)
        assert previous.kind == Kind.BODY and previous.state == p4turn.State.ISS_GRANT
        sent = m.keyed(previous.stream_sample)
        history.append(sent.stream_sample)
        from hfmodem.shrike import p4breakintx
        duration = len(p4breakintx.changeover_packet(sent.payload, sent.status))/FS
        rx.note_emission(Emission(sent.kind, Role.ISS, sent.stream_sample, duration, packet_reply=True))
        assert rx.clock.expects == Expect.WORD and rx.clock.next_due == onset
        record('word_promotion_actual_history', proposal=asdict(sent), duration=duration,
               label_context='semantic actual selected/keyed ordinary BODY grant before real PCM suffix')
        return m
    with (patch.object(p4turn, 'P4Turn', factory), patch.object(p4budget, 'LongDeadline', Granted),
          DispatchTrace() as trace):
        onair._run_p4_link(tx, live, GRID)
    record('word_speculation', kind=kind, reads=reads, branches=trace.rows, charges=live.charges,
           heard=tx.p4_probe.heard, emissions=tx.p4_probe.emissions)
    print(dict(kind=kind, reads=reads, branches=trace.rows, charges=live.charges))
    assert any(r['before']['expects'] == Expect.WORD for r in reads)
    assert any(r['end'] >= packet_at+157440 and r['records'] for r in reads)
    assert any(r['before']['expects'] == Expect.WORD and r['after']['expects'] == Expect.PACKET for r in reads)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
    if kind == 'long':
        # Every LONG deadline at the packet starts with a live wall grant; one
        # already expired is a defect, not a slow host.
        at_packet = [d for d in deadlines if abs(d.onset-packet_at) < FS//20]
        assert at_packet and min(d.granted for d in at_packet) > 0, \
            f'LONG wall grants at the packet: {[d.granted for d in at_packet]}'
        # A decoded LONG can still be published until the stream refuses a key
        # for the turn's reply slot, T[LONG] after the packet: key_notice before
        # it, less a few milliseconds of settling.
        key = packet_at+round(p4turn.T[Kind.LONG]*FS)-live.key_notice
        published = [d.publication.sample_deadline for d in at_packet]
        assert all(key-round(.010*FS) <= at <= key for at in published), (key, published)
        here = [job for job in jobs if abs(job['obs'].stream_sample-packet_at) < FS//20]
        assert here, 'no LONG engine job was started at the packet'
        # The ordinary grant's WORD owner promotes through real prefix PCM.
        # Its new LONG deadline must schedule preparation before the final
        # engine grant, rather than retaining the former SHORT decision time.
        events = rx.live_events
        prepared = next(i for i, event in enumerate(events)
                        if event['event'] == 'sf8 prepare'
                        and abs(event['at']-packet_at) < FS//20)
        finalized = next(i for i, event in enumerate(events)
                         if event['event'] == 'long-engine'
                         and abs(event['at']-packet_at) < FS//20)
        assert prepared < finalized
        if not any(event['event'] == 'long-engine' and event['timely'] and event['crc_valid']
                   and abs(event['at']-packet_at) < FS//20 for event in events):
            # No timely frame. The link must still repeat the BODY grant it
            # last keyed, whole cycles after it.
            first = tx.p4_probe.emissions[0] if tx.p4_probe.emissions else None
            assert first and (first['kind'], first['status'], first['word'], first['refused']) == (
                Kind.BODY, 0x40, None, None)
            assert (first['proposed_at']-history[0]) % round(p4turn.CYCLE[Kind.BODY]*FS) == 0
            found = [row for row in engine if row['crc_valid'] and abs(row['at']-packet_at) < FS//20]
            if found:
                # The frame was in hand inside the live grant. Publication is
                # checked after re-encoding, so a slow finish is late only if
                # its time had really run out; a refusal with time left is the
                # gate's fault.
                left = found[0]['publication_left']
                assert left is not None and left < 0, f'a CRC-valid LONG went unpublished with {left} s left'
                miss = partial(within, left, 'publication time left once the live engine held the frame',
                               at_least=0)
            else:
                # Lifted, the same search must still read the packet. Being here
                # means it needed engine time past what the live grant gave.
                lifted = ((job, job['search'].advance()) for job in here)
                job, result = next(((job, result) for job, result in lifted
                                    if result.frame is not None and result.frame.status.byte == 0x62),
                                   (None, None))
                assert result is not None, 'no LONG search reads the promoted packet even without a deadline'
                # The live engine must have stopped because its work budget was
                # spent, not on an early limit. It may stop short by its 15 ms
                # side-read reserve plus one bounded step it would not start.
                rows = [row for row in engine if row['search'] is job['search']]
                assert rows and rows[-1]['work_left'] <= .030, [row['work_left'] for row in rows]
                spent = max(row['spent'] for row in rows)
                miss = partial(within, result.elapsed_seconds, 'engine time the promoted LONG needed, '
                               'against what its live grant allowed', at_most=spent)
            within((first['keyed_at']-first['proposed_at'])/FS, 'the repeat keyed after its proposed slot',
                   at_most=0)
            miss()
    assert any(r.get('status') == 0x62 and not r['provisional'] for r in tx.p4_probe.heard)
    assert tx.p4_probe.emissions and tx.p4_probe.emissions[0]['word'] in (
        ('X27',) if kind == Kind.LONG else ('X27', 'V27'))
