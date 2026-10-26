# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Actual dispatch branch/work recorder and genuine WORD-origin speculation."""
from dataclasses import asdict
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from hfmodem.shrike import onair, p4turn, p4chirp, p4longtx
from hfmodem.shrike.p4receiver import Expect, Emission, Role
from hfmodem.tests.shrike.test_p4_part_a_1011 import packet
from hfmodem.shrike.p4observation import Kind
from hfmodem.tests.shrike.test_p4_link_0927 import Tx, FS
from hfmodem.tests.shrike.test_p4_opus_driver_1011 import ChargedStream, GRID, test_f2_real_primary_retains_parent_budget as primary
from hfmodem.tests.shrike.reader_owner_support_1011 import record


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


@pytest.mark.parametrize('kind', ['long', 'chirp'])
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
    tx.p4_link_options.p4_ladder, tx.p4_link_options.p4_data = 'none', 'body'
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
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        seed_at = onset-44880-44160
        m.observe(packet(seed_at, Kind.BODY, 0x40), now=seed_at+38640)
        previous = m.decide(onset-44880-m.lead)
        assert previous.kind == Kind.BODY and previous.state == p4turn.State.ISS_GRANT
        sent = m.keyed(previous.stream_sample)
        from hfmodem.shrike import p4breakintx
        duration = len(p4breakintx.changeover_packet(sent.payload, sent.status))/FS
        rx.note_emission(Emission(sent.kind, Role.ISS, sent.stream_sample, duration, packet_reply=True))
        assert rx.clock.expects == Expect.WORD and rx.clock.next_due == onset
        record('word_promotion_actual_history', proposal=asdict(sent), duration=duration,
               label_context='semantic actual selected/keyed ordinary BODY grant before real PCM suffix')
        return m
    with patch.object(p4turn, 'P4Turn', factory), DispatchTrace() as trace:
        onair._run_p4_link(tx, live, GRID)
    record('word_speculation', kind=kind, reads=reads, branches=trace.rows, charges=live.charges,
           heard=tx.p4_probe.heard, emissions=tx.p4_probe.emissions)
    assert any(r['before']['expects'] == Expect.WORD for r in reads)
    assert any(r['end'] >= packet_at+157440 and r['records'] for r in reads)
    assert any(r.get('status') == 0x62 and not r['provisional'] for r in tx.p4_probe.heard)
    assert tx.p4_probe.emissions and tx.p4_probe.emissions[0]['word'] in ('X27', 'V27')
    assert any(r['before']['expects'] == Expect.WORD and r['after']['expects'] == Expect.PACKET for r in reads)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0
