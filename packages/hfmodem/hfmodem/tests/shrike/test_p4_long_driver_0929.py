# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import numpy as np
import pytest
from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, Observation
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx


def exchange(monkeypatch, *, offset=0., established=False, short=False):
    live = Stream(np.zeros(10*FS))
    live.pos = round(.9*FS)
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 8.
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, FS, .330, 3.75 if established else 1.25, 0.)
    starts = [FS+round(offset*FS), FS+round((offset+3.75)*FS)]
    if short:
        starts = [FS]
    turn = p4turn.P4Turn
    def irs(*a, **kw):
        return turn(*a, **{**kw, 'role': 'irs'})
    monkeypatch.setattr(p4turn, 'P4Turn', irs)
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    def packet(audio, origin, partial=False, long_fallback=False):
        at = next((s for s in starts if abs(s-rx.clock.next_due) < FS//10), None)
        if at is None:
            return None
        available = origin+len(audio)-at
        if established and partial:
            return None
        duration = .758 if short else 3.28
        needed = .80 if short else .81 if partial else 3.30
        if available < round(needed*FS):
            return None
        return Observation(Kind.SHORT if short else Kind.LONG, FS, origin, at-origin,
                           at, duration, 0., .8, provisional=partial,
                           value=13 if short else None, walsh_peak=64 if short else None)
    monkeypatch.setattr(rx, '_packet', packet)
    original = Tx._tx
    def key(audio, what):
        original(tx, audio, what)
        live.pos = tx.tx_end
    tx._tx = key
    onair._run_p4_link(tx, live, onair._MasterGrid(0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800))
    return tx, starts


def test_late_long_head_is_rechecked_before_committing_the_short_deadline(monkeypatch):
    tx, starts = exchange(monkeypatch, offset=.025)
    assert [e['audio_start'] for e in tx.p4_probe.emissions] == [at+round(3.4*FS) for at in starts]
    assert all(e['word'] == 'X27' for e in tx.p4_probe.emissions)


def test_established_long_clock_keeps_its_deadline_when_partial_read_is_absent(monkeypatch):
    tx, starts = exchange(monkeypatch, established=True)
    assert [e['audio_start'] for e in tx.p4_probe.emissions] == [at+round(3.4*FS) for at in starts]


def test_positive_short_can_replace_the_long_cycle_deadline(monkeypatch):
    tx, starts = exchange(monkeypatch, established=True, short=True)
    assert tx.p4_probe.emissions[0]['audio_start'] == starts[0]+round(.92*FS)


@pytest.mark.parametrize('cost_ms', [0, 34, 80])
@pytest.mark.parametrize('previous_chirp', [False, True])
def test_recorded_long_repeats_reach_their_own_driver_deadlines(monkeypatch, cost_ms, previous_chirp):
    from pathlib import Path
    path = Path.cwd()/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG long recording absent')
    audio = onair.session.load_wav(str(path), FS)
    at = 2688057
    class ClockedStream(Stream):
        now = 0
        def sample_now(self):
            return max(self.now, self.pos)
    live = ClockedStream(audio)
    live.pos = at-round(.6*FS)
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 8.
    tx.p4_link_options.p4_data = 'body'
    received = []
    tx.p4_link_host.deliver = received.append
    rx = tx.sessrx.p4
    read_window = rx.read_window
    def costly(audio, origin, **kw):
        out = read_window(audio, origin, **kw)
        if origin+len(audio) >= rx.clock.next_due+round(3.28*FS):
            live.now = max(live.now, live.pos)+round(cost_ms*FS/1000)
        return out
    monkeypatch.setattr(rx, 'read_window', costly)
    rx.chirp = previous_chirp
    rx.cfo_hz = 1.54
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, 1.54)
    tx.p4_probe.handoff_audio = at-6*FS, audio[at-6*FS:live.pos].copy()
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **{**kw, 'role': 'irs'}))
    original = Tx._tx
    def key(audio, what):
        original(tx, audio, what)
        live.pos = tx.tx_end
    tx._tx = key
    onair._run_p4_link(tx, live, onair._MasterGrid(0,60000,0,packet_n=36000,cs_n=12000,d_max_n=4800))
    # B: an 80 ms read leaves the first LONG late but keyable, so it is ACKed
    # in its own slot like the cheaper reads, not requested and deferred.
    full = [o for o in tx.p4_probe.heard if o['kind'] == Kind.LONG and not o['provisional']]
    assert received == [b'RMS Trimode 1.4.4.0\r\nW9SSJ has 109 daily minutes remaining with VE3KP']
    assert len(full) >= 2
    assert len(tx.p4_probe.emissions) == 2
    for obs, emitted in zip(full[:2], tx.p4_probe.emissions):
        assert emitted['word'] == 'X27'
        assert emitted['audio_start'] == obs['onset_sample']+round(3.4*FS)
    assert not [e for e in tx.p4_probe.link_events if e['event'] == 'Refused']
