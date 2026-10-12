# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import json
from pathlib import Path

from hfmodem.tests.shrike import p4_stock_peer as stock

import numpy as np
import pytest

from hfmodem.shrike import onair, p4breakin, p4hold, p4oracle
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, P4Receiver, Role
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Tx
from hfmodem.tests.shrike.test_ws8eoc_p4_silent_0927 import Live, grid
from hfmodem.tests.shrike.test_p4turn import machine

ROOT = Path(__file__).resolve().parents[5]
GREETING = b'RMS Trimode 1.4.4.0\r\nW9SSJ has '


def recorded(arm='2232'):
    path = ROOT / ('captures/onair-0928-'+arm)
    if not (path / 'stream.flac').exists():
        pytest.skip('local VE3KPG September 28 capture absent')
    return onair.session.load_wav(str(path / 'stream.flac'), FS), json.loads((path / 'p4-probe.json').read_text())


def receive(audio, at, *, gaps=()):
    rx = P4Receiver(FS, admitted=True, cfo_hz=0., breakin=True)
    rx.enable_hold()
    rx.note_emission(Emission(Kind.SHORT, Role.ISS, at-round(.923*FS), .758))
    for gap in gaps:
        rx.note_gap(*gap)
    turn, _ = machine(data='body')
    for reg in rx.read_window(audio, 0, whole=True):
        turn.observe(reg)
    return rx, turn


@pytest.mark.parametrize('shape', ['bare', 'bad-body', 'split-body', 'censored-body'])
def test_bare_requires_a_complete_word_and_observed_body_interval(shape):
    at = FS
    signal = stock.word('Y27') if shape == 'bare' else stock.body(b'hello', 0)
    if shape != 'bare':
        for start in range(round(.30*FS), len(signal), round(.05*FS)):
            signal[start:start+round(.025*FS)] *= -1
    if shape == 'split-body':
        signal[round(.245*FS):round(.50*FS)] = 0
    audio = np.r_[np.zeros(at), signal, np.zeros(FS)]
    gaps = ()
    if shape == 'censored-body':
        gaps = ((at+round(.245*FS), at+round(.805*FS)),)
        audio[gaps[0][0]:gaps[0][1]] = 0
    rx, turn = receive(audio, at, gaps=gaps)
    head = next(o.stream_sample for o in rx.observations if o.word == 'Y27')
    assert abs(head - at) <= 1
    proposal = turn.decide(head+round(.920*FS)-turn.lead)
    assert not any(o.kind == Kind.BODY for o in rx.observations)
    if shape == 'bare':
        assert proposal is not None and proposal.status == 0x40
        assert proposal.stream_sample == head+round(.920*FS)
    else:
        assert proposal is None


def test_recorded_fragment_is_never_bare_with_full_history():
    audio, _ = recorded()
    onset = 1188236
    end = onset+round(.846*FS)
    origin = end-6*FS
    window = audio[origin:end].copy()
    gap = (1170155-origin, 1183906-origin)
    window[gap[0]:gap[1]] = 0
    obs = p4oracle.word_at(window, FS, origin, (onset-origin)/FS,
                           cfo_hz=-3., whole=True, expected_words={'Y27'})
    assert obs is not None and obs.word == 'Y27' and obs.score > .95
    assert obs.extent_s is None or obs.extent_s > .5


def test_recorded_bare_y27_keeps_its_closed_extent_above_the_tail_noise():
    path = ROOT / 'captures/onair-0927-1744/stream.flac'
    if not path.exists():
        pytest.skip('local September 27 1744 capture absent')
    audio = onair.session.load_wav(str(path), FS)
    origin = round(13.314*FS)
    onset = round(13.495*FS)
    window = audio[origin:onset+round(.846*FS)]
    extent = p4oracle._extent(window, FS, (onset-origin)/FS, word='Y27')
    assert extent is not None and .270 < extent <= .280


def test_refined_recorded_body_is_registered_once_in_its_original_cycle():
    audio, _ = recorded()
    at = 1188234
    origin, end = at-2400, at+round(.846*FS)
    rx = P4Receiver(FS, admitted=True, cfo_hz=-3.-1.4, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, -3.-1.4)
    rx._peer_heads = [at-120000, at-60000]
    records = rx.read_window(audio[origin:end], origin, whole=True)
    assert not any(r.observation.kind == Kind.BODY for r in records)
    head = next(o for o in rx.observations if o.word == 'Y27')
    records = rx.refine(deadline=float("inf"))
    bodies = [r.observation for r in records if r.observation.kind == Kind.BODY]
    assert len(bodies) == 1
    assert bodies[0].frame.data == GREETING and bodies[0].frame.status.byte == 0
    assert bodies[0].stream_sample == head.stream_sample
    assert rx.refine(deadline=float("inf")) == []
    assert rx.read_window(audio[origin:end], origin, whole=True) == []
    turn, _ = machine(data='body')
    turn.observe(next(r for r in records if r.observation.kind == Kind.BODY), now=end)
    decision = bodies[0].stream_sample+round(.846*FS)
    proposal = turn.decide(decision)
    assert proposal is not None and proposal.word == 'X27'
    assert proposal.stream_sample == bodies[0].stream_sample+round(.920*FS)
    late, _ = machine(data='body')
    late.observe(next(r for r in records if r.observation.kind == Kind.BODY), now=decision+1)
    # B: a frame accepted after the lead but before its due keeps its own slot.
    answer = late.decide(decision+1)
    assert answer == proposal and late._deferred is None


def test_refinement_deadline_rejects_a_late_search_without_registering(monkeypatch):
    audio, _ = recorded()
    at = 1188234
    origin, end = at-2400, at+round(.846*FS)
    rx = P4Receiver(FS, admitted=True, cfo_hz=-3.-1.4, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, -3.-1.4)
    rx._peer_heads = [at-120000, at-60000]
    rx.read_window(audio[origin:end], origin, whole=True)
    now = [10.]
    monkeypatch.setattr(p4hold, 'monotonic', lambda: now[0])
    search = p4breakin.search_scores

    def late_search(z, nominal, cfo, *, deadline):
        result = search(z, nominal, cfo)
        assert result is not None
        now[0] = deadline+.001
        return result

    monkeypatch.setattr(p4breakin, 'search_scores', late_search)
    assert rx.refine(deadline=11.) == []
    assert rx.hold.frame is None
    assert any(e['event'] == 'search overrun' for e in rx.hold.events)
    assert not any(o.kind == Kind.BODY for o in rx.observations)


@pytest.mark.parametrize('arm,end', [('2232', 1270000), ('2255', 2080000)])
def test_recorded_driver_never_grants_a_greeting_repeat(monkeypatch, arm, end):
    audio, meta = recorded(arm)
    live = Live(audio[:end])
    live.pos = meta['emissions'][1]['audio_end']
    tx = Tx(live)
    tx.p4_probe.emissions = meta['emissions'][:2]
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_link_timeout = 10
    rx = tx.sessrx.p4
    rx.cfo_hz = meta['peer_offset_hz']
    key = tx._tx
    def emitted(*args):
        key(*args)
        live.pos = tx.tx_end
    monkeypatch.setattr(tx, '_tx', emitted)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    onair._run_p4_link(tx, live, grid())
    keys = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert keys and all(e['kind'] == Kind.WORD and e['word'] == 'X27' for e in keys)
    assert rx.capture_gaps
    for key in keys:
        if not key.get('current_slot_accepted', True):
            assert key['word'] == 'X27'
        else:
            assert any(o.kind in (Kind.BODY, Kind.SHORT)
                       and key['at'] == o.stream_sample+round(.920*FS)
                       for o in rx.observations)


def test_valid_greeting_repeat_acks_again_without_delivering_twice(monkeypatch):
    at = round(.923*FS)
    audio = np.zeros(4*FS)
    wave = stock.body(b'hello', 0)
    for start in (at, at+60000):
        audio[start:start+len(wave)] = wave
    live = Live(audio)
    tx = Tx(live)
    tx.p4_link_options.p4_data = 'body'
    received = []
    tx.p4_link_host.deliver = received.append
    key = tx._tx
    def emitted(*args):
        key(*args)
        live.pos = tx.tx_end
    monkeypatch.setattr(tx, '_tx', emitted)
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: 2131)
    onair._run_p4_link(tx, live, grid())
    keys = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert [e['word'] for e in keys] == ['X27', 'X27']
    assert all(abs(e['at']-want) <= 1 for e, want in zip(keys, (at+44160, at+60000+44160)))
    assert received == [b'hello']
