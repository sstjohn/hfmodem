# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""S5(a,d,f): recorded deferred answers, first longs and off-raster words."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p4frame, p4oracle, p4short, p4turn
from hfmodem.shrike.p4receiver import (
    Clock, Emission, Expect, Kind, Observation, P4Receiver, Role,
)
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

ROOT = Path(__file__).resolve().parents[5]


def recording(arm):
    path = ROOT / f'captures/onair-0929-{arm}'
    if not (path / 'stream.flac').exists():
        pytest.skip(f'local {arm} recording absent')
    return (onair.session.load_wav(str(path / 'stream.flac'), FS),
            json.loads((path / 'p4-probe.json').read_text()))


def tracked(at, cfo):
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    rx.peer_onset = at - 60000
    rx._peer_heads = [at - 120000, at - 60000]
    return rx


def normal_read(rx, audio, *, tail=.81):
    at = rx.clock.next_due
    origin = at - round(.05 * FS)
    end = at + round(.846 * FS)
    records = rx.read_partial(audio[origin:at + round(tail * FS)], origin,
                              deadline=at + round(tail * FS))
    records += rx.read_window(audio[origin:end], origin, whole=True)
    records += rx.refine()
    return [r.observation for r in records if r.observation.kind == Kind.NORMAL
            and r.observation.frame is not None]


@pytest.mark.parametrize('cycles', [1, 2, 5])
def test_1852_deferred_x27_keeps_the_packet_phase_and_reads_c1(cycles):
    audio, meta = recording('1852')
    # At key time the later body had not decoded; it was received at 1056362.
    body = next(r for r in meta['heard'] if r['kind'] == 'body')
    at = body['onset_sample']
    rx = tracked(at, body['cfo_hz'])
    frame = p4frame.decode_field(p4frame.encode_field(bytes.fromhex(body['data']),
                                p4frame.Status.from_byte(body['status'])), data_bytes=31)
    rx.observations.append(Observation(Kind.BODY, FS, at, 0, at, .8,
                                       body['cfo_hz'], body['coherence'], frame=frame))
    key = meta['emissions'][-1]['audio_start'] + (cycles - 1) * 60000
    rx.note_emission(Emission(Kind.WORD, Role.IRS, key, .244))
    expected = at + (cycles + 1) * 60000
    assert rx.clock.next_due == expected
    assert expected / FS == pytest.approx(20.995 + (cycles - 1) * 1.25, abs=.002)
    heard = normal_read(rx, audio)
    assert heard and heard[-1].frame.status.counter == 1
    assert heard[-1].frame.data == b'77 daily minutes remaining with WS8EOC ('


# The twenty N-T slots in scratch/c80_t3-2045-nolive.txt, not twenty
# consecutive slots: the live reader missed these between 65 and 125 s.
MISSED_2045 = [67.245, 68.495, 75.997, 77.247, 78.497, 79.747, 80.997,
               82.247, 83.497, 85.997, 87.247, 88.497, 89.747, 90.997,
               97.247, 109.747, 110.997, 112.247, 113.497, 118.497]


@pytest.mark.parametrize('displacement', [-.020, .020])
def test_2045_lost_tracking_reacquires_all_twenty_crc_valid_slots(displacement):
    audio, meta = recording('2045')
    decoded = []
    for onset in MISSED_2045:
        prior = [r for r in meta['heard'] if r['onset_sample'] < onset * FS
                 and r['kind'] == 'normal'][-1]
        # Exercise tracking loss outside the narrow bracket in both directions.
        rx = tracked(round((onset + displacement) * FS), prior['cfo_hz'])
        rx.acked_counter = 2 if onset < 100 else 3
        # A +20 ms candidate needs its full field before the cached search.
        heard = normal_read(rx, audio, tail=.81 + max(0., -displacement))
        if heard:
            assert abs(heard[-1].stream_sample / FS - onset) < .004
            decoded.append(onset)
    assert decoded == MISSED_2045


@pytest.mark.parametrize('arm,onset', [('1917', 53.503), ('2138', 131.003),
                                     ('1303', 44.747)])
def test_first_recorded_long_opens_capture_and_retains_long_raster(monkeypatch, arm, onset):
    audio, meta = recording(arm)
    at = round(onset * FS)
    cfo = next(r['cfo_hz'] for r in reversed(meta['heard']) if r['onset_sample'] < at)
    live = Stream(audio)
    live.pos = at - round(.6 * FS)
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 8.
    rx = tx.sessrx.p4
    rx.cfo_hz = cfo
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    tx.p4_probe.handoff_audio = at - 6 * FS, audio[at - 6 * FS:live.pos].copy()
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **{**kw, 'role': 'irs'}))
    onair._run_p4_link(tx, live, onair._MasterGrid(0, 60000, 0,
                         packet_n=36000, cs_n=12000, d_max_n=4800))
    reads = [e for e in rx.live_events if e['event'] == 'long decode']
    for k in range(2):
        assert any(abs(e['at'] - at - k * 180000) < round(.035 * FS)
                   and e['symbols'] >= 369 for e in reads), reads
    assert rx.clock.cycle_s == 3.75
    assert tx.p4_probe.emissions and all(e['word'] == 'V27' for e in tx.p4_probe.emissions)
    assert all(e['event'] != 'Keyed' or not e.get('current_slot_accepted', True)
               for e in tx.p4_probe.link_events)


def test_off_raster_word_preserves_heads_and_following_crc_normals():
    audio, _ = recording('2253')
    at = round(24.100 * FS)
    origin = at - round(.08 * FS)
    word = p4oracle.word_at(audio[origin:at + round(.32 * FS)], FS,
                            origin, (at-origin)/FS, cfo_hz=1., whole=True)
    assert word is not None and word.word == 'V27'
    rx = tracked(round(11.25 * FS), 1.)
    heads = rx._peer_heads.copy()
    rx.register(replace(word, origin=0, first_chip=round(10.6 * FS),
                        stream_sample=round(10.6 * FS)))
    assert rx.peer_onset == 10 * FS
    assert rx._peer_heads == heads
    assert rx.finish_cycle(10 * FS) == 0
    frame = p4frame.decode_field(p4frame.encode_field(b'normal'.ljust(40, b'\x1e'),
                                p4frame.Status.from_byte(1)), data_bytes=40)
    assert frame is not None
    for seconds in (11.25, 12.5, 13.75):
        at = round(seconds * FS)
        rx.register(Observation(Kind.NORMAL, FS, 0, at, at, .805, 1., .8,
                                frame=frame, lattice=.8, placed=True))
        assert rx.finish_cycle(at) == 0
        assert rx.peer_onset == at
        assert rx._normal_eligible()


@pytest.mark.parametrize('onset', [67.245, 75.997, 97.247])
def test_2045_wide_retry_finishes_at_partial_read_and_is_cached(monkeypatch, onset):
    audio, meta = recording('2045')
    cfo = next(r['cfo_hz'] for r in reversed(meta['heard'])
               if r['kind'] == 'normal' and r['onset_sample'] < onset * FS)
    rx = tracked(round((onset + .020) * FS), cfo)
    rx.acked_counter = 2
    at = rx.clock.next_due
    origin = at - round(.05 * FS)
    calls = []
    place = p4short.place_short
    def counted(*a, **kw):
        result = place(*a, **kw)
        calls.append((kw.get('acquisition'), result))
        return result
    monkeypatch.setattr(p4short, 'place_short', counted)
    rx.read_partial(audio[origin:at + round(.81 * FS)], origin,
                    deadline=at + round(.81 * FS))
    assert not rx.normal_waiting
    assert any(wide and result is not None for wide, result in calls)
    count = len(calls)
    rx.read_window(audio[origin:at + round(.846 * FS)], origin, whole=True)
    assert len(calls) == count
    assert any(o.kind == Kind.NORMAL and o.frame is not None for o in rx.observations)


@pytest.mark.parametrize('tail', [.81, .846, 1.0])
def test_0124_robust_short_does_not_promote_long_cadence(tail):
    audio, meta = recording('0124')
    at = round(37.254 * FS)
    cfo = next(r['cfo_hz'] for r in reversed(meta['heard']) if r['onset_sample'] < at)
    rx = tracked(at, cfo)
    origin = at - 6 * FS
    end = at + round(tail * FS)
    rx.read_partial(audio[origin:end], origin, deadline=end)
    rx.read_window(audio[origin:end], origin, whole=True)
    assert rx.clock.cycle_s == 1.25


def test_0124_driver_next_short_scores_as_on_base(monkeypatch):
    audio, meta = recording('0124')
    at = round(36.003 * FS)
    cfo = next(r['cfo_hz'] for r in reversed(meta['heard']) if r['onset_sample'] < at)
    live = Stream(audio)
    live.pos = 36 * FS
    tx = Tx(live)
    tx.settle = .044
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 7.
    rx = tx.sessrx.p4
    rx.cfo_hz = cfo
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    rx.peer_onset = at - 60000
    rx._peer_heads = [at - 120000, at - 60000]
    tx.p4_probe.handoff_audio = at - 6 * FS, audio[at - 6 * FS:live.pos].copy()
    cycles = []
    finish = rx.finish_cycle
    def finished(onset):
        silent = finish(onset)
        cycles.append((onset / FS, silent, rx.clock.cycle_s))
        return silent
    monkeypatch.setattr(rx, 'finish_cycle', finished)
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **{**kw, 'role': 'irs'}))
    onair._run_p4_link(tx, live, onair._MasterGrid(0, 60000, 0,
                         packet_n=36000, cs_n=12000, d_max_n=4800))
    next_short = next(c for c in cycles if abs(c[0] - 41.001) < .004)
    assert next_short[1:] == (0, 1.25), cycles
    assert any(e['word'] == 'X27' and abs(e['audio_start']/FS - 41.921) < .004
               for e in tx.p4_probe.emissions)


@pytest.mark.parametrize('arm,first', [('1857', 26.594), ('2034', 40.345),
                                     ('2253', 24.100)])
def test_recorded_off_raster_v27_stream_is_live_then_silence_expires(arm, first):
    audio, meta = recording(arm)
    last = meta['emissions'][-1]
    old_due = last['audio_start'] + round(.330 * FS)
    rx = tracked(old_due, 1.)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, last['audio_start'], .244))
    rx.silent_cycles = 15
    for k in range(11):
        at = round((first + k * 1.25) * FS)
        origin = at - round(.08 * FS)
        obs = p4oracle.word_at(audio[origin:at + round(.32 * FS)], FS,
                               origin, (at-origin)/FS, cfo_hz=1., whole=True)
        assert obs is not None and obs.word == 'V27' and not obs.provisional
        rx.register(obs)
        # A decision can still carry the old packet onset after an off-raster read.
        cycle = at - round(.600 * FS)
        assert rx.finish_cycle(cycle) == 0
        assert rx.finish_cycle(cycle) == 0
        rx.new_cycle()
    for k in range(1, 17):
        assert rx.finish_cycle(cycle + k * 60000) == k


@pytest.mark.parametrize('provisional,weak,unexpected', [(True, False, False),
                                                       (False, True, False),
                                                       (False, False, True)])
def test_unqualified_off_raster_head_cannot_clear_silence(provisional, weak, unexpected):
    rx = tracked(FS, 0.)
    at = FS + round(.6 * FS)
    rx.silent_cycles = 15
    rx.register(Observation(Kind.WORD, FS, 0, at, at, .244, 0., .8,
                            word='V27', provisional=provisional,
                            weak=weak, unexpected=unexpected))
    assert rx.finish_cycle(FS) == 16


def test_off_raster_duplicate_does_not_keep_the_next_cycle_alive():
    audio, _ = recording('2253')
    at = round(24.100 * FS)
    origin = at - round(.08 * FS)
    obs = p4oracle.word_at(audio[origin:at + round(.32 * FS)], FS,
                           origin, (at-origin)/FS, cfo_hz=1., whole=True)
    assert obs is not None and obs.word == 'V27'
    rx = tracked(at - round(.600 * FS), 1.)
    rx.register(obs)
    assert rx.finish_cycle(at - round(.600 * FS)) == 0
    rx.register(obs)
    assert rx.finish_cycle(at + round(.650 * FS)) == 1


def test_2253_driver_does_not_declare_silence_during_the_v27_stream(monkeypatch):
    audio, meta = recording('2253')
    last = meta['emissions'][-1]
    at = last['audio_start'] + round(.330 * FS)
    live = Stream(audio)
    live.pos = at - round(.06 * FS)
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 40.
    rx = tx.sessrx.p4
    rx.cfo_hz = 1.
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, 1.)
    rx.peer_onset = at - 60000
    tx.p4_probe.handoff_audio = live.pos, audio[live.pos:live.pos].copy()
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **{**kw, 'role': 'irs'}))
    onair._run_p4_link(tx, live, onair._MasterGrid(0, 60000, 0,
                         packet_n=36000, cs_n=12000, d_max_n=4800))
    assert tx.p4_probe.reason == 'session ceiling'
    assert rx.silent_cycles < 16
    assert len([o for o in rx.observations if o.word == 'V27' and not o.provisional]) >= 20
    assert tx.p4_probe.emissions and all(e['word'] == 'V27' for e in tx.p4_probe.emissions)
    assert all(e['event'] != 'Keyed' or not e.get('current_slot_accepted', True)
               for e in tx.p4_probe.link_events)
