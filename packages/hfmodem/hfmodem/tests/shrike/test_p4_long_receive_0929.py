# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import p4oracle, session
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Kind, Observation, Registered
from hfmodem.tests.shrike.corpora import n9vc_0802

FS = 48000
ROOT = Path(__file__).resolve().parents[5]


def receiver(at, cfo=1.54):
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, cfo)
    return rx


def provisional(rx, at):
    obs = Observation(Kind.LONG, FS, 0, at, at, .846, 1.54, .5, provisional=True)
    rx.register(obs)
    return obs


@pytest.mark.parametrize('at, lead', [(2688057, 8), (2868057, 2)])
def test_ve0121_long_uses_bounded_context_from_a_rolling_window(at, lead):
    path = ROOT/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG long recording absent')
    audio = session.load_wav(str(path), FS)
    rx = receiver(at)
    obs = rx._packet(audio[at-lead*FS:at+round(3.326*FS)], at-lead*FS)
    assert obs is not None and obs.kind == Kind.LONG and not obs.provisional
    assert abs(obs.stream_sample-at) <= round(.035*FS)
    assert 3.20 <= obs.duration_s <= 3.35
    assert obs.frame is not None and obs.frame.status.counter == 1
    assert obs.frame.crc == 55412


def test_fragmented_envelope_cannot_expire_a_provisional_long_early(monkeypatch):
    at = FS
    rx = receiver(at)
    obs = provisional(rx, at)
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [(1., .91)])
    assert rx.read_window(np.zeros(at+round(1.25*FS)), 0, whole=True) == []
    assert obs in rx.observations
    assert rx.read_window(np.zeros(at+round(3.35*FS)-1), 0, whole=True) == []
    records = rx.read_window(np.zeros(at+round(3.35*FS)), 0, whole=True)
    assert [r.outcome for r in records] == [Registered.EXPIRED]


def test_pending_long_avoids_free_running_word_search(monkeypatch):
    rx = receiver(FS)
    provisional(rx, FS)
    rx.clock.misses = 3
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [])

    def no_scan(*a, **kw):
        pytest.fail('a pending long must not trigger a free-running word scan')

    monkeypatch.setattr(p4oracle, 'word_scan', no_scan)
    assert rx.read_window(np.zeros(round(3*FS)), 0, whole=True) == []


def test_a_positive_short_still_contradicts_a_provisional_long(monkeypatch):
    rx = receiver(FS)
    old = provisional(rx, FS)
    short = replace(old, kind=Kind.SHORT, provisional=False, duration_s=.758,
                    header='A', value=13, walsh_peak=64)
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: short)
    records = rx.read_window(np.zeros(2*FS), 0, whole=True)
    assert [r.outcome for r in records] == [Registered.CONTRADICTED]
    assert records[0].observation.kind == Kind.SHORT


def test_frame_less_normal_is_read_before_free_word_search(monkeypatch):
    rx = receiver(FS)
    rx.clock.misses = 3
    obs = Observation(Kind.NORMAL, FS, 0, FS, FS, .796, 1.54, .1,
                      lattice=.6, placed=True)
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: obs)
    monkeypatch.setattr(p4oracle, 'word_scan', lambda *a, **kw: pytest.fail('normal starved by scan'))
    records = rx.read_window(np.zeros(2*FS), 0, whole=True)
    assert records[0].observation == obs
    assert rx.clock.misses == 0
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
    rx.clock.misses = 3
    assert rx.read_window(np.zeros(2*FS), 0, whole=True) == []
    assert rx.clock.misses == 0


def test_three_genuinely_missed_packets_still_allow_word_reacquisition(monkeypatch):
    rx = receiver(FS)
    scans = []
    monkeypatch.setattr(rx, '_word', lambda *a, **kw: None)
    monkeypatch.setattr(rx, '_packet', lambda *a, **kw: None)
    monkeypatch.setattr(p4oracle, 'word_scan', lambda *a, **kw: scans.append(kw) or [])
    for n in range(3):
        rx.read_window(np.zeros(rx.clock.next_due+FS), 0, whole=True)
        assert rx.clock.misses == n+1
        rx.new_cycle()
    assert len(scans) == 1


def test_weak_ve0121_long_still_fails_full_coherence():
    path = ROOT/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG long recording absent')
    audio = session.load_wav(str(path), FS)
    at = 1788029
    assert receiver(at)._packet(audio[at-4*FS:at+round(3.355*FS)], at-4*FS) is None


def test_placed_packet_bounds_envelope_work(monkeypatch):
    rx = receiver(20*FS)
    lengths = []
    monkeypatch.setattr(p4oracle, 'robust_at', lambda *a, **kw: dict(
        p4_hdr=None, p4_value=None, p4_walsh_peak=0, coherence=0., first_chip=None))
    monkeypatch.setattr(p4oracle, 'placed_despread', lambda *a, **kw: (0., None, None))
    monkeypatch.setattr(rx, '_bursts', lambda audio, origin, **kw: lengths.append(len(audio)) or [])
    assert rx._packet(np.zeros(40*FS), 0) is None
    assert lengths == [round(3.855*FS)]


@pytest.mark.parametrize('negotiated', [False, True])
def test_negotiated_first_chirp_remains_long_until_crc(monkeypatch, negotiated):
    rx = receiver(FS)
    rx.chirp = negotiated
    rx.expected = True
    rx.clock.cycle_s = 3.75
    monkeypatch.setattr(p4oracle, 'robust_at', lambda *a, **kw: dict(
        p4_hdr=None, p4_value=None, p4_walsh_peak=0, coherence=0., first_chip=None))
    monkeypatch.setattr(p4oracle, 'placed_despread', lambda *a, **kw: (0., None, None))
    monkeypatch.setattr(p4oracle, 'occupied_bw', lambda *a, **kw: 2400.)
    monkeypatch.setattr(rx, '_bursts', lambda audio, origin, **kw: [((FS-origin)/FS, .846)])
    obs = rx._packet(np.zeros(round(1.846*FS)), 0, partial=True)
    assert obs.kind == (Kind.CHIRP if negotiated else Kind.NORMAL)
    assert obs.provisional and obs.frame is None


def test_recorded_first_negotiated_chirp_confirms_crc_before_answer(monkeypatch):
    from scipy.signal import hilbert
    from hfmodem.shrike import p4chirprx

    fs, pcm = n9vc_0802()
    assert fs == FS
    audio = pcm[round(4*FS):round(10.3*FS)]/32768
    audio = np.real(hilbert(audio)*np.exp(-2j*np.pi*400*np.arange(len(audio))/FS))
    at = round((6.660-4)*FS)
    rx = receiver(at, 0.)
    rx.chirp = rx.expected = True
    rx.clock.cycle_s = 3.75
    early = rx._packet(audio[:at+round(.846*FS)], 0, partial=True)
    assert early.kind == Kind.CHIRP and early.provisional and early.frame is None
    rx.register(early)
    decode = p4chirprx.decode_packet
    calls = []

    def counted(*a, **kw):
        calls.append(kw)
        return decode(*a, **kw)

    monkeypatch.setattr(p4chirprx, 'decode_packet', counted)
    assert rx._packet(audio[:at+round(3.30*FS)], 0) is None
    assert calls == []
    full = rx._packet(audio[:at+round(3.326*FS)], 0)
    assert full.kind == Kind.CHIRP and not full.provisional
    assert full.frame.data == b'[BPQ-6.0.25.1-B2FWIHJM'
    assert full.frame.status.counter == 1
    assert rx.register(full).outcome == Registered.CONFIRMED
    assert rx._packet(audio[:at+round(3.355*FS)], 0).frame == full.frame
    assert len(calls) == 1
    rx.chirp = False
    assert rx._packet(audio[:at+round(3.355*FS)], 0) is None


def test_negotiated_chirp_crc_failure_is_bounded_and_never_confirmed(monkeypatch):
    from hfmodem.shrike import p4chirprx

    rx = receiver(FS)
    rx.chirp = rx.expected = True
    rx.clock.cycle_s = 3.75
    calls = []
    monkeypatch.setattr(p4chirprx, 'decode_packet', lambda *a, **kw: calls.append(kw) or None)
    monkeypatch.setattr(p4chirprx, 'acquire_header', lambda *a, **kw: p4chirprx.Header(kw['nominal_sample'], 8, .95))
    monkeypatch.setattr(p4oracle, 'robust_at', lambda *a, **kw: dict(
        p4_hdr=None, p4_value=None, p4_walsh_peak=0, coherence=0., first_chip=None))
    monkeypatch.setattr(p4oracle, 'placed_despread', lambda *a, **kw: (0., None, None))
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [])
    for end in [4.31, 4.33, 5.]:
        assert rx._packet(np.zeros(round(end*FS)), 0) is None
    assert len(calls) == 1
    rx.clock.next_due += round(3.75*FS)
    assert rx._packet(np.zeros(9*FS), 0) is None
    assert len(calls) == 2


def test_ws0131_repeated_normal_packets_keep_decode_budget(monkeypatch):
    import json

    directory = ROOT/'captures/onair-0929-0131'
    if not (directory/'stream.flac').exists():
        pytest.skip('local WS8EOC normal-starvation recording absent')
    audio = session.load_wav(str(directory/'stream.flac'), FS)
    heard = json.loads((directory/'p4-probe.json').read_text())['heard']
    packets = [o for o in heard if o['kind'] == 'normal' and o['onset_sample'] >= 1487971]
    rx = receiver(packets[0]['onset_sample'], -3.487)
    rx.peer_onset = rx.clock.next_due-60000
    rx._peer_heads = [rx.peer_onset-60000, rx.peer_onset]
    rx.acked_counter = 2
    clock = [0.]
    rx.hold.now = lambda: clock[0]
    deadline = [0.]
    rx.normal_deadline = lambda at: deadline[0]
    decode = rx.hold._decode
    attempts = []
    scans = []

    def costed_decode(scores):
        attempts.append(rx.hold.last)
        frame = decode(scores)
        clock[0] += .010
        return frame

    def expensive_scan(*a, **kw):
        scans.append(rx.clock.next_due)
        clock[0] += .290
        return []

    monkeypatch.setattr(rx.hold, '_decode', costed_decode)
    monkeypatch.setattr(p4oracle, 'word_scan', expensive_scan)
    for row in packets:
        at = row['onset_sample']
        rx.clock.next_due = at
        rx.clock.cfo_hz = row['cfo_hz']
        rx.clock.misses = 3
        deadline[0] = clock[0]+.250
        origin = at-5*FS
        rx.read_window(audio[origin:at+round(.846*FS)], origin, whole=True)
        assert rx.clock.misses == 0
    assert scans == []
    assert len(set(attempts)) >= 2
    assert rx.hold.frame is not None
    assert rx.hold.frame.status.counter == 2
    assert rx.hold.frame.data == b'(EN72QQ)\r[WL2K-5.0-B2FWIHJM$]\r;PQ: 87047'


@pytest.mark.parametrize('at', [2268068, 2988928, 3168073])
def test_ve0124_long_uses_a_second_bounded_noise_context(at):
    path = ROOT/'captures/onair-0929-0124/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG second long recording absent')
    audio = session.load_wav(str(path), FS)
    rx = receiver(at, -2.3024190086545215)
    early = rx._packet(audio[at-5*FS:at+round(.846*FS)], at-5*FS, partial=True)
    assert early.kind == Kind.LONG and early.provisional
    rx.register(early)
    full = rx._packet(audio[at-5*FS:at+round(3.326*FS)], at-5*FS)
    assert full.kind == Kind.LONG and not full.provisional
    assert rx.register(full).outcome == Registered.CONFIRMED
    assert abs(full.stream_sample-at) <= round(.035*FS)


def test_second_ve0121_long_survives_own_ack_gap_and_clock_offset():
    path = ROOT/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG long recording absent')
    audio = session.load_wav(str(path), FS)
    at = 2867097
    begin = 2688057+round((3.4-.0444)*FS)
    end = 2688057+round((3.4+.2421)*FS)
    audio[begin:end] = 0
    rx = receiver(at)
    rx.clock.cycle_s = 3.75
    rx.note_gap(begin, end)
    origin = at-6*FS
    early = rx._packet(audio[origin:at+round(.846*FS)], origin, partial=True)
    assert early.kind == Kind.LONG and early.provisional
    rx.register(early)
    full = rx._packet(audio[origin:at+round(3.326*FS)], origin)
    assert full.kind == Kind.LONG and not full.provisional
    assert abs(full.stream_sample-2868057) <= round(.005*FS)
    assert full.frame is not None and full.frame.crc == 55412
    assert rx.register(full).outcome == Registered.CONFIRMED


def test_partial_ve0121_long_uses_envelope_not_ambiguous_symbol_phase():
    path = ROOT/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG long recording absent')
    audio = session.load_wav(str(path), FS)
    at = 2688057
    early = receiver(at)._packet(audio[at-6*FS:at+round(.846*FS)], at-6*FS, partial=True)
    assert early.kind == Kind.LONG and early.provisional
    assert abs(early.stream_sample-at) <= round(.005*FS)


@pytest.mark.parametrize('signal', ['silence', 'tone', 'noise'])
def test_placed_long_does_not_confirm_non_packets(signal):
    rng = np.random.default_rng(42)
    audio = rng.normal(0, .01, 5*FS)
    if signal == 'silence':
        audio[:] = 0
    elif signal == 'tone':
        audio[FS:] += np.cos(2*np.pi*1500*np.arange(4*FS)/FS)
    rx = receiver(FS)
    rx.clock.cycle_s = 3.75
    rx.note_gap(round(.6*FS), round(.9*FS))
    audio[round(.6*FS):round(.9*FS)] = 0
    assert rx._packet(audio[:round(4.326*FS)], 0) is None


@pytest.mark.parametrize('error_ms', [-35, -20, -10, 0, 10, 20, 35])
def test_recorded_chirp_header_corrects_the_placed_clock(error_ms):
    from scipy.signal import hilbert

    fs, pcm = n9vc_0802()
    audio = pcm[round(4*fs):round(10.3*fs)]/32768
    audio = np.real(hilbert(audio)*np.exp(-2j*np.pi*400*np.arange(len(audio))/fs))
    at = round((6.660-4)*fs)
    predicted = at+round(error_ms*fs/1000)
    rx = receiver(predicted, 0.)
    rx.chirp = rx.expected = True
    rx.clock.cycle_s = 3.75
    early = rx._packet(audio[:predicted+round(.846*fs)], 0, partial=True)
    assert early.kind == Kind.CHIRP and early.provisional and early.frame is None
    full = rx._packet(audio[:early.stream_sample+round(3.326*fs)], 0)
    assert full.kind == Kind.CHIRP and not full.provisional
    assert full.stream_sample == early.stream_sample
    assert abs(full.stream_sample-at) < round(.011*fs)
    assert full.frame.data == b'[BPQ-6.0.25.1-B2FWIHJM'


def test_incomplete_long_failure_can_retry_then_reuses_accepted_crc(monkeypatch):
    from hfmodem.shrike import p4robustrx

    path = ROOT/'captures/onair-0929-0121/stream.flac'
    if not path.exists():
        pytest.skip('local VE3KPG long recording absent')
    audio = session.load_wav(str(path), FS)
    at = 2867097
    rx = receiver(at)
    rx.clock.cycle_s = 3.75
    origin = at-6*FS
    decode = p4robustrx.decode_symbols
    calls = []

    def counted(*a, **kw):
        calls.append(len(a[0]))
        return decode(*a, **kw)

    monkeypatch.setattr(p4robustrx, 'decode_symbols', counted)
    early = rx._packet(audio[origin:at+round(3.295*FS)], origin)
    assert early is None or early.frame is None
    full = rx._packet(audio[origin:at+round(3.311*FS)], origin)
    assert full is not None and full.frame is not None and full.frame.crc == 55412
    count = len(calls)
    assert count == 2
    assert rx._packet(audio[origin:at+round(3.355*FS)], origin).frame == full.frame
    assert len(calls) == count


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('shift', [-round(.026*FS), 0, 2, round(.026*FS)])
@pytest.mark.parametrize('late', [False, True])
def test_full_long_registry_enrichment_delivers_once_and_keeps_repeat_ack(kind, shift, late):
    from types import SimpleNamespace
    from hfmodem.shrike import p4frame
    from hfmodem.tests.shrike.test_p4turn import machine

    rx = receiver(FS)
    turn, _ = machine(role='irs', data='body')
    received = []
    turn.adapter.host = SimpleNamespace(deliver=received.append, supervisor=lambda *a: None)
    old = Observation(kind, FS, 0, FS, FS, 3.28, 0., .8)
    frame = p4frame.decode_field(p4frame.encode_field(b'gateway data', p4frame.Status.from_byte(0)), data_bytes=12)
    records = []
    rx._emit(old, records)
    for reg in records:
        turn.observe(reg, now=FS+round(3.28*FS))
    assert received == []
    due = FS+round(3.4*FS)
    if late:
        first = turn.decide(due-turn.lead)
        assert (first.purpose if first is not None else None) == (
            None)
    full = replace(old, stream_sample=FS+shift, first_chip=FS+shift, frame=frame)
    records = []
    rx._emit(full, records)
    assert len(records) == 1 and records[0].outcome == Registered.ENRICHED
    assert records[0].observation.stream_sample == FS+shift
    assert records[0].observation.first_chip == FS+shift
    assert records[0].replaced == old
    assert rx.clock.next_due == FS+shift
    turn.observe(records[0], now=due-turn.lead if late else
                 FS+shift+round(3.4*FS)-turn.lead-round(.010*FS))
    # B1: a counter-0 chirp takes V27 by parity; a late frame keys in its own
    # slot, so its repeat answers on the repeat's raster, not a deferred one.
    expected = 'V27' if kind == Kind.CHIRP else 'X27'
    due = FS+shift+round(3.4*FS)
    proposal = turn.decide(due-turn.lead)
    if late:
        assert (proposal.purpose if proposal is not None else None) == (
            None)
        if proposal is not None:
            turn.keyed()
    else:
        assert proposal is not None and proposal.word == expected and proposal.stream_sample == due
        turn.keyed()
    duplicate = []
    rx._emit(full, duplicate)
    assert duplicate == []
    repeated = replace(full, stream_sample=FS+round(3.75*FS), first_chip=FS+round(3.75*FS))
    records = []
    rx._emit(repeated, records)
    turn.observe(records[0], now=repeated.stream_sample+round(3.31*FS))
    due = repeated.stream_sample+round(3.4*FS)
    assert turn.decide(due-turn.lead).stream_sample == due
    assert received == [b'gateway data']


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('valid', [False, True])
@pytest.mark.parametrize('old_raster', [False, True])
def test_checked_long_response_updates_liveness_and_learns_turnaround(kind, valid, old_raster):
    from hfmodem.shrike import p4frame
    from hfmodem.shrike.p4receiver import Emission, Role

    keyed = 5*FS
    at = keyed+round(.350*FS)
    rx = receiver(at)
    rx.turnarounds['word'] = [.330]
    rx.note_emission(Emission(Kind.WORD, Role.IRS, keyed, .242))
    rx.clock.cycle_s = 3.75
    previous = keyed+round((.330-3.75)*FS) if old_raster else None
    rx.peer_onset = previous
    rx.silent_cycles = 2
    frame = p4frame.decode_field(p4frame.encode_field(b'checked data', p4frame.Status.from_byte(0)), data_bytes=12)
    obs = Observation(kind, FS, 0, at, at, 3.28, 0., .8, frame=frame if valid else None)
    reg = rx.register(obs)
    assert reg.responds_to == keyed
    if valid:
        assert at in rx.live_onsets
        assert rx.peer_onset == at
        assert rx.turnarounds['word'] == [.330, .350]
        assert rx.finish_cycle(at) == 0
        assert rx.register(obs).outcome == Registered.DUPLICATE
        assert rx.turnarounds['word'] == [.330, .350]
    else:
        assert at not in rx.live_onsets
        assert rx.peer_onset == previous
        assert rx.turnarounds['word'] == [.330]
        assert rx.finish_cycle(at) == 3


@pytest.mark.parametrize('kind', [Kind.LONG, Kind.CHIRP])
@pytest.mark.parametrize('already_associated', [False, True])
def test_long_crc_refinement_recovers_missing_response_association(kind, already_associated):
    from hfmodem.shrike import p4frame
    from hfmodem.shrike.p4receiver import Emission, Role

    keyed = 5*FS
    old_tau, new_tau = (.344, .370) if already_associated else (.370, .344)
    old_at, decoded_at = keyed+round(old_tau*FS), keyed+round(new_tau*FS)
    rx = receiver(old_at)
    rx.turnarounds['word'] = [.330]
    rx.note_emission(Emission(Kind.WORD, Role.IRS, keyed, .242))
    old = Observation(kind, FS, 0, old_at, old_at, 3.28, 0., .8)
    original = rx.register(old)
    assert original.responds_to == (keyed if already_associated else None)
    frame = p4frame.decode_field(p4frame.encode_field(b'checked data', p4frame.Status.from_byte(0)), data_bytes=12)
    records = []
    rx._emit(replace(old, stream_sample=decoded_at, first_chip=decoded_at, frame=frame), records)
    assert len(records) == 1 and records[0].outcome == Registered.ENRICHED
    assert records[0].observation.stream_sample == decoded_at
    assert records[0].responds_to == keyed
    assert decoded_at in rx.live_onsets
    assert rx.turnarounds['word'] == [.330, new_tau]
    assert rx.clock.next_due == decoded_at
