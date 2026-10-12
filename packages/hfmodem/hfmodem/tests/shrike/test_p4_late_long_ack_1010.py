# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import onair, p4oracle, p4receiver, p4robustrx, p4robusttx, p4rx, p4turn, session
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, P4Receiver, Registered, Role
from hfmodem.shrike.p4observation import RegisteredObservation
from hfmodem.tests.evidence import CAPTURES

FS = 48000
C = 'onair-20261010T233501182327Z-54a54140-3f95-46d2-856b-daed7e6f24f7'
DATA_62 = (b'K4MSU Winlink Gateway at Murray State University\r\n'
           b'W9SSJ has 107 daily')
DATA_03 = b' minutes remaining with K4MSU (EM56UO)\r' + b'\x1e' * 30


def receiver(at=48000, cfo=-5.01):
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, cfo)
    return rx


def recording(name, at, end_s=3.306):
    path = CAPTURES/name/'stream.flac'
    if not path.is_file():
        pytest.skip(f'{name} recording absent')
    origin = at-24000
    audio = session.load_wav(str(path), FS)[origin:at+round(end_s*FS)]
    return audio, origin


def count_sf8(monkeypatch):
    calls = []
    recover = p4rx.recover_robust

    def counted(*a, **kw):
        if kw['spread_factor'] == 8:
            calls.append(kw['data_symbols'])
        return recover(*a, **kw)

    monkeypatch.setattr(p4rx, 'recover_robust', counted)
    return calls


def test_recording_negative_sf8_runs_once(monkeypatch):
    at = 1428171
    audio, origin = recording(C, at)
    rx = receiver(at)
    calls = count_sf8(monkeypatch)
    prefix = audio[:24000+round(3.230*FS)]
    assert rx.prepare_sf8(prefix, origin) is None
    rx.prepare_sf8(prefix, origin)
    rx.prepare_long(prefix, origin)
    rows = rx.read_window(audio, origin, whole=True)
    rx.read_window(audio, origin, whole=True)
    assert len(calls) <= 1, calls
    obs = next(r.observation for r in rows if r.observation.frame is not None)
    assert not obs.sf8 and obs.frame.status.byte == 0x62
    assert obs.frame.data == DATA_62 and len(DATA_62) == 69
    assert abs(obs.stream_sample-at) <= 48
    assert rx._sf8_prepared[(at, -5.01)].reason == 'below-coherence'


def test_recording_uncached_sf8_gate():
    at = 1428171
    audio, _ = recording(C, at)
    z = p4rx.audio_to_baseband(audio[19200:24000+round(3.230*FS)], FS)
    observed = p4rx.recover_robust(z, reference_start=8000, spread_factor=8,
        data_symbols=680, timing_offsets=range(-48, 49), cfo_hz=(-5.01,))
    print(f'uncached SF8 coherence={observed.coherence:.6f}')
    assert observed.coherence < .40


@pytest.mark.parametrize('reason', ['below-coherence', 'energy', 'recovery-error'])
def test_complete_negative_is_cached_but_incomplete_is_retryable(monkeypatch, reason):
    rx = receiver()
    at = rx.clock.next_due
    audio = np.ones(round(3.9*FS))
    calls = []

    def recover(*a, **kw):
        calls.append(kw)
        if reason == 'recovery-error':
            raise ValueError('complete support failure')
        return SimpleNamespace(coherence=.2 if reason == 'below-coherence' else .9)

    monkeypatch.setattr(p4rx, 'recover_robust', recover)
    monkeypatch.setattr(rx, '_sf8_energy_present', lambda *a: False)
    prefix = audio[:24000+round(3.230*FS)]
    origin = at-24000
    assert rx.prepare_sf8(prefix[:-1], origin) is None
    assert not rx._sf8_prepared and not calls
    assert rx.prepare_sf8(prefix, at) is None
    assert not rx._sf8_prepared and not calls
    for _ in range(2):
        assert rx.prepare_sf8(prefix, origin) is None
        assert rx._sf8_long(audio, origin) is None
    assert len(calls) == 1
    assert rx._sf8_prepared[(at, rx.clock.cfo_hz)].reason == reason


def test_negative_keys_phase_gap_pruning_and_positive_selection(monkeypatch):
    rx = receiver()
    at = rx.clock.next_due
    origin = at-24000
    audio = np.ones(24000+round(3.230*FS))
    calls = []

    def recover(*a, **kw):
        calls.append(kw)
        return SimpleNamespace(coherence=.2)

    monkeypatch.setattr(p4rx, 'recover_robust', recover)
    rx.prepare_sf8(audio, origin)
    assert not rx.sf8_prepared()
    rx.clock.cfo_hz += 1
    rx.prepare_sf8(audio, origin)
    rx.clock.next_due -= 1
    rx.prepare_sf8(audio, origin)
    assert len(calls) == 3
    rx.set_receive_phase((1, 1))
    rx.prepare_sf8(audio, origin)
    assert len(calls) == 4
    rx.note_gap(at+100, at+200)
    assert not rx._sf8_prepared
    assert rx._long_hint is None
    rx.clock.next_due = at+7*FS
    rx.prepare_sf8(audio, origin+7*FS)
    assert len(rx._sf8_prepared) == 1
    rx.new_call()
    assert not rx._sf8_prepared


@pytest.mark.parametrize('name,at,cfo,status,data', [
    (C, 1428171, -5.01, 0x62, DATA_62),
    (C, 1788166, -5.01, 0x03, DATA_03),
    ('onair-1002-2150', 1188369, 2.43, 0x61,
     b'RMS Trimode 1.4.4.0 Welcome to K4MSU Winlink Gateway at Murray State '),
    ('onair-1002-2150', 1368446, 2.43, 0x62,
     b'University\r\nW9SSJ has 116 daily minutes remaining with K4MSU (EM56UO)'),
])
def test_recording_split_matches_full_without_deadline_full_recovery(
        monkeypatch, name, at, cfo, status, data):
    audio, origin = recording(name, at)
    full, _ = receiver(at, cfo)._long(audio, origin)
    assert full is not None and full.frame.data == data
    rx = receiver(at, cfo)
    assert rx.prepare_long(audio[:24000+round(3.230*FS)], origin) is not None
    full_calls, tail_calls, decode_calls = [], [], []
    despread, finish, decode = (p4oracle.placed_despread, p4oracle.finish_despread_long,
                               p4robustrx.decode_symbols)

    def counted_full(*a, **kw):
        if kw.get('count') == 373:
            full_calls.append(kw)
        return despread(*a, **kw)

    def counted_decode(*a, **kw):
        decode_calls.append(len(a[0]))
        return decode(*a, **kw)

    def counted_tail(*a, **kw):
        tail_calls.append(True)
        return finish(*a, **kw)

    monkeypatch.setattr(p4oracle, 'placed_despread', counted_full)
    monkeypatch.setattr(p4robustrx, 'decode_symbols', counted_decode)
    monkeypatch.setattr(p4oracle, 'finish_despread_long', counted_tail)
    rows = rx.read_window(audio, origin, whole=True)
    obs = next(r.observation for r in rows if r.observation.frame is not None)
    rx.read_window(audio, origin, whole=True)
    assert not full_calls
    assert len(tail_calls) == 1
    assert decode_calls == [373]
    assert obs.frame == full.frame and obs.frame.status.byte == status
    assert obs.speedlevel == full.speedlevel == '3'
    assert obs.stream_sample == full.stream_sample
    if name == C:
        assert abs(obs.stream_sample-at) <= 48
    split = next(e for e in rx.live_events if e['event'] == 'long decode')
    prep = next(e for e in rx.live_events if e['event'] == 'long prepare')
    print(f'{name} {status:02x}: prefix={prep["prefix_ms"]:.3f}ms '
          f'tail={split["tail_ms"]:.3f}ms decode={split["decode_ms"]:.3f}ms '
          f'overlap={split["overlap_correlation"]:.12f} '
          f'floor={p4oracle.LONG_OVERLAP_CORRELATION_MIN:.2f}')
    assert rx._long_hint == 'sf16'


def test_crc_hint_provenance_lifecycle_and_failed_slot(monkeypatch):
    at = 1428171
    audio, origin = recording(C, at)
    rx = receiver(at)
    obs, _ = rx._long(audio, origin)
    assert obs is not None
    normal = replace(obs, speedlevel=None)
    other = receiver(at)
    other.register(normal)
    assert other._long_hint is None
    rx.register(obs)
    assert rx._long_hint == 'sf16'
    rx.note_emission(Emission(Kind.WORD, Role.IRS, at+round(3.4*FS), .24))
    assert rx._long_hint == 'sf16'
    rx.new_cycle()
    calls = count_sf8(monkeypatch)
    next_at = rx.clock.next_due
    assert rx.prepare_sf8(audio, next_at-24000) is None
    rx.prepare_long(audio[:24000+round(3.230*FS)], next_at-24000)
    with monkeypatch.context() as failed:
        failed.setattr(p4robustrx, 'decode_symbols', lambda *a, **kw: None)
        rx.read_window(audio, next_at-24000, whole=True)
    assert not calls
    assert rx._long_hint is None
    rx.new_cycle()
    rx.prepare_sf8(audio, rx.clock.next_due-24000)
    assert len(calls) == 1
    rx._long_hint = 'sf16'
    rx.register(replace(obs, kind=Kind.NORMAL, stream_sample=rx.clock.next_due,
                        first_chip=rx.clock.next_due-origin))
    assert rx._long_hint is None
    rx._long_hint = 'sf16'
    rx.note_emission(Emission(Kind.BODY, Role.ISS, rx.clock.next_due, .9))
    assert rx._long_hint is None


@pytest.mark.parametrize('format', ['sf16', 'sf8', 'normal'])
def test_only_positive_sf8_selects_early_driver_read(monkeypatch, format):
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx

    sf8 = format == 'sf8'
    recorded_at = 3528586 if sf8 else 2148660 if format == 'normal' else 1428171
    name = 'onair-1002-2154' if sf8 else 'onair-1002-1049' if format == 'normal' else C
    audio, _ = recording(name, recorded_at, 3.700)
    at, cfo = 24000, .69 if sf8 else -6.94 if format == 'normal' else -5.01
    live = Stream(audio)
    tx = Tx(live)
    tx.p4_probe.emissions.clear()
    tx.p4_probe.handoff_audio = (0, audio[:0])
    tx.p4_link_options.p4_link_timeout = 4.2
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_rx_timing_trace = True
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, cfo)
    rx.cfo_hz = cfo
    if format == 'normal':
        monkeypatch.setattr(rx, '_normal_eligible', lambda: True)
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **{**kw, 'role': 'irs'}))
    monkeypatch.setattr(onair, '_prekey_lead', lambda *a: round(.049*FS))
    calls = count_sf8(monkeypatch)
    preparations = []
    for preparation in ('sf8', 'long'):
        method = getattr(rx, f'prepare_{preparation}')
        def counted(*a, _format=preparation, _method=method, **kw):
            preparations.append(_format)
            return _method(*a, **kw)
        monkeypatch.setattr(rx, f'prepare_{preparation}', counted)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    final = at+round((3.280 if sf8 else 3.306)*FS)
    assert any(row['stage'] == 'read_window' and row['support_end'] == final
               for row in tx.p4_probe.timing_trace)
    assert calls == ([680, 24] if sf8 else [680])
    assert preparations == (['sf8', 'long'] if format == 'sf16' else ['sf8'])
    assert any(row['event'] == ('sf8-long' if sf8 else 'long-decode' if format == 'normal' else 'long decode')
               and row['crc_valid'] for row in rx.live_events)


def test_prefix_context_keys_phase_and_nonoverlapping_reply_gap():
    at = 1428171
    audio, origin = recording(C, at)
    prefix = audio[:24000+round(3.230*FS)]
    rx = receiver(at)
    assert rx.prepare_long(prefix[:-1], origin) is None
    assert rx.prepare_long(prefix, at) is None
    prepared = rx.prepare_long(prefix, origin)
    assert prepared is not None and rx.prepare_long(prefix, origin) is prepared
    rx.clock.cfo_hz += 1
    assert rx.prepare_long(prefix, origin) is not prepared
    rx.clock.next_due -= 1
    assert rx.prepare_long(prefix, origin) is not prepared
    rx.set_receive_phase(1)
    assert not rx._long_prepared
    rx.clock = Clock(Expect.PACKET, at, .330, 3.75, -5.01)
    rx.prepare_long(prefix, origin)
    obs, _ = rx._long(audio, origin)
    rx.register(obs)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, at+round(3.4*FS), .24))
    rx.note_gap(at+round(3.306*FS), at+round(3.65*FS))
    assert rx._long_hint == 'sf16' and rx._long_prepared
    rx.note_gap(at+round(3.1*FS), at+round(3.2*FS))
    assert rx._long_hint is None and not rx._long_prepared


def test_tail_recovery_uses_all_candidates_and_ranks_completed_statistics(monkeypatch):
    count = 2017
    starts = np.arange(count)
    symbols = np.ones((count, 352), complex)
    power = np.full((count, 352), p4oracle._SPREAD_ENERGY)
    power[0] *= .99
    prepared = dict(onset=1680, lo=0, cfo=0., starts=starts,
                    symbols=symbols, power=power, power_floor=1e-30)
    spread = np.ones(110000, complex)
    tail_power = np.full(110000, p4oracle._SPREAD_ENERGY)
    offset = round((1680+round(2.950*FS))*p4rx.SAMPLE_RATE/FS)
    tail_power[344*256-offset+256*np.arange(8, 29)] *= 10
    monkeypatch.setattr(p4oracle, '_despread_window', lambda *a: (spread, tail_power, len(spread)))
    coherence, recovered, first, reason = p4oracle.finish_despread_long(np.ones(200000), FS, prepared)
    assert reason == 'prepared' and len(recovered) == 373 and coherence > .9
    assert first != 0
    assert len(prepared['starts']) == count


@pytest.mark.parametrize('offset_ms,cfo,fade', [
    (-35, -5.01, False), (35, 7.25, False), (0, 0., True),
])
def test_synthetic_timing_cfo_fade_split_equivalence(offset_ms, cfo, fade):
    path = Path(__file__).parent/'fixtures/p4-long-0929/symbols.npz'
    with np.load(path) as fixture:
        symbols = fixture['ve3kpg_0'].copy()
    if fade:
        symbols[340:357] *= .35
    waveform = p4robusttx._render(symbols, FS)
    waveform = (hilbert(waveform)*np.exp(2j*np.pi*cfo*np.arange(len(waveform))/FS)).real
    at = 48000
    first = 24000+round(offset_ms*FS/1000)
    audio = np.zeros(24000+round(3.355*FS))
    audio[first:first+len(waveform)] = waveform[:len(audio)-first]
    origin = at-24000
    full, _ = receiver(at, cfo)._long(audio, origin)
    assert full is not None
    rx = receiver(at, cfo)
    assert rx.prepare_long(audio[:24000+round(3.230*FS)], origin) is not None
    split, _ = rx._long(audio, origin)
    assert split is not None
    assert split.frame == full.frame and split.speedlevel == full.speedlevel
    assert split.stream_sample == full.stream_sample


def test_missing_tail_and_unusable_overlap_defer_full_fallback(monkeypatch):
    at = 1428171
    audio, origin = recording(C, at)
    rx = receiver(at)
    prefix = audio[:24000+round(3.230*FS)]
    assert rx.prepare_long(prefix, origin) is not None
    calls = []
    full = p4oracle.placed_despread

    def counted(*a, **kw):
        if kw.get('count') == 373:
            calls.append(kw)
        return full(*a, **kw)

    monkeypatch.setattr(p4oracle, 'placed_despread', counted)
    assert rx._long(prefix, origin)[0] is None
    assert not calls
    damaged = audio.copy()
    damaged[24000+round(2.950*FS):] = 0
    assert rx._long(damaged, origin)[0] is None
    assert not calls
    assert rx._long(audio, origin, fallback=True)[0] is not None
    assert len(calls) == 1


def test_misaligned_prefix_is_rejected(monkeypatch):
    rng = np.random.default_rng(1010)
    spread = rng.normal(size=110000)+1j*rng.normal(size=110000)
    power = np.full(len(spread), p4oracle._SPREAD_ENERGY)
    misaligned_prefix = dict(onset=1680, lo=0, cfo=0., starts=np.arange(2017),
        symbols=np.ones((2017, 352), complex),
        power=np.full((2017, 352), p4oracle._SPREAD_ENERGY), power_floor=1e-30)
    monkeypatch.setattr(p4oracle, '_despread_window', lambda *a: (spread, power, len(spread)))
    diagnostics = {}
    coherence, symbols, first, reason = p4oracle.finish_despread_long(
        np.ones(200000), FS, misaligned_prefix, diagnostics=diagnostics)
    print(f'misaligned prefix maximum={diagnostics["overlap_correlation"]:.12f} '
          f'floor={p4oracle.LONG_OVERLAP_CORRELATION_MIN:.2f}')
    assert 0 < diagnostics['overlap_correlation'] < p4oracle.LONG_OVERLAP_CORRELATION_MIN
    assert reason == 'overlap' and coherence == 0 and not len(symbols) and first is None


@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('expiry', ['none', 'before', 'placement', 'decode'])
def test_post_decision_sf16_recovery_registers_crc_valid_late_observation(
        monkeypatch, prepared, expiry):
    from hfmodem.tests.shrike.test_p4turn import machine

    at = 1428171
    audio, origin = recording(C, at)
    rx = receiver(at)
    prefix = audio[:24000+round(3.230*FS)]
    rx.prepare_sf8(prefix, origin)
    if prepared:
        assert rx.prepare_long(prefix, origin) is not None
    clock = [0.]
    monkeypatch.setattr(p4receiver, 'monotonic', lambda: clock[0])
    rx.sf8_deadline = lambda _: 1.
    decode = p4robustrx.decode_symbols
    calls = []
    full = p4oracle.placed_despread

    def counted(*a, **kw):
        if kw.get('count') == 373:
            calls.append(kw)
            if expiry == 'placement':
                clock[0] = 2.
        return full(*a, **kw)

    monkeypatch.setattr(p4oracle, 'placed_despread', counted)
    if prepared:
        with monkeypatch.context() as failed:
            failed.setattr(p4robustrx, 'decode_symbols', lambda *a, **kw: None)
            assert rx._long(audio, origin)[0] is None
        assert rx._robust_long_cache[(at, -5.01)]['splice_failed']
    assert not calls
    if expiry == 'before':
        clock[0] = 2.

    def recovered(*a, **kw):
        result = decode(*a, **kw)
        if expiry == 'decode':
            clock[0] = 2.
        return result

    monkeypatch.setattr(p4robustrx, 'decode_symbols', recovered)
    rows = rx.read_window(audio, origin, whole=True, long_fallback=True)
    reg, = [r for r in rows if r.observation.frame is not None]
    obs = reg.observation
    assert reg.outcome == Registered.NEW and obs in rx.observations
    assert obs.kind == Kind.LONG and not obs.sf8 and not obs.provisional
    assert obs.frame.data == DATA_62
    turn, _ = machine(role='irs')
    turn.sf8_eligible = lambda _: False
    due = obs.stream_sample+round(turn.t[Kind.LONG]*FS)
    decision = due-turn.lead
    assert turn.decide(decision) is None
    turn.observe(reg, now=due+1)
    assert turn.adapter.result.delivery == DATA_62
    deferred, = [e for e in turn.link_events if e['event'] == 'answer-deferred']
    assert deferred['at'] == obs.stream_sample
    assert deferred['due'] == due+round(3.75*FS)
    answer = turn.decide(deferred['due']-turn.lead)
    assert answer is not None and answer.stream_sample == deferred['due']
    event = next(e for e in reversed(rx.live_events) if e['event'] == 'long decode')
    assert event['preparation'] == 'unprepared' and event['crc_valid']
    rx.read_window(audio, origin, whole=True, long_fallback=True)
    assert len(calls) == 1
    assert rx._long(np.pad(audio, (0, FS)), origin)[0] is obs
    assert len(calls) == 1


def test_long_negative_cache_matches_moved_due_but_retries_more_support(monkeypatch):
    rx = receiver()
    at = rx.clock.next_due
    calls = []

    def missing(*a, **kw):
        calls.append(kw)
        return .2, np.zeros(373), 0

    monkeypatch.setattr(p4oracle, 'placed_despread', missing)
    audio = np.zeros(round(3.300*FS))
    assert rx._long(audio, at)[0] is None
    rx.clock.next_due += 48
    assert rx._long(audio, at)[0] is None
    assert len(calls) == 1
    assert rx._long(np.zeros(round(3.355*FS)), at)[0] is None
    assert len(calls) == 2
    rx.clock.next_due += 48
    assert rx._long(np.zeros(round(3.4*FS)), at)[0] is None
    assert len(calls) == 2
    rx.clock.cfo_hz += 1
    rx._long(audio, at)
    assert len(calls) == 3
def accepted(at, data, status, kind=Kind.LONG):
    from hfmodem.shrike import p4frame
    from hfmodem.shrike.p4observation import Observation, Registered, RegisteredObservation
    frame = p4frame.decode_field(p4frame.encode_field(data,
        p4frame.Status.from_byte(status)), data_bytes=len(data))
    assert frame is not None and frame.data == data
    return RegisteredObservation(Observation(kind, FS, 0, at, at,
        3.28 if kind in (Kind.LONG, Kind.CHIRP) else .8, 0, .9, frame=frame), Registered.NEW)


def held_machine():
    from hfmodem.tests.shrike.test_p4turn import machine
    m, _ = machine(role='irs', data='body', ladder='none', normal_ack='parity',
                   long_ack='fixed-x27', prekey_lead=.049)
    m.require_long_crc = True
    delivered = []
    m.adapter.host = SimpleNamespace(deliver=delivered.append, supervisor=lambda *a: None)
    return m, delivered


def seed_v60(m, key_at):
    from hfmodem.shrike.p4observation import Observation, Registered, RegisteredObservation
    obs = Observation(Kind.SHORT, FS, 0, key_at-round(.920*FS),
                      key_at-round(.920*FS), .796, 0, .9, value=60, header='A')
    m.observe(RegisteredObservation(obs, Registered.NEW))
    assert m.decide(key_at-m.lead).word == 'U27'
    assert m.keyed(key_at).word == 'U27'


@pytest.mark.parametrize('key_at', [1412434, 1412448])
@pytest.mark.parametrize('refuse_first', [False, True])
def test_b1_recorded_long_latest_held_and_synthetic_refusal(key_at, refuse_first):
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    m, delivered = held_machine()
    seed_v60(m, key_at)
    m.observe(accepted(1428171, DATA_62, 0x62), now=1589338)
    answer = m.decide(1589338)
    assert (answer.kind, answer.word, answer.stream_sample, answer.ack_identity, answer.purpose) == (
        Kind.WORD, 'X27', 1591371, (0x62, DATA_62), None)
    assert m._deferred is None and delivered == [DATA_62]
    assert answer.accepted_target == 1428171 and answer.current_slot_accepted
    if refuse_first:
        m.refused('deadline')
        assert m._last_ack is None and m._packet_ack is None
        at = 1428171+180000
        m.observe(unread(Kind.LONG, at), now=at+158400)
        held = m.decide(at+163200-m.lead)
        assert held.word == 'X27' and held.ack_identity == (0x62, DATA_62)
        assert not m.pending[at]['accepted']
        m.keyed()
    else:
        m.keyed()
    assert m._last_ack[0] == (0x62, DATA_62)
    m.observe(accepted(1788166, DATA_03, 0x03), now=1950320)
    answer = m.decide(1950320)
    assert (answer.word, answer.stream_sample, answer.ack_identity) == ('X27', 1951366, (3, DATA_03))
    live = SimpleNamespace(pos=1950320, clamp_late=lambda at: max(0, 1950320+1796-at))
    grid = onair._MasterGrid(1951366, 60000, 0, packet_n=11712, cs_n=11712, d_max_n=4800)
    assert onair._keyable_slot(live, grid, 0, 0, listen=False, shift=False) != 0
    history = m._last_ack, m._packet_ack
    m.refused('deadline')
    assert m._next is None and (m._last_ack, m._packet_ack) == history
    assert m._deferred[1]['due'] == 2131366
    fallback = m.decide(2131366-m.lead)
    assert (fallback.word, fallback.stream_sample, fallback.purpose) == ('X27', 2131366, None)
    m.keyed()
    at = 1788166+360000
    m.observe(accepted(at, DATA_03, 3), now=at+158400)
    assert m.decide(at+163200-m.lead).word == 'X27'
    m.keyed()
    m.silence(at+343200-m.lead, answer_due=at+343200)
    assert m.decide(at+343200-m.lead).word == 'X27'
    assert delivered == [DATA_62, DATA_03[:39]]


def test_b1_recorded_normal_missed_slots_wrap_and_delivery():
    import json
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    path = CAPTURES/'onair-20261005T002601583605Z-6e2ff851-3bcd-4d2e-b30c-e7258f59cac1/p4-probe.json'
    if not path.is_file():
        pytest.skip('N capture absent')
    probe = json.loads(path.read_text())
    fields = [bytes.fromhex(value) for value in (
        '2072656d61696e696e672077697468204b344d53552028454d3536554f290d5b574c324b2d352e30',
        '2d4232465749484a4d245d0d3b50513a2030313933333431330d434d5320766961204b344d535520',
        '3e0d'+'1e'*38)]
    steps = [(1547665, 1591825, 3, fields[0]), (1607668, 1651828, None, None),
             (1667665, 1711825, None, None), (1727668, 1771828, 0, fields[1]),
             (1787678, 1831838, 0, fields[1]), (1847675, 1891835, 1, fields[2])]
    recorded_keys = {e['at']: e['word'] for e in probe['link_events'] if e['event'] == 'Keyed'}
    m, delivered = held_machine()
    words = []
    for at, due, status, data in steps:
        before = list(delivered)
        if status is None:
            m.observe(unread(Kind.NORMAL, at))
        else:
            m.observe(accepted(at, data, status, Kind.NORMAL))
        answer = m.decide(due-m.lead)
        assert answer.stream_sample == due and answer.purpose is None
        assert answer.current_slot_accepted == (status is not None)
        if status is None:
            assert answer.accepted_target == 1547665 and answer.ack_identity == (3, fields[0])
            assert not m.pending[at]['accepted'] and delivered == before
        assert recorded_keys[due] == answer.word
        words.append(m.keyed().word)
    assert words == ['V27', 'V27', 'V27', 'X27', 'X27', 'V27']
    assert delivered == [fields[0], fields[1], b'>\r']


@pytest.mark.parametrize('when', ['observe', 'decide'])
def test_b1_due_equality_bounded_fallback_and_held_survives(when):
    m, delivered = held_machine()
    reg = accepted(FS, DATA_62, 0x62)
    due = FS+163200
    m.observe(reg, now=due if when == 'observe' else FS+158400)
    assert m.decide(due) is None
    assert m._deferred[1]['due'] == due+180000
    assert m.decide(due+180000-m.lead).word == 'X27'
    m.refused('deadline')
    assert m._deferred is None and m._next is None and m._last_ack is None
    m.silence(due+360000-m.lead, answer_due=due+360000)
    answer = m.decide(due+360000-m.lead)
    assert answer.word == 'X27' and answer.ack_identity == (0x62, DATA_62)
    assert delivered == [DATA_62]


@pytest.mark.parametrize('boundary', ['shutdown', 'epoch'])
def test_b1_cleanup_forgets_held_and_proposed_context(boundary):
    m, _ = held_machine()
    m.observe(accepted(FS, DATA_62, 0x62))
    m.decide(FS+163200-m.lead)
    if boundary == 'shutdown':
        m.refused('shutdown')
        assert m._deferred is None
        m.link_down('shutdown')
    else:
        m._close_receive_epoch('test ownership')
        m._open_receive_epoch(5*FS)
    assert m._held_answer is None and m._accepted_proposal is None


def test_b1_refused_short_supersedes_body_before_any_key():
    from hfmodem.tests.shrike.test_p4turn import body
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    from hfmodem.shrike.p4observation import Registered, RegisteredObservation
    m, _ = held_machine()
    m.observe(body(FS, b'prior'))
    assert m.decide(FS+44160-m.lead).word == 'X27'
    m.keyed()
    at = FS+60000
    obs = replace(body(at).observation, kind=Kind.SHORT, frame=None, value=53)
    m.observe(RegisteredObservation(obs, Registered.NEW))
    assert m.decide(at+44160-m.lead).word == 'U27'
    m.refused('deadline')
    m.observe(unread(Kind.BODY, at+60000))
    answer = m.decide(at+104160-m.lead)
    assert answer.word == 'U27' and answer.accepted_target == at
    assert not answer.current_slot_accepted


def test_b1_chirp_h_unread_steps_repeat_held_v_and_x():
    import json
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    path = CAPTURES/'onair-20261010T230608783923Z-512b8e8a-baa0-44bc-9fa3-efa2590e3590/p4-probe.json'
    if not path.is_file():
        pytest.skip('H capture absent')
    probe = json.loads(path.read_text())
    fields = {r['onset_sample']: r for r in probe['heard']
              if r['kind'] == 'chirp' and r.get('status') is not None}
    m, _ = held_machine()
    m.chirp_ack = 'alternate'
    steps = [(6948617, 7111817), (7128595, 7291795), (None, 7471795),
             (7488624, 7651824), (None, 7832000), (7848587, 8011787)]
    words = []
    target = None
    for at, key_at in steps:
        if at is not None:
            row = fields[at]
            m.observe(accepted(at, bytes.fromhex(row['data']), row['status'], Kind.CHIRP))
            target = at
            due = at+163200
        else:
            at = target+180000
            due = at+163200
            m.observe(unread(Kind.CHIRP, at))
        answer = m.decide(due-m.lead)
        assert answer.purpose is None and answer.accepted_target == target
        if at != target:
            assert not answer.current_slot_accepted and not m.pending[at]['accepted']
        words.append(m.keyed(key_at).word)
    assert words == ['X27', 'V27', 'V27', 'X27', 'X27', 'V27']
    keys = {e['at']: e['word'] for e in probe['link_events'] if e['event'] == 'Keyed'}
    assert words == [keys[key_at] for _, key_at in steps]


def test_b1_prior_unread_enrichment_keeps_cumulative_count_and_accepts_once():
    from hfmodem.shrike.p4observation import Registered, RegisteredObservation
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    m, delivered = held_machine()
    at, due = FS, FS+44160
    old = unread(Kind.NORMAL, at)
    m.observe(old)
    seed = m.decide(due-m.lead)
    assert seed.accepted_target is None and not seed.current_slot_accepted
    assert m.unread == 1
    frame = accepted(at, b'late data'.ljust(40, b'\x1e'), 1, Kind.NORMAL)
    m.observe(RegisteredObservation(frame.observation, Registered.ENRICHED,
                                   old.observation), now=due-m.lead+1)
    answer = m.decide(due-m.lead+1)
    assert answer.word == 'V27' and answer.purpose is None and answer.current_slot_accepted
    assert m.unread == 1 and m._deferred is None and delivered == [b'late data']
    m.keyed()
    assert m.unread == 1


def test_b1_proposed_original_expiry_defers_without_expired_key():
    m, delivered = held_machine()
    m.observe(accepted(FS, DATA_62, 0x62))
    assert m.decide(FS+163200-m.lead).stream_sample == FS+163200
    assert m.decide(FS+163200) is None
    assert m._deferred[1]['due'] == FS+343200
    assert m._last_ack is None and delivered == [DATA_62]


def test_b1_invalid_and_older_observations_cannot_replace_latest_held():
    from hfmodem.shrike.p4observation import Registered, RegisteredObservation
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    m, delivered = held_machine()
    m.observe(accepted(FS, DATA_62, 0x62))
    m.decide(FS+163200-m.lead)
    m.keyed()
    held = m._held_answer
    for reg in [unread(Kind.LONG, FS+180000),
                RegisteredObservation(replace(accepted(FS+180000, DATA_03, 3).observation,
                                               provisional=True), Registered.NEW),
                RegisteredObservation(accepted(FS-1, DATA_62, 0x62).observation, Registered.DUPLICATE)]:
        m.observe(reg)
        assert m._held_answer is held
    m.sf8_eligible = lambda obs: False
    m.observe(RegisteredObservation(replace(accepted(FS+360000, DATA_03, 3).observation,
                                           sf8=True), Registered.NEW))
    assert m._held_answer is held and delivered == [DATA_62]


def test_b1_unread_fallback_key_preserves_receiver_hold_and_slot():
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    m, delivered = held_machine()
    rx = P4Receiver(FS, admitted=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, FS, .330, 3.75, 0.)
    m.observe(accepted(FS, DATA_62, 0x62))
    m.decide(FS+163200-m.lead)
    m.refused('deadline')
    at = FS+180000
    head = replace(unread(Kind.LONG, at).observation, provisional=True)
    rx.observations.append(head)
    m.observe(unread(Kind.LONG, at))
    due = FS+343200
    answer = m.decide(due-m.lead)
    assert answer.accepted_target == FS and answer.current_slot == at
    assert not answer.current_slot_accepted
    before = rx.hold.epoch, rx.hold.last, m.first_body
    sent = m.keyed()
    rx.note_emission(Emission(sent.kind, Role.IRS, sent.stream_sample, .244,
        accepted_target=sent.accepted_target, current_slot=sent.current_slot,
        current_slot_accepted=sent.current_slot_accepted))
    assert (rx.hold.epoch, rx.hold.last, m.first_body) == before
    assert at not in rx._answered
    assert rx.last_emission.stream_sample == due and rx.clock.next_due == due+15840
    assert m._last_ack[0] == (0x62, DATA_62) and delivered == [DATA_62]


@pytest.mark.parametrize('policy', [None, 'alternate', 'parity', 'x27'])
def test_b1_held_chirp_selects_its_own_counter_before_first_key(policy):
    from hfmodem.tests.shrike.test_p4_irs_nak_1002 import unread
    m, _ = held_machine()
    if policy is not None:
        m.chirp_ack = policy
    reg = accepted(FS, b'held counter zero'.ljust(22, b'\x1e'), 0, Kind.CHIRP)
    m.observe(reg)
    answer = m.decide(FS+163200-m.lead)
    expected = 'X27' if policy == 'x27' else 'V27'
    assert answer.word == expected
    m.refused('deadline')
    at = FS+180000
    m.observe(unread(Kind.CHIRP, at))
    repeated = m.decide(at+163200-m.lead)
    assert repeated.word == expected and repeated.ack_identity == answer.ack_identity
    assert not repeated.current_slot_accepted and repeated.accepted_target == FS


@pytest.mark.parametrize('sf8', [True, False])
def test_b1_timely_repeat_of_late_deferred_sf8_long_keys_from_the_answering_copy(sf8):
    from hfmodem.tests.shrike.test_p4turn import machine
    m, _ = machine(role='irs', prekey_lead=.049)
    m._receive_epoch, m._receive_active = 1, True
    data = b'payload'.ljust(141, b'\x1e')
    first = replace(accepted(480000, data, 0x61).observation, placed=True, sf8=True)
    live = {first.stream_sample: True}
    m.sf8_eligible = lambda seen: live.get(seen.stream_sample, False)
    m.observe(RegisteredObservation(first, Registered.NEW), now=480000+163200+10)
    assert m._deferred[1]['due'] == 823200
    live[first.stream_sample] = False
    repeat = replace(first, stream_sample=660000, first_chip=first.first_chip+180000, sf8=sf8)
    live[repeat.stream_sample] = True
    m.observe(RegisteredObservation(repeat, Registered.NEW), now=660000+158400)
    answer = m.decide(823200-m.lead)
    assert (answer.word, answer.stream_sample, answer.current_slot_accepted) == ('X27', 823200, True)
    assert m.sf8_source(answer) is (repeat if sf8 else None)
    assert m.can_key(answer)
    assert m.keyed().ack_identity == (0x61, data)


def test_b1_grant_answering_accepted_packet_defers_by_incoming_cadence():
    from hfmodem.tests.shrike.test_p4turn import body, propose_ack
    m, _ = held_machine()
    grant = propose_ack(m, body(FS, b'', 0x40))
    assert (grant.kind, grant.state) == (Kind.BODY, p4turn.State.ISS_GRANT)
    m.refused('deadline')
    assert m._next is None and m._refused_due is None
    assert m._deferred[0] == FS and m._deferred[1]['due'] == grant.stream_sample+60000
    assert m.decide(grant.stream_sample+60000-m.lead).status == 0x40


def test_b1_grant_answering_accepted_long_defers_by_long_cadence():
    from hfmodem.tests.shrike.test_p4turn import propose_ack
    m, _ = held_machine()
    grant = propose_ack(m, accepted(FS, b'', 0x40))
    assert (grant.kind, grant.state, grant.status, grant.payload) == (
        Kind.BODY, p4turn.State.ISS_GRANT, 0x40, b'')
    m.refused('deadline')
    assert m._next is None and m._refused_due is None
    assert m._deferred[0] == FS and m._deferred[1]['due'] == grant.stream_sample+180000
    repeated = m.decide(grant.stream_sample+180000-m.lead)
    assert (repeated.status, repeated.payload, repeated.stream_sample) == (
        0x40, b'', grant.stream_sample+180000)


@pytest.mark.parametrize('case', ['unanswered grant', 'data'])
def test_b1_fallback_keeps_other_refusals_on_their_own_raster(case):
    m, _ = held_machine()
    at = 10*FS
    if case == 'data':
        m.outbox.arq.on_host_data(b'outbound')
        m._data(at, p4turn.State.ISS_DATA_ACK)
    else:
        m._body(at, 0x40, p4turn.State.ISS_GRANT)
    proposal = m._next
    m.refused('deadline')
    assert m._deferred is None and m._accepted_proposal is None
    assert m._next.stream_sample == m._refused_due == proposal.stream_sample+60000
    assert (m._next.status, m._next.payload) == (proposal.status, proposal.payload)
