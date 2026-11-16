# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from math import gcd
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import hilbert, resample_poly

from hfmodem.shrike import (p4c2, p4coding, p4frame, p4longdecode, p4oracle,
    p4robustc2, p4robustlongtx as tx, p4robustrx, p4robusttx, p4rx, p4sf8rx,
    p4sig, spec)
from hfmodem.shrike.p4receiver import Clock, Expect, Kind, P4Receiver

FX = Path(__file__).parent/'fixtures/p4-long-0929/symbols.npz'
STATUSES = (0, 1, 2, 3, 0x40, 0x41, 0x42, 0x43, 0x60, 0x61, 0x62, 0x63)


def field_for(level, seed=1110, status=0x41):
    data = np.random.default_rng(seed).bytes(tx.GEOMETRY[level].data_bytes)
    wire = p4frame.encode_field(data, p4frame.Status.from_byte(status))
    return wire, p4frame.decode_field(wire, data_bytes=len(data))


def read_symbols(body, level, header):
    return (p4sf8rx.decode_symbols(body) if level == 4 else
            p4robustrx.decode_symbols(np.r_[tx.header_symbols(header), body]))


def recover_audio_body(wave, level, fs=48000):
    """Production placed readers, returning the reference and all data symbols."""
    audio = np.r_[np.zeros(round(.1*fs)), wave, np.zeros(round(.1*fs))]
    if level == 4:
        z = p4rx.audio_to_baseband(audio, fs)
        ob = p4rx.recover_robust(z, reference_start=2880+5120,
            spread_factor=8, data_symbols=696, timing_offsets=range(-8, 9))
        raw = ob.symbols
        r = p4sf8rx.decode_symbols(raw)
    else:
        _, symbols, _ = p4oracle.placed_despread(audio, fs, .1, 0, count=373)
        r = p4robustrx.decode_symbols(symbols)
        assert r is not None
        raw = symbols[r.field_start:r.field_start+349]
    assert r is not None
    return r, raw


def quarter_turns(symbols):
    d = symbols[1:]*symbols[:-1].conj()
    unit = d/np.maximum(abs(d), 1e-30)
    residual = np.angle(np.mean(unit**4))/4
    return np.rint(np.angle(d*np.exp(-1j*residual))/(np.pi/2)).astype(int) % 4


@pytest.mark.parametrize('level', [2, 3, 4])
@pytest.mark.parametrize('fs', [48000, 8000])
@pytest.mark.parametrize('mode', ['default', 'repeat', 'override'])
def test_noiseless_audio_all_fields(level, fs, mode):
    field, frame = field_for(level)
    options = ({'repeat': True} if mode == 'repeat' else
               {'repeat': True, 'header': {2: 3, 3: 4, 4: 2}[level]}
               if mode == 'override' else {})
    wave = tx.robust_long(frame.data, frame.status.byte, level, fs=fs, **options)
    r, raw = recover_audio_body(wave, level, fs)
    bits = tx.coded_bits(field, level)
    D = tx.GEOMETRY[level].systematic
    P = (len(bits)-D)//2
    # Separate the termination tail so a valid systematic CRC cannot hide it.
    errors = [int(x.sum()) for x in np.split(bits != (r.fields < 0),
                                             [D-3, D, D+P])]
    turns = quarter_turns(raw)
    expected = quarter_turns(tx.body_symbols(bits, level))
    metric = dict(level=level, fs=fs, mode=mode, fields=len(bits),
                  errors=errors, differential_mismatches=int(np.sum(turns != expected)))
    print('\nLOOPBACK_METRICS '+json.dumps(metric, sort_keys=True), flush=True)
    assert r.frame == frame
    np.testing.assert_array_equal(r.fields < 0, bits)
    np.testing.assert_array_equal(turns, expected)


@pytest.mark.parametrize('level', [2, 3, 4])
def test_geometry_tail_and_puncture(level):
    g = tx.GEOMETRY[level]
    field, _ = field_for(level)
    bits = tx.coded_bits(field, level)
    assert bits.dtype == np.uint8 and len(bits) == (1392 if level == 4 else 696)
    s = bits[:g.systematic]
    np.testing.assert_array_equal(s[:8*len(field)],
        np.unpackbits(np.frombuffer(field, np.uint8), bitorder='little'))
    assert not np.any(s[8*len(field):-3])
    first = p4coding.rsc_encode(s, initial_state=(0, 0, 0))
    assert first.final_state == (0, 0, 0)
    order = (np.concatenate([np.arange(j, 580, 24) for j in range(24)])
             if level == 3 else p4c2.permutation(g.systematic))
    second = p4coding.rsc_encode(s[order], initial_state=(0, 0, 0))
    period, phase = (2, 1) if level == 2 else (10, 4)
    np.testing.assert_array_equal(bits[g.systematic:],
        np.r_[first.parity[::period], second.parity[phase::period]])
    audio = tx.robust_long(b'', 0x40, level)
    assert len(audio) == round(tx.EMISSION_S[level]*48000)+1
    assert abs((len(audio)-1)/48000-tx.EMISSION_S[level]) <= 1/48000
    # Public empty rendering must encode all IDLE, never placement's P3 template.
    np.testing.assert_array_equal(audio,
        tx.robust_long(bytes([spec.IDLE])*g.data_bytes, 0x40, level))
    np.testing.assert_array_equal(tx.robust_long(b'a', 3, level),
        tx.robust_long(b'a'+bytes([spec.IDLE])*(g.data_bytes-1), 3, level))
    with pytest.raises(ValueError):
        tx.robust_long(b'x'*(g.data_bytes+1), 0, level)
    with pytest.raises(ValueError):
        tx.coded_bits(field[:-1], level)
    with pytest.raises(ValueError):
        tx.body_symbols(bits[:-1], level)


@pytest.mark.parametrize('level', [None, 0, 1, 5, 2.0, True])
def test_invalid_level(level):
    with pytest.raises(ValueError):
        tx.robust_long(b'', 0, level)


@pytest.mark.parametrize('header', ['A', 'B', 1, 5])
def test_invalid_header(header):
    with pytest.raises(ValueError):
        tx.robust_long(b'', 0, 2, header=header)


@pytest.mark.parametrize('header', ['RQ', 2, 3, 4])
def test_header_reference_anchor(header):
    steps = tx.HEADER_STEPS[header]
    assert len(steps) == 19 and all(type(step) is int for step in steps)
    head = tx.header_symbols(header)
    assert head.shape == (20,)
    np.testing.assert_allclose(head[1:]*head[:-1].conj(),
                               np.exp(1j*np.radians(steps)), atol=1e-14)
    assert head[-1] == 1 and tx.REFERENCE_STEP == 0


@pytest.mark.parametrize('name,expected', [('n9vc_0', (0, 0, 0)),
    ('n9vc_1', (0, 0, 0)), ('ve3kpg_0', (0, 0, 0)), ('ve3kpg_1', (7, 0, 0))])
def test_all_fixture_bits(name, expected):
    with np.load(FX) as fixture:
        r = p4robustrx.decode_symbols(fixture[name])
    level = r.speedlevel
    bits = tx.coded_bits(p4frame.encode_field(r.frame.data, r.frame.status), level)
    D = tx.GEOMETRY[level].systematic
    P = (len(bits)-D)//2
    errors = np.split(bits != (r.fields < 0), [D, D+P])
    assert tuple(int(e.sum()) for e in errors) == expected


@pytest.mark.parametrize('level', [2, 3, 4])
@pytest.mark.parametrize('header', ['RQ', 2, 3, 4])
def test_seeded_symbol_loopback(level, header):
    for seed in range(64):
        field, frame = field_for(level, seed, STATUSES[seed % len(STATUSES)])
        bits = tx.coded_bits(field, level)
        body = tx.body_symbols(bits, level)
        r = read_symbols(body, level, header)
        assert r is not None and r.frame == frame
        np.testing.assert_array_equal(r.fields < 0, bits)
        if level == 4:
            assert r.quarter_turn == 0 and r.correction == 'systematic'
        else:
            assert (r.speedlevel, r.field_start, r.corrected_bits) == (level, 20, 0)
        if level in (2, 4):
            result = p4robustc2.decode_placed(body, speedlevel=level, iterations=2)
            assert result.reason == 'unique' and result.frame == frame
            rows = [p for p in result.provenance if p.quarter_turn == 0]
            assert {p.method for p in rows} == {'systematic', 'c1-only', 'turbo'}
            assert all(p.crc_valid for p in rows)


@pytest.mark.parametrize('level', [2, 3, 4])
@pytest.mark.parametrize('fs', [48000, 8000])
def test_fixed_gain_peak_and_repeat(level, fs):
    peak = 0
    for seed in range(64):
        _, frame = field_for(level, seed)
        new = tx.robust_long(frame.data, frame.status.byte, level, fs=fs)
        repeat = tx.robust_long(frame.data, frame.status.byte, level, repeat=True, fs=fs)
        for header in tx.HEADER_STEPS:
            wave = tx.robust_long(frame.data, frame.status.byte, level, header=header, fs=fs)
            peak = max(peak, float(np.max(abs(wave))))
        # 320 chips plus pulse and resampling filter support; no peak rescaling.
        end_overlap = int(np.ceil((320/1800 + 128/28800 + 20/min(fs, 28800))*fs))
        assert not np.array_equal(new[:end_overlap], repeat[:end_overlap])
        np.testing.assert_array_equal(new[end_overlap:], repeat[end_overlap:])
        np.testing.assert_array_equal(repeat,
            tx.robust_long(frame.data, frame.status.byte, level, header='RQ', fs=fs))
    assert .8 <= peak <= 1.0
    print(f'peak SL{level} {fs} Hz: {peak:.12f}')


@pytest.mark.parametrize('fs', [48000, 8000])
def test_shared_shaper_preserves_short_samples(fs):
    for header in ('A', 'B'):
        symbols = p4robusttx.phasors(p4robusttx.short_steps(17, header))
        chips = (symbols[:, None]*p4sig.SPREAD16).ravel()
        impulses = np.zeros(16*(len(chips)-1)+1, complex)
        impulses[::16] = chips
        g = gcd(fs, 28800)
        bb = resample_poly(np.convolve(impulses, p4rx.RRC_TAPS), fs//g, 28800//g)
        old = (bb*np.exp(2j*np.pi*1500*np.arange(len(bb))/fs)).real
        old /= max(abs(old))
        np.testing.assert_array_equal(p4robusttx.robust_short(17, header, fs), old)


@pytest.mark.parametrize('level', [2, 3, 4])
@pytest.mark.parametrize('cfo,offset', [(0, 0), (3, -.035), (-3, .035)])
@pytest.mark.parametrize('esn0', [24, 30])
def test_audio_loopback(level, cfo, offset, esn0):
    _, frame = field_for(level)
    wave = tx.robust_long(frame.data, frame.status.byte, level)
    wave = (hilbert(wave)*np.exp(2j*np.pi*cfo*np.arange(len(wave))/48000)).real
    onset = .15+offset
    audio = np.r_[np.zeros(round(onset*48000)), wave, np.zeros(4800)]
    # Real AWGN: Es/N0 uses a spread data symbol's energy, N0=2*sigma^2.
    symbol_samples = 48000*tx.GEOMETRY[level].spread/1800
    variance = np.mean(wave**2)*symbol_samples/(2*10**(esn0/10))
    audio += np.random.default_rng(1110).normal(0, np.sqrt(variance), len(audio))
    if level == 4:
        z = p4rx.audio_to_baseband(audio, 48000)
        ob = p4rx.recover_robust(z, reference_start=round(.15*28800)+5120,
            spread_factor=8, data_symbols=696, timing_offsets=range(-1024, 1025), cfo_hz=(cfo,))
        r = p4sf8rx.decode_symbols(ob.symbols)
    else:
        _, symbols, _ = p4oracle.placed_despread(audio, 48000, .15, cfo, count=373)
        r = p4robustrx.decode_symbols(symbols)
    assert r is not None and r.frame == frame


@pytest.mark.parametrize('level', [2, 3, 4])
def test_receiver_entry_on_long_raster(level):
    _, frame = field_for(level)
    at = 24000
    wave = tx.robust_long(frame.data, frame.status.byte, level)
    audio = np.r_[np.zeros(at), wave, np.zeros(4800)]
    # Channel noise keeps the near-zero pulse tail realistic for the SF8 energy guard.
    variance = np.mean(wave**2) / 10**(20/10) * (48000/2)/2400
    audio += np.random.default_rng(1110).normal(0, np.sqrt(variance), len(audio))
    rx = P4Receiver(48000, admitted=True, cfo_hz=0, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.PACKET, at, .33, 3.75, 0)
    rx.prepare_sf8(audio[:at+round(3.230*48000)], 0)
    records = rx.read_window(audio, 0, whole=True)
    longs = [r.observation for r in records if r.observation.kind == Kind.LONG
             and r.observation.frame == frame]
    assert len(longs) == 1
    assert longs[0].sf8 if level == 4 else longs[0].speedlevel == str(level)
    print(f'P4Receiver SL{level}: Kind.LONG, frame equal, '
          f'sf8={longs[0].sf8}, speedlevel={longs[0].speedlevel}')


def noisy_differentials(bits, level, seed):
    body = tx.body_symbols(bits, level)
    differential = body[1:]*body[:-1].conj()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(differential), 12 if level == 2 else 6, replace=False)
    differential[indices] += rng.normal(size=len(indices)) + 1j*rng.normal(size=len(indices))
    return np.r_[1, np.cumprod(differential/abs(differential))]


@pytest.mark.parametrize('level,seed', [(2, 1), (3, 0), (4, 0)])
def test_c2_is_decisive_under_seeded_noise(level, seed, monkeypatch):
    from dataclasses import replace

    wire, frame = field_for(level)
    bits = tx.coded_bits(wire, level)
    D = tx.GEOMETRY[level].systematic
    P = (len(bits)-D)//2
    body = noisy_differentials(bits, level, seed)
    bad = bits.copy()
    order = np.random.default_rng(711).permutation(D)
    parity = p4coding.rsc_encode(bits[:D][order], initial_state=(0, 0, 0)).parity
    period, phase = (2, 1) if level == 2 else (10, 4)
    bad[D+P:] = parity[phase::period]
    wrong = noisy_differentials(bad, level, seed)
    if level in (2, 4):
        result = p4robustc2.decode_placed(body, speedlevel=level, iterations=2)
        assert result.reason == 'unique' and result.frame == frame
        assert not any(p.crc_valid for p in result.provenance if p.method != 'turbo')
        assert any(p.crc_valid for p in result.provenance if p.method == 'turbo')
        assert p4robustc2.decode_placed(wrong, speedlevel=level, iterations=2).frame is None
        original = p4robustc2._eligible_members

        def erased(raw, geometry):
            members = original(raw, geometry)
            output = []
            for member in members:
                fields = member.fields.copy()
                fields[D+P:] = 0
                np.testing.assert_array_equal(fields[:D+P], member.fields[:D+P])
                output.append(replace(member, fields=fields))
            return tuple(output)

        monkeypatch.setattr(p4robustc2, '_eligible_members', erased)
        assert p4robustc2.decode_placed(body, speedlevel=level, iterations=2).frame is None
    else:
        raw = np.r_[tx.header_symbols(3), body]
        result = p4robustrx.decode_symbols(raw, iterations=2)
        assert result is not None and result.frame == frame and result.corrected_bits > 0
        assert p4robustrx.decode_symbols(raw, iterations=0) is None
        members = p4robustrx.build_members(raw)
        exact = p4longdecode.LongSearch(members, iterations=2).advance()
        assert exact.state == 'unique' and exact.frame == frame
        assert not any(p.crc_valid for p in exact.provenance if p.method != 'turbo')
        assert any(p.crc_valid for p in exact.provenance if p.method == 'turbo')
        assert p4robustrx.decode_symbols(np.r_[tx.header_symbols(3), wrong], iterations=2) is None
        original = p4robustrx.build_members

        def erased(*args, **kwargs):
            output = []
            for member in original(*args, **kwargs):
                fields = member.fields.copy()
                n, parity = member.geometry.systematic, member.geometry.parity
                fields[n+parity:] = 0
                np.testing.assert_array_equal(fields[:n+parity], member.fields[:n+parity])
                output.append(replace(member, fields=fields))
            return tuple(output)

        # Erase only the observed C2 input; both backends still run the real
        # LONG search, including its C1-only and turbo tiers.
        monkeypatch.setattr(p4robustrx, 'build_members', erased)
        assert p4robustrx.decode_symbols(raw, iterations=2) is None
